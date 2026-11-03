#!/usr/bin/env python3
"""Run a fresh desktop EGAS-SLAM experiment, or render an existing saved map.

Source templates remain untouched. Native trainer startup clears its configured
workspace, so every training invocation gets an exclusively created run parent
and a new map child. Receipts and logs are outside that clearable child.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import threading
import uuid

import yaml

PROJECT = Path(__file__).resolve().parents[1]
PROFILES = PROJECT / "configs/desktop/profiles.yaml"


def local_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if any(path == root or root in path.parents
           for root in (Path('/media'), Path('/mnt'), Path('/run/media'))):
        raise ValueError(f"Desktop work must be local; external mount rejected: {path}")
    return path


def workspace_path(root: Path, *parts: str) -> Path:
    path = local_path(root.joinpath(*parts))
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Output escapes the dedicated workspace: {path}')
    return path

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def capture(command: list[str]) -> str:
    try:
        result = subprocess.run(command, cwd=PROJECT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=20)
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def select_profile(dataset: str, scene: str) -> dict:
    profile = yaml.safe_load(PROFILES.read_text())[dataset]
    if scene not in profile['scenes']:
        raise ValueError(f"Unknown {dataset} scene {scene}; choose {profile['scenes']}")
    return profile


def find_input(profile: dict, scene: str, root: Path, override: str | None) -> Path:
    if override:
        return local_path(override)
    name = profile.get('sequence_names', {}).get(scene, scene)
    candidates = [root / 'data' / profile['prepared_directory'] / name]
    if profile.get('legacy_prepared_root'):
        candidates.append(Path.home() / profile['legacy_prepared_root'] / name)
    for candidate in candidates:
        if candidate.is_dir():
            return local_path(candidate)
    raise FileNotFoundError(
        f"No prepared dataset found at {candidates}. For Replica run the prepare "
        "subcommand, or supply --input-dir with a local prepared scene.")


def validate_input(config: dict) -> dict:
    reader = config['READER']
    source = local_path(reader['input_dir'])
    indices = range(reader['start_frame'], reader['end_frame'] + 1, reader['frame_step'])
    import cv2
    color_first = None
    depth_first = None
    for i in indices:
        for kind, directory, prefix, extension in (
            ('color', 'image_path', 'image_prefix', 'image_extension'),
            ('depth', 'depth_path', 'depth_prefix', 'depth_extension')):
            default_prefix = 'frame' if kind == 'color' else 'depth'
            default_extension = '.jpg' if kind == 'color' else '.png'
            filename = (source / reader[directory] /
                        f"{reader.get(prefix, default_prefix)}{i:06d}"
                        f"{reader.get(extension, default_extension)}")
            if not filename.is_file():
                raise FileNotFoundError(f"Missing {kind} frame: {filename}")
            local_path(filename)
            if i == reader['start_frame']:
                decoded = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
                if decoded is None:
                    raise ValueError(f"Cannot decode {filename}")
                shape = [decoded.shape[1], decoded.shape[0]]
                if shape != reader['image_shape']:
                    raise ValueError(f"{filename}: image is {shape}, configuration expects "
                                     f"{reader['image_shape']}; do not halve calibration twice.")
                if kind == 'color':
                    color_first = shape
                else:
                    depth_first = shape
    factor = float(reader['downscale_factor'])
    return dict(input_dir=str(source), frame_count=len(indices),
                source_color_shape=color_first, source_depth_shape=depth_first,
                processing_shape=[round(v / factor) for v in reader['image_shape']],
                processing_intrinsics=[v / factor for v in reader['intrinsics']],
                pose_source=reader['pose_source'], ground_truth_used_for_tracking=False,
                desktop_preparation_manifest=(str(source / 'desktop_preparation.json')
                    if (source / 'desktop_preparation.json').is_file() else None),
                desktop_preparation_sha256=(digest(source / 'desktop_preparation.json')
                    if (source / 'desktop_preparation.json').is_file() else None))


def apply_eval_differences(target: dict, original: dict, evaluation: dict) -> None:
    """Apply only final train-to-eval differences to the actual training config."""
    for key, value in evaluation.items():
        if isinstance(value, dict):
            apply_eval_differences(target.setdefault(key, {}), original.get(key, {}), value)
        elif original.get(key) != value:
            target[key] = copy.deepcopy(value)


class Telemetry:
    """Asynchronous process-RSS/device sampling; never synchronises CUDA."""

    def __init__(self, pid: int, record: Path, device: int):
        self.pid, self.record, self.device = pid, record, device
        self.stop_event = threading.Event()
        self.summary = dict(samples=0, gpu_samples=0, gpu_query_errors=0,
                            peak_process_rss_mib=0.0, peak_device_used_mib=0.0,
                            peak_device_utilization_percent=0.0,
                            peak_device_temperature_c=0.0, peak_device_power_w=0.0,
                            scope='Process RSS is the native PID only, not subprocesses. '
                            'GPU values cover the whole device, including other processes. '
                            'One-second sampling can miss short peaks; these are observed '
                            'sample maxima, not allocation accounting or benchmark cgroup peaks.')
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> dict:
        self.stop_event.set()
        self.thread.join(timeout=4)
        return dict(self.summary)

    def sample(self) -> None:
        try:
            with (self.record / 'telemetry.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['utc', 'process_rss_mib', 'device_timestamp',
                                 'device_memory_used_mib', 'device_utilization_percent',
                                 'device_temperature_c', 'device_power_w', 'query_error'])
                while not self.stop_event.is_set():
                    row = [dt.datetime.now(dt.timezone.utc).isoformat(), '', '', '', '', '', '', '']
                    try:
                        for line in Path(f'/proc/{self.pid}/status').read_text().splitlines():
                            if line.startswith('VmRSS:'):
                                row[1] = float(line.split()[1]) / 1024
                                self.summary['peak_process_rss_mib'] = max(self.summary['peak_process_rss_mib'], row[1])
                                break
                    except (OSError, ValueError):
                        pass
                    try:
                        query = subprocess.run(
                            ['nvidia-smi', '-i', str(self.device),
                             '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                        if query.returncode:
                            raise ValueError(query.stderr.strip() or f'nvidia-smi exit {query.returncode}')
                        values = next(csv.reader(query.stdout.splitlines()))
                        if len(values) != 5:
                            raise ValueError('Unexpected nvidia-smi fields')
                        row[2:7] = [value.strip() for value in values]
                        for i, key in enumerate(('peak_device_used_mib', 'peak_device_utilization_percent',
                                                 'peak_device_temperature_c', 'peak_device_power_w'), start=3):
                            try:
                                self.summary[key] = max(self.summary[key], float(row[i]))
                            except ValueError:
                                pass
                        self.summary['gpu_samples'] += 1
                    except (OSError, ValueError, StopIteration, subprocess.TimeoutExpired) as exc:
                        row[7] = str(exc)
                        self.summary['gpu_query_errors'] += 1
                    self.summary['samples'] += 1
                    writer.writerow(row)
                    stream.flush()
                    self.stop_event.wait(1)
        except Exception as exc:
            # Measurement failure must not terminate the native experiment.
            self.summary['monitor_error'] = str(exc)


def execute(args: argparse.Namespace, config: dict, record: Path,
            template: Path, dataset: str, scene: str, purpose: str) -> int:
    support = validate_input(config)
    binary = local_path(args.binary or PROJECT / 'build_desktop/slam_trainer')
    if not args.dry_run and (not binary.is_file() or not os.access(binary, os.X_OK)):
        raise FileNotFoundError(f"Build the desktop executable first: {binary}")
    record.mkdir(parents=True, exist_ok=True)
    config_path = record / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    environment = os.environ.copy()
    environment.update(NVIDIA_TF32_OVERRIDE='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [str(binary), str(config_path)]
    receipt = dict(status='configured', purpose=purpose, dataset=dataset, scene=scene,
                   created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   command=command, cwd=str(PROJECT), config=str(config_path),
                   config_sha256=digest(config_path), final_template=str(template),
                   final_template_sha256=digest(template),
                   binary_sha256=digest(binary) if binary.is_file() else None,
                   source_commit=capture(['git', 'rev-parse', 'HEAD']),
                   source_status=capture(['git', 'status', '--short']),
                   source_tracked_diff_sha256=hashlib.sha256(capture(['git', 'diff', 'HEAD']).encode()).hexdigest(),
                   environment={k: environment.get(k) for k in (
                       'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'LD_LIBRARY_PATH',
                       'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                   gpu=capture(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
                   input_support=support, workspace_dir=config['workspace_dir'],
                   save_after_train=config['PIPE']['save_after_train'],
                   measurement_note='New desktop run; wall time includes startup and export. '
                   'Native timing logs report separately scoped processing time. '
                   'Half-resolution desktop rates are not Thor benchmark rates. '
                   'Asynchronous one-second process/device telemetry remains enabled; '
                   'its overhead has not been independently bounded.')
    if config['PIPE'].get('demo_capture', {}).get('enabled', False):
        receipt['measurement_note'] += (' Demo capture adds GPU meshing, rendering, '
            'synchronisation and disk IO. These timings are not benchmark throughput.')
    atomic_json(record / 'run.json', receipt)
    print(f"Configuration: {config_path}\nRun record: {record}\n"
          f"Map workspace: {config['workspace_dir']}\n"
          f"Input: {support['frame_count']} frames; processing {support['processing_shape']}", flush=True)
    if args.dry_run:
        print('Configuration validated; --dry-run did not start SLAM.')
        return 0
    # The root-owned environment wrapper resolves desktop Torch runtime libraries.
    # Source scripts/desktop_env.sh before invoking this script (see README).
    start = time.monotonic()
    receipt.update(status='running', started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(record / 'run.json', receipt)
    code = 1
    monitor = None
    try:
        with (record / 'native.log').open('w') as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            monitor = Telemetry(process.pid, record, int(config.get('dev_id', 0)))
            monitor.start()
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code = 130
    except Exception as exc:
        receipt['launch_error'] = str(exc)
        raise
    finally:
        if monitor is not None:
            receipt['telemetry'] = monitor.finish()
        receipt.update(status='completed' if code == 0 else 'failed', returncode=code,
                       command_wall_seconds=time.monotonic() - start,
                       finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(record / 'run.json', receipt)
    print(f"Run {'completed' if code == 0 else 'failed'}; record: {record / 'run.json'}", flush=True)
    return code


def prepare(args: argparse.Namespace, root: Path) -> int:
    profile = select_profile('replica', args.scene)
    raw = local_path(args.raw_root or Path.home() / profile['legacy_raw_root'])
    full_target = workspace_path(root, 'data', 'replica_prepared')
    full_candidates = [full_target / args.scene,
                       Path.home() / profile['legacy_full_root'] / args.scene]
    full = next((local_path(p) for p in full_candidates if p.is_dir()), None)
    if full is None:
        local_path(raw / args.scene)
        command = [sys.executable, str(PROJECT / 'scripts/replica_prepare_scene.py'),
                   '--raw-root', str(raw), '--scene', args.scene,
                   '--output-root', str(full_target), '--target-frames', '2000']
        subprocess.run(command, cwd=PROJECT, check=True)
        full = local_path(full_target / args.scene)
    template = PROJECT / profile['train_template'].format(scene=args.scene)
    full_config = yaml.safe_load(template.read_text())
    full_config['READER']['input_dir'] = str(full)
    support = validate_input(full_config)
    target = workspace_path(root, 'data', profile['prepared_directory'], args.scene)
    if target.exists():
        raise FileExistsError(f'Refusing to replace prepared data: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + '.preparing_' + uuid.uuid4().hex[:8])
    helper = PROJECT / 'scripts/replica_prepare_scene.py'
    spec = importlib.util.spec_from_file_location('egas_replica_prepare', helper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.prepare_half(full, staging, support['frame_count'], False)
    metadata = dict(scene=args.scene, source_full_resolution=str(full),
                    frame_count=support['frame_count'], image_shape=profile['image_shape'],
                    intrinsics=profile['intrinsics'], downscale_factor=1,
                    color_resize='OpenCV INTER_AREA; JPEG re-encoded by inherited prepare_half',
                    depth_resize='OpenCV INTER_NEAREST; depth scale unchanged at 6553.5',
                    helper_sha256=digest(helper),
                    reason='Frozen TSDF engine initialises from reader metadata, so both TSDF '
                    'and Gaussian cameras receive physically half-resolution data and calibration.',
                    created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(staging / 'desktop_preparation.json', metadata)
    staging.rename(target)
    print(f'Prepared local half-resolution Replica: {target}', flush=True)
    return 0


def train(args: argparse.Namespace, root: Path) -> int:
    profile = select_profile(args.dataset, args.scene)
    if args.resolution == 'full' and args.dataset == 'replica':
        profile = copy.deepcopy(profile)
        profile.update(image_shape=[1200, 680], intrinsics=[600, 600, 599.5, 339.5],
                       prepared_directory='replica_prepared',
                       legacy_prepared_root=profile['legacy_full_root'])
    template = PROJECT / profile['train_template'].format(scene=args.scene)
    config = yaml.safe_load(template.read_text())
    config['READER']['input_dir'] = str(find_input(profile, args.scene, root, args.input_dir))
    config['READER']['downscale_factor'] = profile['downscale_factor']
    for key in ('image_shape', 'intrinsics'):
        if key in profile:
            config['READER'][key] = copy.deepcopy(profile[key])
    if args.frames is not None:
        available = (config['READER']['end_frame'] - config['READER']['start_frame']) // config['READER']['frame_step'] + 1
        if not 2 <= args.frames <= available:
            raise ValueError(f"--frames must be between 2 and {available}")
        config['READER']['end_frame'] = config['READER']['start_frame'] + (args.frames - 1) * config['READER']['frame_step']
    if args.no_save:
        config['PIPE']['save_after_train'] = False
    if args.capture_every:
        if args.capture_every < 1:
            raise ValueError('--capture-every must be positive')
        config['PIPE']['demo_capture'] = dict(enabled=True, interval=args.capture_every,
                                             scene=args.scene, dataset=args.dataset)
    # Validate all inputs before reserving a new experiment directory.
    validate_input(config)
    label = args.name or f"{args.dataset}_{args.scene}_{stamp()}_{uuid.uuid4().hex[:6]}"
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', label):
        raise ValueError('--name must be a simple name, not a path')
    run = workspace_path(root, 'runs', label)
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir()  # exclusive: prevents reruns from clearing another experiment
    config['workspace_dir'] = str(workspace_path(root, 'runs', label, 'map'))
    return execute(args, config, run / 'record', template, args.dataset, args.scene, 'train')


def evaluate(args: argparse.Namespace, root: Path) -> int:
    run = local_path(args.run)
    if not run.is_relative_to(workspace_path(root, 'runs')):
        raise ValueError('Evaluation must name a run beneath the selected workspace/runs directory')
    for part in ('record', 'record/run.json', 'record/config.yaml', 'map', 'evaluations'):
        checked = local_path(run / part)
        if not checked.is_relative_to(run):
            raise ValueError(f'Saved run path escapes through a symbolic link: {checked}')
    record = json.loads((run / 'record/run.json').read_text())
    if record.get('purpose') != 'train' or record.get('status') != 'completed' or not record.get('save_after_train'):
        raise ValueError('Evaluation requires a successfully completed training run with saved maps')
    dataset, scene = record['dataset'], record['scene']
    profile = select_profile(dataset, scene)
    if 'eval_template' not in profile:
        raise ValueError('This launcher currently provides saved-map rendering for Replica. '
                         'TUM tracking poses are available directly in map/val/corrected_pose.')
    config_path = run / 'record/config.yaml'
    if digest(config_path) != record.get('config_sha256'):
        raise ValueError('Training configuration changed after launch; refusing saved-map evaluation')
    config = yaml.safe_load(config_path.read_text())
    if Path(config['workspace_dir']).resolve() != run / 'map':
        raise ValueError('Recorded map workspace does not match this run')
    train_template = PROJECT / profile['train_template'].format(scene=scene)
    template = PROJECT / profile['eval_template'].format(scene=scene)
    apply_eval_differences(config, yaml.safe_load(train_template.read_text()), yaml.safe_load(template.read_text()))
    if not (run / 'map/submaps/manifest.json').is_file() or not (run / 'map/tsdf_engine').is_dir():
        raise FileNotFoundError('This run has no complete saved hybrid map and TSDF checkpoint')
    if config['work_mode'] != 'eval':
        raise ValueError('Evaluation template did not select eval mode')
    reader = config['READER']
    available = list(range(reader['start_frame'], reader['end_frame'] + 1, reader['frame_step']))
    if args.views < 0:
        raise ValueError('--views must be nonnegative; 0 selects all recorded frames')
    count = min(args.views or len(available), len(available))
    if count == 1:
        frames = [available[len(available) // 2]]
    else:
        frames = [available[round(i * (len(available) - 1) / (count - 1))] for i in range(count)]
    config['PIPE']['eval_frame_indices'] = frames
    label = f'eval_{stamp()}_{uuid.uuid4().hex[:6]}'
    config['PIPE']['eval_path'] = label
    return execute(args, config, run / 'evaluations' / label, template, dataset, scene, 'eval')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'eval', 'prepare'):
        p = sub.add_parser(name)
        p.add_argument('--workspace-root', default=str(Path.home() / 'egas_workspace'))
        if name != 'prepare':
            p.add_argument('--binary', help='Default: build_desktop/slam_trainer')
            p.add_argument('--dry-run', action='store_true', help='Validate and save configuration without running SLAM')
        if name == 'train':
            p.add_argument('--dataset', choices=['replica', 'tum'], required=True)
            p.add_argument('--scene', required=True)
            p.add_argument('--input-dir', help='Prepared local scene matching --resolution; TUM always uses native resolution')
            p.add_argument('--resolution', choices=['half', 'full'], default='half',
                           help='Replica processing resolution; TUM always uses native resolution')
            p.add_argument('--capture-every', type=int, default=0,
                           help='Enable demonstration capture every N frames; extra work is not benchmark timing')
            p.add_argument('--frames', type=int, help='First N frames; omit for complete scene')
            p.add_argument('--no-save', action='store_true', help='Skip large map export; no later rendering of this run')
            p.add_argument('--name', help='Unique output name; an existing name is always refused')
        elif name == 'eval':
            p.add_argument('--run', required=True, help='Training run directory containing record/ and map/')
            p.add_argument('--views', type=int, default=20, help='Uniform observed views; default 20, 0 means all')
        else:
            p.add_argument('--scene', required=True)
            p.add_argument('--raw-root', help='Local raw Replica root; default existing GPS_SLAM data')
    args = parser.parse_args()
    root = local_path(args.workspace_root)
    if root in (Path('/'), Path.home(), PROJECT) or root in PROJECT.parents:
        raise ValueError('Use a dedicated workspace directory')
    if args.command == 'prepare':
        return prepare(args, root)
    if args.command == 'train':
        return train(args, root)
    return evaluate(args, root)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
