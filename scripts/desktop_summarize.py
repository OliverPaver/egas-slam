#!/usr/bin/env python3
"""CPU-only validation of completed desktop runs; writes summary JSON, no native outputs.

Usage: ./egas summary RUN [RUN ...] [--aggregate PATH]
Only record/validation_summary.json and optional aggregate JSON are written.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import cv2
import numpy as np
import yaml
from ate_general import align
from pose_evaluation_common import read_poses, pose_id


def local(value):
    path = Path(value).expanduser().resolve()
    if any(path == r or r in path.parents for r in
           (Path('/media'), Path('/mnt'), Path('/run/media'))):
        raise ValueError(f'Local paths required: {path}')
    return path


def write_json(path, data):
    path = local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def csv_rows(path, required=True):
    if not path.exists() and not required:
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Non-finite output: {value!r}')
    return result


def finite_tree(value, name=''):
    if isinstance(value, float):
        finite(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, name + '.' + str(key))
    elif isinstance(value, list):
        for item in value:
            finite_tree(item, name)


def statistics(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError('Missing/non-finite timing samples')
    return dict(count=len(array), mean=float(array.mean()), median=float(np.median(array)),
                p95=float(np.percentile(array, 95)), p99=float(np.percentile(array, 99)),
                maximum=float(array.max()), minimum=float(array.min()))


def exact_rows(rows, expected, name):
    actual = [int(row['frame_id']) for row in rows]
    if actual != expected:
        raise ValueError(f'{name}: expected ordered {len(expected)} frame IDs, found {len(actual)}')
    if 'sequence_id' in rows[0]:
        if [int(row['sequence_id']) for row in rows] != list(range(len(expected))):
            raise ValueError(f'{name}: sequence IDs not contiguous from zero')


def pose_summary(directory, expected, gt):
    poses = read_poses(directory, label=directory.name)
    if set(poses) != set(expected):
        raise ValueError(f'{directory}: pose coverage differs from configured frames')
    rotations = np.stack([poses[i][:3, :3] for i in expected])
    ortho = float(np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))))
    det = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    if ortho > 0.01 or det > 0.01:
        raise ValueError(f'{directory}: invalid rotation matrices')
    result = dict(poses=len(poses), exact_configured_coverage=True, finite=True,
                  maximum_rotation_orthogonality_error=ortho,
                  maximum_rotation_determinant_error=det)
    if gt is not None:
        missing = sorted(set(expected) - set(gt))
        if missing:
            raise ValueError(f'Ground truth missing requested frames: {missing[:12]}')
        est_xyz = np.stack([poses[i][:3, 3] for i in expected], axis=1)
        gt_xyz = np.stack([gt[i][:3, 3] for i in expected], axis=1)
        _, _, error = align(est_xyz, gt_xyz)
        result.update(ate_rmse_cm=float(np.sqrt(np.mean(error**2))*100),
                      ate_mean_translation_error_cm=float(error.mean()*100),
                      alignment='One rigid SE(3) alignment; no scale fitting',
                      matched_gt_poses=len(expected))
    return result


def image_summary(run, map_root, expected):
    output = []
    for receipt_path in sorted((run / 'evaluations').glob('*/run.json')):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('status') != 'completed' or receipt.get('returncode') != 0:
            output.append(dict(record=str(receipt_path), status=receipt.get('status'),
                               checked=False, reason='Evaluation has not completed successfully',
                               interruption_note=(json.loads((receipt_path.parent/'interruption_note.json').read_text())
                                                  if (receipt_path.parent/'interruption_note.json').exists() else None)))
            continue
        config_path = receipt_path.parent / 'config.yaml'
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != receipt['config_sha256']:
            raise ValueError('Evaluation configuration changed after launch')
        evaluation = yaml.safe_load(config_path.read_text())
        frame_ids = list(map(int, evaluation['PIPE']['eval_frame_indices']))
        if len(frame_ids) != len(set(frame_ids)) or not set(frame_ids).issubset(expected):
            raise ValueError('Evaluation frame list duplicated/outside training run')
        eval_dir = local(map_root / evaluation['PIPE']['eval_path'].lstrip('/'))
        if not eval_dir.is_relative_to(map_root):
            raise ValueError('Evaluation output escapes map workspace')
        renders = {}
        for path in (eval_dir / 'render').glob('*'):
            if path.is_file() and '.color.' in path.name:
                idx = pose_id(path)
                if idx in renders:
                    raise ValueError(f'Duplicate rendered frame {idx}')
                renders[idx] = path
        if set(renders) != set(frame_ids):
            raise ValueError(f'{eval_dir}: rendered coverage differs from requested views')
        reader = evaluation['READER']
        source = local(reader['input_dir'])
        factor = float(reader['downscale_factor'])
        shape = tuple(int(round(v/factor)) for v in reader['image_shape'])
        rows, perfect = [], 0
        for idx in frame_ids:
            rgb_path = source / reader['image_path'] / f"{reader.get('image_prefix','frame')}{idx:06d}{reader.get('image_extension','.jpg')}"
            depth_path = source / reader['depth_path'] / f"{reader.get('depth_prefix','depth')}{idx:06d}{reader.get('depth_extension','.png')}"
            render = cv2.imread(str(local(renders[idx])), cv2.IMREAD_COLOR)
            reference = cv2.imread(str(local(rgb_path)), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(local(depth_path)), cv2.IMREAD_UNCHANGED)
            if render is None or reference is None or depth is None:
                raise ValueError(f'Cannot decode evaluation images for frame {idx}')
            if factor > 1:
                # Match Camera::loadImage/loadDepth for on-the-fly resizing.
                reference = cv2.resize(reference, None, fx=1/factor, fy=1/factor, interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth.astype(np.float32), None, fx=1/factor, fy=1/factor, interpolation=cv2.INTER_AREA)
            if render.shape != reference.shape or render.shape[:2] != depth.shape[:2] or render.shape[1::-1] != shape:
                raise ValueError(f'Image/depth shape mismatch at frame {idx}')
            if not all(np.isfinite(x).all() for x in (render, reference, depth)):
                raise ValueError(f'Non-finite image data at frame {idx}')
            mask = depth > 0
            if not np.any(mask):
                raise ValueError(f'No valid depth pixels at frame {idx}')
            difference = (render.astype(np.float64)-reference.astype(np.float64))/255.0
            mse = float(np.mean(difference**2))
            masked_mse = float(np.mean((difference * mask[..., None])**2))
            psnr = -10.0*math.log10(mse) if mse > 0 else None
            perfect += mse == 0
            rows.append(dict(frame_id=idx, psnr_db=psnr, mse=mse,
                             depth_masked_psnr_db=-10.0*math.log10(masked_mse) if masked_mse > 0 else None,
                             valid_depth_fraction=float(mask.mean())))
        result = dict(record=str(receipt_path), status='validated', checked=True,
                      views=len(rows), exact_requested_coverage=True, finite_image_values=True,
                      resolution=list(shape), perfect_match_views=int(perfect),
                      psnr_mean_db=None if perfect else float(np.mean([x['psnr_db'] for x in rows])),
                      per_view=rows,
                      metric='Mean per-view RGB PSNR over all pixels, range 1; no depth masking. '
                             'Optional per-view depth_masked_psnr_db zeroes invalid-depth pixels in '
                             'both images with a whole-image MSE denominator. No SSIM/LPIPS computed.',
                      scope='Observed saved-map views at desktop resolution; no novel-view or dense-geometry claim')
        if perfect:
            result['psnr_note'] = 'Perfect matches have infinite PSNR; JSON mean is null.'
        output.append(result)
    return output


def summarise(run):
    run = local(run)
    errors, warnings = [], []
    result = dict(run=str(run), schema_version=1, errors=errors, warnings=warnings,
                  scope='Desktop validation, not the Thor benchmark. Unpaced processing is not '
                        'a real-time guarantee. No quantitative dense-geometry evaluation.')
    try:
        receipt = json.loads((run/'record/run.json').read_text())
        if receipt.get('purpose') != 'train' or receipt.get('status') != 'completed' or receipt.get('returncode') != 0:
            raise ValueError('Requires completed training receipt with return code zero')
        config_path = run/'record/config.yaml'
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != receipt['config_sha256']:
            raise ValueError('Training configuration changed after launch')
        config = yaml.safe_load(config_path.read_text())
        map_root = local(config['workspace_dir'])
        if map_root != run/'map':
            raise ValueError('Configured workspace is not this run/map')
        if yaml.safe_load((map_root/'config.yaml').read_text()) != config:
            raise ValueError('Native saved configuration differs from launch configuration')
        reader = config['READER']
        expected = list(range(reader['start_frame'], reader['end_frame']+1, reader['frame_step']))
        if len(expected) < 2:
            raise ValueError('Insufficient configured frames')
        result.update(dataset=receipt['dataset'], scene=receipt['scene'],
                      source_commit=receipt['source_commit'], binary_sha256=receipt['binary_sha256'],
                      configured_frames=len(expected), first_frame=expected[0], last_frame=expected[-1],
                      processing_shape=receipt['input_support']['processing_shape'],
                      estimated_pose_tracking=not config['PIPE']['TSDF']['use_gt_pose'])
        timings = csv_rows(map_root/'timing.csv')
        tracking = csv_rows(map_root/'tracking.csv')
        assignments = csv_rows(map_root/'frame_assignments.csv')
        for name, rows in [('timing',timings),('tracking',tracking),('frame assignments',assignments)]:
            exact_rows(rows, expected, name)
        for row in timings:
            for key, value in row.items():
                if key != 'gaussianScheduleMode':
                    finite(value)
        service = [finite(row['algorithmicService_ms']) for row in timings]
        if any(value < 0 for value in service):
            raise ValueError('Negative service time')
        resources = json.loads((map_root/'runtime_resources.json').read_text())
        finite_tree(resources)
        time_log = (map_root/'time_log.txt').read_text()
        fps_match = re.search(r'^\[PIPELINE AVG TIME\].*\bFPS:\s*([0-9.eE+-]+)', time_log, re.MULTILINE)
        if not fps_match:
            raise ValueError('Missing native pipeline completion/FPS summary')
        fps = finite(fps_match.group(1))
        algorithm_seconds = finite(resources['sequence_algorithmic_wall_ms'])/1000
        if algorithm_seconds <= 0 or fps <= 0 or not math.isclose(fps,len(expected)/algorithm_seconds,rel_tol=1e-5):
            raise ValueError('Native FPS and exact frame count/runtime summary disagree')
        wall_seconds = finite(receipt['command_wall_seconds'])
        result['timing'] = dict(native_processing_fps=fps,
                                sequence_algorithmic_seconds=algorithm_seconds,
                                native_scope='Includes terminal algorithm completion and CUDA/paging '
                                             'synchronisation; excludes startup, map export and '
                                             'separate online-end trajectory export.',
                                whole_command_seconds=wall_seconds,
                                whole_command_frames_per_second=len(expected)/wall_seconds,
                                whole_command_scope='Includes process startup and saved-map export.',
                                algorithmic_service_ms=statistics(service),
                                service_over_33_333_ms_frames=sum(value>1000/30 for value in service),
                                service_scope='Individual algorithmicService_ms calls; excludes queued '
                                              'sensor-delay analysis and does not prove deadline compliance.')
        statuses = {0:'FAILED',1:'POOR',2:'GOOD'}
        result['tracking'] = {}
        for field in ('tracking_result','raw_primary_tracking_result','policy_primary_tracking_result'):
            counts = Counter(int(row[field]) for row in tracking)
            if not set(counts).issubset(statuses):
                raise ValueError(f'Unknown tracking status in {field}')
            result['tracking'][field] = {name:counts.get(number,0) for number,name in statuses.items()}
        result['tracking']['maximum_raw_failed_streak'] = max(int(row['raw_failed_streak']) for row in tracking)
        if result['tracking']['raw_primary_tracking_result']['FAILED']:
            warnings.append('Raw primary tracking failures occurred; inspect raw/policy counts and ATE.')
        gt_path = local(reader['input_dir'])/reader['pose_path']
        gt = read_poses(gt_path,label='ground truth') if any(gt_path.glob('pose*.txt')) else None
        if gt is None:
            warnings.append('No GT pose files found; trajectory finiteness checked without ATE.')
        result['trajectories'] = {}
        for name,directory in [('raw',map_root/'val/pose'),
                               ('terminal_corrected',map_root/'val/corrected_pose'),
                               ('online_end_corrected',map_root/'pose_graph_live/online_end_poses')]:
            if name == 'online_end_corrected' and not directory.exists():
                continue
            result['trajectories'][name] = pose_summary(directory,expected,gt)
        result['trajectory_scope'] = ('Raw and retrospectively corrected trajectories are distinct; '
                                      'online-end corrected history is not a first-return causal trajectory.')
        metadata = csv_rows(map_root/'submaps/metadata.csv', required=bool(receipt['save_after_train']))
        if receipt['save_after_train']:
            for row in metadata:
                if int(row['gaussians']) > 0:
                    model_file = map_root/'submaps'/('submap_'+row['id'])/'gaussian/model.pt'
                    if not model_file.is_file() or model_file.stat().st_size == 0:
                        raise ValueError(f'Missing/nonempty Gaussian checkpoint required: {model_file}')
        manifest_path = map_root/'submaps/manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            finite_tree(manifest)
            if len(manifest.get('frame_assignments',[])) != len(expected):
                raise ValueError('Checkpoint frame-assignment count differs from full trajectory')
        result['submaps'] = dict(hybrid_registry_entries=len(metadata),
                                retired_hybrid_entries=sum(int(row['retired'])!=0 for row in metadata),
                                retained_hybrid_entries=sum(int(row['retired'])==0 for row in metadata),
                                nonempty_gaussian_submaps=sum(int(row['gaussians'])>0 for row in metadata),
                                saved_gaussians=sum(int(row['gaussians']) for row in metadata),
                                final_gaussian_count=resources['final_gaussian_count'])
        if receipt['save_after_train'] and result['submaps']['saved_gaussians'] != resources['final_gaussian_count']:
            raise ValueError('Saved Gaussian count differs from completed native count')
        stats = csv_rows(map_root/'submap_stats.csv',required=False)
        result['submaps']['final_sdf_map_records'] = sum(row['stage']=='end_of_sequence' for row in stats)
        result['submaps']['note'] = ('SDF map records may include an unpromoted candidate; hybrid registry '
                                    'entries include retired candidates. Neither counts successful handoffs.')
        events_path = map_root/'submap_events.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
        result['submaps']['event_counts'] = dict(Counter(row['event'] for row in events))
        updates = csv_rows(map_root/'pose_graph_live/updates.csv',required=False)
        result['pose_graph'] = dict(update_attempts=len(updates),
                                   applied_updates=sum(int(row['applied'])!=0 for row in updates),
                                   terminal_applied_updates=sum(int(row['applied'])!=0 and row['source']=='live_end_of_sequence' for row in updates),
                                   maximum_loop_constraints=max([int(row['loop_constraints']) for row in updates] or [0]),
                                   final_pending_transform_updates=int(updates[-1]['transform_updates_pending']) if updates else 0)
        if result['pose_graph']['final_pending_transform_updates']:
            errors.append('Terminal graph update reports pending transforms')
        paging = csv_rows(map_root/'paging.csv',required=False)
        result['paging'] = dict(event_rows=len(paging),
                               action_counts=dict(Counter(row['action'] for row in paging)),
                               configured=bool(config['PIPE']['TSDF']['multi']['paging']))
        if not paging:
            result['paging']['note'] = 'Paging configured but this run produced no paging events.'
        result['resources'] = dict(sampled_telemetry=receipt.get('telemetry'),
                                   native_resource_summary=resources,
                                   native_memory_note='Native field names retained: device-wide memory and '
                                                      'process RSS differ; neither is the historical cgroup peak.')
        finite_tree(result['resources'])
        text = (run/'record/native.log').read_text(errors='replace')
        failures = re.findall(r'^.*(?:CUDA error:|CUDA Error|out of memory|illegal memory access|SLAM failed|what\(\):).*$',
                              text, flags=re.MULTILINE|re.IGNORECASE)
        if failures:
            errors.append('Native error markers: '+'; '.join(failures[:5]))
        result['image_evaluations'] = image_summary(run,map_root,expected)
        result['excluded_evaluation_attempts'] = sum(not row.get('checked', False)
                                                     for row in result['image_evaluations'])
    except Exception as exc:
        errors.append(str(exc))
    result['passed'] = not errors
    result['status'] = 'validated' if result['passed'] else 'failed'
    write_json(run/'record/validation_summary.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    parser.add_argument('--aggregate',type=Path,help='Optional JSON containing all requested summaries')
    args = parser.parse_args()
    summaries = [summarise(run) for run in args.runs]
    if args.aggregate:
        write_json(args.aggregate,dict(passed=all(row['passed'] for row in summaries),runs=summaries))
    for row in summaries:
        print(f"{row['status']}: {row['run']}")
        if row['passed']:
            timing = row['timing']
            ate = row['trajectories']['terminal_corrected'].get('ate_rmse_cm')
            print(f"  {row['configured_frames']} frames; native {timing['native_processing_fps']:.2f} FPS; "
                  f"command {timing['whole_command_frames_per_second']:.2f} FPS; corrected ATE {ate} cm")
            print(f"  Raw primary tracking: {row['tracking']['raw_primary_tracking_result']}")
            for evaluation in row['image_evaluations']:
                if evaluation.get('checked'):
                    print(f"  Observed-view PSNR: {evaluation['psnr_mean_db']} dB ({evaluation['views']} views)")
        for error in row['errors']:
            print('  ERROR: '+error)
        for warning in row['warnings']:
            print('  NOTE: '+warning)
    return 0 if all(row['passed'] for row in summaries) else 1


if __name__ == '__main__':
    raise SystemExit(main())
