#!/usr/bin/env python3
"""Export a continuous GitHub demonstration from verified local EGAS captures.

Original RGB frames provide continuous motion; map states use the latest past
capture. The correction comparison uses recorded transforms, never tweened ones.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import numpy as np
import yaml

from demo_export import PROJECT, atomic_json, capture, capture_identity, sha, validate_media


def local_path(path):
    path = Path(path).expanduser().resolve()
    if any(root in path.parents for root in (Path('/media'), Path('/mnt'), Path('/run/media'))):
        raise ValueError(f'Presentation inputs and execution must remain local: {path}')
    return path


def link_file(source, destination):
    source = local_path(source)
    assert source.is_file(), source
    replace_old_link = destination.is_symlink() and destination.resolve() == source
    if destination.exists() and not destination.is_symlink():
        if os.path.samefile(source, destination) or sha(source) == sha(destination):
            return
    if (destination.exists() or destination.is_symlink()) and not replace_old_link:
        raise FileExistsError(f'Refusing to replace unrelated presentation asset: {destination}')
    # The HTTP server correctly confines resolved paths to its presentation
    # directory. Hardlinks preserve that boundary without duplicating storage;
    # crossing a local filesystem boundary requires an ordinary copy instead.
    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.partial')
    try:
        try:
            os.link(source, temporary)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def past_index(records, frame):
    index = bisect_right([record['frame'] for record in records], frame) - 1
    # Prefer an online state if a terminal snapshot shares its frame.
    while index > 0 and records[index]['stage'] != 'online':
        index -= 1
    assert index >= 0 and records[index]['frame'] <= frame
    return index


def comparison_path(records, before_index, after_index):
    before = records[before_index]
    after = records[after_index]
    placements = [{s['sdf_id']: np.asarray(s['sdf_local_to_global']).reshape(4, 4)
                   for s in record['submaps']} for record in (before, after)]
    result = []
    for record in records[:before_index + 1]:
        local = np.asarray(record['camera_to_owner']).reshape(4, 4)
        row = {'frame': record['frame'], 'owner': record['owner']}
        for name, matrices in zip(('before', 'after'), placements):
            assert record['owner'] in matrices
            row[name] = (matrices[record['owner']] @ local)[:3, 3].tolist()
        result.append(row)
    return {'description': 'Identical recorded camera-to-owner samples through frame 1919, '
            'placed using actual transforms at frames 1919 and 1924. Not ground-truth errors.',
            'before_frame': before['frame'], 'after_frame': after['frame'], 'samples': result}


def prepare(args):
    destination = local_path(args.presentation)
    destination.mkdir(parents=True, exist_ok=True)
    runs = {'room': local_path(args.room), 'office': local_path(args.office)}
    story = {'schema': 1, 'runs': {}, 'referenceImage': '/presentation/reference.png'}
    provenance = {'captures': {}, 'sources': {}, 'presentation_root': str(destination)}
    for key, run in runs.items():
        timeline, records, _ = capture(run)
        identity = capture_identity(run)
        config = yaml.safe_load((run / 'record/config.yaml').read_text())['READER']
        assert config['start_frame'] == 0 and config['frame_step'] == 1
        assert config['test_split_interval'] == -1
        root = local_path(config['input_dir'])
        extension = config.get('image_extension', '.jpg')
        images = root / config['image_path']
        count = int(config['end_frame']) + 1
        target = destination / key
        target.mkdir(exist_ok=True)
        digests = {}
        for frame in range(count):
            name = f'frame{frame:06d}{extension}'
            source = images / name
            link_file(source, target / name)
            # Hash every linked source so provenance also covers reference and
            # manually inspected frames outside the edited excerpts.
            digests[name] = sha(source)
        for row, record in zip(timeline, records):
            record['captureFile'] = row['file']
        story['runs'][key] = {'id': run.name, 'snapshots': records,
                             'sourcePrefix': f'/presentation/{key}/',
                             'sourceExtension': extension}
        provenance['captures'][key] = {'run': str(run), 'identity': identity,
                                      'snapshot_count': len(records)}
        provenance['sources'][key] = {'root': str(root), 'frame_count': count,
                                     'rgb_sha256': digests}
        if key == 'office':
            metadata_file = root / 'metadata.json'
            metadata = json.loads(metadata_file.read_text())
            timestamps = [row['rgb_timestamp'] for row in metadata['frame_records']]
            assert len(timestamps) == count
            assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
            story['runs'][key]['timestamps'] = timestamps
            provenance['sources'][key]['metadata_sha256'] = sha(metadata_file)
            link_file(images / 'frame000001.png', destination / 'reference.png')
    records = story['runs']['office']['snapshots']
    before = next(i for i, row in enumerate(records) if row['frame'] == 1919)
    after = next(i for i, row in enumerate(records) if row['frame'] == 1924)
    assert records[before]['applied_transform_count'] == 0
    assert records[after]['applied_transform_count'] == 19
    assert any(row['from'] == 0 and row['to'] == 18
               for row in records[after]['loop_constraints'])
    story['comparison'] = {'beforeIndex': before, 'afterIndex': after,
                           'beforeFrame': 1919, 'afterFrame': 1924,
                           'eventStart': 1920, 'eventEnd': 1924,
                           'revisitOwners': [0, 18], 'referenceFrame': 1,
                           'matchedFrame': 1920}
    story['nativePathComparison'] = comparison_path(records, before, after)
    timestamps = story['runs']['office']['timestamps']
    story['timing'] = {'fps': args.fps, 'mappingFrames': [600, 1999],
                       'mappingSeconds': 1400 / 30, 'mappingSourceFramesPerSecond': 30,
                       'revisitFrames': [1800, 2040],
                       'revisitSeconds': timestamps[2040] - timestamps[1800],
                       'comparisonSeconds': 10}
    story['scope'] = ('Actual estimated-pose recordings, edited for presentation. Original RGB '
        'frames remain continuous; map surfaces update at recorded capture intervals. '
        'Inspection camera motion is editorial, with no interpolation of map transforms. '
        'Replica replay pace is selected for presentation; TUM follows recorded timestamps. '
        'No real-time performance or accuracy claim is derived from this video.')
    atomic_json(destination / 'story.json', story)
    atomic_json(destination / 'source_provenance.json', provenance)
    return story, provenance


def timeline(story, fps):
    result = []
    offset = 0
    definitions = [('mapping', story['timing']['mappingSeconds']),
                   ('revisit', story['timing']['revisitSeconds']),
                   ('comparison', story['timing']['comparisonSeconds'])]
    timestamps = story['runs']['office']['timestamps']
    for phase, requested_duration in definitions:
        count = int(math.ceil(requested_duration * fps - 1e-9))
        states = []
        for index in range(count):
            progress = index / max(1, count - 1)
            elapsed = index / fps
            if phase == 'mapping':
                frame = min(1999, 600 + int(math.floor(elapsed * 30 + 1e-7)))
                records = story['runs']['room']['snapshots']
                orbit = 0.10 * progress
            elif phase == 'revisit':
                target = timestamps[1800] + elapsed
                right = min(2040, bisect_right(timestamps, target))
                left = max(1800, right - 1)
                frame = min((left, right), key=lambda i: abs(timestamps[i] - target))
                records = story['runs']['office']['snapshots']
                orbit = 0
            else:
                frame = 1924
                records = story['runs']['office']['snapshots']
                orbit = 0.10 * progress
            state = {'phase': phase, 'frame': frame,
                     'captureIndex': past_index(records, frame),
                     't': elapsed, 'progress': progress, 'orbit': orbit}
            assert records[state['captureIndex']]['frame'] <= frame
            states.append(state)
        result.append({'phase': phase, 'start_frame': offset, 'frames': count,
                       'start_seconds': offset / fps, 'duration_seconds': count / fps,
                       'requested_duration_seconds': requested_duration, 'states': states})
        offset += count
    assert result[0]['states'][0]['frame'] == 600
    assert result[1]['states'][0]['frame'] == 1800
    for section in result:
        source_frames = [state['frame'] for state in section['states']]
        assert source_frames == sorted(source_frames)
        if section['phase'] != 'comparison':
            run_length = longest = 1
            for previous, current in zip(source_frames, source_frames[1:]):
                run_length = run_length + 1 if previous == current else 1
                longest = max(longest, run_length)
            # Repeated output frames are allowed only for ordinary rate
            # conversion / short gaps in the original sensor timestamps.
            assert longest <= max(1, math.ceil(.12 * fps)), (section['phase'], longest)
    return result


def renderer_hash():
    files = [PROJECT / 'viewer' / name for name in ('github.html', 'github.css', 'github.js')]
    files += sorted((PROJECT / 'viewer/scenes').glob('*.json'))
    files += [Path(__file__)]
    return hashlib.sha256(b''.join(path.read_bytes() for path in files)).hexdigest()


def render_section(page, section, output, fps, cache_key, browser_errors):
    name = section['phase']
    media = output / f'{name}.mp4'
    receipt = output / f'{name}.json'
    if media.exists() and receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get('cache_key') == cache_key and previous.get('sha256') == sha(media):
            print(f'Reusing validated {name} segment', flush=True)
            return media
    temporary = output / f'.{name}.partial.mp4'
    log = output / f'{name}_ffmpeg.log'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
               '-f', 'image2pipe', '-vcodec', 'png', '-framerate', str(fps), '-i', 'pipe:0',
               '-an', '-c:v', 'libx264', '-crf', '19', '-preset', 'medium',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    start = time.monotonic()
    with log.open('w') as error_log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=error_log)
        try:
            for index, state in enumerate(section['states']):
                returned = page.evaluate('(state) => githubDemo.show(state)', state)
                if returned is not None:
                    assert returned.get('phase', state['phase']) == state['phase'], returned
                    assert returned.get('frame', state['frame']) == state['frame'], returned
                assert not browser_errors, browser_errors
                process.stdin.write(page.screenshot(type='png'))
                if index % 60 == 0 or index + 1 == len(section['states']):
                    print(f'{name}: {index + 1}/{len(section["states"])} frames '
                          f'({time.monotonic() - start:.1f}s)', flush=True)
            process.stdin.close()
            assert process.wait(timeout=120) == 0, log.read_text()
        except BaseException:
            process.kill()
            process.wait()
            raise
    checks = validate_media(temporary)
    assert abs(checks['duration'] - section['duration_seconds']) < 1 / fps + .002
    temporary.replace(media)
    atomic_json(receipt, {'cache_key': cache_key, 'sha256': sha(media),
                         'frames': section['frames'], 'validation': checks})
    return media


def export(args, story, provenance):
    output = local_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    chunks = output / 'segments'
    chunks.mkdir(exist_ok=True)
    sections = timeline(story, args.fps)
    source_hash = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    viewer_hash = renderer_hash()
    decisions = {'schema': 1, 'fps': args.fps, 'sections': sections,
                 'duration_seconds': sum(section['duration_seconds'] for section in sections),
                 'renderer_sha256': viewer_hash, 'source_identity': source_hash,
                 'scope': story['scope'], 'no_editorial_freeze_holds': True,
                 'map_transform_interpolation': False,
                 'rgb_sampling': 'Nearest original TUM timestamp; original Replica frame at 30 frames/s',
                 'map_sampling': 'Latest captured online state at or before the displayed source frame'}
    atomic_json(output / '.edit_decisions.pending.json', decisions)
    atomic_json(output / 'source_provenance.json', provenance)
    from playwright.sync_api import sync_playwright
    errors = []
    segment_paths = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chrome, headless=True,
                      args=['--no-sandbox', '--use-angle=gl', '--enable-webgl'])
        page = browser.new_page(viewport={'width': 1920, 'height': 1080}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function('window.githubReady', timeout=120000)
        for section in sections:
            key = hashlib.sha256((viewer_hash + source_hash +
                                  json.dumps(section, sort_keys=True)).encode()).hexdigest()
            segment_paths.append(render_section(page, section, chunks, args.fps, key, errors))
        browser.close()
    assert not errors, errors
    concat = chunks / 'concat.txt'
    # Filenames are controlled constants. Relative paths avoid shell or quoting
    # ambiguities even if the destination directory contains apostrophes.
    concat.write_text(''.join(f"file '{path.name}'\n" for path in segment_paths))
    temporary = output / '.EGAS_SLAM_GitHub_Preview.partial.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'concat',
                    '-safe', '0', '-i', str(concat), '-c', 'copy', '-movflags', '+faststart',
                    str(temporary)], check=True)
    checks = validate_media(temporary)
    assert checks['width'] == 1920 and checks['height'] == 1080
    assert abs(checks['duration'] - decisions['duration_seconds']) < 1 / args.fps + .01
    mp4 = output / 'EGAS_SLAM_GitHub_Preview.mp4'
    temporary.replace(mp4)
    poster = output / 'EGAS_SLAM_Poster.png'
    temporary = output / '.EGAS_SLAM_Poster.partial.png'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-ss',
                    str((1790 - 600) / 30), '-i', str(mp4), '-frames:v', '1',
                    '-update', '1', str(temporary)], check=True)
    temporary.replace(poster)
    gif = output / 'EGAS_SLAM_GitHub_Preview.gif'
    temporary = output / '.EGAS_SLAM_GitHub_Preview.partial.gif'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-i', str(mp4),
        '-filter_complex', 'fps=10,scale=960:-1:flags=lanczos,split[a][b];'
        '[a]palettegen=max_colors=128:stats_mode=diff[p];'
        '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
        '-loop', '0', str(temporary)], check=True)
    gif_checks = validate_media(temporary)
    assert abs(gif_checks['duration'] - decisions['duration_seconds']) < .21
    temporary.replace(gif)
    report = {'passed': True, 'browser_errors': errors, 'renderer_sha256': viewer_hash,
              'source_identity': source_hash, 'duration_seconds': decisions['duration_seconds'],
              'videos': {mp4.name: checks, gif.name: gif_checks},
              'files': {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
                        for path in (mp4, gif, poster)},
              'timing_checks': {'no_editorial_holds': True, 'monotonic_source_frames': True,
                                'map_states_never_from_future': True,
                                'correction_transform_interpolation': False}}
    atomic_json(output / 'media_validation.json', report)
    (output / '.edit_decisions.pending.json').replace(output / 'edit_decisions.json')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path.home() / 'egas_workspace'
    parser.add_argument('--room', type=Path, default=root / 'runs/demo_replica_room0_full')
    parser.add_argument('--office', type=Path, default=root / 'runs/demo_tum_office')
    parser.add_argument('--presentation', type=Path, default=root / 'presentation/github_v3')
    parser.add_argument('--output', type=Path,
                        default=root / 'deliverables/EGAS_SLAM_GitHub_v3_2026-11-03')
    parser.add_argument('--url', default='http://127.0.0.1:8766/github.html')
    parser.add_argument('--chrome', default='/usr/bin/google-chrome')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--draft', action='store_true', help='Render at 10 fps into a separate draft folder')
    parser.add_argument('--prepare-only', action='store_true', help='Link source frames and validate the edit plan')
    args = parser.parse_args()
    if args.draft:
        args.fps = 10
        args.output = args.output.with_name(args.output.name + '_draft')
    if not 5 <= args.fps <= 60:
        parser.error('--fps must be between 5 and 60')
    story, provenance = prepare(args)
    sections = timeline(story, args.fps)
    print(json.dumps({'presentation': str(args.presentation),
                      'sections': [{key: value for key, value in section.items() if key != 'states'}
                                   for section in sections]}, indent=2), flush=True)
    if not args.prepare_only:
        export(args, story, provenance)


if __name__ == '__main__':
    main()
