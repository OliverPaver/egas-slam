#!/usr/bin/env python3
"""Render a traceable interview film and GitHub preview from genuine demo captures.

Requires a running demo_viewer.py server, system Chrome, Playwright and FFmpeg.
Edit decisions refer to immutable capture indices; playback timing is editorial.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

def capture(run):
    root=run/'map/demo'
    timeline=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()]
    records=[json.loads((root/x['file']).read_text()) for x in timeline]
    events=[]
    if (root/'transforms.jsonl').exists():
        events=[json.loads(x) for x in (root/'transforms.jsonl').read_text().splitlines()]
    receipt=json.loads((run/'record/run.json').read_text())
    assert receipt['status']=='completed' and receipt['returncode']==0
    assert records[-1]['stage']=='terminal'
    return timeline,records,events

def find_nearest(records,frame):
    return min(range(len(records)),key=lambda i:abs(records[i]['frame']-frame))

def edit_decisions(room, tum):
    _, rr, _ = capture(room)
    _, tr, events = capture(tum)
    assert events, 'A genuine applied correction is required for the loop-closure segment'
    groups = []
    for e in events:
        if not groups or e['frame'] - groups[-1][-1]['frame'] > 8:
            groups.append([])
        groups[-1].append(e)

    def displacement(e):
        a = np.linalg.inv(np.array(e['before_global_to_local']).reshape(4, 4))
        b = np.linalg.inv(np.array(e['after_global_to_local']).reshape(4, 4))
        return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))

    event = max(groups, key=lambda group: max(map(displacement, group)))
    start, end = event[0]['frame'], event[-1]['frame']
    before = max(i for i, d in enumerate(tr) if d['frame'] < start)
    after = min(i for i, d in enumerate(tr) if d['frame'] >= end)
    correction = {'startFrame': start, 'endFrame': end}
    changes = list(range(before + 1, after + 1))
    online = [i for i, d in enumerate(rr) if d['stage'] == 'online']
    hero = find_nearest(rr, 1790)
    # Add holds at actual owner handoffs, then a little later so the new map
    # has acquired visible geometry. All snapshots remain chronological.
    handoffs = [i for i in online[1:] if rr[i]['owner'] != rr[i - 1]['owner']]
    holds = {}
    for j, birth in enumerate(handoffs):
        last = handoffs[j + 1] - 1 if j + 1 < len(handoffs) else online[-1]
        growth_frames = 200 if j == 1 else 120
        settled = min(range(birth, last + 1),
                      key=lambda i: abs(rr[i]['frame'] - rr[birth]['frame'] - growth_frames))
        holds[birth - 1] = (2.0, 'Each local map keeps its own geometry and Gaussian detail.')
        holds[settled] = (3.0, 'A new local map grows alongside the previous one.')
    hold_time = sum(x[0] for x in holds.values())
    assert hold_time < 60, 'Too many owner changes for this edit; adjust the hold schedule'
    weights = [max(1, rr[online[j + 1]]['frame'] - rr[i]['frame'])
               if j + 1 < len(online) else 1 for j, i in enumerate(online)]
    full = []

    def add(items, run, i, duration, title, section, **opts):
        items.append({'run': run.name, 'index': int(i), 'duration': float(duration),
                      'section': section, 'options': {'title': title, **opts}})

    add(full, room, hero, 4, 'EGAS-SLAM · A detailed map built from colour and depth.', 'Opening')
    for i, weight in zip(online, weights):
        title = 'Building the first local map from colour and depth images.' if rr[i]['owner'] == 0 else \
                'The map grows as neighbouring local maps are added.'
        add(full, room, i, (60 - hold_time) * weight / sum(weights), title, 'Room reconstruction')
        if i in holds:
            duration, title = holds[i]
            add(full, room, i, duration, title, 'Room reconstruction')
    # Keep the last online map state: do not jump backwards to a prettier map.
    # Only the image pair uses a labelled earlier captured viewpoint.
    add(full, room, online[-1], 8,
        'Overlapping submaps · separate local maps, placed together in one global map.',
        'Inspect overlapping submaps', imageIndex=hero, focus='submaps')
    growth = list(range(before + 1))
    for i in growth:
        add(full, tum, i, 25 / len(growth),
            'A real office scan · follow the camera and the growing map.', 'Office approach')
    add(full, tum, before, 5, 'Before loop correction · note the current submap placement.',
        'Before correction', correctionPhase='before', correctionWindow=correction, focus='correction')
    for i in changes:
        add(full, tum, i, 12 / len(changes),
            'Loop closure · a verified revisit corrects the submap placements.',
            'Recorded correction', correctionPhase='during', correctionWindow=correction, focus='correction')
    add(full, tum, after, 3,
        'After loop correction · geometry and Gaussian detail have moved together.',
        'After correction', correctionPhase='after', correctionWindow=correction, focus='correction')
    add(full, tum, after, 2,
        'Correction comparison · selected submap at its recorded BEFORE placement.',
        'After correction', correctionPhase='compare-before', correctionWindow=correction, focus='correction')
    add(full, tum, after, 2,
        'Correction comparison · the same local contents at their corrected placement.',
        'After correction', correctionPhase='compare-after', correctionWindow=correction, focus='correction')
    # A restrained nine-degree orbit avoids rotating walls into the viewpoint.
    # This is explicitly a completed-map inspection, not a new online state.
    orbit_frames = 216
    orbit_angle = math.radians(9)
    for i in range(orbit_frames):
        progress = i / (orbit_frames - 1)
        ease = .5 - .5 * math.cos(math.pi * progress)
        add(full, room, len(rr) - 1, 18 / orbit_frames,
            'Completed reconstruction · a slow inspection of the mapped room.',
            'Completed-map orbit', orbit=orbit_angle * ease, imageIndex=hero)
    add(full, room, len(rr) - 1, 5,
        'EGAS-SLAM · Detailed reconstruction with correctable local maps.',
        'Closing', orbit=orbit_angle, imageIndex=hero)
    assert abs(sum(x['duration'] for x in full) - 144) < 1e-6

    short = []
    add(short, room, hero, 3, 'EGAS-SLAM · Colour and depth become a detailed map.', 'Opening')
    # The preview has a shorter growth segment, but still pauses at each actual
    # new-owner state. Preserve every captured growth state and its order.
    preview_holds = {i: value for i, value in holds.items() if value[0] == 3.0}
    preview_hold_time = 1.5 * len(preview_holds)
    assert preview_hold_time < 16
    for i, weight in zip(online, weights):
        add(short, room, i, (16 - preview_hold_time) * weight / sum(weights),
            'Colour + depth → local hybrid maps → a growing reconstruction.', 'Room reconstruction')
        if i in preview_holds:
            add(short, room, i, 1.5, 'A new local map grows alongside its neighbour.', 'Room reconstruction')
    add(short, room, online[-1], 5,
        'Overlapping local maps · geometry and Gaussian detail stay together.',
        'Inspect overlapping submaps', imageIndex=hero, focus='submaps')
    add(short, tum, before, 3, 'Office sequence · before a verified loop correction.',
        'Before correction', correctionPhase='before', correctionWindow=correction, focus='correction')
    for i in changes:
        add(short, tum, i, 6 / len(changes),
            'Loop closure corrects the placements of intact local maps.',
            'Recorded correction', correctionPhase='during', correctionWindow=correction, focus='correction')
    add(short, tum, after, 4, 'Actual recorded correction · geometry and appearance move together.',
        'After correction', correctionPhase='after', correctionWindow=correction, focus='correction')
    add(short, room, len(rr) - 1, 3, 'EGAS-SLAM · Detailed reconstruction. Global correction.',
        'Closing', imageIndex=hero)
    assert abs(sum(x['duration'] for x in short) - 40) < 1e-6

    def chapters(items):
        result = []
        elapsed = 0.0
        for item in items:
            if not result or result[-1]['title'] != item['section']:
                result.append({'title': item['section'], 'start_seconds': elapsed, 'duration_seconds': 0.0})
            result[-1]['duration_seconds'] += item['duration']
            elapsed += item['duration']
        for chapter in result:
            chapter['start_seconds'] = round(chapter['start_seconds'], 6)
            chapter['duration_seconds'] = round(chapter['duration_seconds'], 6)
        return result

    return {'full': full, 'preview': short,
            'chapters': {'full': chapters(full), 'preview': chapters(short)},
            'poster': {'run': room.name, 'index': hero,
                       'options': {'title': 'Detailed reconstruction. Local maps that can move together.'}},
            'selected_correction': {'start_frame': start, 'end_frame': end,
                'before_capture_frame': tr[before]['frame'], 'after_capture_frame': tr[after]['frame'],
                'owner_updates': len(event),
                'max_owner_origin_translation_metres': max(map(displacement, event))},
            'room_owner_handoffs': [{'frame': rr[i]['frame'], 'owner': rr[i]['owner']} for i in handoffs],
            'scope': 'Real estimated-pose runs. Editorial time compression and holds; no real-time claim. '
                'Display SDF meshes use 1 cm clustering; Gaussian ellipsoids are sampled. '
                'Native hybrid images use all current-owner Gaussians. Completed-map orbit is a replay.'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def atomic_json(path, value):
    temporary = path.with_name('.' + path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def capture_identity(run):
    """Bind render reuse to the actual completed capture, not only its name."""
    root = run / 'map/demo'
    files = [run / 'record/run.json', run / 'record/config.yaml', root / 'frames.jsonl']
    if (root / 'transforms.jsonl').exists():
        files.append(root / 'transforms.jsonl')
    files += [root / json.loads(line)['file']
              for line in (root / 'frames.jsonl').read_text().splitlines()]
    manifest = run / 'record/demo_asset_sha256.json'
    assert manifest.exists(), f'Validate this capture before export: {run}'
    validation = json.loads((run / 'record/demo_validation.json').read_text())
    assert validation['passed'], f'Capture validation did not pass: {run}'
    files += [manifest, run / 'record/demo_validation.json']
    digests = {str(p.relative_to(run)): sha(p) for p in files}
    # The validator already hashed each large asset. Include current file
    # metadata too so replacing an asset invalidates an old render cache even
    # before its digest manifest is regenerated. This is cache identity, not
    # a substitute for the capture validator's content checks.
    asset_stats = {name: [p.stat().st_size, p.stat().st_mtime_ns]
                   for name in json.loads(manifest.read_text())
                   for p in [root / name]}
    identity = {'run': str(run.resolve()), 'metadata_sha256': digests,
                'validated_asset_manifest_sha256': sha(manifest), 'asset_stats': asset_stats}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

def validate_media(path):
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], text=True))
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                            text=True, capture_output=True)
    assert result.returncode == 0 and not result.stderr, (path.name, result.stderr)
    return {'width': probe['streams'][0]['width'], 'height': probe['streams'][0]['height'],
            'duration': float(probe['format']['duration']),
            'codec': probe['streams'][0]['codec_name'], 'decode_errors': []}

def export(args):
    args.room = args.room.expanduser().resolve()
    args.office = args.office.expanduser().resolve()
    out=args.output.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    cache=out/'rendered_frames';cache.mkdir(exist_ok=True)
    decisions=edit_decisions(args.room,args.office)
    if args.timeline_only:
        print(json.dumps({key: decisions[key] for key in ['chapters', 'selected_correction', 'room_owner_handoffs']}, indent=2))
        return
    identities = {run.name: capture_identity(run) for run in [args.room, args.office]}
    decisions['capture_identities'] = identities
    atomic_json(out / '.edit_decisions.pending.json', decisions)
    from playwright.sync_api import sync_playwright
    browser_errors=[];rendered={}
    # Version cache with the complete renderer and camera configuration.
    renderer_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((PROJECT/'viewer').rglob('*')) if p.is_file())).hexdigest()
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=args.chrome,headless=True,
            args=['--no-sandbox','--use-angle=gl','--enable-webgl'])
        page=browser.new_page(viewport={'width':1920,'height':1080},device_scale_factor=1)
        page.on('pageerror',lambda e:browser_errors.append(str(e)))
        page.goto(args.url);page.wait_for_function('window.demoReady',timeout=120000)
        page.evaluate('demo.record(true)');page.wait_for_timeout(300)
        active=None
        jobs=decisions['full']+decisions['preview']+[decisions['poster']]
        for j,item in enumerate(jobs):
            spec={k:v for k,v in item.items() if k!='duration'}
            key=hashlib.sha256((json.dumps(spec,sort_keys=True)+renderer_hash+identities[item['run']]).encode()).hexdigest()[:20]
            file=cache/f'{key}.png';rendered[json.dumps(spec,sort_keys=True)]=file
            if not file.exists():
                if active!=item['run']:
                    page.evaluate('(id)=>demo.loadRun(id)',item['run']);active=item['run']
                state=page.evaluate('(s)=>demo.show(s.index,s.options)',spec)
                assert state['index']==item['index'] and state['run']==item['run']
                assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()
                temporary = file.with_name('.' + file.stem + '.partial.png')
                page.screenshot(path=str(temporary))
                temporary.replace(file)
            if j%25==0:print(f'Render {j+1}/{len(jobs)}',flush=True)
        browser.close()
    assert not browser_errors,browser_errors
    def path(item):
        return rendered[json.dumps({k:v for k,v in item.items() if k!='duration'},sort_keys=True)]
    report={'renderer_sha256':renderer_hash,'capture_identities':identities,
            'browser_errors':browser_errors,'videos':{},'files':{}}
    for key,name in [('full','EGAS_SLAM_Interview.mp4'),('preview','EGAS_SLAM_GitHub_Preview.mp4')]:
        lines=[]
        for item in decisions[key]:
            filename=str(path(item)).replace("'", "'\\''")
            lines.extend([f"file '{filename}'",f"duration {item['duration']:.9f}"])
        lines.append(lines[-2])
        concat=out/f'{key}_frames.txt';concat.write_text('\n'.join(lines)+'\n')
        temporary = out / ('.' + Path(name).stem + '.partial.mp4')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','warning','-y','-f','concat','-safe','0',
            '-i',str(concat),'-vf','fps=30','-c:v','libx264','-crf','19','-preset','medium',
            '-pix_fmt','yuv420p','-movflags','+faststart','-t',str(sum(x['duration'] for x in decisions[key])),
            str(temporary)],check=True)
        report['videos'][name] = validate_media(temporary)
        expected = sum(x['duration'] for x in decisions[key])
        assert abs(report['videos'][name]['duration'] - expected) <= .1
        temporary.replace(out / name)
    poster=out/'EGAS_SLAM_Poster.png'
    temporary = out / '.EGAS_SLAM_Poster.partial.png'
    temporary.write_bytes(path(decisions['poster']).read_bytes())
    temporary.replace(poster)
    temporary = out / '.EGAS_SLAM_GitHub_Preview.partial.gif'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','warning','-y','-i',str(out/'EGAS_SLAM_GitHub_Preview.mp4'),
        '-filter_complex','fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
        '-loop','0',str(temporary)],check=True)
    report['videos']['EGAS_SLAM_GitHub_Preview.gif'] = validate_media(temporary)
    assert abs(report['videos']['EGAS_SLAM_GitHub_Preview.gif']['duration'] - 40) <= .2
    temporary.replace(out / 'EGAS_SLAM_GitHub_Preview.gif')
    for p in out.glob('EGAS_SLAM_*'):
        if p.is_file():report['files'][p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    report['selected_correction']=decisions['selected_correction']
    atomic_json(out / 'media_validation.json', report)
    (out / '.edit_decisions.pending.json').replace(out / 'edit_decisions.json')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    root=Path.home()/'egas_workspace'
    p.add_argument('--room',type=Path,default=root/'runs/demo_replica_room0_full')
    p.add_argument('--office',type=Path,default=root/'runs/demo_tum_office')
    p.add_argument('--output',type=Path,default=root/'deliverables/EGAS_SLAM_Demo_v2_2026-11-03')
    p.add_argument('--timeline-only',action='store_true',help='Print the edit plan without launching the browser or encoding media')
    p.add_argument('--url',default='http://127.0.0.1:8766')
    p.add_argument('--chrome',default='/usr/bin/google-chrome')
    export(p.parse_args())
