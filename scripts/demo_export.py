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

def edit_decisions(room,tum):
    rt,rr,_=capture(room);tt,tr,events=capture(tum)
    assert events,'A genuine applied correction is required for the loop-closure segment'
    groups=[]
    for e in events:
        if not groups or e['frame']-groups[-1][-1]['frame']>8:groups.append([])
        groups[-1].append(e)
    def displacement(e):
        a=np.linalg.inv(np.array(e['before_global_to_local']).reshape(4,4))
        b=np.linalg.inv(np.array(e['after_global_to_local']).reshape(4,4))
        return float(np.linalg.norm(a[:3,3]-b[:3,3]))
    event=max(groups,key=lambda group:max(map(displacement,group)))
    start,end=event[0]['frame'],event[-1]['frame']
    before=max(i for i,d in enumerate(tr) if d['frame']<start)
    after=min(i for i,d in enumerate(tr) if d['frame']>=end)
    hero=find_nearest(rr,1790)
    full=[]
    def add(run,i,duration,title,**opts):
        full.append({'run':run.name,'index':int(i),'duration':float(duration),
                     'options':{'title':title,**opts}})
    add(room,hero,3,'Detailed reconstruction. Local maps that can move together.')
    online=[i for i,d in enumerate(rr) if d['stage']=='online']
    for i in online:add(room,i,30/len(online),'Building a room from colour and depth images.')
    add(room,hero,4,'Overlapping submaps keep their own surfaces and Gaussian detail.')
    growth=list(range(before+1))
    for i in growth:add(tum,i,9/len(growth),'A real office scan: tracking, mapping and searching for revisits.')
    add(tum,before,2,'Before correction: the local maps retain their current placements.')
    changes=list(range(before+1,after+1))
    for i in changes:add(tum,i,7/max(1,len(changes)),'Loop closure: a verified revisit repositions the local maps.')
    add(tum,after,3,'After correction: local contents stay intact; global placements change.')
    # The overview is a completed map; the image pair is explicitly labelled
    # with its original online frame so an attractive recorded view can be used.
    for i in range(120):
        angle=-.38+(.76*i/119)
        add(room,len(rr)-1,9/120,'Completed reconstruction · inspect the room from outside.',
            orbit=angle,imageIndex=hero)
    add(room,len(rr)-1,3,'EGAS-SLAM · Detailed reconstruction with correctable local maps.',imageIndex=hero)
    assert 60<=sum(x['duration'] for x in full)<=90
    short=[]
    def shortadd(run,i,duration,title,**opts):
        short.append({'run':run.name,'index':int(i),'duration':float(duration),
                      'options':{'title':title,**opts}})
    shortadd(room,hero,1,'Detailed reconstruction. Local maps that can move together.')
    for i in online:shortadd(room,i,9/len(online),'Colour + depth → local hybrid maps → a growing reconstruction.')
    shortadd(room,hero,2,'Overlapping local maps. Geometry and appearance stay together.')
    shortadd(tum,before,1.5,'Office sequence · before a verified loop closure.')
    for i in changes:shortadd(tum,i,3/max(1,len(changes)),'Loop closure corrects the placements of intact local maps.')
    shortadd(tum,after,1.5,'Actual recorded correction · no exaggerated displacement.')
    shortadd(room,len(rr)-1,2,'EGAS-SLAM · Detailed reconstruction. Global correction.',imageIndex=hero)
    return {'full':full,'preview':short,'poster':{'run':room.name,'index':hero,
            'options':{'title':'Detailed reconstruction. Local maps that can move together.'}},
            'selected_correction':{'start_frame':start,'end_frame':end,
            'owner_updates':len(event),'max_owner_origin_translation_metres':max(map(displacement,event))},
            'scope':'Real estimated-pose runs. Editorial time compression and holds; no real-time claim. '
            'Display SDF meshes use 1 cm clustering; Gaussian ellipsoids are sampled. '
            'Native hybrid images use all current-owner Gaussians. Completed-map orbit is a replay.'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def export(args):
    out=args.output.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    cache=out/'rendered_frames';cache.mkdir(exist_ok=True)
    decisions=edit_decisions(args.room,args.office)
    (out/'edit_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
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
            key=hashlib.sha256((json.dumps(spec,sort_keys=True)+renderer_hash).encode()).hexdigest()[:20]
            file=cache/f'{key}.png';rendered[json.dumps(spec,sort_keys=True)]=file
            if not file.exists():
                if active!=item['run']:
                    page.evaluate('(id)=>demo.loadRun(id)',item['run']);active=item['run']
                state=page.evaluate('(s)=>demo.show(s.index,s.options)',spec)
                assert state['index']==item['index'] and state['run']==item['run']
                assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()
                page.screenshot(path=str(file))
            if j%25==0:print(f'Render {j+1}/{len(jobs)}',flush=True)
        browser.close()
    assert not browser_errors,browser_errors
    def path(item):
        return rendered[json.dumps({k:v for k,v in item.items() if k!='duration'},sort_keys=True)]
    for key,name in [('full','EGAS_SLAM_Interview.mp4'),('preview','EGAS_SLAM_GitHub_Preview.mp4')]:
        lines=[]
        for item in decisions[key]:
            filename=str(path(item)).replace("'", "'\\''")
            lines.extend([f"file '{filename}'",f"duration {item['duration']:.9f}"])
        lines.append(lines[-2])
        concat=out/f'{key}_frames.txt';concat.write_text('\n'.join(lines)+'\n')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','warning','-y','-f','concat','-safe','0',
            '-i',str(concat),'-vf','fps=30','-c:v','libx264','-crf','19','-preset','medium',
            '-pix_fmt','yuv420p','-movflags','+faststart','-t',str(sum(x['duration'] for x in decisions[key])),
            str(out/name)],check=True)
    poster=out/'EGAS_SLAM_Poster.png';poster.write_bytes(path(decisions['poster']).read_bytes())
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','warning','-y','-i',str(out/'EGAS_SLAM_GitHub_Preview.mp4'),
        '-filter_complex','fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
        '-loop','0',str(out/'EGAS_SLAM_GitHub_Preview.gif')],check=True)
    report={'renderer_sha256':renderer_hash,'browser_errors':browser_errors,'videos':{},'files':{}}
    for name in ['EGAS_SLAM_Interview.mp4','EGAS_SLAM_GitHub_Preview.mp4','EGAS_SLAM_GitHub_Preview.gif']:
        file=out/name
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(file)],text=True))
        result=subprocess.run(['ffmpeg','-v','error','-i',str(file),'-f','null','-'],text=True,capture_output=True)
        assert result.returncode==0 and not result.stderr,(name,result.stderr)
        report['videos'][name]={'width':probe['streams'][0]['width'],'height':probe['streams'][0]['height'],
            'duration':float(probe['format']['duration']),'codec':probe['streams'][0]['codec_name'],'decode_errors':[]}
    for p in out.glob('EGAS_SLAM_*'):
        if p.is_file():report['files'][p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    report['selected_correction']=decisions['selected_correction']
    (out/'media_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    root=Path.home()/'egas_workspace'
    p.add_argument('--room',type=Path,default=root/'runs/demo_replica_room0_full')
    p.add_argument('--office',type=Path,default=root/'runs/demo_tum_office')
    p.add_argument('--output',type=Path,default=root/'deliverables/EGAS_SLAM_Demo_2026-11-03')
    p.add_argument('--url',default='http://127.0.0.1:8766')
    p.add_argument('--chrome',default='/usr/bin/google-chrome')
    export(p.parse_args())
