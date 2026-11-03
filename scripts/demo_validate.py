#!/usr/bin/env python3
"""Validate causal demo capture assets, transforms, image support and source receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def validate(run):
 run=Path(run).resolve();root=run/'map/demo';receipt=json.loads((run/'record/run.json').read_text())
 assert receipt['status']=='completed' and receipt['returncode']==0,'Incomplete native run'
 assert digest(run/'record/config.yaml')==receipt['config_sha256'],'Changed run configuration'
 timeline=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()]
 assert timeline and timeline[-1]['stage']=='terminal','Missing completed capture'
 checked=set();records=[];hashes={};psnr=[];max_delta=0.;previous=-1
 for row in timeline:
  d=json.loads((root/row['file']).read_text());records.append(d)
  assert d['frame']>=previous and d['frame']==row['frame'];previous=d['frame']
  owners={s['sdf_id'] for s in d['submaps']};assert d['owner'] in owners
  for s in d['submaps']:
   assert s['mesh_frame']<=d['frame'],'Future geometry in online snapshot'
   for key in ['local_to_global','sdf_local_to_global']:
    t=np.array(s[key]).reshape(4,4);assert np.isfinite(t).all()
    assert np.allclose(t[3],[0,0,0,1],atol=1e-5)
    assert abs(np.linalg.det(t[:3,:3])-1)<.003,'Non-rigid placement'
   for key in ['mesh','gaussians']:
    if not s[key] or s[key] in checked:continue
    path=(root/s[key]).resolve();assert path.is_relative_to(root) and path.is_file()
    raw=path.read_bytes();hashes[s[key]]=hashlib.sha256(raw).hexdigest();checked.add(s[key])
    if key=='mesh':
     h=np.frombuffer(raw,dtype='<u4',count=3);assert h[0]==0x3148534d
     n,k=map(int,h[1:]);assert len(raw)==16+n*24+k*4
     a=np.frombuffer(raw,dtype='<f4',count=n*6,offset=16).reshape(-1,6)
     idx=np.frombuffer(raw,dtype='<u4',count=k,offset=16+n*24)
     assert np.isfinite(a).all() and (not len(idx) or idx.max()<n) and k%3==0
    else:
     a=np.frombuffer(raw,dtype='<f4').reshape(-1,14);assert np.isfinite(a).all()
     # Learned quaternion parameters are unnormalised; both native rendering
     # and the viewer normalise them. Their magnitude is not a quality gate.
     assert (a[:,3:6]>0).all() and (np.linalg.norm(a[:,6:10],axis=1)>1e-8).all()
  images=[]
  for key in ['input','hybrid','sdf']:
   path=root/d[key]
   with Image.open(path) as im:
    assert im.size==(d['width'],d['height'])
    a=np.asarray(im.convert('RGB'),dtype=np.float32)/255.;images.append(a)
   hashes[d[key]]=digest(path)
  mse=float(np.mean((images[0]-images[1])**2));psnr.append(-10*np.log10(max(mse,1e-12)))
 events=[]
 if (root/'transforms.jsonl').exists():
  events=[json.loads(x) for x in (root/'transforms.jsonl').read_text().splitlines()]
  for e in events:
   before=np.linalg.inv(np.array(e['before_global_to_local']).reshape(4,4));after=np.linalg.inv(np.array(e['after_global_to_local']).reshape(4,4))
   max_delta=max(max_delta,float(np.linalg.norm(before[:3,3]-after[:3,3])))
 assert records[-1]['applied_transform_count']==len(events)
 previous_count=0
 for d in records:
  count=d['applied_transform_count'];assert count>=previous_count
  changed={e['owner']:e for e in events[previous_count:count]}
  for owner,e in changed.items():
   s=next(s for s in d['submaps'] if s['sdf_id']==owner)
   expected=np.linalg.inv(np.array(e['after_global_to_local']).reshape(4,4))
   assert np.allclose(np.array(s['sdf_local_to_global']).reshape(4,4),expected,atol=1e-4),'Uncommitted transform event'
   assert np.allclose(np.array(s['local_to_global']).reshape(4,4),expected,atol=1e-4),'Uncoupled Gaussian placement'
  previous_count=count
 assert previous==receipt['input_support']['frame_count']-1,'Capture does not cover the full run'
 summary={'passed':True,'run':str(run),'captures':len(records),'last_frame':previous,
 'processing_shape':receipt['input_support']['processing_shape'],'estimated_poses':not receipt['input_support']['ground_truth_used_for_tracking'],
 'unique_meshes':sum(k.endswith('.mesh.bin') for k in checked),'gaussian_snapshots':sum(k.endswith('.gaussians.bin') for k in checked),
 'captured_image_mean_psnr_db':float(np.mean(psnr)),'applied_owner_transform_events':len(events),
 'maximum_actual_owner_translation_metres':max_delta,'capture_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
 'scope':'Observed capture images and display meshes only; mesh simplified at 1 cm, sampled Gaussian ellipsoids. Capture overhead invalidates benchmark-speed use.',
 'binary_sha256':receipt['binary_sha256']}
 (run/'record/demo_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
 (run/'record/demo_asset_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print(json.dumps(summary,indent=2));return summary
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+');a=p.parse_args()
 for run in a.runs:validate(run)
