import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
const $=id=>document.getElementById(id), palette=['#65b9f0','#ed9cb7','#e3bb6b','#8ac8a3','#aa9cde','#ee9572','#72c8cc'];
const error=e=>{$('error').hidden=false;$('error').textContent=String(e?.stack||e);console.error(e)};
window.addEventListener('unhandledrejection',e=>error(e.reason));
const matrix=a=>new THREE.Matrix4().fromArray(a).transpose();
const colour=id=>palette[((id%palette.length)+palette.length)%palette.length];
let run='',frames=[],index=0,playing=false,busy=false,showGaussians=true,lastData=null,history=[],fixedViews=null;
let events=[],cameraPath=[],lastTimestamp=0;
const assetCache=new Map(), frameCache=new Map();
const sphere=new THREE.SphereGeometry(1,7,5);
const sceneConfigs={};
const fetchJSON=async url=>{const r=await fetch(url);if(!r.ok)throw new Error(`${r.status}: ${url}`);return r.json()};
const root=()=>`/data/${encodeURIComponent(run)}/`;
async function frameData(i){const key=frames[i].file;if(!frameCache.has(key))frameCache.set(key,await fetchJSON(root()+key));return frameCache.get(key)}
async function asset(file,type){
 const key=run+'/'+file;if(assetCache.has(key))return assetCache.get(key);
 const r=await fetch(root()+file);if(!r.ok)throw new Error(`Missing captured asset ${file}`);
 const buffer=await r.arrayBuffer(),a=new Float32Array(buffer);let value;
 if(type==='mesh'){
  const head=new Uint32Array(buffer,0,3);let n,start,indices=null;
  if(head[0]===0x3148534d){n=head[1];start=4;indices=new Uint32Array(buffer,16+n*24,head[2]);}
  else{if(a.length%18)throw new Error('Invalid mesh stride');n=a.length/6;start=0;}
  const p=new Float32Array(n*3),c=new Float32Array(n*3);
  for(let i=0;i<n;i++){p.set(a.subarray(start+i*6,start+i*6+3),i*3);c.set(a.subarray(start+i*6+3,start+i*6+6),i*3)}
  value=new THREE.BufferGeometry();value.setAttribute('position',new THREE.BufferAttribute(p,3));value.setAttribute('color',new THREE.BufferAttribute(c,3));if(indices)value.setIndex(new THREE.BufferAttribute(indices,1));value.computeVertexNormals();value.computeBoundingBox();
 }else{if(a.length%14)throw new Error('Invalid Gaussian stride');value=a}
 assetCache.set(key,value);return value;
}
function clear(group){for(const child of [...group.children]){group.remove(child);child.traverse(o=>{if(o.material){for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose()}if(o.geometry?.userData?.temporary)o.geometry.dispose()})}}
function line(points,col,opacity=1){const g=new THREE.BufferGeometry().setFromPoints(points);g.userData.temporary=true;return new THREE.Line(g,new THREE.LineBasicMaterial({color:col,transparent:opacity<1,opacity,depthTest:false}))}
function makeView(id){
 const el=$(id),scene=new THREE.Scene();scene.background=new THREE.Color('#101c25');
 const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.localClippingEnabled=true;
 el.appendChild(renderer.domElement);const camera=new THREE.PerspectiveCamera(42,1,.01,100);camera.up.set(0,-1,0);camera.position.set(2,-3,-3);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;controls.target.set(0,0,1);controls.update();
 scene.add(new THREE.AmbientLight(0xffffff,2));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(-2,-5,-3);scene.add(light);
 const group=new THREE.Group();scene.add(group);
 const view={id,el,scene,renderer,camera,controls,group,clip:null};
 const resize=()=>{const w=el.clientWidth,h=el.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);draw(view)};
 new ResizeObserver(resize).observe(el);controls.addEventListener('change',()=>draw(view));resize();return view;
}
const views=[makeView('global'),makeView('local'),makeView('correction')];
function draw(view){
 const r=view.renderer,w=view.el.clientWidth,h=view.el.clientHeight;
 if(view.id==='local'&&view.group.children.length===2){
  r.setScissorTest(true);view.camera.aspect=(w/2)/h;view.camera.updateProjectionMatrix();
  for(let j=0;j<2;j++){view.group.children.forEach((g,k)=>g.visible=j===k);r.setViewport(j*w/2,0,w/2,h);r.setScissor(j*w/2,0,w/2,h);r.render(view.scene,view.camera)}
  view.group.children.forEach(g=>g.visible=true);r.setScissorTest(false);r.setViewport(0,0,w,h);
 }else{r.setScissorTest(false);r.setViewport(0,0,w,h);view.camera.aspect=w/h;view.camera.updateProjectionMatrix();r.render(view.scene,view.camera)}
}

function setView(view,position,target,up=[0,-1,0]){view.camera.up.fromArray(up);view.camera.position.fromArray(position);view.controls.target.fromArray(target);view.controls.update()}
function autoFit(view,scale=1){
 const box=new THREE.Box3().setFromObject(view.group);if(box.isEmpty())return;
 const centre=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
 const radius=Math.max(size.x,size.y,size.z,.5)*scale;
 const offset=new THREE.Vector3(.85,-.9,-1.05).normalize().multiplyScalar(radius*1.55);
 setView(view,centre.clone().add(offset).toArray(),centre.toArray());
}
function meshObject(geom,submap,mode='natural',opacity=1,before=null){
 const m=new THREE.MeshStandardMaterial({color:mode==='natural'?0xffffff:mode==='neutral'?0x90a1ac:colour(submap.id),vertexColors:mode==='natural',roughness:1,metalness:0,side:THREE.DoubleSide,transparent:opacity<1,opacity,polygonOffset:true,polygonOffsetFactor:mode==='natural'?0:1});
 if(mode==='local')m.color.lerp(new THREE.Color('#84929e'),.5);
 const mesh=new THREE.Mesh(geom,m);mesh.matrixAutoUpdate=false;mesh.matrix.copy(before||matrix(submap.sdf_local_to_global));return mesh;
}
function gaussianObject(data,s){
 const n=data.length/14,stride=Math.max(1,Math.ceil(n/2500)),count=Math.ceil(n/stride);
 const mesh=new THREE.InstancedMesh(sphere,new THREE.MeshStandardMaterial({color:colour(s.id),roughness:.7,metalness:.05,transparent:true,opacity:.87}),count);
 const p=new THREE.Vector3(),q=new THREE.Quaternion(),scale=new THREE.Vector3(),m=new THREE.Matrix4();let j=0;
 for(let i=0;i<n;i+=stride){const off=i*14;p.fromArray(data,off);scale.fromArray(data,off+3).multiplyScalar(2);q.set(data[off+7],data[off+8],data[off+9],data[off+6]).normalize();m.compose(p,q,scale);mesh.setMatrixAt(j++,m)}
 mesh.matrixAutoUpdate=false;mesh.matrix.copy(matrix(s.local_to_global));mesh.instanceMatrix.needsUpdate=true;return mesh;
}
function cameraIcon(pose){
 const group=new THREE.Group(),coords=[[0,0,0],[-.09,-.055,.14],[.09,-.055,.14],[.09,.055,.14],[-.09,.055,.14]];
 for(const idx of [[0,1],[0,2],[0,3],[0,4],[1,2,3,4,1]])group.add(line(idx.map(i=>new THREE.Vector3(...coords[i])),'#62dbc2'));
 group.matrixAutoUpdate=false;group.matrix.copy(matrix(pose));return group;
}
function correctedCamera(d){const s=d.submaps.find(s=>s.sdf_id===d.owner);return s?matrix(s.sdf_local_to_global).multiply(matrix(d.camera_to_owner)):matrix(d.camera_to_world_raw)}
function pathFor(d){return history.filter(x=>x.frame<=d.frame&&x.stage==='online').map(x=>{
 const owner=d.submaps.find(s=>s.sdf_id===x.owner);const m=owner?matrix(owner.sdf_local_to_global).multiply(matrix(x.camera_to_owner)):matrix(x.camera_to_world_raw);return new THREE.Vector3().setFromMatrixPosition(m)
})}
function matrixRows(m){return m.clone().transpose().toArray()}
function currentEvent(d){const before=events.filter(e=>e.frame<=d.frame);if(!before.length)return null;const latest=before[before.length-1];let start=latest.frame;for(let i=before.length-2;i>=0;i--){if(start-before[i].frame>8)break;start=before[i].frame}return {start,end:latest.frame,items:before.filter(e=>e.frame>=start),recent:d.frame-latest.frame<100}}
function setImages(d){return Promise.all(['input','hybrid'].map(id=>new Promise((resolve,reject)=>{const im=$(id);im.onload=resolve;im.onerror=()=>reject(new Error(`Image failed ${d[id]}`));im.src=root()+d[id];if(im.complete&&im.naturalWidth)resolve()})))}
async function show(i,opts={}){
 if(!frames.length)return;
 index=Math.min(Math.max(0,i),frames.length-1);const d=await frameData(index);lastData=d;
 const cfg=sceneConfigs[run];
 for(const v of views)clear(v.group);
 const useful=d.submaps.filter(s=>s.mesh&&!s.candidate);
 const owner=d.submaps.find(s=>s.sdf_id===d.owner),rank=useful.findIndex(s=>s.id===owner?.id);
 let pair=useful.slice(Math.max(0,rank-1),Math.max(0,rank-1)+2);if(pair.length<2)pair=useful.slice(-2);
 const event=currentEvent(d),needed=new Set();
 const localGroups=new Map();for(const s of pair){const group=new THREE.Group();views[1].group.add(group);localGroups.set(s.id,group)}
 for(const s of useful){needed.add(run+'/'+s.mesh);const g=await asset(s.mesh,'mesh');
  views[0].group.add(meshObject(g,s,'natural'));
  if(cfg?.correctionOwner===undefined||s.sdf_id===cfg.correctionOwner)
   views[2].group.add(meshObject(g,s,s.sdf_id===(cfg?.correctionOwner??d.owner)?'owner':'neutral',1));
  if(cfg?.correctionOwner===s.sdf_id&&s.gaussians){needed.add(run+'/'+s.gaussians);views[2].group.add(gaussianObject(await asset(s.gaussians,'gaussian'),s))}
  if(pair.some(p=>p.id===s.id)){
   localGroups.get(s.id).add(meshObject(g,s,showGaussians?'local':'owner',1));
   if(showGaussians&&s.gaussians){needed.add(run+'/'+s.gaussians);localGroups.get(s.id).add(gaussianObject(await asset(s.gaussians,'gaussian'),s))}
  }
 }
 // Actual old placements, no fabricated warp or exaggerated displacement.
 if(event?.recent){const first=new Map();for(const e of event.items)if(!first.has(e.owner))first.set(e.owner,e);
  for(const s of useful){const e=first.get(s.sdf_id);if(!e||(cfg?.correctionOwner!==undefined&&s.sdf_id!==cfg.correctionOwner))continue;const geom=await asset(s.mesh,'mesh');const ghost=meshObject(geom,s,'owner',.16,matrix(e.before_global_to_local).invert());ghost.material.color.set('#ffffff');ghost.material.wireframe=true;ghost.material.depthWrite=false;views[2].group.add(ghost)}
 }
 const path=pathFor(d);if(path.length>1){views[0].group.add(line(path,'#62dbc2'));views[2].group.add(line(path,'#62dbc2',.8))}
 const pose=matrixRows(correctedCamera(d));views[0].group.add(cameraIcon(pose));
 // Verified graph edges connect the actual submap coordinate origins.
 for(const c of d.loop_constraints){const a=d.submaps.find(s=>s.id===c.from),b=d.submaps.find(s=>s.id===c.to);if(!a||!b)continue;
  const pa=new THREE.Vector3().setFromMatrixPosition(matrix(a.sdf_local_to_global)),pb=new THREE.Vector3().setFromMatrixPosition(matrix(b.sdf_local_to_global));
  views[2].group.add(line([pa,pb],'#f2bd69'));
 }
 if(opts.fit||!fixedViews){for(const v of views)autoFit(v,v.id==='local' ? .87 : 1);fixedViews=true}
 if(cfg){for(const v of views){const c=cfg[v.id];if(c)setView(v,c.position,c.target,c.up)}}
 if(opts.orbit){for(const v of [views[0],views[1]]){const delta=v.camera.position.clone().sub(v.controls.target);delta.applyAxisAngle(v.camera.up,opts.orbit);v.camera.position.copy(v.controls.target).add(delta);v.controls.update()}}
 if(opts.camera){for(const v of views){if(opts.camera[v.id]){const c=opts.camera[v.id];setView(v,c.position,c.target,c.up)}}}
 for(const v of views){
  const clip=cfg?.clip;const planes=clip?[new THREE.Plane(new THREE.Vector3(...clip.normal),clip.constant)]:[];
  if(cfg?.bounds){for(let axis=0;axis<3;axis++){const n=new THREE.Vector3();n.setComponent(axis,1);planes.push(new THREE.Plane(n,-cfg.bounds.min[axis]));planes.push(new THREE.Plane(n.clone().negate(),cfg.bounds.max[axis]))}}
  v.group.traverse(o=>{if(o.isMesh)o.material.clippingPlanes=planes});draw(v);
 }
 $('scene').textContent=d.dataset==='replica'?`REPLICA · ${d.scene.toUpperCase()}`:'TUM RGB-D · OFFICE';
 $('framebadge').textContent=`FRAME ${String(d.frame).padStart(4,'0')}  /  ${d.stage==='terminal'?'COMPLETED MAP':'RECORDED ONLINE STATE'}`;
 $('chaptertitle').textContent=opts.title||(d.dataset==='replica'?'Building a room, one local map at a time.':'Recognise a revisit. Correct the map.');
 $('submaps').textContent=useful.length;$('gaussians').textContent=d.submaps.reduce((n,s)=>n+s.gaussian_count,0).toLocaleString();$('resident').textContent=d.resident_sdf_maps;
 $('state').textContent=event?.recent?(d.pending_transforms?'APPLYING RECORDED CORRECTION':'RECORDED MAP CORRECTION'):d.stage==='terminal'?'RECONSTRUCTION COMPLETE':'RECONSTRUCTING';
 $('local').classList.toggle('split',pair.length===2);
 $('locallabels').innerHTML=pair.map(s=>`<span style="color:${colour(s.id)}">SUBMAP ${s.id}</span>`).join('');
 $('ownerlegend').innerHTML=pair.map(s=>`<span class="dot" style="background:${colour(s.id)}"></span> Submap ${s.id}${s.frozen?' · frozen':s.sdf_id===d.owner?' · primary':' · overlap'}`).join('&nbsp;&nbsp;&nbsp;')+`<span class="rightnote">${showGaussians?'Same viewpoint · SDF + sampled 2σ Gaussians':'Owner-coloured SDF surfaces'}</span>`;
 $('eventstate').textContent=event?.recent?'VERIFIED REVISIT · ACTUAL CORRECTION':event?'MAP CORRECTION RETAINED':'SEARCHING FOR REVISITS';
 $('eventtext').textContent=event?.recent?'Geometry and appearance move together':event?'Corrected submap placements':'Local maps share a global frame';
 $('eventdetail').textContent=event?.recent?`${new Set(event.items.map(e=>e.owner)).size} owners updated so far · ${d.pending_transforms} queued. Faint wireframe: previous placement.`:'The highlighted local map belongs to the current camera. A verified revisit can adjust submap placements without rebuilding their contents.';
 if(cfg?.correctionOwner!==undefined){
  const selected=event?.items.find(e=>e.owner===cfg.correctionOwner);
  if(selected&&event.recent){const old=matrix(selected.before_global_to_local).invert(),s=d.submaps.find(s=>s.sdf_id===cfg.correctionOwner);
   const shift=new THREE.Vector3().setFromMatrixPosition(old).distanceTo(new THREE.Vector3().setFromMatrixPosition(matrix(s.sdf_local_to_global)));
   $('eventdetail').textContent=`${new Set(event.items.map(e=>e.owner)).size} submaps updated · selected submap: ${(shift*100).toFixed(1)} cm translation. White outline: previous placement.`;
  }else if(!event){$('eventtext').textContent=`Submap ${cfg.correctionOwner}: correction detail`;$('eventdetail').textContent='Selected local geometry and Gaussian detail. This owner will be compared at its recorded placements.'}
 }
 $('loopnote').textContent=(cfg?.correctionOwner!==undefined?`Detail: submap ${cfg.correctionOwner} · `:'')+(event?'rigid correction · actual displacement':'No applied loop correction in the captured state');
 $('seek').value=index;
 const imageData=opts.imageIndex!==undefined?await frameData(opts.imageIndex):d;
 $('inputdesc').textContent=opts.imageIndex!==undefined?`Recorded input and native hybrid pair · frame ${imageData.frame}`:'The same viewpoint, captured at the same processing stage';
 await setImages(imageData);
 // Bound CPU/GPU asset retention. Keep current views only plus a small working set.
 for(const [key,value] of assetCache){if(assetCache.size<=18)break;if(!needed.has(key)){if(value.isBufferGeometry)value.dispose();assetCache.delete(key)}}
 window.demoState={run,index,frame:d.frame,stage:d.stage,submaps:useful.length,applied:d.applied_transform_count,errors:0};
 return window.demoState;
}
async function loadRun(id,{fit=true}={}){
 playing=false;const changed=run!==id;run=id;frames=await fetchJSON(`/api/frames?run=${encodeURIComponent(id)}`);frameCache.clear();history=[];if(changed)fixedViews=null;
 $('run').value=id;$('seek').max=Math.max(0,frames.length-1);
 try{const r=await fetch(root()+'transforms.jsonl');events=r.ok?(await r.text()).trim().split('\n').filter(Boolean).map(JSON.parse):[]}catch{events=[]}
 // Metadata only; no future geometry is shown. Used to draw the past camera path.
 for(let i=0;i<frames.length;i++)history.push(await frameData(i));
 if(frames.length){const first=await frameData(0);if(!sceneConfigs[id]){try{const r=await fetch(`/scenes/${first.dataset}_${first.scene}.json`);if(r.ok)sceneConfigs[id]=await r.json()}catch{}}
  await show(Math.min(index,frames.length-1),{fit});}
 return frames.length;
}
async function refresh(){
 const catalog=await fetchJSON('/api/catalog');const selection=$('run');
 if(selection.options.length!==catalog.length){selection.innerHTML='';for(const r of catalog){const option=new Option(r.id,r.id);selection.add(option)}}
 if(!run&&catalog.length)await loadRun(catalog[0].id);
 if($('follow').checked&&run&&!busy){busy=true;try{const next=await fetchJSON(`/api/frames?run=${encodeURIComponent(run)}`);if(next.length>frames.length){await loadRun(run,{fit:false});await show(frames.length-1)}}finally{busy=false}}
}
$('run').onchange=()=>loadRun($('run').value).catch(error);
$('seek').oninput=async()=>{playing=false;if(busy)return;busy=true;try{await show(Number($('seek').value))}catch(e){error(e)}finally{busy=false}};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play'};
$('fit').onclick=()=>{for(const v of views){autoFit(v);draw(v)}};
$('gs').onclick=()=>{showGaussians=!showGaussians;show(index).catch(error)};
async function tick(t){if(playing&&!busy&&t-lastTimestamp>100){lastTimestamp=t;busy=true;try{await show((index+1)%frames.length)}catch(e){error(e)}finally{busy=false}}requestAnimationFrame(tick)}
window.demo={loadRun,show,frames:()=>frames,config:(id,cfg)=>sceneConfigs[id]=cfg,setGaussians:v=>showGaussians=v,
 cameras:()=>Object.fromEntries(views.map(v=>[v.id,{position:v.camera.position.toArray(),target:v.controls.target.toArray(),up:v.camera.up.toArray()}])),
 record:v=>document.body.classList.toggle('recording',v),setDisclosure:t=>$('disclosure').textContent=t};
await refresh();setInterval(()=>refresh().catch(error),1500);requestAnimationFrame(tick);window.demoReady=true;
