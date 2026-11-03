import * as THREE from 'three';
const $=id=>document.getElementById(id), palette=['#65b9f0','#ed9cb7','#e3bb6b','#8ac8a3','#aa9cde','#ee9572','#72c8cc'];
const colour=id=>palette[((id%palette.length)+palette.length)%palette.length];
const matrix=a=>new THREE.Matrix4().fromArray(a).transpose();
const json=async p=>{const r=await fetch(p);if(!r.ok)throw new Error(`${r.status}: ${p}`);return r.json()};
const fail=e=>{$('error').hidden=false;$('error').textContent=String(e.stack||e);console.error(e)};
window.addEventListener('unhandledrejection',e=>fail(e.reason));
const assets=new Map(), sphere=new THREE.SphereGeometry(1,7,5);
let story,configs,phase='',captureKey='',currentData=null,pair=[];
const views={};
function makeView(id){
 const el=$(id),renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.localClippingEnabled=true;el.appendChild(renderer.domElement);
 const overlay=document.createElement('canvas');overlay.className='overlay';el.appendChild(overlay);
 const scene=new THREE.Scene();scene.background=new THREE.Color('#111f29');scene.add(new THREE.AmbientLight(0xffffff,2));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(-2,-5,-3);scene.add(light);
 const camera=new THREE.PerspectiveCamera(42,1,.01,100),group=new THREE.Group();scene.add(group);
 return {id,el,renderer,overlay,scene,camera,group,path:[],markers:[],edge:null,arrows:[]};
}
for(const id of ['map','local','returnmap','before','after'])views[id]=makeView(id);
function clear(v){for(const g of [...v.group.children]){v.group.remove(g);g.traverse(o=>{if(o.material)o.material.dispose();if(o.isInstancedMesh)o.dispose()})}v.path=[];v.ghostPath=[];v.markers=[];v.edge=null;v.arrows=[]}
async function asset(run,file,type,needed){
 const key=run+'/'+file;needed.add(key);if(assets.has(key))return assets.get(key);
 const r=await fetch(`/data/${run}/${file}`);if(!r.ok)throw new Error(`Missing ${key}`);const buffer=await r.arrayBuffer();let value;
 if(type==='mesh'){const h=new Uint32Array(buffer,0,4);if(h[0]!==0x3148534d)throw new Error('Expected indexed captured mesh');const a=new Float32Array(buffer,16,h[1]*6),positions=new Float32Array(h[1]*3),colors=new Float32Array(h[1]*3);for(let i=0;i<h[1];i++){positions.set(a.subarray(i*6,i*6+3),i*3);colors.set(a.subarray(i*6+3,i*6+6),i*3)}value=new THREE.BufferGeometry();value.setAttribute('position',new THREE.BufferAttribute(positions,3));value.setAttribute('color',new THREE.BufferAttribute(colors,3));value.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer,16+h[1]*24,h[2]),1));value.computeVertexNormals();value.computeBoundingBox();}
 else{value=new Float32Array(buffer);if(value.length%14)throw new Error('Invalid Gaussian stride')}
 assets.set(key,value);return value;
}
function clips(cfg){const p=cfg.clip?[new THREE.Plane(new THREE.Vector3(...cfg.clip.normal),cfg.clip.constant)]:[];if(cfg.bounds)for(let a=0;a<3;a++){const n=new THREE.Vector3();n.setComponent(a,1);p.push(new THREE.Plane(n,-cfg.bounds.min[a]),new THREE.Plane(n.clone().negate(),cfg.bounds.max[a]))}return p}
function mesh(g,s,cfg,tinted=false,placement=null){const mat=new THREE.MeshStandardMaterial({color:tinted?colour(s.id):0xffffff,vertexColors:!tinted,roughness:1,side:THREE.DoubleSide,clippingPlanes:clips(cfg)});if(tinted)mat.color.lerp(new THREE.Color('#84929e'),.32);const o=new THREE.Mesh(g,mat);o.matrixAutoUpdate=false;o.matrix.copy(matrix(placement||s.sdf_local_to_global));return o}
function gaussians(a,s,cfg,placement=null){const n=a.length/14,stride=Math.max(1,Math.ceil(n/1800)),count=Math.ceil(n/stride);const o=new THREE.InstancedMesh(sphere,new THREE.MeshStandardMaterial({color:colour(s.id),roughness:.7,transparent:true,opacity:.88,clippingPlanes:clips(cfg)}),count);const p=new THREE.Vector3(),q=new THREE.Quaternion(),scale=new THREE.Vector3(),m=new THREE.Matrix4();let j=0;for(let i=0;i<n;i+=stride){const k=i*14;p.fromArray(a,k);scale.fromArray(a,k+3).multiplyScalar(2);q.set(a[k+7],a[k+8],a[k+9],a[k+6]).normalize();o.setMatrixAt(j++,m.compose(p,q,scale))}o.matrixAutoUpdate=false;o.matrix.copy(matrix(placement||s.local_to_global));return o}
function pathFor(snapshots,d,through=d.frame){return snapshots.filter(x=>x.stage==='online'&&x.frame<=through).map(x=>{const s=d.submaps.find(s=>s.sdf_id===x.owner),m=s?matrix(s.sdf_local_to_global).multiply(matrix(x.camera_to_owner)):matrix(x.camera_to_world_raw);return new THREE.Vector3().setFromMatrixPosition(m)})}
function origin(d,owner){const s=d.submaps.find(s=>s.sdf_id===owner);return s?new THREE.Vector3().setFromMatrixPosition(matrix(s.sdf_local_to_global)):null}
function prune(needed){for(const [key,a] of assets)if(!needed.has(key)){if(a.isBufferGeometry)a.dispose();assets.delete(key)}}
async function buildMapping(d){
 const run=story.runs.room.id,cfg=configs.room,needed=new Set();clear(views.map);clear(views.local);
 const useful=d.submaps.filter(s=>s.mesh&&!s.candidate),owner=useful.find(s=>s.sdf_id===d.owner),rank=useful.findIndex(s=>s.id===owner?.id);pair=useful.slice(Math.max(0,rank-1),Math.max(0,rank-1)+2);if(pair.length<2)pair=useful.slice(-2);
 const groups=new Map();for(const s of pair){const g=new THREE.Group();views.local.group.add(g);groups.set(s.id,g)}
 for(const s of useful){const g=await asset(run,s.mesh,'mesh',needed);views.map.group.add(mesh(g,s,cfg));if(groups.has(s.id)){groups.get(s.id).add(mesh(g,s,cfg,true));if(s.gaussians)groups.get(s.id).add(gaussians(await asset(run,s.gaussians,'gs',needed),s,cfg))}}
 views.map.path=pathFor(story.runs.room.snapshots,d);views.map.pathColor='#67ddc5';views.map.markers=[{point:views.map.path.at(-1),colour:'#67ddc5'}];
 $('local').classList.toggle('split',pair.length===2);$('localnames').innerHTML=pair.map(s=>`<span style="color:${colour(s.id)}">SUBMAP ${s.id}</span>`).join('');
 $('localnote').innerHTML=pair.map(s=>`<i class="dot" style="background:${colour(s.id)}"></i>${s.frozen?'Frozen appearance':s.sdf_id===d.owner?'Current map':'Overlapping map'}`).join(' &nbsp; ');
 $('mapcount').textContent=`${useful.length} local map${useful.length===1?'':'s'} · ${d.resident_sdf_maps} SDF map${d.resident_sdf_maps===1?'':'s'} on GPU`;
 await setImage('hybrid',`/data/${run}/${d.hybrid}`);$('hybridframe').textContent=`Captured map update · frame ${d.frame}`;prune(needed);
}
async function buildRevisit(d){
 const run=story.runs.office.id,cfg=configs.office,needed=new Set(),v=views.returnmap;clear(v);
 for(const s of d.submaps.filter(s=>s.mesh&&!s.candidate)){const g=await asset(run,s.mesh,'mesh',needed);const o=mesh(g,s,cfg,s.sdf_id===0||s.sdf_id===18);if(s.sdf_id===0||s.sdf_id===18)o.material.color.set(s.sdf_id===0?'#f2bc73':'#67ddc5');v.group.add(o);}
 v.path=pathFor(story.runs.office.snapshots,d);v.pathColor='#67ddc5';v.markers=[{point:origin(d,0),label:'Earlier submap 0',colour:'#f2bc73'},{point:origin(d,18),label:'Current submap 18',colour:'#67ddc5'}];
 const loop=d.loop_constraints.some(c=>c.from===0&&c.to===18);v.edge=loop?[origin(d,0),origin(d,18)]:null;$('edgelegend').textContent=loop?'Orange connection: verified revisit between submaps 0 and 18':'Watch for a verified revisit';
 $('matchstate').innerHTML=loop?'RGB-D MATCH<br>VERIFIED':'RETURNING TO<br>THE SAME DESK';document.querySelector('.match').classList.toggle('verified',loop);
 $('revisitstatus').textContent=loop?'Colour and depth verify the earlier view':'A new view of the previously mapped desk';
 $('revisitMapTitle').textContent=d.applied_transform_count?'Loop closure corrects the map':loop?'A verified revisit links two submaps':'Recognise a previously mapped place';
 $('revisitMapNote').textContent=d.applied_transform_count?'Earlier submap placements and camera poses are updated.':loop?'The match adds a constraint between submaps 0 and 18.':'The current view is compared with stored earlier views.';
 prune(needed);
}
async function buildComparison(){
 const r=story.runs.office,old=r.snapshots[story.comparison.beforeIndex],now=r.snapshots[story.comparison.afterIndex],cfg=configs.office,needed=new Set();clear(views.before);clear(views.after);
 // Identical retained local geometry: only native owner placements differ.
 for(const s of old.submaps.filter(s=>s.mesh&&!s.candidate)){const after=now.submaps.find(x=>x.sdf_id===s.sdf_id);if(!after)continue;const g=await asset(r.id,s.mesh,'mesh',needed);views.before.group.add(mesh(g,s,cfg));views.after.group.add(mesh(g,s,cfg,false,after.sdf_local_to_global));}
 views.before.path=pathFor(r.snapshots,old,old.frame);views.before.pathColor='#f2bc73';views.after.path=pathFor(r.snapshots,now,old.frame);views.after.pathColor='#67ddc5';views.after.ghostPath=views.before.path.map(p=>p.clone());prune(needed);
}
function setCamera(v,cfg,kind,orbit=0){
 const c=cfg[kind],target=new THREE.Vector3(...c.target),delta=new THREE.Vector3(...c.position).sub(target),up=new THREE.Vector3(...c.up);
 // Fixed inspection cameras only; owner transforms are never interpolated.
 const aspect=v.el.clientWidth/v.el.clientHeight;
 let factor=v.id==='map'?Math.max(1,1.85/aspect):v.id==='local'?Math.max(1,2.9/aspect):v.id==='returnmap'?1.02:1.12;
 delta.multiplyScalar(factor).applyAxisAngle(up,orbit);v.camera.up.copy(up);v.camera.position.copy(target).add(delta);v.camera.lookAt(target);const w=v.el.clientWidth,h=v.el.clientHeight;if(v.id==='map'||v.id==='local')v.camera.setViewOffset(v.id==='local'?w/2:w,h,0,h*.10,v.id==='local'?w/2:w,h);else v.camera.clearViewOffset();
}
function projected(v,p){if(!p)return null;const q=p.clone().project(v.camera);return {x:(q.x+1)*v.el.clientWidth/2,y:(1-q.y)*v.el.clientHeight/2,visible:q.z>-1&&q.z<1}}
function paintOverlay(v){
 const w=v.el.clientWidth,h=v.el.clientHeight,c=v.overlay.getContext('2d');if(v.overlay.width!==w||v.overlay.height!==h){v.overlay.width=w;v.overlay.height=h}c.clearRect(0,0,w,h);
 if(v.ghostPath?.length){c.strokeStyle='#f2bc73';c.lineWidth=3;c.setLineDash([7,5]);c.beginPath();v.ghostPath.forEach((p,i)=>{const q=projected(v,p);if(i)c.lineTo(q.x,q.y);else c.moveTo(q.x,q.y)});c.stroke();c.setLineDash([])}
 if(v.path.length){c.strokeStyle=v.pathColor;c.lineWidth=2.5;c.beginPath();v.path.forEach((p,i)=>{const q=projected(v,p);if(i)c.lineTo(q.x,q.y);else c.moveTo(q.x,q.y)});c.stroke()}
 if(v.edge){const [a,b]=v.edge.map(p=>projected(v,p));c.strokeStyle='#f2bc73';c.lineWidth=4;c.setLineDash([9,6]);c.beginPath();c.moveTo(a.x,a.y);c.lineTo(b.x,b.y);c.stroke();c.setLineDash([])}
 for(const mark of v.markers){const p=projected(v,mark.point);if(!p?.visible)continue;c.fillStyle=mark.colour;c.beginPath();c.arc(p.x,p.y,6,0,Math.PI*2);c.fill();if(mark.label){c.font='14px Arial';const width=c.measureText(mark.label).width;let x=Math.max(8,Math.min(w-width-16,p.x+12)),y=Math.max(22,Math.min(h-15,p.y-12));c.fillStyle='#0b141be8';c.fillRect(x-5,y-17,width+10,23);c.fillStyle=mark.colour;c.fillText(mark.label,x,y)}}
 for(const [pa,pb] of v.arrows){const a=projected(v,pa),b=projected(v,pb);const angle=Math.atan2(b.y-a.y,b.x-a.x);c.strokeStyle='#f6f1df';c.fillStyle='#f6f1df';c.lineWidth=2.5;c.beginPath();c.moveTo(a.x,a.y);c.lineTo(b.x,b.y);c.stroke();c.beginPath();c.moveTo(b.x,b.y);c.lineTo(b.x-7*Math.cos(angle-.45),b.y-7*Math.sin(angle-.45));c.lineTo(b.x-7*Math.cos(angle+.45),b.y-7*Math.sin(angle+.45));c.closePath();c.fill()}
}
function draw(v){const w=v.el.clientWidth,h=v.el.clientHeight;if(!w||!h)return;if(v.renderer.domElement.width!==w||v.renderer.domElement.height!==h)v.renderer.setSize(w,h,false);const split=v.id==='local'&&v.group.children.length===2;v.camera.aspect=(split?w/2:w)/h;v.camera.updateProjectionMatrix();if(split){v.renderer.setScissorTest(true);for(let j=0;j<2;j++){v.group.children.forEach((g,k)=>g.visible=j===k);v.renderer.setViewport(j*w/2,0,w/2,h);v.renderer.setScissor(j*w/2,0,w/2,h);v.renderer.render(v.scene,v.camera)}v.group.children.forEach(g=>g.visible=true);v.renderer.setScissorTest(false)}else{v.renderer.setViewport(0,0,w,h);v.renderer.render(v.scene,v.camera)}paintOverlay(v)}
async function setImage(id,url){const im=$(id);if(im.getAttribute('src')===url&&im.complete&&im.naturalWidth)return;im.src=url;await im.decode()}
function source(r,frame){return `${r.sourcePrefix}frame${String(frame).padStart(6,'0')}${r.sourceExtension}`}
async function show(s){
 if(phase!==s.phase){phase=s.phase;document.body.dataset.phase=phase;captureKey='';await new Promise(requestAnimationFrame);const order=['mapping','revisit','comparison'];document.querySelectorAll('nav div').forEach(el=>{el.classList.toggle('active',el.dataset.step===phase);el.classList.toggle('done',order.indexOf(el.dataset.step)<order.indexOf(phase))});for(const v of Object.values(views))clear(v)}
 if(phase==='mapping'){
  const r=story.runs.room,d=r.snapshots[s.captureIndex];if(captureKey!==String(s.captureIndex)){await buildMapping(d);captureKey=String(s.captureIndex)}currentData=d;
  await setImage('rgb',source(r,s.frame));$('rgbframe').textContent=`Continuous camera sequence · frame ${s.frame}`;
  $('headline').textContent=s.t<14?'Build a detailed map from colour and depth.':'New submaps overlap with earlier ones as mapping continues.';
  $('dataset').textContent='REPLICA · ROOM0 · 1200 × 680';$('footleft').textContent='SDF geometry + Gaussian appearance · estimated camera poses';$('scope').textContent='Recorded excerpt · map updates sampled · presentation timing';
  setCamera(views.map,configs.room,'global',s.orbit);setCamera(views.local,configs.room,'local',s.orbit);draw(views.map);draw(views.local);
 }else if(phase==='revisit'){
  const r=story.runs.office,d=r.snapshots[s.captureIndex];if(captureKey!==String(s.captureIndex)){await buildRevisit(d);captureKey=String(s.captureIndex)}currentData=d;
  await Promise.all([setImage('reference',story.referenceImage),setImage('current',source(r,s.frame))]);$('revisitframe').textContent=`Continuous camera sequence · frame ${s.frame}`;
  $('headline').textContent='Loop closure starts by recognising a place seen before.';$('dataset').textContent='TUM RGB-D · REAL OFFICE';$('footleft').textContent='Earlier view: frame 1 · supporting revisit: frame 1920';$('scope').textContent='Camera images follow recorded sensor timestamps · map updates sampled';
  setCamera(views.returnmap,configs.office,'global',s.orbit);draw(views.returnmap);
 }else if(phase==='comparison'){
  if(!captureKey){await buildComparison();captureKey='comparison'}currentData=story.runs.office.snapshots[story.comparison.afterIndex];
  $('headline').textContent='The verified revisit corrects the map and past camera poses.';$('dataset').textContent='TUM RGB-D · RECORDED LOOP CORRECTION';$('footleft').textContent='Before / after comparison · same captured contents and viewing angle';$('scope').textContent='Inspection camera moves · actual placement changes, without exaggeration';
  for(const id of ['before','after']){setCamera(views[id],configs.office,'global',s.orbit);draw(views[id])}
 }else throw new Error(`Unknown phase ${phase}`);
 const total=story.timing.mappingSeconds+story.timing.revisitSeconds+story.timing.comparisonSeconds;
 const elapsed=(phase==='mapping'?0:phase==='revisit'?story.timing.mappingSeconds:story.timing.mappingSeconds+story.timing.revisitSeconds)+s.t;$('progress').style.width=`${Math.min(100,100*elapsed/total)}%`;
 window.githubState={phase,frame:s.frame,captureFrame:currentData.frame,captureIndex:s.captureIndex,comparison:phase==='comparison',beforeIndex:story.comparison.beforeIndex,afterIndex:story.comparison.afterIndex};return window.githubState;
}
try{[story,configs]=await Promise.all([json('/presentation/story.json'),Promise.all([json('/scenes/replica_room0.json'),json('/scenes/tum_fr3_office.json')]).then(([room,office])=>({room,office}))]);window.githubDemo={show,story};window.githubReady=true}catch(e){fail(e)}
