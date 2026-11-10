"""Standalone HTML 3D viewer: the part mesh plus a clickable feature list.

Clicking a feature highlights its faces on the model, so every row of
the feature table can be checked visually. Loads three.js from a CDN;
all model data is embedded in the page.
"""

import base64
import json
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def mesh_arrays(faces):
    """Concatenate each face's triangulation. Returns positions, indices, per-vertex face id."""
    pos, idx, fid = [], [], []
    offset = 0
    for i, face in enumerate(faces):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        n = tri.NbNodes()
        pts = np.empty((n, 3), dtype=np.float32)
        for k in range(1, n + 1):
            p = tri.Node(k).Transformed(trsf)
            pts[k - 1] = (p.X(), p.Y(), p.Z())
        tris = np.empty((tri.NbTriangles(), 3), dtype=np.uint32)
        rev = face.Orientation() == TopAbs_REVERSED
        for k in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(k).Get()
            tris[k - 1] = (a - 1, c - 1, b - 1) if rev else (a - 1, b - 1, c - 1)
        pos.append(pts)
        idx.append(tris + offset)
        fid.append(np.full(n, i, dtype=np.uint16))
        offset += n
    return np.vstack(pos), np.vstack(idx), np.concatenate(fid)


def write_viewer(result, faces, face_colors, path):
    pos, idx, fid = mesh_arrays(faces)
    colors = [face_colors.get(i, (160, 160, 160)) for i in range(len(faces))]
    feats = [{k: v for k, v in f.items() if k != "colors"} for f in result["features"]]
    data = {
        "part": result["part"],
        "features": feats,
        "patterns": result["patterns"],
        "faceColors": colors,
        "pos": _b64(pos.astype(np.float32)),
        "idx": _b64(idx.astype(np.uint32)),
        "fid": _b64(fid.astype(np.uint16)),
    }
    html = _TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    Path(path).write_text(html, encoding="utf-8")
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>STEP Feature Viewer</title>
<style>
:root{--bg:#f6f6f4;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e2dd;--hi:#ff7a00;--sel:#fff3e6}
*{box-sizing:border-box}
body{margin:0;font:13px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg);display:grid;grid-template-columns:420px 1fr;height:100vh}
#side{display:flex;flex-direction:column;border-right:1px solid var(--line);background:var(--panel);min-height:0}
header{padding:14px 16px;border-bottom:1px solid var(--line)}
h1{font-size:15px;margin:0 0 4px}
.meta{color:var(--muted);font-size:12px}
.controls{display:flex;gap:6px;padding:10px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
input,select{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:#fff}
input{flex:1;min-width:120px}
#list{overflow:auto;flex:1}
#groups{padding:10px 16px;border-bottom:1px solid var(--line)}
#groups .hd{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
#groups .hd b{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
#groups .hd button{font:inherit;font-size:11px;padding:2px 8px;border:1px solid var(--line);border-radius:5px;background:#fff;cursor:pointer;margin-left:4px}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:3px 9px 3px 6px;border:1px solid var(--line);border-radius:999px;cursor:pointer;font-size:12px;user-select:none;background:#fff}
.chip i{width:11px;height:11px;border-radius:50%;display:inline-block;border:1px solid rgba(0,0,0,.15)}
.chip.on{border-color:currentColor;box-shadow:0 0 0 1px currentColor inset;font-weight:600}
.chip .n{color:var(--muted);font-weight:400}
#patSel{margin-top:8px;width:100%}
.row{display:grid;grid-template-columns:38px 64px 1fr;gap:6px;padding:6px 16px;border-bottom:1px solid #f0f0ec;cursor:pointer}
.row:hover{background:#fafaf7}.row.sel{background:var(--sel)}
.id{color:var(--muted);font-variant-numeric:tabular-nums}
.type{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.pos{color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
#detail{border-top:1px solid var(--line);max-height:34vh;overflow:auto;padding:10px 16px;font:12px/1.45 ui-monospace,Consolas,monospace;white-space:pre-wrap;background:#fbfbf9}
#view{position:relative;min-width:0}
canvas{display:block;width:100%;height:100%}
#hint{position:absolute;right:12px;bottom:10px;color:var(--muted);font-size:11px;background:rgba(255,255,255,.8);padding:4px 8px;border-radius:6px}
#legend{position:absolute;left:12px;top:10px;background:rgba(255,255,255,.9);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:11px}
#legend div{display:flex;align-items:center;gap:6px}
#legend i{width:12px;height:12px;border-radius:3px;border:1px solid rgba(0,0,0,.2);display:inline-block}
@media (max-width:800px){body{grid-template-columns:1fr;grid-template-rows:55vh 45vh}#side{order:2}}
</style></head><body>
<div id="side">
  <header><h1 id="title"></h1><div class="meta" id="summary"></div></header>
  <div id="groups">
    <div class="hd"><b>Highlight groups</b><span><button id="gAll">All</button><button id="gNone">Clear</button></span></div>
    <div class="chips" id="chips"></div>
    <select id="patSel"><option value="">Highlight a pattern (identical features)...</option></select>
  </div>
  <div class="controls">
    <input id="q" placeholder="Search callout, e.g. Ø6.8 or pocket">
    <select id="type"><option value="">All types</option></select>
    <label class="meta" style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="showColors" checked style="flex:none;min-width:0">CAD colours</label>
    <label class="meta" style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="onlyHi" style="flex:none;min-width:0">List highlighted only</label>
  </div>
  <div id="list"></div>
  <div id="detail">Click a feature to highlight it on the model.</div>
</div>
<div id="view"><div id="legend"></div><div id="hint">drag: rotate · right-drag: pan · wheel: zoom · click model: select face's feature</div></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/controls/OrbitControls.js"></script>
<script>
const D=/*DATA*/null;
function dec(b64,T){const s=atob(b64),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new T(u.buffer)}
const pos=dec(D.pos,Float32Array),idx=dec(D.idx,Uint32Array),fid=dec(D.fid,Uint16Array);
document.getElementById('title').textContent=D.part.file;
const s=D.part.stock_size;
document.getElementById('summary').textContent=`${s.x} × ${s.y} × ${s.z} mm · ${D.features.length} features · ${D.part.face_count} faces`;
const faceToFeat={};D.features.forEach(f=>f.face_ids.forEach(i=>faceToFeat[i]=f.id));
const byId={};D.features.forEach(f=>byId[f.id]=f);

// legend of CAD colours
const cols={};D.faceColors.forEach(c=>{const k=c.join(',');cols[k]=(cols[k]||0)+1});
const cadLegend='<b>CAD face colours</b>'+Object.entries(cols).sort((a,b)=>b[1]-a[1]).map(([k,n])=>`<div><i style="background:rgb(${k})"></i>rgb(${k}) · ${n} faces</div>`).join('');
document.getElementById('legend').innerHTML=cadLegend;

// three.js scene
const view=document.getElementById('view');
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(devicePixelRatio);
view.prepend(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(0xf6f6f4);
const camera=new THREE.PerspectiveCamera(35,1,1,1e6);
const controls=new THREE.OrbitControls(camera,renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x888880,0.9));
const dl=new THREE.DirectionalLight(0xffffff,0.6);dl.position.set(1,1.5,2);scene.add(dl);
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
geo.setIndex(new THREE.BufferAttribute(idx,1));
geo.computeVertexNormals();
const colorAttr=new THREE.BufferAttribute(new Float32Array(pos.length),3);geo.setAttribute('color',colorAttr);
const mesh=new THREE.Mesh(geo,new THREE.MeshLambertMaterial({vertexColors:true,side:THREE.DoubleSide}));
scene.add(mesh);
const edges=new THREE.LineSegments(new THREE.EdgesGeometry(geo,25),new THREE.LineBasicMaterial({color:0x444440,transparent:true,opacity:.35}));
scene.add(edges);
geo.computeBoundingBox();const bb=geo.boundingBox,center=bb.getCenter(new THREE.Vector3()),size=bb.getSize(new THREE.Vector3()).length();
function home(){controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(.6,-.9,.8).normalize().multiplyScalar(size*1.3));camera.up.set(0,0,1);controls.update()}
home();

let selected=null;
// Feature groups (chip order). Where groups overlap, the more specific one colours the face.
const isHole=f=>f.type==='hole';
const GROUPS=[
  ['through','Through holes','#e03131',f=>isHole(f)&&f.subtype.includes('through')],
  ['blind','Blind holes','#1c7ed6',f=>isHole(f)&&f.subtype==='blind'],
  ['tapped','Tapped holes (likely)','#7048e8',f=>isHole(f)&&/tapped/.test(f.thread_hint||'')],
  ['cbore','Counterbore / countersink','#0c8599',f=>isHole(f)&&!!(f.counterbore||f.countersink)],
  ['pocket','Pockets','#2f9e44',f=>f.type==='pocket'],
  ['step','Steps & channels','#94d82d',f=>f.type==='step'||f.type==='channel'],
  ['slot','Slots','#f783ac',f=>f.type==='slot'],
  ['cutout','Through openings','#ae3ec9',f=>f.type==='cutout'],
  ['boss','Bosses','#5c940d',f=>f.type==='boss'],
  ['chamfer','Chamfers','#f59f00',f=>f.type==='chamfer'],
  ['fillet','Fillets (internal)','#15aabf',f=>f.type==='fillet'],
  ['round','Rounds (external)','#845ef7',f=>f.type==='round'],
  ['surface','Other surfaces','#868e96',f=>f.type==='surface'],
].map(([key,label,color,test])=>({key,label,color,test,members:D.features.filter(test)})).filter(g=>g.members.length);
const active=new Set();let patternSel=null;
const hex=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16)/255);
function faceHighlights(){
  const m=new Map();
  if(patternSel)D.features.filter(f=>f.pattern_id===patternSel).forEach(f=>f.face_ids.forEach(i=>m.set(i,[.12,.12,.12])));
  // specific groups (counterbore, tapped) paint over the general hole groups they overlap
  const SPECIFIC=new Set(['cbore','tapped']);
  [...GROUPS.filter(g=>SPECIFIC.has(g.key)),...GROUPS.filter(g=>!SPECIFIC.has(g.key))].forEach(g=>{if(!active.has(g.key))return;const c=hex(g.color);g.members.forEach(f=>f.face_ids.forEach(i=>{if(!m.has(i))m.set(i,c)}))});
  return m}
function highlightedIds(){
  const s=new Set();
  GROUPS.forEach(g=>{if(active.has(g.key))g.members.forEach(f=>s.add(f.id))});
  if(patternSel)D.features.forEach(f=>{if(f.pattern_id===patternSel)s.add(f.id)});
  return s}
function paint(){
  const showC=document.getElementById('showColors').checked;
  const sel=selected?new Set(byId[selected].face_ids):null;
  const hi=faceHighlights(),dim=!!sel||hi.size>0;
  for(let v=0;v<fid.length;v++){
    const f=fid[v];let r,g,b;
    if(sel&&sel.has(f)){r=1;g=.1;b=.55}
    else if(hi.has(f)){[r,g,b]=hi.get(f)}
    else{const c=showC&&!dim?D.faceColors[f]:[200,200,196];r=c[0]/255;g=c[1]/255;b=c[2]/255;if(dim){r=r*.3+.66;g=g*.3+.66;b=b*.3+.66}}
    colorAttr.setXYZ(v,r,g,b)}
  colorAttr.needsUpdate=true;renderLegend()}
function renderLegend(){
  const L=document.getElementById('legend');
  const on=GROUPS.filter(g=>active.has(g.key));
  if(on.length||patternSel){
    L.innerHTML='<b>Highlighted</b>'+on.map(g=>`<div><i style="background:${g.color}"></i>${g.label} · ${g.members.length}</div>`).join('')
      +(patternSel?`<div><i style="background:#1f1f1f"></i>Pattern ${patternSel}</div>`:'')
      +(selected?`<div><i style="background:#ff1a8c"></i>Selected #${selected}</div>`:'');
  }else L.innerHTML=cadLegend}
function toggleGroup(k){active.has(k)?active.delete(k):active.add(k);renderChips();paint();renderList()}
function renderChips(){
  document.getElementById('chips').innerHTML=GROUPS.map(g=>`<span class="chip${active.has(g.key)?' on':''}" data-g="${g.key}" style="${active.has(g.key)?'color:'+g.color:''}"><i style="background:${g.color}"></i><span style="color:var(--ink)">${g.label}</span><span class="n">${g.members.length}</span></span>`).join('');
  document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>toggleGroup(c.dataset.g))}
const ps=document.getElementById('patSel');
D.patterns.forEach(p=>{const o=document.createElement('option');o.value=p.pattern_id;o.textContent=`${p.pattern_id}: ${p.callout}`;ps.append(o)});
ps.onchange=()=>{patternSel=ps.value||null;paint();renderList()};
document.getElementById('gAll').onclick=()=>{GROUPS.forEach(g=>active.add(g.key));renderChips();paint();renderList()};
document.getElementById('gNone').onclick=()=>{active.clear();patternSel=null;ps.value='';selected=null;renderChips();paint();renderList();document.getElementById('detail').textContent='Click a feature to highlight it on the model.'};
function focus(f){
  const set=new Set(f.face_ids),box=new THREE.Box3(),v=new THREE.Vector3();
  for(let i=0;i<fid.length;i++)if(set.has(fid[i]))box.expandByPoint(v.set(pos[3*i],pos[3*i+1],pos[3*i+2]));
  if(box.isEmpty())return;const c=box.getCenter(new THREE.Vector3()),r=Math.max(box.getSize(new THREE.Vector3()).length(),size*.08);
  const dir=camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(c);camera.position.copy(c).add(dir.multiplyScalar(r*2.2));controls.update()}
function select(id,fly){
  selected=id;paint();
  document.querySelectorAll('.row').forEach(r=>r.classList.toggle('sel',+r.dataset.id===id));
  const f=byId[id];const show={...f};
  document.getElementById('detail').textContent=`#${f.id} ${f.callout}\n\n`+JSON.stringify(show,null,1);
  if(fly)focus(f);
  const row=document.querySelector(`.row[data-id="${id}"]`);if(row)row.scrollIntoView({block:'nearest'})}

// list
const types=[...new Set(D.features.map(f=>f.type))];
const ts=document.getElementById('type');types.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=`${t} (${D.features.filter(f=>f.type===t).length})`;ts.append(o)});
function renderList(){
  const q=document.getElementById('q').value.toLowerCase(),t=ts.value;
  const only=document.getElementById('onlyHi').checked,hs=highlightedIds();
  const L=document.getElementById('list');L.innerHTML='';
  D.features.filter(f=>(!only||!hs.size||hs.has(f.id))&&(!t||f.type===t)&&(!q||(f.callout+' '+f.subtype+' '+(f.pattern_id||'')).toLowerCase().includes(q))).forEach(f=>{
    const d=document.createElement('div');d.className='row'+(f.id===selected?' sel':'');d.dataset.id=f.id;
    const p=f.position_from_datum?f.position_from_datum.map(x=>x.toFixed(1)).join(', '):'';
    d.innerHTML=`<span class="id">${f.id}${f.pattern_id?'<br><span class=type>'+f.pattern_id+'</span>':''}</span><span class="type">${f.type}</span><span>${f.callout}<div class="pos">${p}</div></span>`;
    d.onclick=()=>select(f.id,true);L.append(d)})}
document.getElementById('q').oninput=renderList;ts.onchange=renderList;
document.getElementById('showColors').onchange=paint;
document.getElementById('onlyHi').onchange=renderList;
renderChips();renderList();paint();
const g0=location.hash.match(/g=([\w,]+)/);if(g0){g0[1].split(',').forEach(k=>active.add(k));renderChips();paint();renderList()}
const m0=location.hash.match(/f=(\d+)/);if(m0&&byId[+m0[1]])select(+m0[1],true);

// click on model
const ray=new THREE.Raycaster(),m=new THREE.Vector2();let down=null;
renderer.domElement.addEventListener('pointerdown',e=>down=[e.clientX,e.clientY]);
renderer.domElement.addEventListener('pointerup',e=>{
  if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>4)return;
  const r=renderer.domElement.getBoundingClientRect();m.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);
  ray.setFromCamera(m,camera);const hit=ray.intersectObject(mesh)[0];
  if(hit){const f=fid[hit.face.a];if(faceToFeat[f]!==undefined)select(faceToFeat[f],false)}});
function resize(){const w=view.clientWidth,h=view.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix()}
addEventListener('resize',resize);resize();
(function loop(){requestAnimationFrame(loop);renderer.render(scene,camera)})();
</script></body></html>
"""
