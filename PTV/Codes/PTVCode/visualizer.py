"""
visualizer.py
=============
Genera un visualizador HTML interactivo con slider temporal,
trayectorias de tracks y vectores de velocidad.
"""
from __future__ import annotations
import base64
import json
from pathlib import Path

import cv2

from .models import Track
from .image_utils import natural_key


def create_interactive_visualizer(
    ann_dir: Path,
    tracks: list[Track],
    out_path: Path,
    width_px: int,
    height_px: int,
    fps: float,
) -> None:
    """
    Crea visualizer.html con:
    - Slider de frames con play/pause y atajos de teclado
    - Trayectorias coloreadas por track ID
    - Vectores de velocidad opcionales
    - Panel de estadísticas y lista de tracks activos
    """
    ann_images = sorted(ann_dir.glob("*.png"), key=lambda p: natural_key(p.name))
    if not ann_images:
        print("[WARN] No hay imágenes anotadas para el visualizador", flush=True)
        return

    print(f"[PTV] Generando visualizador con {len(ann_images)} frames...", flush=True)

    frames_data = []
    for img_path in ann_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        _, buffer = cv2.imencode(".png", img)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        frames_data.append({
            "name": img_path.stem,
            "data": f"data:image/png;base64,{img_b64}",
        })

    tracks_by_frame: dict[int, list] = {}
    for track in tracks:
        for rec in track.history:
            fi = rec.frame_idx
            if fi not in tracks_by_frame:
                tracks_by_frame[fi] = []
            history_pts = [
                {"x": r.x, "y": r.y}
                for r in track.history if r.frame_idx <= fi
            ]
            tracks_by_frame[fi].append({
                "track_id": track.track_id,
                "x": rec.x,
                "y": rec.y,
                "vx": rec.vx,
                "vy": rec.vy,
                "angle_deg": rec.angle_deg,
                "history": history_pts,
            })

    n_frames = len(frames_data)
    first_name = frames_data[0]["name"] if frames_data else ""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>PTV Tracking Visualizer</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#1a1a1a;color:#e0e0e0;display:flex;flex-direction:column;height:100vh;overflow:hidden}}
#header{{background:#2d2d2d;padding:12px 20px;border-bottom:2px solid #404040}}
h1{{font-size:18px;font-weight:600;color:#00d4ff}}
#main{{flex:1;display:flex;overflow:hidden}}
#canvas-container{{flex:1;display:flex;align-items:center;justify-content:center;background:#0a0a0a}}
canvas{{max-width:100%;max-height:100%;border:1px solid #404040}}
#sidebar{{width:280px;background:#2d2d2d;border-left:1px solid #404040;padding:16px;overflow-y:auto}}
#controls{{background:#2d2d2d;padding:16px 20px;border-top:1px solid #404040}}
.ctrl{{margin-bottom:14px}}
label{{display:block;margin-bottom:6px;font-size:12px;color:#b0b0b0}}
input[type=range]{{width:100%;height:5px;border-radius:3px;background:#404040;outline:none;-webkit-appearance:none}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#00d4ff;cursor:pointer}}
.row{{display:flex;justify-content:space-between;font-size:13px;margin-top:6px}}
.val{{color:#00d4ff;font-weight:600}}
.play-row{{display:flex;gap:8px;margin-top:12px}}
button{{flex:1;padding:8px;background:#404040;border:none;border-radius:4px;color:#e0e0e0;font-size:12px;cursor:pointer}}
button:hover{{background:#505050}}
button.active{{background:#00d4ff;color:#1a1a1a}}
.chk{{display:flex;align-items:center;gap:8px;margin-bottom:10px;font-size:12px}}
.stats{{margin-top:14px;padding:12px;background:#1a1a1a;border-radius:4px;font-size:12px}}
.stats h3{{font-size:13px;color:#00d4ff;margin-bottom:8px}}
.stat{{display:flex;justify-content:space-between;margin-bottom:6px}}
.track-list{{margin-top:14px}}
.track-item{{padding:7px;margin-bottom:4px;background:#1a1a1a;border-radius:3px;font-size:11px;display:flex;justify-content:space-between}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px}}
</style>
</head>
<body>
<div id="header"><h1>PTV Tracking Visualizer</h1></div>
<div id="main">
  <div id="canvas-container"><canvas id="cv" width="{width_px}" height="{height_px}"></canvas></div>
  <div id="sidebar">
    <div class="chk"><input type="checkbox" id="chkTraj" checked><label style="margin:0">Trayectorias</label></div>
    <div class="chk"><input type="checkbox" id="chkIDs" checked><label style="margin:0">IDs</label></div>
    <div class="chk"><input type="checkbox" id="chkVec"><label style="margin:0">Vectores velocidad</label></div>
    <div class="ctrl"><label>Grosor: <span id="lwVal">2</span>px</label><input type="range" id="lw" min="1" max="5" value="2" step="0.5"></div>
    <div class="ctrl"><label>Histórico: <span id="tlVal">Completo</span></label><input type="range" id="tl" min="0" max="100" value="0"></div>
    <div class="stats">
      <h3>Frame actual</h3>
      <div class="stat"><span style="color:#b0b0b0">Frame</span><span class="val" id="sFr">1</span></div>
      <div class="stat"><span style="color:#b0b0b0">Tracks</span><span class="val" id="sTr">0</span></div>
      <div class="stat"><span style="color:#b0b0b0">Vel. media</span><span class="val" id="sVel">0 px/s</span></div>
    </div>
    <div class="track-list"><div id="tList"><strong style="font-size:12px;color:#00d4ff">Tracks visibles</strong></div></div>
  </div>
</div>
<div id="controls">
  <div class="ctrl">
    <label>Frame: <span class="val" id="fLabel">1 / {n_frames}</span></label>
    <input type="range" id="fSlider" min="0" max="{n_frames - 1}" value="0" step="1">
    <div class="row"><span id="fName">{first_name}</span><span id="tLabel">0.00 s</span></div>
  </div>
  <div class="play-row">
    <button id="bPrev">◄</button>
    <button id="bPlay">▶ Play</button>
    <button id="bNext">►</button>
  </div>
  <div class="ctrl" style="margin-top:12px">
    <label>Velocidad: <span id="spVal">1x</span></label>
    <input type="range" id="spSlider" min="0.25" max="4" value="1" step="0.25">
  </div>
</div>
<script>
const FD={json.dumps(frames_data)};
const TBF={json.dumps(tracks_by_frame)};
const FPS={fps};
const cv=document.getElementById('cv');
const ctx=cv.getContext('2d');
let cur=0,playing=false,iv=null,spd=1;
const tc={{}};
function tcolor(id){{if(!tc[id]){{const h=(id*137.508)%360;tc[id]=`hsl(${{h}},70%,60%)`;}}return tc[id];}}
function draw(fi){{
  ctx.clearRect(0,0,cv.width,cv.height);
  if(fi>=FD.length)return;
  const img=new Image();
  img.onload=()=>{{
    ctx.drawImage(img,0,0);
    const tracks=TBF[fi]||[];
    const showT=document.getElementById('chkTraj').checked;
    const showI=document.getElementById('chkIDs').checked;
    const showV=document.getElementById('chkVec').checked;
    const lw=parseFloat(document.getElementById('lw').value);
    const tl=parseInt(document.getElementById('tl').value);
    if(showT){{tracks.forEach(t=>{{
      let h=t.history;
      if(tl>0&&h.length>tl)h=h.slice(-tl);
      if(h.length<2)return;
      ctx.strokeStyle=tcolor(t.track_id);ctx.lineWidth=lw;
      ctx.beginPath();ctx.moveTo(h[0].x,h[0].y);
      for(let i=1;i<h.length;i++)ctx.lineTo(h[i].x,h[i].y);
      ctx.stroke();
    }});}}
    tracks.forEach(t=>{{
      const c=tcolor(t.track_id);
      ctx.fillStyle=c;ctx.beginPath();ctx.arc(t.x,t.y,5,0,Math.PI*2);ctx.fill();
      if(showV){{const vm=Math.sqrt(t.vx*t.vx+t.vy*t.vy);if(vm>0.1){{
        const sc=2,dx=t.vx*sc,dy=t.vy*sc,ang=Math.atan2(dy,dx),as=8;
        ctx.strokeStyle=c;ctx.lineWidth=2;
        ctx.beginPath();ctx.moveTo(t.x,t.y);ctx.lineTo(t.x+dx,t.y+dy);ctx.stroke();
        ctx.beginPath();ctx.moveTo(t.x+dx,t.y+dy);
        ctx.lineTo(t.x+dx-as*Math.cos(ang-Math.PI/6),t.y+dy-as*Math.sin(ang-Math.PI/6));
        ctx.moveTo(t.x+dx,t.y+dy);
        ctx.lineTo(t.x+dx-as*Math.cos(ang+Math.PI/6),t.y+dy-as*Math.sin(ang+Math.PI/6));
        ctx.stroke();
      }}}}
      if(showI){{ctx.fillStyle=c;ctx.font='bold 12px monospace';ctx.fillText('ID '+t.track_id,t.x+8,t.y-8);}}
    }});
    const tk=TBF[fi]||[];
    document.getElementById('sFr').textContent=fi+1;
    document.getElementById('sTr').textContent=tk.length;
    const av=tk.length?tk.reduce((s,t)=>s+Math.sqrt(t.vx*t.vx+t.vy*t.vy),0)/tk.length:0;
    document.getElementById('sVel').textContent=av.toFixed(1)+' px/s';
    document.getElementById('tList').innerHTML='<strong style="font-size:12px;color:#00d4ff">Tracks visibles</strong>'+
      tk.map(t=>`<div class="track-item"><span><span class="dot" style="background:${{tcolor(t.track_id)}}"></span>Track ${{t.track_id}}</span><span style="color:#b0b0b0">${{Math.sqrt(t.vx*t.vx+t.vy*t.vy).toFixed(1)}} px/s</span></div>`).join('');
  }};
  img.src=FD[fi].data;
}}
function upd(fi){{
  cur=fi;
  document.getElementById('fSlider').value=fi;
  document.getElementById('fLabel').textContent=(fi+1)+' / '+FD.length;
  document.getElementById('fName').textContent=FD[fi].name;
  document.getElementById('tLabel').textContent=(fi/FPS).toFixed(2)+' s';
  draw(fi);
}}
function play(){{
  if(playing){{playing=false;document.getElementById('bPlay').textContent='▶ Play';document.getElementById('bPlay').classList.remove('active');clearInterval(iv);return;}}
  playing=true;document.getElementById('bPlay').textContent='⏸ Pause';document.getElementById('bPlay').classList.add('active');
  iv=setInterval(()=>{{upd(cur>=FD.length-1?0:cur+1);}},1000/(FPS*spd));
}}
document.getElementById('fSlider').addEventListener('input',e=>{{if(playing)play();upd(parseInt(e.target.value));}});
document.getElementById('bPrev').addEventListener('click',()=>{{if(playing)play();upd(Math.max(0,cur-1));}});
document.getElementById('bNext').addEventListener('click',()=>{{if(playing)play();upd(Math.min(FD.length-1,cur+1));}});
document.getElementById('bPlay').addEventListener('click',play);
document.getElementById('spSlider').addEventListener('input',e=>{{spd=parseFloat(e.target.value);document.getElementById('spVal').textContent=spd+'x';if(playing){{play();play();}}}});
document.getElementById('lw').addEventListener('input',e=>{{document.getElementById('lwVal').textContent=e.target.value;draw(cur);}});
document.getElementById('tl').addEventListener('input',e=>{{const v=parseInt(e.target.value);document.getElementById('tlVal').textContent=v===0?'Completo':v+' frames';draw(cur);}});
['chkTraj','chkIDs','chkVec'].forEach(id=>document.getElementById(id).addEventListener('change',()=>draw(cur)));
document.addEventListener('keydown',e=>{{
  if(e.key===' '){{e.preventDefault();play();}}
  else if(e.key==='ArrowLeft'){{if(playing)play();upd(Math.max(0,cur-1));}}
  else if(e.key==='ArrowRight'){{if(playing)play();upd(Math.min(FD.length-1,cur+1));}}
}});
upd(0);
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"[PTV] Visualizador creado: {out_path}", flush=True)
