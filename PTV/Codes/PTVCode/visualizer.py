"""
visualizer.py
=============
Genera un visualizador HTML interactivo con slider temporal,
trayectorias de tracks y vectores de velocidad.

Unidades
--------
Los TrackRecord guardan posición en mm (x_mm, y_mm) y velocidad en mm/s
(vx_mm_s, vy_mm_s). Los frames anotados están en píxeles, así que todo lo que
se dibuja sobre el canvas se convierte con px_per_mm. Las velocidades que se
muestran como texto quedan en mm/s.

Correspondencia frame ↔ registro
--------------------------------
Cada PNG anotado se llama <stem de la imagen>.png y cada TrackRecord guarda el
image_name de la imagen en que se detectó. Se emparejan por ese stem, así que
funciona igual con schedule por regiones temporales (frames saltados) que sin él.
"""
from __future__ import annotations
import base64
import json
from pathlib import Path

import cv2

from .models import Track
from .image_utils import natural_key


# Longitud de las flechas de velocidad: desplazamiento que recorrería la fibra
# en este intervalo. 0.05 s → una fibra a 100 mm/s con 7 px/mm dibuja ~35 px.
ARROW_SECONDS = 0.05


def create_interactive_visualizer(
    ann_dir: Path,
    tracks: list[Track],
    out_path: Path,
    width_px: int,
    height_px: int,
    fps: float,
    px_per_mm: float,
    jpeg_quality: int = 85,
) -> None:
    """
    Crea visualizer.html con:
    - Slider de frames con play/pause y atajos de teclado
    - Control de velocidad de reproducción
    - Trayectorias coloreadas por track ID
    - Vectores de velocidad opcionales
    - Panel de estadísticas y lista de tracks activos

    Args:
        ann_dir:      carpeta con los PNG anotados (uno por frame procesado).
        tracks:       tracks filtrados; su history está en mm y mm/s.
        out_path:     ruta del HTML a escribir.
        width_px:     ancho del frame en píxeles.
        height_px:    alto del frame en píxeles.
        fps:          frecuencia de adquisición de la cámara [Hz].
        px_per_mm:    escala para convertir mm → px al dibujar.
        jpeg_quality: calidad de los frames embebidos. Se usa JPEG en vez de
                      PNG porque el HTML lleva todos los frames en base64:
                      con PNG un video de ~1000 frames de 1024×1024 supera
                      los 2 GB y el navegador no lo abre.
    """
    if px_per_mm <= 0:
        raise ValueError(f"px_per_mm debe ser > 0 (recibido {px_per_mm})")

    ann_images = sorted(ann_dir.glob("*.png"), key=lambda p: natural_key(p.name))
    if not ann_images:
        print("[WARN] No hay imágenes anotadas para el visualizador", flush=True)
        return

    print(f"[PTV] Generando visualizador con {len(ann_images)} frames...", flush=True)

    # 1. Frames embebidos en base64 (JPEG)
    frames_data: list[dict] = []
    pos_by_stem: dict[str, int] = {}
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    total_bytes = 0
    for img_path in ann_images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] No se pudo leer {img_path.name}, se omite", flush=True)
            continue
        ok, buffer = cv2.imencode(".jpg", img, encode_params)
        if not ok:
            print(f"[WARN] No se pudo codificar {img_path.name}, se omite", flush=True)
            continue
        img_b64 = base64.b64encode(buffer).decode("ascii")
        total_bytes += len(img_b64)
        pos_by_stem[img_path.stem] = len(frames_data)
        frames_data.append({
            "name": img_path.stem,
            "data": f"data:image/jpeg;base64,{img_b64}",
            "t": None,          # timestamp real, se completa con los registros
        })

    if not frames_data:
        print("[WARN] Ninguna imagen anotada se pudo leer", flush=True)
        return

    # 2. Tracks: trayectoria completa una sola vez + estado por frame
    #    TRK[id]  = [[pos, x_px, y_px], ...]   (ordenado por pos)
    #    TBF[pos] = [{id, x, y, vx, vy, v}, ...] con x/y/vx/vy en px y v en mm/s
    tracks_polyline: dict[int, list[list[float]]] = {}
    tracks_by_frame: dict[int, list[dict]] = {}
    unmatched = 0

    for track in tracks:
        points: list[list[float]] = []
        for rec in sorted(track.history, key=lambda r: r.frame_idx):
            pos = pos_by_stem.get(Path(rec.image_name).stem)
            if pos is None:
                unmatched += 1
                continue

            x_px = rec.x_mm * px_per_mm
            y_px = rec.y_mm * px_per_mm
            vx_px = rec.vx_mm_s * px_per_mm
            vy_px = rec.vy_mm_s * px_per_mm
            speed_mm_s = (rec.vx_mm_s ** 2 + rec.vy_mm_s ** 2) ** 0.5

            points.append([pos, round(x_px, 2), round(y_px, 2)])
            tracks_by_frame.setdefault(pos, []).append({
                "id": track.track_id,
                "x":  round(x_px, 2),
                "y":  round(y_px, 2),
                "vx": round(vx_px, 2),
                "vy": round(vy_px, 2),
                "v":  round(speed_mm_s, 2),
            })
            if frames_data[pos]["t"] is None:
                frames_data[pos]["t"] = round(float(rec.timestamp_s), 5)

        if points:
            points.sort(key=lambda p: p[0])
            tracks_polyline[track.track_id] = points

    if unmatched:
        print(
            f"[WARN] {unmatched} registros de tracks sin PNG anotado "
            f"correspondiente; no se dibujan",
            flush=True,
        )

    n_frames = len(frames_data)
    n_tracks = len(tracks_polyline)
    first_name = frames_data[0]["name"]
    playback_fps = max(1, min(int(round(fps)), 30))
    size_mb = total_bytes / 1e6
    print(f"[PTV] Tamaño estimado del visualizador: {size_mb:.0f} MB", flush=True)
    if size_mb > 1000:
        print(
            "[WARN] El visualizador supera 1 GB; puede no abrir en el navegador. "
            "Baja jpeg_quality o procesa menos frames.",
            flush=True,
        )

    def _js(obj) -> str:
        # Evita que un "</script>" dentro de los datos cierre el bloque de script.
        return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")

    # 3. Generación del HTML
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PTV Tracking Visualizer</title>
<style>
:root{{
  --bg:#0f1216; --panel:#171b21; --panel-2:#1f242c; --border:#2b313b;
  --text:#e7e9ec; --muted:#98a1ab; --accent:#5ab8f0; --accent-2:#8fd0f7;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{height:100%}}
body{{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);
  display:flex;flex-direction:column;overflow:hidden;font-size:13px;-webkit-font-smoothing:antialiased}}
header{{display:flex;align-items:baseline;gap:14px;padding:12px 20px;background:var(--panel);border-bottom:1px solid var(--border)}}
header h1{{font-size:16px;font-weight:600;letter-spacing:.2px}}
.chips{{display:flex;gap:8px;flex-wrap:wrap}}
.chip{{font-size:11px;color:var(--muted);background:var(--panel-2);border:1px solid var(--border);border-radius:999px;padding:2px 9px}}
#main{{flex:1;display:flex;min-height:0}}
#stage{{flex:1;display:flex;align-items:center;justify-content:center;padding:16px;min-width:0;
  background:radial-gradient(ellipse at center,#161a20 0%,var(--bg) 70%)}}
canvas{{max-width:100%;max-height:100%;border-radius:4px;box-shadow:0 8px 30px rgba(0,0,0,.55)}}
aside{{width:290px;background:var(--panel);border-left:1px solid var(--border);display:flex;flex-direction:column;min-height:0}}
.section{{padding:14px 16px;border-bottom:1px solid var(--border)}}
.section h2{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:10px}}
.chk{{display:flex;align-items:center;gap:9px;margin-bottom:8px;cursor:pointer;user-select:none}}
.chk input{{accent-color:var(--accent);width:14px;height:14px;cursor:pointer}}
.field{{margin-top:10px}}
.field label{{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-bottom:6px}}
.field label span{{color:var(--text);font-variant-numeric:tabular-nums}}
input[type=range]{{width:100%;height:4px;border-radius:2px;background:var(--border);outline:none;-webkit-appearance:none;appearance:none;cursor:pointer}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--accent);border:2px solid var(--bg);box-shadow:0 0 0 1px var(--accent)}}
input[type=range]::-moz-range-thumb{{width:12px;height:12px;border-radius:50%;background:var(--accent);border:2px solid var(--bg)}}
.stats{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}}
.stat{{background:var(--panel-2);border:1px solid var(--border);border-radius:6px;padding:8px}}
.stat .k{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}}
.stat .v{{font-size:15px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}}
.stat .u{{font-size:10px;color:var(--muted);font-weight:400;margin-left:2px}}
#tracksSection{{flex:1;display:flex;flex-direction:column;min-height:0;border-bottom:none}}
#tList{{overflow-y:auto;min-height:0;flex:1;margin:0 -6px;padding:0 6px}}
.track-item{{display:flex;justify-content:space-between;align-items:center;padding:6px 8px;border-radius:5px;font-size:12px;font-variant-numeric:tabular-nums}}
.track-item:hover{{background:var(--panel-2)}}
.dot{{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:8px;box-shadow:0 0 0 2px rgba(255,255,255,.12)}}
.empty{{color:var(--muted);font-size:12px;padding:6px 2px}}
footer{{background:var(--panel);border-top:1px solid var(--border);padding:12px 20px 14px}}
.timeline{{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:8px;font-variant-numeric:tabular-nums}}
#fLabel{{font-weight:600}}
#fName{{color:var(--muted);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;flex:1;text-align:center}}
#tLabel{{font-weight:600;color:var(--accent-2)}}
.transport{{display:flex;align-items:center;gap:14px;margin-top:12px}}
button{{padding:7px 14px;background:var(--panel-2);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:12px;cursor:pointer;min-width:92px}}
button:hover{{border-color:var(--accent);color:#fff}}
button.active{{background:var(--accent);border-color:var(--accent);color:#0b1014;font-weight:600}}
.speed{{display:flex;align-items:center;gap:10px;margin-left:auto;color:var(--muted);font-size:12px;width:320px}}
.speed input{{flex:1}}
#fpsVal{{color:var(--text);font-variant-numeric:tabular-nums;min-width:22px;text-align:right}}
kbd{{font-family:inherit;font-size:10px;color:var(--muted);border:1px solid var(--border);border-bottom-width:2px;border-radius:3px;padding:0 4px;margin-left:6px}}
</style>
</head>
<body>
<header>
  <h1>PTV Tracking Visualizer</h1>
  <div class="chips">
    <span class="chip">{n_frames} frames</span>
    <span class="chip">{n_tracks} tracks</span>
    <span class="chip">adquisición {fps:g} Hz</span>
    <span class="chip">{px_per_mm:g} px/mm</span>
  </div>
</header>
<div id="main">
  <div id="stage"><canvas id="cv" width="{width_px}" height="{height_px}"></canvas></div>
  <aside>
    <div class="section">
      <h2>Capas</h2>
      <label class="chk"><input type="checkbox" id="chkTraj" checked>Trayectorias</label>
      <label class="chk"><input type="checkbox" id="chkIDs" checked>Identificadores</label>
      <label class="chk"><input type="checkbox" id="chkVec">Vectores de velocidad ({ARROW_SECONDS * 1000:.0f} ms)</label>
      <div class="field"><label>Grosor <span id="lwVal">2</span></label><input type="range" id="lw" min="1" max="5" value="2" step="0.5"></div>
      <div class="field"><label>Histórico <span id="tlVal">Completo</span></label><input type="range" id="tl" min="0" max="100" value="0"></div>
    </div>
    <div class="section">
      <h2>Frame actual</h2>
      <div class="stats">
        <div class="stat"><div class="k">Frame</div><div class="v" id="sFr">1</div></div>
        <div class="stat"><div class="k">Tracks</div><div class="v" id="sTr">0</div></div>
        <div class="stat"><div class="k">Vel. media</div><div class="v"><span id="sVel">0</span><span class="u">mm/s</span></div></div>
      </div>
    </div>
    <div class="section" id="tracksSection">
      <h2>Tracks visibles</h2>
      <div id="tList"></div>
    </div>
  </aside>
</div>
<footer>
  <div class="timeline">
    <span id="fLabel">1 / {n_frames}</span>
    <span id="fName">{first_name}</span>
    <span id="tLabel">—</span>
  </div>
  <input type="range" id="fSlider" min="0" max="{n_frames - 1}" value="0" step="1">
  <div class="transport">
    <button id="bPrev">◀ Anterior<kbd>←</kbd></button>
    <button id="bPlay">▶ Play<kbd>espacio</kbd></button>
    <button id="bNext">Siguiente ▶<kbd>→</kbd></button>
    <div class="speed">Reproducción<input type="range" id="fpsSlider" min="1" max="60" value="{playback_fps}" step="1"><span id="fpsVal">{playback_fps}</span> fps</div>
  </div>
</footer>
<script>
const FD={_js(frames_data)};
const TRK={_js(tracks_polyline)};
const TBF={_js(tracks_by_frame)};
const ARROW_S={ARROW_SECONDS};
const cv=document.getElementById('cv');
const ctx=cv.getContext('2d');
let cur=0, playing=false, iv=null;
let targetFPS={playback_fps};

const tc={{}};
function tcolor(id){{if(!tc[id]){{const h=(id*137.508)%360;tc[id]=`hsl(${{h}},78%,62%)`;}}return tc[id];}}
const HALO='rgba(10,12,15,0.75)';

function strokePath(pts,color,width,alpha){{
  if(pts.length<2)return;
  ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width;
  ctx.beginPath();ctx.moveTo(pts[0][1],pts[0][2]);
  for(let i=1;i<pts.length;i++)ctx.lineTo(pts[i][1],pts[i][2]);
  ctx.stroke();
}}

function drawTrail(h,color,lw){{
  // halo continuo y cola en tramos que se atenúan hacia el pasado
  strokePath(h,HALO,lw+3,1);
  const bands=Math.min(6,h.length-1);
  for(let b=0;b<bands;b++){{
    const a=Math.floor(b*(h.length-1)/bands), z=Math.floor((b+1)*(h.length-1)/bands);
    const w=(b+1)/bands;
    strokePath(h.slice(a,z+1),color,lw*(0.55+0.45*w),0.25+0.75*w);
  }}
  ctx.globalAlpha=1;
}}

function drawLabel(text,x,y,color){{
  ctx.font='600 12px system-ui,-apple-system,Segoe UI,sans-serif';
  const w=ctx.measureText(text).width+10;
  ctx.globalAlpha=0.8;ctx.fillStyle='#0b0e12';ctx.fillRect(x,y-15,w,18);
  ctx.globalAlpha=1;ctx.fillStyle=color;ctx.fillRect(x,y-15,3,18);
  ctx.fillStyle='#f2f4f6';ctx.fillText(text,x+6,y-2);
}}

function draw(fi){{
  if(fi>=FD.length)return;
  const img=new Image();
  img.onload=()=>{{
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.drawImage(img,0,0);
    ctx.lineCap='round';ctx.lineJoin='round';
    const tracks=TBF[fi]||[];
    const showT=document.getElementById('chkTraj').checked;
    const showI=document.getElementById('chkIDs').checked;
    const showV=document.getElementById('chkVec').checked;
    const lw=parseFloat(document.getElementById('lw').value);
    const tl=parseInt(document.getElementById('tl').value);

    if(showT){{
      tracks.forEach(t=>{{
        let h=(TRK[t.id]||[]).filter(p=>p[0]<=fi);
        if(tl>0&&h.length>tl)h=h.slice(-tl);
        if(h.length>=2)drawTrail(h,tcolor(t.id),lw);
      }});
    }}

    tracks.forEach(t=>{{
      const c=tcolor(t.id);
      if(showV&&t.v>0.1){{
        const dx=t.vx*ARROW_S,dy=t.vy*ARROW_S,ang=Math.atan2(dy,dx),as=7;
        const x2=t.x+dx,y2=t.y+dy;
        [[HALO,4.5],[c,2]].forEach(([col,wid])=>{{
          ctx.strokeStyle=col;ctx.lineWidth=wid;
          ctx.beginPath();ctx.moveTo(t.x,t.y);ctx.lineTo(x2,y2);
          ctx.moveTo(x2,y2);ctx.lineTo(x2-as*Math.cos(ang-Math.PI/6),y2-as*Math.sin(ang-Math.PI/6));
          ctx.moveTo(x2,y2);ctx.lineTo(x2-as*Math.cos(ang+Math.PI/6),y2-as*Math.sin(ang+Math.PI/6));
          ctx.stroke();
        }});
      }}
      ctx.fillStyle=HALO;ctx.beginPath();ctx.arc(t.x,t.y,7,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#ffffff';ctx.beginPath();ctx.arc(t.x,t.y,5.5,0,Math.PI*2);ctx.fill();
      ctx.fillStyle=c;ctx.beginPath();ctx.arc(t.x,t.y,3.8,0,Math.PI*2);ctx.fill();
      if(showI)drawLabel('ID '+t.id,t.x+10,t.y-8,c);
    }});

    document.getElementById('sFr').textContent=fi+1;
    document.getElementById('sTr').textContent=tracks.length;
    const av=tracks.length?tracks.reduce((s,t)=>s+t.v,0)/tracks.length:0;
    document.getElementById('sVel').textContent=av.toFixed(1);
    const list=document.getElementById('tList');
    list.innerHTML=tracks.length
      ? tracks.slice().sort((a,b)=>a.id-b.id).map(t=>`<div class="track-item"><span><span class="dot" style="background:${{tcolor(t.id)}}"></span>Track ${{t.id}}</span><span>${{t.v.toFixed(1)}} mm/s</span></div>`).join('')
      : '<div class="empty">Sin tracks en este frame</div>';
  }};
  img.src=FD[fi].data;
}}

function upd(fi){{
  cur=fi;
  document.getElementById('fSlider').value=fi;
  document.getElementById('fLabel').textContent='Frame '+(fi+1)+' / '+FD.length;
  document.getElementById('fName').textContent=FD[fi].name;
  document.getElementById('tLabel').textContent=FD[fi].t===null?'—':'t = '+FD[fi].t.toFixed(3)+' s';
  draw(fi);
}}

function startTimer(){{
  clearInterval(iv);
  iv=setInterval(()=>{{upd(cur>=FD.length-1?0:cur+1);}},1000/targetFPS);
}}

function play(){{
  const b=document.getElementById('bPlay');
  if(playing){{
    playing=false;b.innerHTML='▶ Play<kbd>espacio</kbd>';b.classList.remove('active');clearInterval(iv);return;
  }}
  playing=true;b.innerHTML='❚❚ Pausa<kbd>espacio</kbd>';b.classList.add('active');startTimer();
}}

document.getElementById('fSlider').addEventListener('input',e=>{{if(playing)play();upd(parseInt(e.target.value));}});
document.getElementById('bPrev').addEventListener('click',()=>{{if(playing)play();upd(Math.max(0,cur-1));}});
document.getElementById('bNext').addEventListener('click',()=>{{if(playing)play();upd(Math.min(FD.length-1,cur+1));}});
document.getElementById('bPlay').addEventListener('click',play);

document.getElementById('fpsSlider').addEventListener('input',e=>{{
  targetFPS=parseFloat(e.target.value);
  document.getElementById('fpsVal').textContent=targetFPS;
  if(playing)startTimer();
}});

document.getElementById('lw').addEventListener('input',e=>{{document.getElementById('lwVal').textContent=e.target.value;draw(cur);}});
document.getElementById('tl').addEventListener('input',e=>{{const v=parseInt(e.target.value);document.getElementById('tlVal').textContent=v===0?'Completo':v+' frames';draw(cur);}});
['chkTraj','chkIDs','chkVec'].forEach(id=>document.getElementById(id).addEventListener('change',()=>draw(cur)));

document.addEventListener('keydown',e=>{{
  if(e.target.tagName==='INPUT'&&e.target.type==='range'&&(e.key==='ArrowLeft'||e.key==='ArrowRight'))return;
  if(e.key===' '){{e.preventDefault();play();}}
  else if(e.key==='ArrowLeft'){{if(playing)play();upd(Math.max(0,cur-1));}}
  else if(e.key==='ArrowRight'){{if(playing)play();upd(Math.min(FD.length-1,cur+1));}}
}});

upd(0);
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"[PTV] visualizer.html escrito ({out_path.stat().st_size / 1e6:.0f} MB)", flush=True)
