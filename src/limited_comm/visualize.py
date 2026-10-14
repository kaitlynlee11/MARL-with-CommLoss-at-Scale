"""Generate a standalone browser replay of paired random-navigation smoke runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from .env import CommunicationSpread


def collect_replay(n, loss, horizon, period, seed):
    runs = []
    for schedule in ("none", "full", "fixed"):
        env = CommunicationSpread(n=n, loss=loss, horizon=horizon)
        rng = np.random.default_rng(seed)
        frames = []
        attempts = delivered = payload_bytes = 0
        task_return = 0.0
        try:
            env.reset(seed=seed)

            def snapshot(links, sent):
                return {"agents": [a.state.p_pos.tolist() for a in env.world.agents],
                        "agent_radii": [float(a.size) for a in env.world.agents],
                        "landmarks": [l.state.p_pos.tolist() for l in env.world.landmarks],
                        "coverage_radius": env.coverage_radius,
                        "links": links, "sent": sent, "attempts": attempts,
                        "delivered": delivered, "payload_bytes": payload_bytes,
                        "task_return": task_return, **env.diagnostics()}

            frames.append(snapshot([], False))
            for t in range(horizon):
                gate = int(schedule == "full" or (schedule == "fixed" and t % period == 0))
                actions = {a: np.asarray([rng.integers(5), gate]) for a in env.agents}
                _, _, _, _, infos = env.step(actions)
                info = infos[env.possible_agents[0]]
                attempts += info["attempted_links"]
                delivered += info["delivered_links"]
                payload_bytes += info["payload_bytes"]
                task_return += info["task_reward"]
                # Cache age 1 identifies packets delivered on this step, not older cached packets.
                links = [[sender, receiver] for (receiver, sender), (_, age)
                         in env.channel.cache.items() if age == 1]
                frames.append(snapshot(links, bool(gate)))
            runs.append({"schedule": schedule, "frames": frames})
        finally:
            env.close()
    extent = max(abs(coordinate) for run in runs for frame in run["frames"]
                 for point in frame["agents"] + frame["landmarks"] for coordinate in point)
    return {"n": n, "loss": loss, "horizon": horizon, "period": period, "seed": seed,
            "bounds": max(1.2, extent + 0.25), "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--loss", type=float, default=0.3)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/demo.html"))
    args = parser.parse_args()
    if args.agents < 2 or not 0 <= args.loss <= 1 or min(args.horizon, args.period) < 1:
        parser.error("require agents >= 2, loss in [0, 1], horizon and period >= 1")
    if args.output.exists():
        parser.error("output already exists; choose a new --output filename")
    replay = collect_replay(args.agents, args.loss, args.horizon, args.period, args.seed)
    # Embedding is safe even if future metadata contains a closing script tag.
    serialized = json.dumps(replay, allow_nan=False).replace("<", "\\u003c")
    html = HTML.replace("__REPLAY_DATA__", serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        file.write(html)
    print(f"Replay saved to {args.output.resolve()}")
    print("Open this HTML file in a browser. Movement is random; no learning is shown.")


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Communication scheduling replay</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#17243a;font:16px system-ui,sans-serif}
main{max-width:1440px;margin:auto;padding:28px}h1{margin:0 0 8px;font-size:28px}
p{line-height:1.5;margin:8px 0}.muted{color:#56647a}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:14px;background:white;border:1px solid #dce2ed;border-radius:12px;padding:16px;margin:20px 0}
button,select{font:inherit;padding:8px 14px;border:1px solid #bac7da;border-radius:6px;background:white;cursor:pointer}
button#play{background:#215ac9;color:white;border-color:#215ac9;min-width:86px}input[type=range]{flex:1;min-width:130px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.panel{background:white;border:1px solid #dce2ed;border-radius:12px;overflow:hidden}
h2{margin:18px 18px 4px;font-size:20px}.description{margin:0 18px 12px;font-size:14px;color:#56647a;min-height:42px}
canvas{display:block;width:100%;aspect-ratio:1;background:#fafcff}.stats{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:18px;font-size:13px}
.stats strong{display:block;font-size:20px;font-variant-numeric:tabular-nums}.legend{display:flex;flex-wrap:wrap;gap:16px;margin:18px 0;font-size:14px}.notes{max-width:1050px;font-size:14px}
@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:18px}canvas{max-height:500px}}
</style>
</head>
<body><main>
<h1>Watch communication schedules</h1>
<p class="muted" id="config"></p>
<p>All three teams make the same random movements. Change the frame to see how communication traffic differs.</p>
<div class="controls">
<button id="play">Play</button><button id="step">Step</button><button id="reset">Reset</button>
<label for="timeline">Frame</label><input id="timeline" type="range" min="0" value="0"><output id="frame-label"></output>
<label>Speed <select id="speed"><option value="2">2 steps/s</option><option value="5" selected>5 steps/s</option><option value="10">10 steps/s</option><option value="20">20 steps/s</option></select></label>
<label><input type="checkbox" id="links" checked> Show messages</label>
<label><input type="checkbox" id="sensing"> Show sensing radius</label>
</div>
<div class="grid" id="panels"></div>
<div class="legend"><span>🔵 Numbered circles: agents</span><span>◇ Diamonds: landmarks</span><span style="color:#167948">Green landmark: covered</span><span style="color:#177eba">Solid arrow: delivered packet</span><span style="color:#b25462">Dashed arrow: dropped packet</span></div>
<div class="notes muted">
<p>Arrows show packets from the previous action arriving at the current frame. For readability, they connect the agents' current positions. Messages contain the sender's position and velocity before movement. Arrows disappear on silent steps; older messages can remain cached for five steps.</p>
<p>Coverage means an agent is within 0.1 units of a landmark. Collision pairs count agents whose circles overlap in this frame. Traffic and task return are cumulative; task return excludes communication penalties. Each directed packet contains 16 payload bytes, excluding headers.</p>
<p><strong>This is a smoke replay, not a learned policy.</strong> Navigation ignores messages, so coverage, collisions and task return match across schedules. The viewer shows the full world for explanation; individual agents have limited sensing.</p>
</div>
</main>
<script id="replay-data" type="application/json">__REPLAY_DATA__</script>
<script>
const data=JSON.parse(document.getElementById('replay-data').textContent);
const names={none:'No communication',full:'Full communication',fixed:'Fixed schedule'};
const descriptions={none:'Agents never send messages.',full:'Each agent sends to every other agent each step.',fixed:`Each agent sends every ${data.period} steps, starting with the first action.`};
const colors=['#2866cc','#8b51bf','#d47b17','#008778','#bd4d76','#576b91'];
let frame=0,playing=false,last=0;
const slider=document.getElementById('timeline'),play=document.getElementById('play');
slider.max=data.horizon;
document.getElementById('config').textContent=`${data.n} agents · ${Math.round(data.loss*100)}% packet loss per link · seed ${data.seed} · ${data.horizon} steps`;
const panels=data.runs.map(run=>{
 const panel=document.createElement('section');panel.className='panel';
 const title=document.createElement('h2');title.textContent=names[run.schedule];panel.append(title);
 const desc=document.createElement('p');desc.className='description';desc.textContent=descriptions[run.schedule];panel.append(desc);
 const canvas=document.createElement('canvas');canvas.width=600;canvas.height=600;canvas.setAttribute('aria-label',names[run.schedule]+' navigation replay');panel.append(canvas);
 const stats=document.createElement('div');stats.className='stats';panel.append(stats);
 const fields=[['coverage','Coverage'],['collision_pairs','Collision pairs now'],['attempts','Attempted packets'],['delivered','Delivered packets'],['payload_bytes','Payload bytes sent'],['task_return','Task return']];
 const values={};fields.forEach(([key,label])=>{const box=document.createElement('div');box.textContent=label;const val=document.createElement('strong');box.append(val);stats.append(box);values[key]=val;});
 document.getElementById('panels').append(panel);return {run,canvas,values};
});
function arrow(ctx,a,b,dropped){
 const dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy);if(length<25)return;
 const ux=dx/length,uy=dy/length;
 // Offset reciprocal arrows so the two directions can be distinguished.
 const start=[a[0]+ux*20-uy*5,a[1]+uy*20+ux*5],end=[b[0]-ux*20-uy*5,b[1]-uy*20+ux*5];
 ctx.strokeStyle=dropped?'#bd6070':'#1688bb';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2;ctx.setLineDash(dropped?[6,6]:[]);
 ctx.beginPath();ctx.moveTo(...start);ctx.lineTo(...end);ctx.stroke();ctx.setLineDash([]);
 ctx.beginPath();ctx.moveTo(...end);ctx.lineTo(end[0]-ux*9-uy*4,end[1]-uy*9+ux*4);ctx.lineTo(end[0]-ux*9+uy*4,end[1]-uy*9-ux*4);ctx.closePath();ctx.fill();
}
function draw({run,canvas,values}){
 const f=run.frames[frame],ctx=canvas.getContext('2d'),size=canvas.width,scale=(size-64)/(2*data.bounds);
 const xy=p=>[size/2+p[0]*scale,size/2-p[1]*scale];ctx.clearRect(0,0,size,size);
 ctx.strokeStyle='#e5ebf4';ctx.lineWidth=1;
 for(let v=-Math.floor(data.bounds);v<=data.bounds;v++){
  ctx.beginPath();ctx.moveTo(...xy([v,-data.bounds]));ctx.lineTo(...xy([v,data.bounds]));ctx.stroke();
  ctx.beginPath();ctx.moveTo(...xy([-data.bounds,v]));ctx.lineTo(...xy([data.bounds,v]));ctx.stroke();
 }
 if(document.getElementById('sensing').checked)f.agents.forEach(p=>{const [x,y]=xy(p);ctx.beginPath();ctx.arc(x,y,0.6*scale,0,2*Math.PI);ctx.fillStyle='#2866cc09';ctx.fill();ctx.strokeStyle='#2866cc33';ctx.stroke();});
 f.landmarks.forEach((p,i)=>{
  const covered=f.agents.some(a=>Math.hypot(a[0]-p[0],a[1]-p[1])<=f.coverage_radius),[x,y]=xy(p);
  ctx.beginPath();ctx.arc(x,y,f.coverage_radius*scale,0,2*Math.PI);ctx.strokeStyle=covered?'#229660':'#bdc8d9';ctx.stroke();
  ctx.beginPath();ctx.moveTo(x,y-7);ctx.lineTo(x+7,y);ctx.lineTo(x,y+7);ctx.lineTo(x-7,y);ctx.closePath();ctx.fillStyle=covered?'#229660':'#63738d';ctx.fill();
  ctx.font='14px system-ui';ctx.fillStyle='#43516a';ctx.fillText('L'+i,x+11,y+5);
 });
 if(document.getElementById('links').checked&&f.sent){
  const delivered=new Set(f.links.map(([s,r])=>s+','+r));
  f.agents.forEach((a,s)=>f.agents.forEach((b,r)=>{if(s!==r)arrow(ctx,xy(a),xy(b),!delivered.has(s+','+r));}));
 }
 f.agents.forEach((p,i)=>{
  const [x,y]=xy(p),radius=f.agent_radii[i]*scale;ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.fillStyle=colors[i%colors.length];ctx.fill();
  ctx.strokeStyle='#ffffff';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='white';ctx.font='bold 16px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(i,x,y);ctx.textAlign='start';ctx.textBaseline='alphabetic';
 });
 for(const key in values)values[key].textContent=key==='coverage'?`${Math.round(f[key]*100)}%`:key==='task_return'?f[key].toFixed(2):f[key].toLocaleString();
}
function render(){slider.value=frame;document.getElementById('frame-label').textContent=`${frame} / ${data.horizon}`;panels.forEach(draw);}
function pause(){playing=false;play.textContent='Play';}
play.onclick=()=>{if(playing)pause();else{if(frame===data.horizon)frame=0;playing=true;last=performance.now();play.textContent='Pause';render();}};
document.getElementById('step').onclick=()=>{pause();frame=Math.min(frame+1,data.horizon);render();};
document.getElementById('reset').onclick=()=>{pause();frame=0;render();};
slider.oninput=()=>{pause();frame=Number(slider.value);render();};
document.getElementById('links').onchange=render;document.getElementById('sensing').onchange=render;
function tick(now){if(playing&&now-last>=1000/Number(document.getElementById('speed').value)){last=now;frame++;if(frame>=data.horizon){frame=data.horizon;pause();}render();}requestAnimationFrame(tick);}
render();requestAnimationFrame(tick);
</script></body></html>
"""


if __name__ == "__main__":
    main()
