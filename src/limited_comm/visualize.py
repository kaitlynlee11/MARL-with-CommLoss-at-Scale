"""Generate a standalone browser replay of paired random-navigation smoke runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from .env import CommunicationSpread
from .rewards import TASK_REWARD_VERSION


def collect_replay(n, loss, horizon, period, seed, model=None, radius=0.6, min_landmark_distance=0.4,
                   models=None, schedules=None, terminate_on_collision=False, min_agent_distance=0.4, layout_sampler="whole",
                   communication_mode=None, communication_radius=1.):
    runs = []
    if schedules is None:
        schedules = ("adaptive",) if model is not None and model.spec.get("adaptive") else ("none", "full", "fixed")
    for schedule in schedules:
        if models is not None:
            model = models[schedule]
        mode = communication_mode or ("landmark" if getattr(model, "spec", {}).get("obs_dim") == 44 else "legacy")
        channel_options = {"communication_mode": mode, "communication_radius": communication_radius} if mode == "landmark" else {}
        env = CommunicationSpread(n=n, loss=loss, horizon=horizon, radius=radius,
                                  min_landmark_distance=min_landmark_distance,
                                  terminate_on_collision=terminate_on_collision, min_agent_distance=min_agent_distance,
                                  layout_sampler=layout_sampler, **channel_options)
        rng = np.random.default_rng(seed)
        frames = []
        attempts = delivered = payload_bytes = total_collisions = 0
        task_return = 0.0
        try:
            observations, _ = env.reset(seed=seed)
            if model is not None:
                model.reset_memory()

            def snapshot(links, sent, gates=None):
                return {"agents": [a.state.p_pos.tolist() for a in env.world.agents],
                        "agent_radii": [float(a.size) for a in env.world.agents],
                        "landmarks": [l.state.p_pos.tolist() for l in env.world.landmarks],
                        "coverage_radius": env.coverage_radius,
                        "links": links, "sent": sent, "sent_senders": np.flatnonzero(gates).tolist() if gates is not None else [], "attempts": attempts,
                        "delivered": delivered, "payload_bytes": payload_bytes,
                        "task_return": task_return, "total_collisions": total_collisions,
                        **env.diagnostics()}

            frames.append(snapshot([], False))
            failed = bool(terminate_on_collision and frames[0]["collision_pairs"] > 0)
            if failed:
                total_collisions = frames[0]["collision_pairs"]
                frames[0]["total_collisions"] = total_collisions
                frames[0]["success"] = False
            for t in range(0 if failed else horizon):
                gate = int(schedule == "full" or (schedule == "fixed" and t % period == 0))
                if schedule == "adaptive":
                    moves, gates = model.actions(np.stack([observations[a] for a in env.possible_agents]))
                else:
                    moves = (rng.integers(5, size=n) if model is None else
                             model.moves(np.stack([observations[a] for a in env.possible_agents])))
                    gates = np.full(n, gate)
                actions = {a: np.asarray([moves[i], gates[i]]) for i, a in enumerate(env.agents)}
                observations, _, _, _, infos = env.step(actions)
                info = infos[env.possible_agents[0]]
                attempts += info["attempted_links"]
                delivered += info["delivered_links"]
                payload_bytes += info["payload_bytes"]
                task_return += info["task_reward"]
                total_collisions += info["collision_pairs"]
                # Cache age 1 identifies packets delivered on this step, not older cached packets.
                links = [[sender, receiver] for (receiver, sender), (_, age)
                         in env.channel.cache.items() if age == 1]
                frames.append(snapshot(links, bool(np.any(gates)), gates))
                if terminate_on_collision and info["collision_pairs"] > 0:
                    failed = True
                    frames[-1]["success"] = False
                    break
            runs.append({"schedule": schedule, "frames": frames, "active_steps": len(frames)-1,
                         "failed_by_collision": failed})
        finally:
            env.close()
    extent = max(abs(coordinate) for run in runs for frame in run["frames"]
                 for point in frame["agents"] + frame["landmarks"] for coordinate in point)
    return {"n": n, "loss": loss, "horizon": horizon, "period": period, "seed": seed,
            "bounds": max(1.2, extent + 0.25), "runs": runs, "radius": radius,
            "policy": "random" if model is None else "trained_mappo",
            "bytes_per_packet": 16 if mode == "legacy" else None,
            "packet_description": ("Each directed packet contains 16 payload bytes, excluding headers." if mode == "legacy" else
                "Landmark packets contain a 28-byte header and 16 bytes per discovered landmark. Range is checked at send time before movement; cached coverage includes report age."),
            "communication_mode": mode, "communication_radius": communication_radius if mode == "landmark" else None,
            "min_landmark_distance": min_landmark_distance,
            "separately_trained": models is not None, "terminate_on_collision": terminate_on_collision,
            "min_agent_distance": min_agent_distance, "task_reward_version": TASK_REWARD_VERSION,
            "layout_sampler": layout_sampler}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--loss", type=float, default=0.3)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, help="Replay a trained MAPPO policy instead of random movement")
    parser.add_argument("--min-landmark-distance", type=float, default=0.4)
    parser.add_argument("--output", type=Path, default=Path("results/demo.html"))
    parser.add_argument("--min-agent-distance", type=float, default=0.4)
    parser.add_argument("--layout-sampler", choices=("whole", "sequential"), default="whole")
    parser.add_argument("--communication-radius", type=float, default=None)
    args = parser.parse_args()
    if args.agents < 2 or not 0 <= args.loss <= 1 or min(args.horizon, args.period) < 1:
        parser.error("require agents >= 2, loss in [0, 1], horizon and period >= 1")
    if args.output.exists():
        parser.error("output already exists; choose a new --output filename")
    model, radius, spacing = None, 0.6, args.min_landmark_distance
    if args.checkpoint:
        from .mappo import load_policy
        import torch
        torch.set_num_threads(1)
        model, checkpoint = load_policy(args.checkpoint)
        if model.spec["obs_dim"] not in (28, 44):
            parser.error("checkpoint must support a known local actor observation protocol")
        radius = checkpoint["config"]["radius"]
        spacing = checkpoint["config"].get("min_landmark_distance", 0.0)
    replay = collect_replay(args.agents, args.loss, args.horizon, args.period, args.seed, model, radius, spacing,
                            min_agent_distance=args.min_agent_distance, layout_sampler=args.layout_sampler,
                            communication_radius=(args.communication_radius if args.communication_radius is not None else
                                                  checkpoint["config"].get("communication_radius", 1.) if args.checkpoint else 1.))
    # Embedding is safe even if future metadata contains a closing script tag.
    serialized = json.dumps(replay, allow_nan=False).replace("<", "\\u003c")
    html = HTML.replace("__REPLAY_DATA__", serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        file.write(html)
    print(f"Replay saved to {args.output.resolve()}")
    print("Open this HTML file in a browser. Policy: " + replay["policy"])


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
<p id="policy-description">Change the frame to compare the schedules.</p>
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
<p id="message-note"></p>
<p>Coverage means an agent is within 0.1 units of a landmark. Collision pairs count agents whose circles overlap in this frame. Total collisions sums overlapping pairs after each action through the current step; continued overlap counts again each step, and initial-frame overlaps are excluded. Traffic and task return are cumulative; task return excludes communication penalties. <span id="packet-note"></span></p>
<p id="task-reward-note"></p>
<p id="policy-note"></p>
</div>
</main>
<script id="replay-data" type="application/json">__REPLAY_DATA__</script>
<script>
const data=JSON.parse(document.getElementById('replay-data').textContent);
document.getElementById('task-reward-note').textContent=data.task_reward_version==='coverage-gated-distance-v1'?'Task return stays constant while full coverage is maintained without collisions. Losing coverage resumes distance costs; collisions remain penalized. Reaching coverage does not end the episode.'+(data.return_recalculation?' Task returns were recalculated under this rule; the original episode outcomes and packet counts are unchanged.':''):'';
document.getElementById('panels').style.gridTemplateColumns=`repeat(${data.runs.length}, minmax(0,1fr))`;
const names={adaptive:'Learned adaptive scheduler',none:'No communication',full:'Full communication',fixed:'Fixed schedule'};
const descriptions={adaptive:'Each agent learns when to send from its local observation and recurrent memory.',none:'Agents never send messages.',full:'Each agent sends to every other agent each step.',fixed:`Each agent sends every ${data.period} steps, starting with the first action.`};
const colors=['#2866cc','#8b51bf','#d47b17','#008778','#bd4d76','#576b91'];
let frame=0,playing=false,last=0;
const slider=document.getElementById('timeline'),play=document.getElementById('play');
slider.max=data.horizon;
document.getElementById('config').textContent=`${data.n} agents · ${Math.round(data.loss*100)}% packet loss per link · seed ${data.seed} · ${data.horizon} steps · nearby landmarks only · landmark spacing ≥ ${data.min_landmark_distance??0}`;
document.getElementById('policy-description').textContent=data.terminate_on_collision?'Collision-terminated evaluation: success requires collision-free survival through step 100 and final full coverage. Each panel freezes at its actual termination.':data.runs[0].schedule==='adaptive'?'Jointly learned navigation and transmission under independent packet loss.':data.separately_trained?'Three separately trained MAPPO policies, each evaluated under its own communication schedule. All panels start from the same world.':data.policy==='trained_mappo'?'One full-communication-trained MAPPO policy, evaluated with three communication schedules. All panels start from the same world.':'All three teams make the same random movements. Change the frame to see how communication traffic differs.';
document.getElementById('policy-note').textContent=(data.policy==='trained_mappo'?'Actions come from the trained actor using local sensing and received messages. '+(data.separately_trained?'Each panel uses its own checkpoint.':'The same checkpoint is used in all panels.'):'This is a smoke replay, not a learned policy. Navigation ignores messages, so coverage, collisions and task return match across schedules.')+' The viewer shows the full world for explanation; individual agents have limited sensing.';
document.getElementById('message-note').textContent='Arrows show packets arriving from the previous action, connecting current agent positions for readability. Messages contain sender position and velocity. Arrows disappear on silent steps; cached messages expire after five steps. A recurrent actor can remember prior observations.';
document.getElementById('packet-note').textContent=data.packet_description??`Each directed packet contains ${data.bytes_per_packet} payload bytes, excluding headers.`;
const panels=data.runs.map(run=>{
 const panel=document.createElement('section');panel.className='panel';
 const title=document.createElement('h2');title.textContent=run.label??names[run.schedule];panel.append(title);
 const desc=document.createElement('p');desc.className='description';desc.textContent=(run.description??descriptions[run.schedule])+(data.terminate_on_collision?` Episode ${run.failed_by_collision?'failed by collision':'finished'} at step ${run.active_steps}; final coverage ${Math.round(run.frames.at(-1).coverage*100)}%.`:'');panel.append(desc);
 const canvas=document.createElement('canvas');canvas.width=600;canvas.height=600;canvas.setAttribute('aria-label',names[run.schedule]+' navigation replay');panel.append(canvas);
 const stats=document.createElement('div');stats.className='stats';panel.append(stats);
 const fields=[['coverage','Coverage'],['collision_pairs','Collision pairs now'],['total_collisions','Total collisions (so far)'],['attempts','Attempted packets'],['delivered','Delivered packets'],['payload_bytes','Payload bytes sent'],['task_return','Task return']];
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
 const f=run.frames[Math.min(frame,run.frames.length-1)],ctx=canvas.getContext('2d'),size=canvas.width,scale=(size-64)/(2*data.bounds);
 const xy=p=>[size/2+p[0]*scale,size/2-p[1]*scale];ctx.clearRect(0,0,size,size);
 ctx.strokeStyle='#e5ebf4';ctx.lineWidth=1;
 for(let v=-Math.floor(data.bounds);v<=data.bounds;v++){
  ctx.beginPath();ctx.moveTo(...xy([v,-data.bounds]));ctx.lineTo(...xy([v,data.bounds]));ctx.stroke();
  ctx.beginPath();ctx.moveTo(...xy([-data.bounds,v]));ctx.lineTo(...xy([data.bounds,v]));ctx.stroke();
 }
 if(document.getElementById('sensing').checked)f.agents.forEach(p=>{const [x,y]=xy(p);ctx.beginPath();ctx.arc(x,y,data.radius*scale,0,2*Math.PI);ctx.fillStyle='#2866cc09';ctx.fill();ctx.strokeStyle='#2866cc33';ctx.stroke();});
 f.landmarks.forEach((p,i)=>{
  const covered=f.agents.some(a=>Math.hypot(a[0]-p[0],a[1]-p[1])<=f.coverage_radius),[x,y]=xy(p);
  ctx.beginPath();ctx.arc(x,y,f.coverage_radius*scale,0,2*Math.PI);ctx.strokeStyle=covered?'#229660':'#bdc8d9';ctx.stroke();
  ctx.beginPath();ctx.moveTo(x,y-7);ctx.lineTo(x+7,y);ctx.lineTo(x,y+7);ctx.lineTo(x-7,y);ctx.closePath();ctx.fillStyle=covered?'#229660':'#63738d';ctx.fill();
  ctx.font='14px system-ui';ctx.fillStyle='#43516a';ctx.fillText('L'+i,x+11,y+5);
 });
 if(document.getElementById('links').checked&&f.sent){
  const delivered=new Set(f.links.map(([s,r])=>s+','+r));
  f.agents.forEach((a,s)=>f.agents.forEach((b,r)=>{if(s!==r&&(!f.sent_senders||f.sent_senders.includes(s)))arrow(ctx,xy(a),xy(b),!delivered.has(s+','+r));}));
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
