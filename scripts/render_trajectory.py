"""Render saved comparison runs as self-contained HTML animations (SVG + JS).

Reads <run_dir>/result.json written by src/compare/common_runner.run_method
(e.g. outputs/logs/compare_<batch>/runs/<run_id>/) and the scenario it names,
and never writes next to them. Everything drawn comes from those two inputs:
the grid (checked against the run's grid_sha256), agent ids, starts, goals,
the executed trajectory, the executed actions, action_sources (IL / CBS /
SAFETY), transitions (TRIGGER, IL_RESUME, SAFE_STOP), CBS call times, the
Monitor's own trigger time, Safety-only holds and the termination.

Before rendering, every run is checked: ids constant, trajectory[0] ==
starts, positions in bounds and off walls, every position change equals the
executed action's (row, col) delta, all_at_goal == (final state == goals) and
SUCCESS <-> all_at_goal. A failed check raises RenderError; nothing is drawn
from data that does not add up. Per step and agent the page shows MOVE, WAIT
(action 4) or BLOCKED (a move action that left the agent in place).

Output: a NEW directory (refused if it exists), default
outputs/videos/render_<timestamp>/, with one <run_id>.html per run and
render_manifest.json (inputs + sha256). No external resources are loaded.

Usage (from the repository root):
    python -m scripts.render_trajectory --run-dir outputs/logs/compare_<batch>/runs/<run_id> [...]
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_il import load_scenario
from src.common.spec import ACTION_DELTA
from src.compare.common_runner import _sha256_grid

DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "videos"
ACTION_LABELS = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "WAIT"}
WAIT = 4
REQUIRED = ("method", "run_id", "scenario", "grid_sha256", "starts", "goals", "trajectory", "actions",
            "action_sources", "termination", "stop_reason", "steps", "max_steps", "all_at_goal",
            "transitions", "cbs_calls", "monitor", "safety_holds", "dev_settings")


class RenderError(ValueError):
    """The saved run is inconsistent; it is not rendered."""


def _fail(run_id, message):
    raise RenderError(f"{run_id}: {message}")


def _state(mapping):
    return {int(a): (int(p[0]), int(p[1])) for a, p in mapping.items()}


def load_run_data(run_dir):
    """Validated, render-ready data for one saved run (see module docstring)."""
    run_dir = Path(run_dir)
    path = run_dir / "result.json"
    run_id = run_dir.name
    if not path.is_file():
        _fail(run_id, f"missing {path}")
    raw_bytes = path.read_bytes()
    try:
        result = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(run_id, f"result.json is not valid JSON ({exc})")
    missing = [k for k in REQUIRED if k not in result]
    if missing:
        _fail(run_id, f"result.json missing field(s): {', '.join(missing)}")
    run_id = result["run_id"]

    grid, scen_starts, scen_goals = load_scenario(result["scenario"])
    if _sha256_grid(grid) != result["grid_sha256"]:
        _fail(run_id, "scenario grid differs from the run's grid_sha256")
    starts, goals = _state(result["starts"]), _state(result["goals"])
    if starts != scen_starts or goals != scen_goals:
        _fail(run_id, "starts/goals differ from the scenario file")
    ids = sorted(starts)
    traj = [_state(s) for s in result["trajectory"]]
    actions = [{int(a): int(v) for a, v in step.items()} for step in result["actions"]]
    sources = list(result["action_sources"])
    steps = result["steps"]
    height, width = grid.shape

    if len(traj) != steps + 1 or len(actions) != steps or len(sources) != steps:
        _fail(run_id, "trajectory / actions / action_sources lengths disagree with steps")
    if set(sources) - {"IL", "CBS", "SAFETY"}:
        _fail(run_id, f"unknown action source(s) {sorted(set(sources) - {'IL', 'CBS', 'SAFETY'})}")
    if traj[0] != starts:
        _fail(run_id, "trajectory[0] is not the start state")
    for t, state in enumerate(traj):
        if sorted(state) != ids:
            _fail(run_id, f"agent ids change at t={t}")
        for aid, (r, c) in state.items():
            if not (0 <= r < height and 0 <= c < width) or grid[r, c] != 0:
                _fail(run_id, f"agent {aid} off the map or on a wall at t={t}: {(r, c)}")

    kinds = []
    for t in range(1, steps + 1):
        if sorted(actions[t - 1]) != ids:
            _fail(run_id, f"actions at step {t} do not cover every agent")
        row = []
        for aid in ids:
            a = actions[t - 1][aid]
            if a not in ACTION_DELTA:
                _fail(run_id, f"unknown action {a} at step {t}")
            (pr, pc), (cr, cc) = traj[t - 1][aid], traj[t][aid]
            if (cr, cc) == (pr, pc):
                row.append("WAIT" if a == WAIT else "BLOCKED")
            elif (cr - pr, cc - pc) == ACTION_DELTA[a] and a != WAIT:
                row.append("MOVE")
            else:
                _fail(run_id, f"agent {aid} step {t}: {(pr, pc)} -> {(cr, cc)} is not action {a}")
        kinds.append(row)

    at_goal = [[traj[t][a] == goals[a] for a in ids] for t in range(steps + 1)]
    if result["all_at_goal"] != all(at_goal[-1]):
        _fail(run_id, "all_at_goal disagrees with the final state")
    if (result["termination"] == "SUCCESS") != result["all_at_goal"]:
        _fail(run_id, "termination and all_at_goal disagree")

    events = []
    for c in result["cbs_calls"]:
        if "request_timestep" in c:
            events.append({"t": c["request_timestep"], "kind": "CBS",
                           "label": f"CBS request at S_{c['request_timestep']} (initial state): {c['status']}"})
        else:
            events.append({"t": c["trigger_timestep"], "kind": "CBS",
                           "label": f"CBS request after IL stall Trigger: {c['status']}"})
    for tr in result["transitions"]:
        if tr["to"] == "TRIGGER":
            events.append({"t": tr["t"], "kind": "TRIGGER", "label": f"TRIGGER ({tr['reason']})"})
        elif tr["to"] == "IL_RESUME":
            events.append({"t": tr["t"], "kind": "RESUME", "label": f"IL_RESUME ({tr['reason']})"})
        elif tr["to"] == "SAFE_STOP":
            events.append({"t": tr["t"], "kind": "STOP", "label": f"SAFE_STOP ({tr['reason']})"})
    mon_t = result["monitor"].get("trigger_timestep")
    if mon_t is not None and not any(e["kind"] == "TRIGGER" for e in events):
        events.append({"t": mon_t, "kind": "MONITOR",
                       "label": f"Monitor {result['monitor'].get('trigger_reason_code')} (observed)"})
    for t, aid in result["safety_holds"]:
        events.append({"t": t, "kind": "SAFETY", "label": f"Safety-only WAIT forced on agent {aid}"})
    events.append({"t": steps, "kind": "END", "label": f"{result['termination']}"
                   + (f" ({result['stop_reason']})" if result["stop_reason"] else "")})
    order = {"MONITOR": 0, "TRIGGER": 0, "CBS": 1, "SAFETY": 2, "RESUME": 3, "STOP": 4, "END": 5}
    events.sort(key=lambda e: (e["t"], order[e["kind"]]))   # same t: causal order

    return {
        "run_id": run_id,
        "method": result["method"],
        "is_dev_setting": bool(result["dev_settings"]),
        "dev_settings": result["dev_settings"],
        "scenario": Path(result["scenario"]).name,
        "termination": result["termination"],
        "stop_reason": result["stop_reason"],
        "steps": steps,
        "max_steps": result["max_steps"],
        "all_at_goal": result["all_at_goal"],
        "grid": grid.astype(int).tolist(),
        "agents": ids,
        "starts": [list(starts[a]) for a in ids],
        "goals": [list(goals[a]) for a in ids],
        "positions": [[list(traj[t][a]) for a in ids] for t in range(steps + 1)],
        "actions": [[ACTION_LABELS[actions[t][a]] for a in ids] for t in range(steps)],
        "kinds": kinds,
        "sources": sources,
        "at_goal": at_goal,
        "overlap": [len(set(state.values())) != len(state) for state in traj],
        "events": events,
        "source_result_json": str(path),
        "source_result_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f5;--panel:#fff;--fg:#1c2320;--muted:#5d6863;--line:#d6dcd8;--wall:#3a423e;--free:#eef1ee;
--il:#3d6fb3;--cbs:#c46a1c;--safety:#7b4fa8;--ok:#2e7a3c;--bad:#b23a2c;--warn:#9a6b00;
--font:system-ui,-apple-system,"Segoe UI",Roboto,"Malgun Gothic",sans-serif;--mono:ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#141917;--panel:#1c2220;--fg:#e2e7e4;--muted:#9aa6a0;--line:#313a36;
--wall:#c9d1cc;--free:#232a27;--il:#7aa6e6;--cbs:#eba25b;--safety:#b99ae0;--ok:#86cf8f;--bad:#ef8a7c;--warn:#e3b35a;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 var(--font);padding:24px max(16px,env(safe-area-inset-right)) 40px max(16px,env(safe-area-inset-left))}
main{max-width:1000px;margin:0 auto;display:grid;gap:16px}
h1{font-size:1.25rem;margin:0;text-wrap:balance}
.meta{display:flex;flex-wrap:wrap;gap:6px;color:var(--muted);font-size:.85rem}
.chip{padding:1px 8px;border:1px solid var(--line);border-radius:999px;background:var(--panel);white-space:nowrap}
.chip.dev{color:var(--warn);border-color:var(--warn)} .chip.ok{color:var(--ok);border-color:var(--ok)} .chip.bad{color:var(--bad);border-color:var(--bad)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.stage{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px}
@media (max-width:760px){.stage{grid-template-columns:1fr}}
svg{width:100%;height:auto;display:block}
.cell-free{fill:var(--free)} .cell-wall{fill:var(--wall)}
.agent{transition:transform .22s ease}
@media (prefers-reduced-motion:reduce){.agent{transition:none}}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button,select{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--il);outline-offset:2px}
input[type=range]{flex:1;min-width:120px}
.tnum{font-variant-numeric:tabular-nums;font-family:var(--mono)}
.timeline{display:flex;gap:1px;height:18px;margin-top:6px}
.timeline div{flex:1;min-width:2px;cursor:pointer;opacity:.85}
.timeline div.cur{outline:2px solid var(--fg);opacity:1}
.src-IL{background:var(--il)} .src-CBS{background:var(--cbs)} .src-SAFETY{background:var(--safety)}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:.82rem;color:var(--muted)}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line)} th{color:var(--muted);font-weight:500}
.events{margin:0;padding-left:1.1em;font-size:.85rem} .events li.now{font-weight:600}
.status{font-size:.95rem} .foot{color:var(--muted);font-size:.78rem;word-break:break-all}
</style></head><body><main>
<header>
  <h1 id="title"></h1>
  <div class="meta" id="meta"></div>
</header>
<section class="stage">
  <div class="panel">
    <svg id="board" role="img" aria-label="grid animation"></svg>
    <div class="controls" style="margin-top:10px">
      <button id="first" aria-label="first step">|&lt;</button>
      <button id="prev" aria-label="previous step">&lt;</button>
      <button id="play">Play</button>
      <button id="next" aria-label="next step">&gt;</button>
      <button id="last" aria-label="last step">&gt;|</button>
      <label for="speed">speed</label>
      <select id="speed"><option value="900">0.5x</option><option value="450" selected>1x</option><option value="200">2x</option><option value="80">5x</option></select>
    </div>
    <div class="controls" style="margin-top:8px">
      <span class="tnum" id="tlabel"></span>
      <input id="slider" type="range" min="0" step="1" aria-label="timestep">
    </div>
    <div class="timeline" id="timeline" aria-label="action source per step"></div>
    <div class="legend" style="margin-top:8px">
      <span><span class="sw src-IL"></span>IL action</span>
      <span><span class="sw src-CBS"></span>CBS action</span>
      <span><span class="sw src-SAFETY"></span>Safety-only forced WAIT (+ IL)</span>
      <span>circle = agent at t, dashed square = its goal</span>
    </div>
  </div>
  <div class="panel" style="display:grid;gap:10px;align-content:start">
    <div class="status" id="status"></div>
    <table><thead><tr><th>agent</th><th>(row,col)</th><th>action</th><th>result</th><th>goal</th></tr></thead><tbody id="agents"></tbody></table>
    <div><strong style="font-size:.85rem">Events</strong><ol class="events" id="events"></ol></div>
  </div>
</section>
<div class="foot" id="foot"></div>
</main>
<script type="application/json" id="run-data">__RUN_DATA__</script>
<script>
(function(){
const D=JSON.parse(document.getElementById('run-data').textContent);
const palette=['#d1495b','#2a9d8f','#e9a03b','#5465ff','#8f5db7','#3c9d4a','#c2569b','#6b7f2a'];
const color=i=>palette[i%palette.length];
const H=D.grid.length,W=D.grid[0].length,C=Math.max(20,Math.min(56,Math.floor(640/W)));
const svgNS='http://www.w3.org/2000/svg',board=document.getElementById('board');
board.setAttribute('viewBox',`0 0 ${W*C} ${H*C}`);
const el=(n,a)=>{const e=document.createElementNS(svgNS,n);for(const k in a)e.setAttribute(k,a[k]);return e;};
for(let r=0;r<H;r++)for(let c=0;c<W;c++)board.appendChild(el('rect',{x:c*C,y:r*C,width:C-1,height:C-1,class:D.grid[r][c]?'cell-wall':'cell-free'}));
D.goals.forEach(([r,c],i)=>{board.appendChild(el('rect',{x:c*C+3,y:r*C+3,width:C-7,height:C-7,fill:'none',stroke:color(i),'stroke-width':2,'stroke-dasharray':'4 3'}));
  const t=el('text',{x:c*C+6,y:r*C+C-6,'font-size':C*0.26,fill:color(i)});t.textContent='G'+D.agents[i];board.appendChild(t);});
const glyphs=D.agents.map((id,i)=>{const g=el('g',{class:'agent'});
  g.appendChild(el('circle',{cx:C/2,cy:C/2,r:C*0.34,fill:color(i),stroke:'#000','stroke-opacity':0.25}));
  const t=el('text',{x:C/2,y:C/2+C*0.12,'text-anchor':'middle','font-size':C*0.34,fill:'#fff','font-weight':'700'});t.textContent=id;g.appendChild(t);
  board.appendChild(g);return g;});
const $=id=>document.getElementById(id);
$('title').textContent=`${D.method} · ${D.scenario}`;
const meta=[[D.termination+(D.stop_reason?` (${D.stop_reason})`:''),D.termination==='SUCCESS'?'ok':'bad'],
  [`steps ${D.steps} / max_steps ${D.max_steps}`,''],[`all_at_goal=${D.all_at_goal}`,D.all_at_goal?'ok':'bad'],
  [D.is_dev_setting?'DEV setting, not an agreed policy':'baseline (no DEV setting)',D.is_dev_setting?'dev':''],[D.run_id,'']];
meta.forEach(([txt,cls])=>{const s=document.createElement('span');s.className='chip '+cls;s.textContent=txt;$('meta').appendChild(s);});
$('foot').textContent=`source: ${D.source_result_json} · sha256 ${D.source_result_sha256}`;
const tl=$('timeline');for(let t=1;t<=D.steps;t++){const d=document.createElement('div');d.className='src-'+D.sources[t-1];d.title=`step ${t}: ${D.sources[t-1]}`;d.onclick=()=>show(t);tl.appendChild(d);}
D.events.forEach(e=>{const li=document.createElement('li');li.dataset.t=e.t;li.textContent=`t=${e.t}  ${e.label}`;li.style.cursor='pointer';li.onclick=()=>show(e.t);$('events').appendChild(li);});
const slider=$('slider');slider.max=D.steps;let cur=0,timer=null;
function show(t){cur=Math.max(0,Math.min(D.steps,t));slider.value=cur;
  D.positions[cur].forEach(([r,c],i)=>glyphs[i].setAttribute('transform',`translate(${c*C},${r*C})`));
  const src=cur===0?'initial state S_0':`step ${cur} by ${D.sources[cur-1]}`;
  $('tlabel').textContent=`t=${String(cur).padStart(3,' ')} / ${D.steps}`;
  const atg=D.at_goal[cur].filter(Boolean).length;
  $('status').innerHTML='';const s=document.createElement('div');
  s.textContent=`${src} · at goal ${atg}/${D.agents.length}`+(D.overlap[cur]?' · OVERLAP':'')+(cur===D.steps?` · end: ${D.termination}`:'');
  if(D.overlap[cur])s.style.color='var(--bad)';$('status').appendChild(s);
  const tb=$('agents');tb.innerHTML='';
  D.agents.forEach((id,i)=>{const tr=document.createElement('tr');const p=D.positions[cur][i];
    const cells=[id,`(${p[0]},${p[1]})`,cur?D.actions[cur-1][i]:'-',cur?D.kinds[cur-1][i]:'-',D.at_goal[cur][i]?'yes':'no'];
    cells.forEach((v,k)=>{const td=document.createElement('td');td.textContent=v;if(k===0)td.style.color=color(i);if(k===1)td.className='tnum';tr.appendChild(td);});tb.appendChild(tr);});
  [...tl.children].forEach((d,k)=>d.classList.toggle('cur',k===cur-1));
  [...$('events').children].forEach(li=>li.classList.toggle('now',Number(li.dataset.t)===cur));}
function stop(){if(timer){clearInterval(timer);timer=null;}$('play').textContent='Play';}
function play(){if(cur>=D.steps)show(0);stop();$('play').textContent='Pause';
  timer=setInterval(()=>{if(cur>=D.steps){stop();return;}show(cur+1);},Number($('speed').value));}
$('play').onclick=()=>timer?stop():play();
$('prev').onclick=()=>{stop();show(cur-1);};$('next').onclick=()=>{stop();show(cur+1);};
$('first').onclick=()=>{stop();show(0);};$('last').onclick=()=>{stop();show(D.steps);};
$('speed').onchange=()=>{if(timer)play();};slider.oninput=()=>{stop();show(Number(slider.value));};
show(0);
})();
</script></body></html>
"""


def render_html(data):
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(f"{data['method']} {data['scenario']}")
    return TEMPLATE.replace("__TITLE__", title).replace("__RUN_DATA__", payload)


def render_runs(run_dirs, out_dir, *, command=None):
    """Validate every run first, then write <run_id>.html files into a NEW out_dir."""
    datas = [load_run_data(d) for d in run_dirs]
    if not datas:
        raise RenderError("no run directories given")
    ids = [d["run_id"] for d in datas]
    if len(set(ids)) != len(ids):
        raise RenderError("duplicate run_id")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)  # never reuse an output directory
    written = []
    for d in datas:
        path = out_dir / f"{d['run_id']}.html"
        with path.open("x", encoding="utf-8") as fh:
            fh.write(render_html(d))
        written.append(path)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "note": "rendered only from each run's result.json and scenario; DEV settings are not agreed policies",
        "runs": [{"run_id": d["run_id"], "method": d["method"], "scenario": d["scenario"],
                  "termination": d["termination"], "steps": d["steps"], "html": p.name,
                  "source_result_json": d["source_result_json"],
                  "source_result_sha256": d["source_result_sha256"]} for d, p in zip(datas, written)],
    }
    with (out_dir / "render_manifest.json").open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return written


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, action="append", required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="new directory (default: outputs/videos/render_<timestamp>)")
    args = ap.parse_args(argv)
    out_dir = args.out_dir or DEFAULT_OUT_ROOT / f"render_{datetime.now():%Y%m%d_%H%M%S}"
    if out_dir.exists():
        print(f"ERROR: output directory already exists, refusing to overwrite: {out_dir}")
        return 2
    command = " ".join([sys.executable, "-m", "scripts.render_trajectory", *argv])
    try:
        written = render_runs(args.run_dir, out_dir, command=command)
    except RenderError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"out_dir: {out_dir}")
    for p in written:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
