# Serves a browser interface for annotating rollouts.
# Created: 2026-10-22

import argparse
import json
import mimetypes
import re
import shutil
import subprocess
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


# Find the recording directory requested by the browser.
def choose_record_dir():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        tk = None
    if tk is not None:
        try:
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askdirectory(title="Select a rollout recording folder")
            root.destroy()
            return Path(path).expanduser() if path else None
        except tk.TclError:
            pass
    dialogs = (
        ("zenity", ["--file-selection", "--directory", "--title=Select a rollout recording folder"]),
        ("kdialog", ["--getexistingdirectory", ".", "--title", "Select a rollout recording folder"]),
        ("yad", ["--file", "--directory", "--title=Select a rollout recording folder"]),
    )
    for program, options in dialogs:
        executable = shutil.which(program)
        if executable:
            result = subprocess.run([executable, *options], capture_output=True, text=True)
            if result.returncode:
                return None
            return Path(result.stdout.strip()).expanduser()
    raise RuntimeError("No graphical folder picker is available; install zenity or run with --record-dir.")


# Build the rollout review page from episode metadata.
def build_page(record_dir):
    manifest_path = record_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest.json in {record_dir}")
    manifest = json.loads(manifest_path.read_text())
    hz = float(manifest.get("control_hz") or manifest.get("policy_hz") or manifest.get("video_fps") or 20)
    task = manifest.get("metadata", {}).get("task", record_dir.name)
    episodes = []
    for index, item in enumerate(manifest.get("episodes", [])):
        episode_id = str(item.get("episode_id", f"traj_{index}"))
        sidecar_path = record_dir / "data" / f"ep_{index}.json"
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
        conditions = sidecar.get("failure_subconditions", item.get("failure_subconditions", {})) or {}
        first = sidecar.get("first_failure_timesteps", item.get("first_failure_timesteps", {})) or {}
        events = [{"name": name, "step": int(step)} for name, step in first.items()
                  if conditions.get(name, False) and step is not None]
        video = item.get("video")
        if video:
            video_path = (record_dir / video).resolve()
            if not video_path.is_relative_to(record_dir.resolve()) or not video_path.is_file():
                video = None
        steps = int(item.get("steps") or sidecar.get("steps") or 0)
        if steps < 1:
            raise ValueError(f"Episode {episode_id} has no positive step count")
        episodes.append({
            "id": episode_id,
            "seed": item.get("seed", sidecar.get("seed")),
            "success": bool(item.get("success", sidecar.get("success", False))),
            "outcome": str(item.get("outcome") or sidecar.get("outcome") or
                           ("success" if item.get("success", sidecar.get("success", False)) else "failure")).upper(),
            "reasons": item.get("failure_reasons") or [],
            "steps": steps,
            "video": video,
            "video_fps": float(manifest.get("video_fps", hz)),
            "video_frame_count": int(item.get("video_frame_count") or steps),
            "variance": sidecar.get("executed_variance_by_env_timestep"),
            "events": events,
        })
    if not episodes:
        raise ValueError(f"No episodes listed in {manifest_path}")
    payload = json.dumps(episodes, separators=(",", ":")).replace("</", "<\\/")
    html = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rollout annotator</title><style>
:root{--ink:#172331;--muted:#697887;--line:#dce3e9;--paper:#fff;--bg:#eef2f5;--blue:#2866d5;--red:#b63c38;--green:#23764f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}button,select,input{font:inherit}button{border:1px solid var(--line);border-radius:8px;background:white;padding:9px 13px;font-weight:650;cursor:pointer}button:hover{background:#f4f7f9}header{position:sticky;top:0;z-index:2;background:#172331;color:white;padding:12px max(18px,calc((100vw - 1540px)/2));display:flex;align-items:center;gap:12px;box-shadow:0 2px 9px #10182025}h1{font-size:17px;margin:0 auto 0 0}header label{font-size:13px;color:#d4dce3}.top-select{margin-left:7px;padding:8px;border:1px solid #40515f;border-radius:7px;background:#243440;color:white;max-width:330px}.save{background:#e3edff;color:#1749a1;border-color:#e3edff}main{max-width:1540px;margin:20px auto;padding:0 18px}.summary,.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.summary{margin:4px 0 12px}.episode-id{font-size:18px;font-weight:750}.pill{padding:4px 10px;border-radius:99px;background:#e8edf1;font-size:12px;font-weight:700}.success{background:#e0f2e8;color:#17623f}.failure{background:#fde8e5;color:#a2332f}.muted{font-size:13px;color:var(--muted)}.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:15px 17px;margin:12px 0;box-shadow:0 3px 12px #25384a08}.title{font-weight:750;margin-bottom:10px}.video-wrap{min-height:150px;display:grid;place-items:center;background:#101418;border-radius:8px;overflow:hidden}.video{display:block;width:100%;max-height:60vh;background:#101418}.empty{padding:40px 15px;text-align:center;color:#d4dce3}.chart{display:block;width:100%;height:245px;cursor:crosshair}.controls label{display:flex;align-items:center;gap:7px}.controls input{width:78px;padding:8px;border:1px solid var(--line);border-radius:7px}.danger{background:#fff0ef;color:var(--red);border-color:#f1d1cf}.positive{background:#eaf5ee;color:var(--green);border-color:#d7eadd}.status{margin-left:auto}.legend{display:flex;gap:15px;flex-wrap:wrap;margin-top:7px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.callout{margin-top:12px;padding:10px 12px;background:#f4f7f9;border-left:3px solid var(--blue);border-radius:4px;font-size:13px;color:#536272}@media(max-width:700px){header{position:static;flex-wrap:wrap;padding:12px 14px}header h1{width:100%}main{padding:0 10px}.card{padding:12px}.status{margin-left:0;width:100%}}
</style></head><body>
<style>
main.layout{display:grid;grid-template-columns:235px minmax(0,1fr);gap:14px;align-items:start}.episode-rail{position:sticky;top:82px;height:max(600px,calc(100vh - 150px));max-height:calc(100vh - 100px);overflow:auto;background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px}.rail-title{font-size:13px;font-weight:750;color:#566574;padding:4px 7px 10px}.episode-filter{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin:0 0 12px}.episode-filter button{padding:6px 3px;font-size:11px}.episode-filter button.active{background:#e8edf1;border-color:#cbd1d6}.episode-groups{display:grid;gap:15px}.episode-group-title{display:flex;justify-content:space-between;align-items:center;padding:5px 7px 6px;color:#566574;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase}.episode-group-title .count{color:#89939b;font-weight:600;letter-spacing:0}.episode-list{display:grid;gap:4px}.episode-item{display:flex;width:100%;align-items:center;text-align:left;gap:8px;padding:8px;border:1px solid transparent;border-radius:8px;background:transparent;font-size:12px}.episode-item:hover{background:#f4f7f9}.episode-item.active{background:#edf3ff;border-color:#d5e2ff;color:#1749a1}.episode-state{width:8px;height:8px;flex:none;border-radius:50%;background:#82909b}.episode-state.success{background:#25835a}.episode-state.failure{background:#c84442}.episode-mark{margin-left:auto;padding:2px 5px;border-radius:5px;background:#e1f3e8;color:#17623f;font-size:9px;font-weight:800;letter-spacing:.04em}.workspace{min-width:0}.inspection-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(240px,.32fr);gap:14px;align-items:stretch}.media-column{display:grid;grid-template-rows:minmax(0,3fr) minmax(220px,1fr);gap:14px;min-width:0;height:calc(100vh - 190px);min-height:600px}.media-column>.card{height:auto;min-height:0;margin:0;overflow:auto}.media-column>.card:first-child{display:flex;flex-direction:column}.media-column .video-wrap{flex:1;min-height:0}.media-column .video{max-height:100%;object-fit:contain}.media-column .chart{height:130px}.annotation-panel{height:calc(100vh - 190px);min-height:600px;margin:0;overflow:auto}.annotation-panel .controls{align-items:stretch;flex-direction:column;gap:8px}.annotation-panel .controls button{text-align:left}.annotation-panel .controls label{justify-content:space-between}.annotation-panel .status{margin-left:0}.annotation-panel .description{min-height:120px}.top-select{display:none!important}@media(max-width:1000px){.inspection-grid{grid-template-columns:minmax(0,1fr) minmax(240px,.42fr)}.media-column,.annotation-panel{height:calc(100vh - 190px);min-height:560px}}@media(max-width:760px){main.layout{grid-template-columns:1fr}.episode-rail{position:static;height:auto;max-height:300px}.episode-groups{grid-template-columns:1fr 1fr;align-items:start}.episode-list{grid-template-columns:1fr}.inspection-grid{grid-template-columns:1fr}.media-column{height:75vh;min-height:560px}.annotation-panel{height:auto;min-height:unset}}
.episode-item.active{background:#f1f3f4;border-color:#cbd1d6;color:var(--ink)}.episode-item.excluded{color:#89939b;text-decoration:line-through}.episode-state.success{background:#25835a}.episode-state.failure{background:#c84442}.episode-mark{background:#edf0f2;color:#46515b}.episode-mark.removed{background:#f0e9e9;color:#704d4d}.pill.success{background:#e0f2e8;color:#17623f}.pill.failure{background:#fde8e5;color:#a2332f}.danger,.positive,.save{background:#fff;color:var(--ink);border-color:var(--line)}.save{font-weight:700}.notes-label{display:block;font-size:13px;font-weight:650;margin:12px 0 6px}.description{width:100%;min-height:76px;resize:vertical;padding:10px 11px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink);font:14px/1.4 system-ui,sans-serif}.description:focus{outline:2px solid #aeb8c1;outline-offset:1px}
</style></head><body>
<header><h1 id="title">Rollout annotator</h1><span id="episodeCount" class="muted" style="color:#d4dce3"></span><button class="save" id="save">Save annotations</button><span id="saved" class="muted" style="color:#e0e7ed"></span></header>
<main class="layout"><aside class="episode-rail"><div class="rail-title">EPISODES</div><div class="episode-filter" role="group" aria-label="Filter episodes"><button class="active" data-filter="all">All</button><button data-filter="success">Success</button><button data-filter="failure">Failure</button></div><div class="episode-groups"><section id="successSection"><div class="episode-group-title">Success <span id="successCount" class="count"></span></div><div id="successList" class="episode-list"></div></section><section id="failureSection"><div class="episode-group-title">Failure <span id="failureCount" class="count"></span></div><div id="failureList" class="episode-list"></div></section></div></aside><div class="workspace"><div class="summary" id="info"></div><div class="inspection-grid">
<div class="media-column"><section class="card"><div class="title">Episode video</div><div class="video-wrap"><div class="empty" id="emptyVideo">No video is available for this episode.</div><video id="video" class="video" controls preload="auto" hidden></video></div><div id="videoStatus" class="muted" style="margin-top:7px"></div></section>
<section class="card"><div class="title">Episode timeline <span id="cursor" class="muted"></span></div><canvas id="chart" class="chart"></canvas><div class="legend muted"><span><i class="dot" style="background:#2866d5"></i>variance</span><span><i class="dot" style="background:#963ca8"></i>recorded conditions</span><span>Outcome mark</span><span>Episode end</span><span>Shaded tail is excluded</span></div></section></div>
<section class="card annotation-panel"><div class="title">Annotation</div><div class="controls"><button id="success">Set success</button><button id="setFailure">Set failure</button><button id="markSuccess">Mark success here &amp; cut</button><button id="markFailure">Mark failure here &amp; cut</button><label>Keep after mark <input id="buffer" type="number" min="0" max="60" step="0.5" value="2"> s</label><button id="review">Review full episode</button><button id="clear">Reset label and cut</button><button id="removeEpisode">Remove episode</button><span id="status" class="muted status"></span></div><label class="notes-label" for="description">Description (optional)</label><textarea id="description" class="description" placeholder="Add a note about what you observed in this episode."></textarea></section></div></div></main>
<script>
const episodes=__DATA__,hz=__HZ__,task=__TASK__;let current=0,step=0,annotations={},reviewFull=false,pendingSeek=null,episodeFilter='all';
const video=document.getElementById('video'),canvas=document.getElementById('chart'),buffer=document.getElementById('buffer'),successList=document.getElementById('successList'),failureList=document.getElementById('failureList'),description=document.getElementById('description'),episodeButtons=[];
episodes.forEach((e,i)=>{const b=document.createElement('button');b.className='episode-item';b.onclick=()=>choose(i);episodeButtons[i]=b});document.getElementById('episodeCount').textContent=`${episodes.length} episodes`;document.getElementById('title').textContent=`${task} · rollout annotator`;
document.querySelectorAll('.episode-filter button').forEach(button=>button.onclick=()=>{episodeFilter=button.dataset.filter;document.querySelectorAll('.episode-filter button').forEach(item=>item.classList.toggle('active',item===button));updateEpisodeList();const ep=episodes[current],label=annotations[ep.id]?.annotated_outcome||(ep.success?'success':'failure');if(episodeFilter!=='all'&&label!==episodeFilter){const next=episodes.findIndex(item=>(annotations[item.id]?.annotated_outcome||(item.success?'success':'failure'))===episodeFilter);if(next>=0)choose(next)}});
fetch('/api/annotations').then(r=>r.json()).then(x=>{annotations=x.annotations||{};render()}).catch(()=>render());
function fit(){const d=devicePixelRatio||1,r=canvas.getBoundingClientRect();canvas.width=Math.max(1,r.width*d);canvas.height=Math.max(1,r.height*d);const c=canvas.getContext('2d');c.setTransform(d,0,0,d,0,0);return{c,w:r.width,h:r.height}}
function endStep(e,a){return a&&a.episode_end_step!=null?Math.min(e.steps-1,a.episode_end_step):e.steps-1}
function draw(){const e=episodes[current],a=annotations[e.id],{c,w,h}=fit(),left=55,right=18,top=12,bottom=28,pw=w-left-right,ph=h-top-bottom,n=Math.max(1,e.steps),cutEnd=endStep(e,a),end=reviewFull?e.steps-1:cutEnd,vals=e.variance,xp=t=>left+t/Math.max(1,n-1)*pw;c.clearRect(0,0,w,h);if(!reviewFull&&cutEnd<n-1){c.fillStyle='#edf0f2';c.fillRect(xp(cutEnd),top,left+pw-xp(cutEnd),ph)}const vmax=Math.max(...(vals||[]).filter(Number.isFinite),1e-12);c.strokeStyle='#e0e5e9';c.fillStyle='#697887';c.font='11px system-ui';for(let j=0;j<=4;j++){const y=top+ph*j/4;c.beginPath();c.moveTo(left,y);c.lineTo(left+pw,y);c.stroke();if(vals)c.fillText((vmax*(1-j/4)).toPrecision(3),3,y+4)}c.strokeStyle='#8996a0';c.beginPath();c.moveTo(left,top);c.lineTo(left,top+ph);c.lineTo(left+pw,top+ph);c.stroke();if(vals){c.strokeStyle='#2866d5';c.lineWidth=1.7;c.beginPath();let started=false;vals.forEach((v,i)=>{if(i>end||v==null||!Number.isFinite(v)){started=false;return}const x=xp(i),y=top+ph-v/vmax*ph;if(started)c.lineTo(x,y);else c.moveTo(x,y);started=true});c.stroke()}e.events.forEach((ev,j)=>{if(ev.step>end)return;const x=xp(ev.step);c.strokeStyle='#963ca8';c.setLineDash([3,4]);c.beginPath();c.moveTo(x,top);c.lineTo(x,top+ph);c.stroke();c.setLineDash([]);c.save();c.translate(x+2,top+10+(j%3)*22);c.rotate(Math.PI/2);c.fillStyle='#783685';c.fillText(ev.name,0,0);c.restore()});const mark=a?.failure_onset_step??a?.success_mark_step;if(mark!=null){c.strokeStyle=a.annotated_outcome==='failure'?'#d77b19':'#59636c';c.lineWidth=2;c.beginPath();c.moveTo(xp(mark),top);c.lineTo(xp(mark),top+ph);c.stroke()}if(cutEnd<n-1){c.strokeStyle='#59636c';c.lineWidth=2;c.beginPath();c.moveTo(xp(cutEnd),top);c.lineTo(xp(cutEnd),top+ph);c.stroke()}c.strokeStyle='#202a33';c.lineWidth=1;c.beginPath();c.moveTo(xp(step),top);c.lineTo(xp(step),top+ph);c.stroke();document.getElementById('cursor').textContent=`· step ${step}/${end} · ${(step/hz).toFixed(2)} s`}
function updateEpisodeList(){const successIndices=[],failureIndices=[];episodes.forEach((ep,i)=>{const ann=annotations[ep.id],outcome=ann?.annotated_outcome||(ep.success?'success':'failure'),excluded=ann?.excluded===true,b=episodeButtons[i];b.classList.toggle('active',i===current);b.classList.toggle('excluded',excluded);b.innerHTML=`<i class="episode-state ${outcome}"></i><span>${ep.id}</span>${excluded?'<span class="episode-mark removed">REMOVED</span>':ann?'<span class="episode-mark">DONE</span>':''}`;(outcome==='success'?successIndices:failureIndices).push(i)});successIndices.forEach(i=>successList.appendChild(episodeButtons[i]));failureIndices.forEach(i=>failureList.appendChild(episodeButtons[i]));document.getElementById('successCount').textContent=successIndices.length;document.getElementById('failureCount').textContent=failureIndices.length;document.getElementById('successSection').hidden=episodeFilter==='failure';document.getElementById('failureSection').hidden=episodeFilter==='success'}
function render(){const e=episodes[current],a=annotations[e.id],label=a?.annotated_outcome||(e.success?'success':'failure'),end=endStep(e,a),markStep=a?.failure_onset_step??a?.success_mark_step,hasCutMark=a?.failure_onset_step!=null||a?.success_mark_step!=null;document.getElementById('info').innerHTML=`<span class="episode-id">${e.id}</span><span class="pill ${label}">${label.toUpperCase()}</span>${a?.excluded?'<span class="pill">REMOVED</span>':''}<span class="pill">seed ${e.seed??'unknown'}</span><span class="pill">${end+1} effective / ${e.steps} recorded steps</span>${e.reasons.length?`<span class="muted">Recorded: ${e.reasons.join(', ')}</span>`:''}`;updateEpisodeList();description.value=a?.description||'';document.getElementById('status').textContent=a?.excluded?'Removed from review set · recording unchanged':hasCutMark?`${Math.max(0,e.steps-end-1)} tail steps excluded · ${label} mark ${markStep} · end ${end}`:a?.annotated_outcome?`${label==='success'?'Success':'Failure'} label · full episode retained`:a?.description?'Description added':'No edits';document.getElementById('review').textContent=reviewFull?'Use cut boundary':'Review full episode';document.getElementById('removeEpisode').textContent=a?.excluded?'Restore episode':'Remove episode';draw()}
function videoTimeForStep(e,t){return t/Math.max(1,e.steps-1)*Math.max(0,e.video_frame_count-1)/Math.max(1,e.video_fps)}function stepForVideoTime(e,t){return Math.round(t*Math.max(1,e.video_fps)/Math.max(1,e.video_frame_count-1)*Math.max(1,e.steps-1))}function seekVideo(e,t,play=false){const seek=()=>{if(episodes[current].id!==e.id)return;video.currentTime=videoTimeForStep(e,t);if(play)video.play().catch(()=>{})};if(video.readyState<1)pendingSeek={episodeId:e.id,step:t,play};else seek()}
function choose(i){current=(i+episodes.length)%episodes.length;step=0;reviewFull=false;pendingSeek=null;const e=episodes[current];video.pause();if(e.video){video.hidden=false;document.getElementById('emptyVideo').hidden=true;video.src='/'+e.video.split('/').map(encodeURIComponent).join('/');video.load();document.getElementById('videoStatus').textContent=e.video}else{video.hidden=true;video.removeAttribute('src');video.load();document.getElementById('emptyVideo').hidden=false;document.getElementById('videoStatus').textContent=''}render()}
function save(){fetch('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({schema_version:1,task,control_hz:hz,annotations})}).then(r=>{if(!r.ok)throw Error('save failed');document.getElementById('saved').textContent='Saved'}).catch(err=>{document.getElementById('saved').textContent=err.message})}
function setOutcome(outcome){const e=episodes[current],end=e.steps-1;reviewFull=true;annotations[e.id]={...(annotations[e.id]||{}),recorded_outcome:e.success?'success':'failure',post_mark_buffer_seconds:null,post_failure_buffer_seconds:null,post_success_buffer_seconds:null,failure_onset_step:null,failure_onset_seconds:null,success_mark_step:null,success_mark_seconds:null,episode_end_step:end,effective_end_step:end,discarded_steps:0,episode_end_seconds:end/hz,annotated_outcome:outcome};video.pause();render()}
function markOutcomeAndCut(outcome){const e=episodes[current],seconds=Math.max(0,Number(buffer.value)||0),end=Math.min(e.steps-1,step+Math.round(seconds*hz)),isFailure=outcome==='failure',a={...(annotations[e.id]||{}),recorded_outcome:e.success?'success':'failure',post_mark_buffer_seconds:seconds,post_failure_buffer_seconds:isFailure?seconds:null,post_success_buffer_seconds:isFailure?null:seconds,failure_onset_step:isFailure?step:null,failure_onset_seconds:isFailure?step/hz:null,success_mark_step:isFailure?null:step,success_mark_seconds:isFailure?null:step/hz,episode_end_step:end,effective_end_step:end,discarded_steps:e.steps-end-1,episode_end_seconds:end/hz,annotated_outcome:outcome};reviewFull=false;annotations[e.id]=a;render();seekVideo(e,step,true)}
document.getElementById('markSuccess').onclick=()=>markOutcomeAndCut('success');document.getElementById('markFailure').onclick=()=>markOutcomeAndCut('failure');document.getElementById('success').onclick=()=>setOutcome('success');document.getElementById('setFailure').onclick=()=>setOutcome('failure');document.getElementById('removeEpisode').onclick=()=>{const e=episodes[current],a=annotations[e.id]||{};a.excluded=!a.excluded;annotations[e.id]=a;video.pause();document.getElementById('saved').textContent='Unsaved changes';if(a.excluded){const next=episodes.findIndex((item,i)=>!annotations[item.id]?.excluded&&i!==current);if(next>=0)choose(next);else render()}else render()};document.getElementById('clear').onclick=()=>{const e=episodes[current],old=annotations[e.id]||{},keep={};if(old.description)keep.description=old.description;if(old.excluded)keep.excluded=true;delete annotations[e.id];if(Object.keys(keep).length)annotations[e.id]=keep;reviewFull=false;video.pause();render()};document.getElementById('review').onclick=()=>{reviewFull=!reviewFull;render()};document.getElementById('save').onclick=save;description.addEventListener('input',()=>{const e=episodes[current],value=description.value;if(value.trim())annotations[e.id]={...(annotations[e.id]||{}),description:value};else if(annotations[e.id]){delete annotations[e.id].description;if(!Object.keys(annotations[e.id]).length)delete annotations[e.id]}updateEpisodeList();document.getElementById('status').textContent=value.trim()?'Description added':'Description cleared';document.getElementById('saved').textContent='Unsaved changes'});
function syncVideoStep(){const e=episodes[current],a=annotations[e.id];step=Math.max(0,Math.min(e.steps-1,stepForVideoTime(e,video.currentTime)));if(!reviewFull&&a&&a.episode_end_step!=null&&step>=a.episode_end_step){video.pause();step=a.episode_end_step;const endTime=videoTimeForStep(e,a.episode_end_step);if(Math.abs(video.currentTime-endTime)>0.02)video.currentTime=endTime}draw()}
canvas.addEventListener('click',ev=>{const r=canvas.getBoundingClientRect(),x=(ev.clientX-r.left-55)/(r.width-73),e=episodes[current],max=reviewFull?e.steps-1:endStep(e,annotations[e.id]);step=Math.max(0,Math.min(max,Math.round(x*(e.steps-1))));seekVideo(e,step);draw()});video.addEventListener('loadedmetadata',()=>{if(pendingSeek&&episodes[current].id===pendingSeek.episodeId){const p=pendingSeek;pendingSeek=null;seekVideo(episodes[current],p.step,p.play)}else if(video.currentTime===0&&video.duration>0)video.currentTime=Math.min(0.01,video.duration)});video.addEventListener('timeupdate',syncVideoStep);video.addEventListener('seeked',syncVideoStep);window.addEventListener('resize',draw);choose(0);
</script></body></html>'''

    return html.replace("__DATA__", payload).replace("__HZ__", str(hz)).replace("__TASK__", json.dumps(task))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory, page, annotations_path, **kwargs):
        self.page = page.encode()
        self.annotations_path = annotations_path
        super().__init__(*args, directory=str(directory), **kwargs)

    # Handle an HTTP GET request.
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.page)
            return
        if path == "/api/annotations":
            data = json.loads(self.annotations_path.read_text()) if self.annotations_path.is_file() else {"annotations": {}}
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/videos/"):
            self.serve_video()
            return
        super().do_GET()

    # Handle an HTTP HEAD request.
    def do_HEAD(self):
        if urlsplit(self.path).path.startswith("/videos/"):
            self.serve_video(send_body=False)
            return
        super().do_HEAD()

    # Stream the requested episode video with range support.
    def serve_video(self, send_body=True):
        path = Path(self.translate_path(urlsplit(self.path).path)).resolve()
        root = Path(self.directory).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        header = self.headers.get("Range")
        if header and header.strip() != "bytes=0-":
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
            if not match or not any(match.groups()):
                self.send_error(416, "Invalid byte range")
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                length = int(last)
                start = max(0, size - length)
            if start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if send_body:
            with path.open("rb") as video_file:
                video_file.seek(start)
                remaining = length
                while remaining:
                    chunk = video_file.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    # Handle an HTTP POST request.
    def do_POST(self):
        if self.path != "/api/annotations":
            self.send_error(404)
            return
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        if not isinstance(data.get("annotations"), dict):
            self.send_error(400, "annotations must be an object")
            return
        self.annotations_path.write_text(json.dumps(data, indent=2) + "\n")
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description='Annotate rollout outcomes and optional episode cut points.')
    parser.add_argument("--record-dir", type=Path, help="folder containing manifest.json; opens a folder chooser if omitted")
    parser.add_argument("--port", type=int, default=0, help="local HTTP port (default: choose an available port)")
    parser.add_argument("--no-browser", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    selected_dir = args.record_dir or choose_record_dir()
    if selected_dir is None:
        print("Folder selection cancelled.")
        return
    record_dir = selected_dir.expanduser().resolve()
    page = build_page(record_dir)
    handler = lambda *a, **kw: Handler(*a, directory=record_dir, page=page,
                                        annotations_path=record_dir / "failure_annotations.json", **kw)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Annotating {record_dir}\nOpen {url} (Ctrl-C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAnnotator stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
