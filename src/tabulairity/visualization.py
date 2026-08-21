"""
visualization.py — Live force-directed visualization for TabulAIrity.

Default OFF. Call vizOn() after `import tabulairity` to start a local HTTP
server that serves a D3.js force-directed graph + scrolling prompt/response
panel. Call vizOff() to stop it.

Architecture:
  Python core (walkChatNet / processNodeStep / buildChatNet / askChatQuestion)
    → viz_emit(type, payload)  (no-op when disabled, non-blocking when enabled)
    → VizServer broadcasts JSON over Server-Sent Events (SSE) to browser
    → Frontend (D3 v7) renders force graph, handles per-chatnet animation delay,
      color states, and FIFO side-panel.

No new Python dependencies — uses stdlib http.server + threading.
Frontend loads D3 from CDN; single-file HTML is embedded in this module.
"""
from __future__ import annotations

import json
import queue
import re
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Global viz state
# ---------------------------------------------------------------------------
_viz_enabled: bool = False
_viz_server: "VizServer | None" = None
_viz_server_thread: threading.Thread | None = None
_next_chatnet_id: int = 1
_chatnet_titles: dict[int, str] = {}
_viz_lock = threading.Lock()

# SSE clients — each is a queue.Queue that the handler drains
_sse_clients: list[queue.Queue] = []
_sse_clients_lock = threading.Lock()

# Replay buffer for late-joining browsers (graph_load etc. before they connected)
_event_log: list[str] = []
_event_log_lock = threading.Lock()
_EVENT_LOG_MAX = 200

# Mapping from id(G) -> chatnet_id for graphs that haven't stored it in G.graph
_graph_id_map: dict[int, int] = {}

# ---------------------------------------------------------------------------
# Low-level emit / helpers
# ---------------------------------------------------------------------------

def viz_is_enabled() -> bool:
    return _viz_enabled


def viz_emit(event_type: str, payload: dict) -> None:
    """Thread-safe broadcast. No-op when viz is off. Never raises."""
    if not _viz_enabled:
        return
    msg = {"type": event_type, "payload": payload, "ts": time.time()}
    data = json.dumps(msg, default=str)
    # keep replay buffer (graph_load, node_queued etc. — cap to avoid unbounded growth)
    with _event_log_lock:
        _event_log.append(data)
        if len(_event_log) > _EVENT_LOG_MAX:
            _event_log.pop(0)
    with _sse_clients_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        # drop full clients (should not happen with unbounded queue)
        for q in dead:
            _sse_clients.remove(q)


def _allocate_chatnet_id() -> int:
    global _next_chatnet_id
    with _viz_lock:
        cid = _next_chatnet_id
        _next_chatnet_id += 1
    return cid


def _get_or_create_chatnet_id(G) -> int:
    """Get G.graph['viz_id'] or allocate one."""
    try:
        cid = G.graph.get("viz_id")
        if cid is not None:
            return cid
    except Exception:
        pass
    # fallback to id(G) map
    gid = id(G)
    with _viz_lock:
        if gid in _graph_id_map:
            return _graph_id_map[gid]
        cid = _next_chatnet_id
        # inline allocate to avoid double-lock
        globals()["_next_chatnet_id"] += 1
        _graph_id_map[gid] = cid
        try:
            G.graph["viz_id"] = cid
        except Exception:
            pass
        return cid


def _generate_title_sync(G, chatnet_id: int) -> str:
    """Generate a short title for a chatNet via LLM, non-blocking for caller."""
    # Collect prompts heuristic first — used as fallback
    try:
        prompts = []
        for _, data in G.nodes(data=True):
            p = data.get("prompt", "")
            if isinstance(p, str) and p.strip():
                prompts.append(p.strip()[:120])
            if len(prompts) >= 5:
                break
        heuristic = prompts[0][:40] if prompts else f"ChatNet {chatnet_id}"
        heuristic = heuristic.split("\n")[0].strip()
        if len(heuristic) > 50:
            heuristic = heuristic[:47] + "..."
    except Exception:
        heuristic = f"ChatNet {chatnet_id}"

    # Try LLM title in background thread so we don't block the caller
    def _llm_title():
        try:
            from . import core as _core
            persona = "You are a concise label generator. Return ONLY a 2-4 word title, no punctuation, no quotes."
            combined = " | ".join(prompts[:3]) if prompts else heuristic
            prompt = f"Generate a short descriptive title for this extraction network. Prompts: {combined}"
            # Bypass askChatQuestion's viz hooks by calling the cache layer directly
            # (askChatQuestion would emit standalone_prompt events and also temporarily
            # flip _viz_enabled globally — we must not do that while a walk is in progress).
            messages = [
                {"role": "system", "content": persona},
                {"role": "user", "content": prompt[:350000]},
            ]
            cacheKey = f"getChatContent({messages},20,'{_core.modelName}',None,None,timeout=600,extra_params=None)"
            title = _core.queryToCache(
                cacheKey,
                _core.getChatContent,
                args=(messages, 20, _core.modelName),
                kwargs={"temperature": None, "seed": None, "timeout": 600, "extra_params": None},
                tolerant=False,
            )
            title = str(title).strip().strip('"').strip("'")
            title = title.split("\n")[0][:50]
            if title:
                _chatnet_titles[chatnet_id] = title
                viz_emit("graph_title", {"chatnet_id": chatnet_id, "title": title})
        except Exception:
            pass

    # Cache heuristic immediately so frontend has something
    _chatnet_titles[chatnet_id] = heuristic
    threading.Thread(target=_llm_title, daemon=True).start()
    return heuristic


# ---------------------------------------------------------------------------
# Public hooks called from core.py
# ---------------------------------------------------------------------------

def viz_notify_graph_load(G) -> int:
    """Called from core.buildChatNet after G is built. Returns chatnet_id."""
    if not _viz_enabled:
        return _get_or_create_chatnet_id(G)
    try:
        cid = _get_or_create_chatnet_id(G)
        title = _generate_title_sync(G, cid)

        nodes = []
        for n, data in G.nodes(data=True):
            nodes.append({
                "id": str(n),
                "label": str(n),
                "chatnet_id": cid,
                "prompt": (str(data.get("prompt", ""))[:500] if data.get("prompt") else ""),
                "fx": str(data.get("fx", "")),
            })

        edges = []
        # MultiDiGraph: edges may have duplicate (u,v). Include key index to dedup id.
        for idx, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):
            # data may be the key if using add_edges_from with edict — handle both
            fx = ""
            prompt = ""
            if isinstance(data, dict):
                fx = str(data.get("fx", k if isinstance(k, str) else ""))
                prompt = str(data.get("prompt", ""))[:200]
            else:
                fx = str(k)
            edge_id = f"{u}-{v}__{idx}"
            # Also store canonical label like "Start-Chat" for color mapping
            canonical = f"{u}-{v}"
            edges.append({
                "id": edge_id,
                "canonical": canonical,
                "source": str(u),
                "target": str(v),
                "label": fx,
                "prompt": prompt,
                "chatnet_id": cid,
            })

        viz_emit("graph_load", {
            "chatnet_id": cid,
            "title": title,
            "nodes": nodes,
            "edges": edges,
        })
        return cid
    except Exception:
        return 0


def viz_notify_node_queued(chatnet_id: int, node_id: str):
    viz_emit("node_queued", {"chatnet_id": chatnet_id, "node_id": str(node_id)})


def viz_notify_node_start(chatnet_id: int, node_id: str, prompt: str = "", persona: str = "", fullPrompt: str = None):
    viz_emit("node_start", {
        "chatnet_id": chatnet_id,
        "node_id": str(node_id),
        "prompt": (prompt or "")[:500],
        "persona": (persona or "")[:2000],
        "fullPrompt": (fullPrompt if fullPrompt is not None else prompt or "")[:5000],
    })


def viz_notify_node_complete(chatnet_id: int, node_id: str, response: str = "", cleaned: str = ""):
    viz_emit("node_complete", {
        "chatnet_id": chatnet_id,
        "node_id": str(node_id),
        "response": (response or "")[:500],
        "cleaned": (cleaned or "")[:500],
    })


def viz_notify_node_error(chatnet_id: int, node_id: str, error: str = ""):
    viz_emit("node_error", {
        "chatnet_id": chatnet_id,
        "node_id": str(node_id),
        "error": (error or "")[:500],
    })


def viz_notify_edge_evaluated(chatnet_id: int, edge_id: str, result: bool, fx: str | None = None):
    viz_emit("edge_evaluated", {
        "chatnet_id": chatnet_id,
        "edge_id": str(edge_id),
        "canonical": str(edge_id),
        "result": bool(result),
        "fx": str(fx) if fx is not None else None,
    })


def viz_notify_prompt_response(chatnet_id: int, node_id: str, prompt: str, response: str, cleaned: str = "", persona: str = "", fx: str = "", fullPrompt: str = None):
    viz_emit("prompt_response", {
        "chatnet_id": chatnet_id,
        "node_id": str(node_id),
        "prompt": (prompt or "")[:500],
        "response": (response or "")[:500],
        "cleaned": (cleaned or "")[:500],
        "persona": (persona or "")[:2000],
        "fx": str(fx or ""),
        "fullPrompt": (fullPrompt if fullPrompt is not None else prompt or "")[:5000],
    })


def viz_notify_chatnet_complete(chatnet_id: int, success: bool = True):
    viz_emit("chatnet_complete", {"chatnet_id": chatnet_id, "success": bool(success)})


def viz_notify_chatnet_reset(chatnet_id: int):
    viz_emit("chatnet_reset", {"chatnet_id": chatnet_id})


def viz_notify_standalone_prompt(prompt: str, persona: str, model: str):
    if not _viz_enabled:
        return
    viz_emit("standalone_prompt", {
        "prompt": (prompt or "")[:500],
        "persona": (persona or "")[:200],
        "model": str(model),
    })


def viz_notify_standalone_response(prompt: str, response: str):
    if not _viz_enabled:
        return
    viz_emit("standalone_response", {
        "prompt": (prompt or "")[:500],
        "response": (response or "")[:500],
    })


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>TabulAIrity Live Viz</title>
<script src="/d3.v7.min.js"></script>
<script>if(typeof d3==='undefined'){document.write('<script src="https://d3js.org/d3.v7.min.js"><\/script>')}</script>
<script>if(typeof d3==='undefined'){document.write('<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"><\/script>')}</script>
<style>
  *{box-sizing:border-box}
  html,body{margin:0;height:100vh;overflow:hidden;font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#0f1419;color:#e6eef6}
  #bar{height:42px;display:flex;align-items:center;gap:12px;padding:0 14px;background:#1a2332;border-bottom:1px solid #2a3a4d}
  #bar h1{font-size:14px;font-weight:600;letter-spacing:.04em;margin:0;color:#9ec5ff}
  #bar .pill{font-size:11px;padding:3px 8px;border-radius:999px;background:#223044;border:1px solid #2e415c;color:#9fb4cc}
  #bar .spacer{flex:1}
  #bar button{font-size:12px;padding:5px 10px;border-radius:6px;border:1px solid #2e415c;background:#1e2d44;color:#cfe0f5;cursor:pointer}
  #bar button:hover{background:#24406a}
  #wrap{display:flex;flex:1;height:calc(100vh - 42px);min-height:0;overflow:hidden}
  #graphWrap{flex:1 1 0;position:relative;overflow:hidden;background:radial-gradient(1200px 800px at 40% 40%, #162233 0%, #0f1419 70%);min-height:420px;min-width:0;min-height:0}
  #graphWrap svg{width:100%;height:100%;min-height:420px;display:block}
  #d3Error{position:absolute;top:12px;left:50%;transform:translateX(-50%);background:#3d1a1a;border:1px solid #7a2a2a;color:#ffb4b4;padding:8px 12px;border-radius:8px;font-size:12px;display:none;z-index:10;max-width:80%;text-align:center}
  #tooltip{position:absolute;display:none;pointer-events:none;z-index:20;max-width:380px;background:rgba(17,26,38,.98);border:1px solid #2a3a4d;border-radius:8px;padding:8px 10px;font-size:11px;line-height:1.45;color:#e6eef6;box-shadow:0 8px 24px rgba(0,0,0,.45)}
  #tooltip .tt-title{font-weight:700;color:#9ec5ff;margin-bottom:4px;font-size:12px}
  #tooltip .tt-meta{color:#7aa0c6;font-size:10px;margin-bottom:6px}
  #tooltip .tt-body{white-space:pre-wrap;word-break:break-word;max-height:180px;overflow:auto;color:#cfe0f5}
  #tooltip .tt-row{margin:2px 0}
  #popupOverlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:30;align-items:center;justify-content:center}
  #popupOverlay.open{display:flex}
  #popup{width:min(820px,92vw);max-height:88vh;overflow:auto;background:#0f1419;border:1px solid #2a3a4d;border-radius:12px;box-shadow:0 16px 48px rgba(0,0,0,.6);padding:16px}
  #popup h3{margin:0 0 8px;color:#9ec5ff;font-size:14px}
  #popup .popup-meta{color:#7aa0c6;font-size:11px;margin-bottom:10px}
  #popup .popup-section{margin:10px 0}
  #popup .popup-label{font-size:10px;letter-spacing:.04em;color:#8fbcff;margin-bottom:4px;font-weight:600}
  #popup pre{white-space:pre-wrap;word-break:break-word;background:#111a26;border:1px solid #1e2d44;border-radius:8px;padding:10px;font-size:11.5px;line-height:1.5;max-height:240px;overflow:auto;color:#cfe0f5}
  #popup .closeBtn{float:right;background:#1a2332;border:1px solid #2a3a4d;color:#9ec5ff;border-radius:6px;padding:4px 8px;cursor:pointer}
  #popup .closeBtn:hover{background:#1e2d44}
  #side{flex:0 0 420px;width:420px;min-width:320px;max-width:520px;border-left:1px solid #2a3a4d;background:#0f1419;display:flex;flex-direction:column;min-height:0;overflow:hidden;height:100%}
  #sideHead{padding:10px 12px;border-bottom:1px solid #1e2d44;font-size:12px;color:#9fb4cc;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
  #sideHead .count{font-variant-numeric:tabular-nums;color:#7aa0c6}
  #panel{flex:1 1 auto;overflow-y:auto;overflow-x:hidden;padding:8px 10px;display:flex;flex-direction:column;gap:8px;min-height:0;align-items:stretch}
  .entry{border:1px solid #1e2d44;border-radius:8px;overflow:hidden;background:#111a26;flex-shrink:0}
  .entryHead{padding:6px 8px;font-size:11px;font-weight:600;letter-spacing:.03em;background:#132034;border-bottom:1px solid #1e2d44;display:flex;justify-content:space-between;color:#9ec5ff}
  .entryHead .cid{font-weight:400;color:#7aa0c6}
  .promptBox,.respBox{padding:7px 8px;font-size:11.5px;line-height:1.45;white-space:pre-wrap;word-break:break-word;max-height:120px;overflow:auto}
  .promptBox{background:#142033;color:#cfe0f5;border-left:3px solid #3a7bd5}
  .respBox{background:#14301e;color:#d4f5df;border-left:3px solid #2ecc71}
  .empty{color:#6b86a3;font-size:12px;text-align:center;padding:24px 12px}
  /* legend overlay */
  #legend{position:absolute;left:10px;bottom:10px;background:rgba(17,26,38,.92);border:1px solid #1e2d44;border-radius:8px;padding:8px 10px;font-size:11px;color:#9fb4cc;backdrop-filter:blur(6px)}
  #legend .row{display:flex;align-items:center;gap:6px;margin:3px 0}
  .dot{width:10px;height:10px;border-radius:50%;display:inline-block;border:1px solid rgba(255,255,255,.15)}
  .line{width:18px;height:3px;border-radius:2px;display:inline-block}
  #stats{position:absolute;right:10px;bottom:10px;background:rgba(17,26,38,.92);border:1px solid #1e2d44;border-radius:8px;padding:6px 8px;font-size:11px;color:#7aa0c6}
  /* node flash */
  @keyframes flash{0%{filter:brightness(2.2)}100%{filter:brightness(1)}}
  .flash{animation:flash .45s ease 2}
</style>
</head>
<body>
<div id="bar">
  <h1>◈ TABULAIRITY — LIVE</h1>
  <span class="pill" id="statusPill">● connecting…</span>
  <span class="pill" id="graphPill">0 chatNets</span>
  <span class="pill" id="nodePill">0 nodes</span>
  <div class="spacer"></div>
  <button id="resetBtn" title="Reset view">Reset view</button>
  <button id="clearBtn" title="Clear side panel">Clear panel</button>
</div>
<div id="debug" style="position:absolute;top:44px;left:10px;max-width:520px;max-height:120px;overflow:auto;background:rgba(0,0,0,.75);color:#ff8f8f;font-family:monospace;font-size:10px;padding:6px 8px;border-radius:6px;z-index:5;display:none;white-space:pre-wrap;pointer-events:none"></div>
<div id="wrap">
  <div id="graphWrap">
    <div id="tooltip"></div>
    <div id="d3Error">⚠ D3.js failed to load (offline?). Graphs need <code>d3.v7.min.js</code> from CDN. Check network or vendor D3 locally.</div>
    <svg id="svg"></svg>
    <div id="legend">
      <div class="row"><span class="dot" style="background:#6aa3d9"></span> idle</div>
      <div class="row"><span class="dot" style="background:#f5c842;box-shadow:0 0 8px #f5c842"></span> queued (flash on add)</div>
      <div class="row"><span class="dot" style="background:#ff8c42;box-shadow:0 0 10px #ff7a18"></span> processing</div>
      <div class="row"><span class="dot" style="background:#2ecc71"></span> completed</div>
      <div class="row"><span class="dot" style="background:#ff4d4d"></span> error</div>
      <div class="row"><span class="line" style="background:#8a8f98"></span> idle edge</div>
      <div class="row"><span class="line" style="background:#3a7bd5"></span> edge true</div>
      <div class="row"><span class="line" style="background:#ff4d4d"></span> edge false</div>
      <div class="row" style="margin-top:6px;border-top:1px solid #1e2d44;padding-top:6px;color:#7aa0c6">drag node to move · dbl-click to release · scroll to zoom · drag bg to pan</div>
    </div>
    <div id="stats"></div>
  </div>
  <div id="side">
    <div id="sideHead"><span>Prompts &amp; Responses <span style="color:#5a7896">(first 500 chars)</span></span><span class="count" id="panelCount">0 / 20</span></div>
    <div id="panel"><div class="empty">No prompts yet. Run a chatNet with <code>walkChatNet</code> or <code>askChatQuestion</code>.</div></div>
  </div>
</div>
<div id="popupOverlay" onclick="if(event.target===this) closePopup()"><div id="popup"><button class="closeBtn" onclick="closePopup()">✕ Close</button><h3 id="popupTitle"></h3><div id="popupMeta" class="popup-meta"></div><div class="popup-section"><div class="popup-label">PROMPT (full)</div><pre id="popupPrompt"></pre></div><div class="popup-section"><div class="popup-label">SYSTEM PROMPT (persona)</div><pre id="popupPersona"></pre></div><div class="popup-section"><div class="popup-label">RESPONSE (raw)</div><pre id="popupContext"></pre></div></div></div>
<script>
const MAX_PANEL = 20;
const ANIM_DELAY = 2000; // per-chatnet

function dbg(msg, isError=false){
  console.log(msg);
  if(!isError) return;
  const el=document.getElementById('debug');
  if(!el) return;
  el.style.display='block';
  el.textContent += msg + "\n";
  el.scrollTop = el.scrollHeight;
}
function showTip(html, evt){
  const tip=document.getElementById('tooltip');
  if(!tip) return;
  tip.innerHTML=html;
  tip.style.display='block';
  moveTip(evt);
}
function moveTip(evt){
  const tip=document.getElementById('tooltip');
  if(!tip || tip.style.display==='none') return;
  const pad=14;
  let x=evt.clientX+pad, y=evt.clientY+pad;
  const r=tip.getBoundingClientRect();
  if(x+r.width > window.innerWidth-10) x=evt.clientX - r.width - pad;
  if(y+r.height > window.innerHeight-10) y=evt.clientY - r.height - pad;
  tip.style.left=x+'px';
  tip.style.top=y+'px';
}
function hideTip(){ const t=document.getElementById('tooltip'); if(t) t.style.display='none'; }
function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function nodeTip(d){
  const ch=chatnets.get(String(d.chatnet_id));
  const title=ch?ch.title:'';
  return `<div class="tt-title">${esc(d.label)}</div><div class="tt-meta">ChatNet: ${esc(title)} (id ${esc(d.chatnet_id)}) · state: ${esc(d.state)} · fx: ${esc(d.cdenFx||'')}</div><div class="tt-body">${esc((d.prompt||'').slice(0,500))||'<i>no prompt</i>'}</div>`;
}
function edgeTip(d){
  const ch=chatnets.get(String(d.chatnet_id));
  const title=ch?ch.title:'';
  return `<div class="tt-title">${esc(d.canonical)}</div><div class="tt-meta">ChatNet: ${esc(title)} · fx: ${esc(d.label)} · state: ${esc(d.state)}</div><div class="tt-body">${esc((d.prompt||'').slice(0,300))||'<i>no edge prompt</i>'}</div>`;
}
window.addEventListener('error', e=> dbg('JS ERROR: '+(e.message||e.error||e.filename+':'+e.lineno), true));
window.addEventListener('unhandledrejection', e=> dbg('REJECTION: '+(e.reason&&e.reason.message||e.reason), true));

if(typeof d3==='undefined'){
  document.getElementById('d3Error').style.display='block';
  document.getElementById('statusPill').textContent='● D3 missing';
  document.getElementById('statusPill').style.background='#3d1a1a';
  dbg('D3 not loaded — will use fallback list view', true);
}
let _svgEl = document.getElementById("svg");
let svg = (typeof d3!=='undefined') ? d3.select("#svg") : null;
let gWrap = null, gEdges=null, gEdgeLabels=null, gNodes=null, gNodeLabels=null;
if(svg){
  // diffuse glow via CSS drop-shadow (no SVG filter needed — kept minimal defs to avoid parse errors)
  svg.append("defs");
  gWrap = svg.append("g");
  gEdges = gWrap.append("g").attr("id","gEdges");
  gEdgeLabels = gWrap.append("g").attr("id","gEdgeLabels");
  gNodes = gWrap.append("g").attr("id","gNodes");
  gNodeLabels = gWrap.append("g").attr("id","gNodeLabels");
}

// data (must be before resize which may reference chatnets)
let nodes = []; // {id,label,chatnet_id, state}
let edges = []; // {id,canonical,source,target,label,chatnet_id, state}
let nodeById = new Map();
let edgeByCanonical = new Map();

// per-chatnet state
// chatnetId -> {title, colorIdx, center:{x,y}, animating, timer, pending:[]}
const chatnets = new Map();
const palette = ["#6aa3d9","#a78bfa","#f59e0b","#34d399","#f472b6","#60a5fa","#f97316","#a3e635"];
let colorPtr = 0;

let width=800, height=600;
function resize(){
  const el = document.getElementById("graphWrap");
  const r = el.getBoundingClientRect();
  width = r.width; height = r.height;
  // fallback when flex hasn't laid out yet (0)
  if(width < 50 || height < 50){
    width = Math.max(600, window.innerWidth - 440);
    height = Math.max(420, window.innerHeight - 60);
  }
  if(svg) svg.attr("viewBox", `0 0 ${width} ${height}`);
  // nudge simulation centers if exists
  try{
    if(chatnets.size){
      const R = Math.min(width,height)*0.28;
      const cx = width/2, cy = height/2;
      let i=0; const n=chatnets.size;
      for(const [id,ch] of chatnets){
        const ang = (i / n)*Math.PI*2 - Math.PI/2;
        ch.center.x = cx + Math.cos(ang)*R;
        ch.center.y = cy + Math.sin(ang)*R;
        i++;
      }
    }
  } catch(e){ /* chatnets not yet init during early resize */ }
}
resize();
window.addEventListener("resize", resize);
setTimeout(resize, 120);
setTimeout(resize, 600);

// zoom/pan
let zoom=null;
if(svg && typeof d3!=='undefined'){
  zoom = d3.zoom().scaleExtent([0.15,4]).on("zoom", e=> gWrap.attr("transform", e.transform));
  svg.call(zoom);
}

function ensureChatnet(cid, title){
  cid = String(cid);
  if(!chatnets.has(cid)){
    const c = palette[colorPtr % palette.length]; colorPtr++;
    // place centers on circle
    const n = chatnets.size + 1;
    // recompute all centers in circle
    const allIds = [...chatnets.keys(), cid];
    const R = Math.min(width,height)*0.28;
    const cx = width/2, cy = height/2;
    allIds.forEach((id,i)=>{
      const ang = (i / allIds.length)*Math.PI*2 - Math.PI/2;
      const ch = chatnets.get(id) || {center:{x:cx+Math.cos(ang)*R, y:cy+Math.sin(ang)*R}};
      ch.center = {x: cx + Math.cos(ang)*R, y: cy + Math.sin(ang)*R};
      if(chatnets.has(id)) chatnets.set(id,ch);
    });
    const ang = ((allIds.length-1)/allIds.length)*Math.PI*2 - Math.PI/2;
    chatnets.set(cid, {title: title||`ChatNet ${cid}`, color:c, center:{x:cx+Math.cos(ang)*R, y:cy+Math.sin(ang)*R}, animating:false, timer:null, pending:[]});
    // start anim delay timer
    const entry = chatnets.get(cid);
    entry.timer = setTimeout(()=> {
      entry.animating = true;
      // flush pending
      const pend = entry.pending.slice(); entry.pending=[];
      pend.forEach(fn=>fn());
      updateGraph(true);
    }, ANIM_DELAY);
    document.getElementById("graphPill").textContent = `${chatnets.size} chatNets`;
  } else if(title && chatnets.get(cid).title.startsWith("ChatNet")){
    chatnets.get(cid).title = title;
  }
  return chatnets.get(cid);
}

let simulation = null;

// drag handlers — keep nodes interactive: click+drag to reposition,
// double-click to release pin; hides tooltip while dragging so it
// doesn't obscure the node. Works with zoom/pan (drag on nodes,
// zoom on background).
function dragstarted(event, d){
  hideTip();
  if(!event.active && simulation) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
  // show grabbing cursor
  if(event.sourceEvent && event.sourceEvent.target) event.sourceEvent.target.style.cursor='grabbing';
}
function dragged(event, d){
  d.fx = event.x;
  d.fy = event.y;
}
function dragended(event, d){
  if(!event.active && simulation) simulation.alphaTarget(0);
  // keep pinned at dragged position — double-click to release
  if(event.sourceEvent && event.sourceEvent.target) event.sourceEvent.target.style.cursor='grab';
}

function makeSim(){
  if(!svg || typeof d3==='undefined') return;
  if(simulation) simulation.stop();
  simulation = d3.forceSimulation(nodes)
    .force("link",
      d3.forceLink(edges)
        .id(d => d.id)
        .distance(110)
        .strength(0.35)
    )
    .force("charge",
      d3.forceManyBody()
        .strength(-450)
        .distanceMin(20)   // prevent 1/d^2 blowup when jittered nodes start near-coincident
    )
    .force("collision",
      d3.forceCollide()
        .radius(32)
        .strength(0.8)
    )
    .alphaDecay(0.0228)   // back to d3 default — was cooling the sim before it could spread out
    .velocityDecay(0.45)  // back near d3 default — was over-damping every tick
    .alpha(0.6)
    .on("tick", ticked);
}

function ticked(){
  if(!svg) return;
  for(const d of nodes){
    if(!isFinite(d.x) || !isFinite(d.y)){
      d.x = width / 2;
      d.y = height / 2;
    }
  }
  gEdges.selectAll("line")
    .attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
    .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  // edge label positions mid+offset perp
  gEdgeLabels.selectAll("text")
    .attr("x",d=> (d.source.x + d.target.x)/2)
    .attr("y",d=> (d.source.y + d.target.y)/2 - 6);
  gNodes.selectAll("circle")
    .attr("cx",d=>d.x).attr("cy",d=>d.y);
  gNodeLabels.selectAll("text")
    .attr("x",d=>d.x).attr("y",d=>d.y + 4);

  // chatnet title hulls/badges — update positions
  gWrap.selectAll("text.chatnetTitle")
    .attr("x",d=>d.center.x).attr("y",d=>d.center.y - (Math.min(width,height)*0.28) - 18);
}

function colorForNode(d){
  const ch = chatnets.get(String(d.chatnet_id));
  const base = ch? ch.color : "#6aa3d9";
  switch(d.state){
    case "queued": return "#f5c842";
    case "processing": return "#ff8c42";
    case "completed": return "#2ecc71";
    case "error": return "#ff4d4d";
    default: return base;
  }
}
function colorForEdge(d){
  if(d.state==="true") return "#3a7bd5";
  if(d.state==="false") return "#ff4d4d";
  return "#8a8f98";
}

function renderFallbackList(){
  // shown when D3 missing — render node/edge list as HTML
  const wrap = document.getElementById("graphWrap");
  let fb = document.getElementById("fallbackList");
  if(!fb){
    fb = document.createElement("div");
    fb.id="fallbackList";
    fb.style.cssText="position:absolute;inset:10px;overflow:auto;background:rgba(17,26,38,.96);border:1px solid #1e2d44;border-radius:8px;padding:10px;font-size:11px;color:#9fb4cc";
    wrap.appendChild(fb);
  }
  let html = `<div style="margin-bottom:8px;color:#9ec5ff;font-weight:600">${nodes.length} nodes · ${edges.length} edges · ${chatnets.size} nets (fallback — D3 unavailable)</div>`;
  for(const [cid,ch] of chatnets){
    html += `<div style="margin:8px 0 4px;font-weight:600;color:${ch.color}">${ch.title} <span style="font-weight:400;color:#7aa0c6">(id ${cid})</span></div>`;
    const cn = nodes.filter(n=> String(n.chatnet_id)===String(cid));
    const ce = edges.filter(e=> String(e.chatnet_id)===String(cid));
    html += `<div>Nodes: ${cn.map(n=>`<span style="display:inline-block;margin:2px 4px;padding:2px 6px;border-radius:999px;background:${colorForNode(n)};color:#0b1220;font-weight:600">${n.label}:${n.state}</span>`).join("")}</div>`;
    html += `<div style="margin-top:4px">Edges: ${ce.map(e=>`<span style="display:inline-block;margin:2px 4px;padding:2px 6px;border-radius:4px;border:1px solid ${colorForEdge(e)};color:${colorForEdge(e)}">${e.canonical}[${e.label}]:${e.state}</span>`).join("")}</div>`;
  }
  fb.innerHTML = html;
}
function updateGraph(isStructural=false){
  if(!svg || typeof d3==='undefined'){
    renderFallbackList();
    document.getElementById("nodePill").textContent = `${nodes.length} nodes`;
    document.getElementById("graphPill").textContent = `${chatnets.size} chatNets`;
    return;
  }
  // data joins
  const link = gEdges.selectAll("line").data(edges, d=>d.id);
  link.enter().append("line")
    .attr("stroke-width",4).attr("stroke-opacity",0.9)
    .attr("stroke-linecap","round")
    .on("mousemove", (evt,d)=>{ showTip(edgeTip(d), evt); })
    .on("mouseenter", (evt,d)=>{ showTip(edgeTip(d), evt); })
    .on("mouseleave", hideTip)
    .merge(link)
    .attr("stroke", d=>colorForEdge(d))
    .attr("stroke-dasharray", d=> d.state==="false" ? "6 4" : null)
    .on("mousemove", (evt,d)=>{ showTip(edgeTip(d), evt); })
    .on("mouseenter", (evt,d)=>{ showTip(edgeTip(d), evt); })
    .on("mouseleave", hideTip);
  link.exit().remove();

  const eLabel = gEdgeLabels.selectAll("text").data(edges, d=>d.id);
  eLabel.enter().append("text")
    .attr("text-anchor","middle").attr("font-size","10px").attr("fill","#e6eef6").attr("pointer-events","none")
    .attr("paint-order","stroke").attr("stroke","#0f1419").attr("stroke-width",2)
    .merge(eLabel).text(d=>d.label||"");
  eLabel.exit().remove();

  const n = gNodes.selectAll("circle").data(nodes, d=>d.id);
  const nEnter = n.enter().append("circle")
    .attr("r",22).attr("stroke","#0f1419").attr("stroke-width",1.5)
    .style("cursor","grab")
    .on("mousemove", (evt,d)=>{ showTip(nodeTip(d), evt); })
    .on("mouseenter", (evt,d)=>{ showTip(nodeTip(d), evt); })
    .on("mouseleave", hideTip)
    .on("dblclick", (evt,d)=>{
      // double-click releases pin and lets force reflow
      d.fx = null; d.fy = null;
      if(simulation) simulation.alphaTarget(0.3).restart();
      evt.stopPropagation();
    });
  // attach drag to entering nodes
  if(typeof d3!=='undefined' && d3.drag){
    const drag = d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
    nEnter.call(drag);
    n.call(drag);
  }
  nEnter.merge(n)
    .attr("fill", d=>colorForNode(d))
    .attr("opacity", d=>{
      if(d.state==="queued") return 0.95;
      if(d.state==="processing") return 1;
      return 0.92;
    })
    .classed("flash", d=> d._flash)
    .style("filter", d=>{
      const base = colorForNode(d);
      if(d.state==="processing") return "drop-shadow(0 0 10px rgba(255,140,66,.95)) drop-shadow(0 0 18px rgba(255,140,66,.45))";
      if(d.state==="queued") return "drop-shadow(0 0 9px rgba(245,200,66,.9)) drop-shadow(0 0 16px rgba(245,200,66,.5))";
      if(d.state==="completed") return `drop-shadow(0 0 7px ${base}99) drop-shadow(0 0 14px ${base}55)`;
      if(d.state==="error") return "drop-shadow(0 0 8px rgba(255,77,77,.7))";
      return `drop-shadow(0 0 8px ${base}CC) drop-shadow(0 0 16px ${base}66)`;
    });
  n.exit().remove();

  const nl = gNodeLabels.selectAll("text").data(nodes, d=>d.id);
  nl.enter().append("text")
    .attr("text-anchor","middle").attr("font-size","11px").attr("font-weight","700").attr("fill","#ffffff").attr("stroke","#0b1220").attr("stroke-width",0.6).attr("paint-order","stroke").attr("pointer-events","none")
    .merge(nl).text(d=>d.label);
  nl.exit().remove();

  // chatnet titles (one per chatnet) — render as badges near center
  const titles = [...chatnets.entries()].map(([id,ch])=> ({id, ...ch}));
  const tSel = gWrap.selectAll("text.chatnetTitle").data(titles, d=>d.id);
  tSel.enter().append("text").attr("class","chatnetTitle")
    .attr("text-anchor","middle").attr("font-size","11px").attr("font-weight","700").attr("fill","#9ec5ff")
    .attr("paint-order","stroke").attr("stroke","#0f1419").attr("stroke-width",3)
    .merge(tSel).text(d=> d.title);
  tSel.exit().remove();

  document.getElementById("nodePill").textContent = `${nodes.length} nodes`;
  if(!simulation) makeSim();
  else if(isStructural){
    simulation.nodes(nodes);
    simulation.force("link").links(edges);
    // Small, controlled reheating — do not inject 0.9
    simulation.alpha(Math.max(simulation.alpha(), 0.12)).restart();
  } // else: state-only change — just update colors, no reheat (prevents twitch)
}

// side panel — FIFO 20, clickable popup for full prompt/system/context
const panel = document.getElementById("panel");
const panelCount = document.getElementById("panelCount");
let panelEntries = 0;
const popupStore = new Map(); // `${cid}::${nodeId}` -> {prompt, persona, response, cleaned, fullPrompt}
function ensurePanelNotEmpty(){
  const empt = panel.querySelector(".empty");
  if(empt) empt.remove();
}
function openPopupFor(cid, nodeId){
  const key = `${cid}::${nodeId}`;
  const d = popupStore.get(key);
  if(!d) return;
  document.getElementById('popupTitle').textContent = nodeId + ' — ' + (chatnets.get(String(cid))?.title || ('ChatNet '+cid));
  document.getElementById('popupMeta').textContent = 'ChatNet ' + cid + ' · fx: ' + (d.fx||'—') + ' · state: ' + (d.state||'');
  document.getElementById('popupPrompt').textContent = d.fullPrompt || d.prompt || '';
  document.getElementById('popupPersona').textContent = d.persona || '(no system prompt)';
  document.getElementById('popupContext').textContent = d.response || '';
  document.getElementById('popupOverlay').classList.add('open');
}
function closePopup(){ document.getElementById('popupOverlay').classList.remove('open'); }
document.addEventListener('keydown', e=>{ if(e.key==='Escape') closePopup(); });
function addPanelEntry(cid, nodeId, prompt, response, cleaned, extra={}){
  ensurePanelNotEmpty();
  const key = `${cid}::${nodeId}`;
  // store full data for popup (keep latest)
  popupStore.set(key, {prompt, response, cleaned, persona: extra.persona||'', fx: extra.fx||'', state: extra.state||'completed', fullPrompt: extra.fullPrompt||prompt});
  const div = document.createElement("div");
  div.className="entry";
  div.style.cursor='pointer';
  div.title='Click for full prompt / system prompt / context';
  div.onclick = ()=> openPopupFor(cid, nodeId);
  const title = chatnets.get(String(cid))?.title || `ChatNet ${cid}`;
  div.innerHTML = `<div class="entryHead"><span>${nodeId}</span><span class="cid">${title}</span></div>
    <div class="promptBox"><strong style="font-size:10px;letter-spacing:.04em;color:#6aa3d9">PROMPT</strong>\n${escapeHtml((prompt||"").slice(0,500))}</div>
    <div class="respBox"><strong style="font-size:10px;letter-spacing:.04em;color:#2ecc71">RESPONSE${cleaned? " → "+escapeHtml(cleaned.slice(0,120)):""}</strong>\n${escapeHtml((response||"").slice(0,500))}</div>`;
  panel.appendChild(div);
  panelEntries++;
  if(panelEntries > MAX_PANEL){
    panel.firstElementChild.remove();
    panelEntries--;
  }
  panel.scrollTop = panel.scrollHeight;
  panelCount.textContent = `${panelEntries} / ${MAX_PANEL}`;
}
function addStandaloneEntry(prompt, response, persona){
  ensurePanelNotEmpty();
  const div = document.createElement("div");
  div.className="entry";
  div.innerHTML = `<div class="entryHead"><span>askChatQuestion</span><span class="cid" style="color:#7aa0c6">${escapeHtml((persona||"").slice(0,40))}</span></div>
    <div class="promptBox"><strong style="font-size:10px;letter-spacing:.04em;color:#6aa3d9">PROMPT</strong>\n${escapeHtml((prompt||"").slice(0,500))}</div>
    <div class="respBox"><strong style="font-size:10px;letter-spacing:.04em;color:#2ecc71">RESPONSE</strong>\n${escapeHtml((response||"").slice(0,500))}</div>`;
  panel.appendChild(div);
  panelEntries++;
  if(panelEntries > MAX_PANEL){ panel.firstElementChild.remove(); panelEntries--; }
  panel.scrollTop = panel.scrollHeight;
  panelCount.textContent = `${panelEntries} / ${MAX_PANEL}`;
}
function escapeHtml(s){ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

// state helpers with anim-delay gate
function withGate(cid, fn){
  const ch = chatnets.get(String(cid));
  if(!ch) { fn(); return; }
  if(ch.animating) fn();
  else ch.pending.push(fn);
}
function applyInstantly(cid, fn){
  const ch = chatnets.get(String(cid));
  if(!ch){ fn(); return; }
  // if gate not yet fired and chatnet completed fast, cancel timer and apply instantly
  if(!ch.animating){
    clearTimeout(ch.timer);
    ch.animating = true;
    // apply this and all pending instantly (no flash animation)
    ch.pending.forEach(p=>p());
    ch.pending=[];
    fn();
    updateGraph();
  } else {
    fn();
  }
}

// SSE
let es=null, reconnectTimer=null;
function connectSSE(){
  if(es) try{es.close();}catch(e){}
  es = new EventSource("/events");
  const pill = document.getElementById("statusPill");
  es.onopen = ()=>{ pill.textContent="● live"; pill.style.background="#14301e"; pill.style.color="#2ecc71"; pill.style.borderColor="#1e4d2f"; };
  es.onerror = ()=>{
    pill.textContent="● reconnecting…"; pill.style.background="#33260f"; pill.style.color="#f5c842";
    try{es.close();}catch(e){}
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connectSSE, 1500);
  };
  es.onmessage = (ev)=>{
    try{
      const msg = JSON.parse(ev.data);
      handleEvent(msg);
    }catch(e){ console.error(e, ev.data); }
  };
}
function handleEvent(msg){
  const t = msg.type, p = msg.payload;
  if(t==="graph_load"){
    ensureChatnet(p.chatnet_id, p.title);
    // add nodes/edges (avoid duplicates if re-sent)
    for(const n of p.nodes){
      const key = `${p.chatnet_id}::${n.id}`;
      if(!nodeById.has(key)){
        const obj = {id: key, rawId:n.id, label:n.label, chatnet_id:String(p.chatnet_id), state:"idle", _flash:false, prompt:n.prompt||"", cdenFx:n.fx||"", x: (chatnets.get(String(p.chatnet_id)).center.x + (Math.random()-0.5)*160), y: (chatnets.get(String(p.chatnet_id)).center.y + (Math.random()-0.5)*160)};
        nodes.push(obj); nodeById.set(key,obj);
      }
    }
    for(const e of p.edges){
      if(!edges.find(x=>x.id===e.id)){
        // source/target are raw ids like "Start"; map to keyed ids
        const sKey = `${p.chatnet_id}::${e.source}`;
        const tKey = `${p.chatnet_id}::${e.target}`;
        const sNode = nodeById.get(sKey), tNode = nodeById.get(tKey);
        if(sNode && tNode){
          const obj = {id:e.id, canonical:e.canonical, source:sNode.id, target:tNode.id, _srcKey:sKey, _tgtKey:tKey, label:e.label, prompt:e.prompt||"", chatnet_id:String(p.chatnet_id), state:"idle"};
          // D3 link needs source/target as object or id; we use id strings and forceLink id accessor
          obj.source = sNode.id; obj.target = tNode.id;
          edges.push(obj);
          // keyed by canonical+fx to distinguish parallel edges like two End edges with isYes/isNo
          edgeByCanonical.set(`${p.chatnet_id}::${e.canonical}::${e.label}`, obj);
          // also keep a fallback entry for canonical alone (last one wins) for backward compat
          edgeByCanonical.set(`${p.chatnet_id}::${e.canonical}`, obj);
        }
      }
    }
    updateGraph(true);
  } else if(t==="graph_title"){
    const ch = chatnets.get(String(p.chatnet_id));
    if(ch){ ch.title = p.title; updateGraph(); }
  } else if(t==="node_queued"){
    withGate(p.chatnet_id, ()=>{
      const key = `${p.chatnet_id}::${p.node_id}`;
      const n = nodeById.get(key);
      if(n){ n.state="queued"; n._flash=true; setTimeout(()=>{n._flash=false; updateGraph();}, 900); updateGraph(); }
    });
  } else if(t==="node_start"){
    withGate(p.chatnet_id, ()=>{
      const key = `${p.chatnet_id}::${p.node_id}`;
      const n = nodeById.get(key);
      if(n){ n.state="processing"; updateGraph(); }
      // store persona/prompt for popup even before response
      const pk = `${p.chatnet_id}::${p.node_id}`;
      const prev = popupStore.get(pk) || {};
      popupStore.set(pk, {...prev, prompt: p.prompt||prev.prompt||'', persona: p.persona||prev.persona||'', fx: n? n.cdenFx : '', fullPrompt: p.fullPrompt||p.prompt||''});
    });
  } else if(t==="node_complete"){
    withGate(p.chatnet_id, ()=>{
      const key = `${p.chatnet_id}::${p.node_id}`;
      const n = nodeById.get(key);
      if(n){ n.state="completed"; updateGraph(); }
    });
  } else if(t==="node_error"){
    withGate(p.chatnet_id, ()=>{
      const key = `${p.chatnet_id}::${p.node_id}`;
      const n = nodeById.get(key);
      if(n){ n.state="error"; updateGraph(); }
    });
  } else if(t==="edge_evaluated"){
    withGate(p.chatnet_id, ()=>{
      // distinguish parallel edges by fx when provided
      let e = null;
      if(p.fx){
        e = edgeByCanonical.get(`${p.chatnet_id}::${p.canonical}::${p.fx}`);
      }
      if(!e){
        const key = `${p.chatnet_id}::${p.canonical}`;
        e = edgeByCanonical.get(key);
      }
      // fallback: first edge with that canonical
      if(!e) e = edges.find(x=> x.canonical===p.canonical && String(x.chatnet_id)===String(p.chatnet_id));
      // if fx provided and there are parallel edges, prefer the one with matching label
      if(!e && p.fx) e = edges.find(x=> x.canonical===p.canonical && x.label===p.fx && String(x.chatnet_id)===String(p.chatnet_id));
      if(e){ e.state = p.result ? "true":"false"; updateGraph(); }
      // if we emitted per-edge but there are parallel edges sharing canonical, ensure only the matching fx edge updated (above handles it)
    });
  } else if(t==="prompt_response"){
    // side panel always shows immediately (not gated); graph color already handled by node_complete
    addPanelEntry(p.chatnet_id, p.node_id, p.prompt, p.response, p.cleaned, {persona: p.persona||'', fx: p.fx||'', fullPrompt: p.fullPrompt||p.prompt, state: 'completed'});
  } else if(t==="standalone_prompt"){
    // store for pairing with response? just note
    window._lastStandalone = p;
  } else if(t==="standalone_response"){
    const lp = window._lastStandalone || {};
    addStandaloneEntry(lp.prompt||p.prompt, p.response, lp.persona);
    window._lastStandalone=null;
  } else if(t==="chatnet_complete"){
    const ch = chatnets.get(String(p.chatnet_id));
    if(ch){
      // if not yet animating, cancel timer and snap to completion instantly (no animation overhead)
      if(!ch.animating){
        clearTimeout(ch.timer);
        ch.animating = true;
        // flush all pending instantly to final states, then apply completion colors
        ch.pending.forEach(fn=>fn());
        ch.pending=[];
        // mark all nodes of this chatnet completed if not error
        for(const n of nodes){ if(String(n.chatnet_id)===String(p.chatnet_id) && n.state!=="error") n.state="completed"; }
        updateGraph();
      }
    }
    // schedule reset after 4s
    setTimeout(()=>{
      handleEvent({type:"chatnet_reset", payload:{chatnet_id:p.chatnet_id}});
    }, 4000);
  } else if(t==="chatnet_reset"){
    // return colors to idle
    for(const n of nodes){ if(String(n.chatnet_id)===String(p.chatnet_id)) n.state="idle"; }
    for(const e of edges){ if(String(e.chatnet_id)===String(p.chatnet_id)) e.state="idle"; }
    updateGraph();
  }
}

connectSSE();

document.getElementById("resetBtn").addEventListener("click", ()=>{
  if(simulation){ simulation.alpha(1).restart(); }
  if(svg && zoom) svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity);
});
document.getElementById("clearBtn").addEventListener("click", ()=>{
  panel.innerHTML='<div class="empty">Cleared.</div>'; panelEntries=0;
  panelCount.textContent="0 / 20";
});
function updateStats(){
  const s = document.getElementById("stats");
  if(!s) return;
  s.textContent = `${nodes.length} nodes · ${edges.length} edges · ${chatnets.size} nets`;
  requestAnimationFrame(updateStats);
}
updateStats();
</script>
</body>
</html>
"""

# map raw path -> content
_STATIC = {
    "/": (HTML_PAGE, "text/html; charset=utf-8"),
    "/index.html": (HTML_PAGE, "text/html; charset=utf-8"),
}


class _VizHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # strip query string
        path = self.path.split("?")[0]
        if path == "/events":
            self._handle_sse()
            return
        if path == "/health":
            body = json.dumps({"ok": True, "chatnets": len(_chatnet_titles)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/d3.v7.min.js", "/static/d3.v7.min.js"):
            # serve vendored D3 for offline use
            import pathlib
            cand = pathlib.Path(__file__).parent / "static" / "d3.v7.min.js"
            if cand.exists():
                try:
                    data = cand.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/javascript")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.end_headers()
                    self.wfile.write(data)
                    return
                except Exception:
                    pass
            # fallback: redirect to CDN (client will try next script tag)
            self.send_response(302)
            self.send_header("Location", "https://d3js.org/d3.v7.min.js")
            self.end_headers()
            return
        if path in _STATIC:
            body, ctype = _STATIC[path]
            body_b = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body_b)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body_b)
            return
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"not found")

    def _handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        # CORS for file:// or other origins
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        # flush headers
        try:
            self.wfile.flush()
        except Exception:
            pass

        q: queue.Queue = queue.Queue(maxsize=1000)
        with _sse_clients_lock:
            _sse_clients.append(q)
        # send initial retry + comment
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
        except Exception:
            pass
        # send hello
        try:
            hello = json.dumps({"type": "hello", "payload": {"msg": "connected"}})
            self.wfile.write(f"data: {hello}\n\n".encode())
            self.wfile.flush()
        except Exception:
            pass
        # replay buffered events for late joiners (so graphs built before connect still appear)
        try:
            with _event_log_lock:
                buffered = list(_event_log)
            for data in buffered:
                try:
                    self.wfile.write(f"data: {data}\n\n".encode())
                except Exception:
                    break
            self.wfile.flush()
        except Exception:
            pass

        # block and forward
        try:
            while True:
                try:
                    data = q.get(timeout=15)
                    # heartbeat comment every 15s if no data, to keep alive
                    payload = f"data: {data}\n\n".encode()
                    self.wfile.write(payload)
                    self.wfile.flush()
                except queue.Empty:
                    # heartbeat
                    try:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    except Exception:
                        break
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _sse_clients_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    def log_message(self, format, *args):
        # suppress noisy logging; only errors
        if "events" in format % args:
            return
        # comment out to silence entirely:
        # super().log_message(format, *args)
        pass


class VizServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self.httpd: ThreadingHTTPServer | None = None

    def start(self) -> str:
        # find free port if 0
        self.httpd = ThreadingHTTPServer((self.host, self.port), _VizHandler)
        # allow reuse
        self.httpd.daemon_threads = True
        # retrieve actual port
        actual_port = self.httpd.server_address[1]
        self.port = actual_port
        url = f"http://{self.host}:{actual_port}/"
        return url

    def serve_forever(self):
        if self.httpd:
            self.httpd.serve_forever()

    def shutdown(self):
        if self.httpd:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
            self.httpd = None


# ---------------------------------------------------------------------------
# Public API: vizOn / vizOff
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def vizOn(host: str = "127.0.0.1", port: int = 0, open_browser: bool = True, verbose: bool = True) -> str | None:
    """Start the live visualization server.

    After `import tabulairity`, call `tabulairity.vizOn()` to enable the viz.
    A browser tab will be opened automatically (disable with open_browser=False).
    Returns the URL, or None if already running.
    """
    global _viz_enabled, _viz_server, _viz_server_thread
    if _viz_enabled and _viz_server is not None:
        if verbose:
            print(f"[Viz] Already running at http://{_viz_server.host}:{_viz_server.port}/")
        return f"http://{_viz_server.host}:{_viz_server.port}/"

    # also flip the flag in core so walkChatNet can cheap-check without import
    try:
        from . import core as _core
        _core._viz_enabled = True  # type: ignore
    except Exception:
        pass

    _viz_enabled = True

    server = VizServer(host=host, port=port)
    url = server.start()
    _viz_server = server
    t = threading.Thread(target=server.serve_forever, daemon=True, name="TabulAIrity-Viz")
    t.start()
    _viz_server_thread = t

    if verbose:
        print(f"[Viz] Live visualization ON  →  {url}")
        print(f"[Viz] Waiting for chatNets… (call vizOff() to stop)")

    if open_browser:
        try:
            # delay slightly so server is ready
            def _open():
                time.sleep(0.35)
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
            threading.Thread(target=_open, daemon=True).start()
        except Exception:
            pass

    return url


def vizOff(verbose: bool = True) -> None:
    """Stop the live visualization server and disable event emission."""
    global _viz_enabled, _viz_server, _viz_server_thread
    _viz_enabled = False
    try:
        from . import core as _core
        _core._viz_enabled = False  # type: ignore
    except Exception:
        pass

    srv = _viz_server
    _viz_server = None
    _viz_server_thread = None
    if srv is not None:
        try:
            srv.shutdown()
        except Exception:
            pass
    # clear SSE clients
    with _sse_clients_lock:
        _sse_clients.clear()
    if verbose:
        print("[Viz] Live visualization OFF")