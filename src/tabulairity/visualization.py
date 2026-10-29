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
import hashlib
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
_topology_events: dict[tuple, str] = {}
_active_graph_ids: set[str] = set()
_node_activity: dict[tuple, dict] = {}
_event_sequence = 0

# Mapping from id(G) -> chatnet_id for graphs that haven't stored it in G.graph
_graph_id_map: dict[int, int] = {}

# Deduplication for cyclical reads — hash ChatNet definition to avoid double-render
_chatnet_hashes: dict[str, int] = {}
_chatnet_graphs: set[int] = set()
_chatnet_links: set[tuple] = set()
_chatnet_fingerprints: dict[str, int] = {}


def viz_graph_fingerprint(G) -> str:
    """Stable identity for a graph definition, independent of object/viz IDs."""
    nodes = []
    for node, data in G.nodes(data=True):
        fields = {key: data.get(key) for key in
                  ("prompt", "persona", "fx", "fx_params", "model", "tokens", "extra_params")}
        nodes.append((str(node), fields))
    edges = []
    for source, target, key, data in G.edges(keys=True, data=True):
        fields = {name: data.get(name) for name in ("prompt", "fx", "fx_params")}
        edges.append((str(source), str(target), str(key), fields))
    definition = {
        "nodes": sorted(nodes, key=lambda item: item[0]),
        "edges": sorted(edges, key=lambda item: (item[0], item[1], item[2])),
    }
    packed = json.dumps(definition, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def viz_graph_shape_fingerprint(G) -> str:
    """Identity of the visible workflow, ignoring prompt-level variations."""
    nodes = sorted((str(node), str(data.get("fx", ""))) for node, data in G.nodes(data=True))
    edges = sorted((str(source), str(target), str(data.get("fx", "")))
                   for source, target, data in G.edges(data=True))
    packed = json.dumps({"nodes": nodes, "edges": edges}, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _chatnet_hash(chatnet) -> str:
    """Hash ChatNet DataFrame/definition for viz deduplication."""
    try:
        import hashlib

        # Try pandas-aware hash for DataFrames (deterministic)
        try:
            import pandas as pd  # type: ignore

            if isinstance(chatnet, pd.DataFrame):
                # hash_pandas_object is stable across runs for same content
                base = pd.util.hash_pandas_object(chatnet, index=True).values.tobytes()
                # mix shape/columns to avoid collisions on empty frames
                extra = f"{chatnet.shape}{list(chatnet.columns)}".encode()
                return hashlib.md5(base + extra).hexdigest()
        except Exception:
            pass
        return hashlib.md5(str(chatnet).encode()).hexdigest()
    except Exception:
        return str(id(chatnet))

# ---------------------------------------------------------------------------
# Low-level emit / helpers
# ---------------------------------------------------------------------------

def viz_is_enabled() -> bool:
    return _viz_enabled


def viz_emit(event_type: str, payload: dict) -> None:
    """Thread-safe broadcast. No-op when viz is off. Never raises."""
    if not _viz_enabled:
        return
    global _event_sequence
    with _event_log_lock:
        _event_sequence += 1
        msg = {"type": event_type, "payload": payload, "ts": time.time(), "seq": _event_sequence}
        data = json.dumps(msg, default=str)
        # Topology must survive arbitrarily many loop events and reconnects.
        if event_type in {"graph_load", "graph_title"}:
            _topology_events[(event_type, str(payload["chatnet_id"]))] = data
        elif event_type == "graph_link":
            _topology_events[(event_type, str(payload["parent_chatnet_id"]),
                              str(payload["parent_node_id"]), str(payload["child_chatnet_id"]))] = data
        else:
            _event_log.append(data)
            if len(_event_log) > _EVENT_LOG_MAX:
                _event_log.pop(0)
        if event_type in {"node_queued", "node_start", "node_complete", "node_error"}:
            cid, nid = str(payload["chatnet_id"]), str(payload["node_id"])
            _active_graph_ids.add(cid)
            state = _node_activity.setdefault((cid, nid),
                {"active": set(), "queued": set(), "completed": 0, "failed": 0})
            run = str(payload.get("run_id") or json.dumps(payload.get("trace", [])))
            if event_type == "node_queued":
                state["queued"].add(run)
            elif event_type == "node_start":
                state["queued"].discard(run)
                state["active"].add(run)
            else:
                state["queued"].discard(run)
                state["active"].discard(run)
                state["completed" if event_type == "node_complete" else "failed"] += 1
    with _sse_clients_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        # Force a reconnect and state replay instead of silently abandoning a
        # full client's queue while its browser still displays a live status.
        for q in dead:
            _sse_clients.remove(q)
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(None)
            except queue.Full:
                pass


def _replay_events():
    with _event_log_lock:
        events = list(_topology_events.values())
        events += [json.dumps({"type": "graph_active", "payload": {"chatnet_id": cid}})
                   for cid in sorted(_active_graph_ids)]
        events += [json.dumps({**json.loads(data), "replay": True}) for data in _event_log]
        for (cid, nid), state in _node_activity.items():
            snapshot = {key: sorted(value) if isinstance(value, set) else value
                        for key, value in state.items()}
            events.append(json.dumps({"type": "node_snapshot", "payload": {
                "chatnet_id": cid, "node_id": nid, **snapshot}}))
        return events


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
    """Use authored or node names; visualization never adds model requests."""
    title = str(G.graph.get("name") or "").strip()
    if not title:
        candidates = [(str(n), data) for n, data in G.nodes(data=True) if str(n) != "Start"]
        meaningful = [name for name, data in candidates
                      if not str(data.get("prompt", "")).strip().startswith("recall:")]
        title = (meaningful or [name for name, _ in candidates] or ["CDEN"])[0]
        title = title.replace("_", " ").strip().capitalize()
    _chatnet_titles[chatnet_id] = title[:80]
    return _chatnet_titles[chatnet_id]


# ---------------------------------------------------------------------------
# Public hooks called from core.py
# ---------------------------------------------------------------------------

def viz_notify_graph_load(G, chatnet=None) -> int:
    """Called from core.buildChatNet after G is built. Returns chatnet_id."""
    if not _viz_enabled:
        return _get_or_create_chatnet_id(G)

    try:
        # Deduplicate visualization initialization by ChatNet definition.
        if chatnet is not None:
            chatnet_hash = _chatnet_hash(chatnet)

            with _viz_lock:
                cid = _chatnet_hashes.get(chatnet_hash)

                if cid is None:
                    cid = _next_chatnet_id
                    globals()["_next_chatnet_id"] += 1
                    _chatnet_hashes[chatnet_hash] = cid
                    is_new = True
                else:
                    is_new = False

            # Always attach the existing ID to this execution's graph.
            G.graph["viz_id"] = cid

            # Same ChatNet already exists in the visualization.
            # Do not initialize/render it again.
            if not is_new:
                return cid

        else:
            # Backwards-compatible path for callers that don't provide
            # the source DataFrame.
            cid = _get_or_create_chatnet_id(G)

        with _viz_lock:
            if cid in _chatnet_graphs:
                return cid
            _chatnet_graphs.add(cid)

        title = _generate_title_sync(G, cid)
        graph_fingerprint = viz_graph_fingerprint(G)
        shape_fingerprint = viz_graph_shape_fingerprint(G)
        _chatnet_fingerprints[graph_fingerprint] = cid

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
            "segments": G.graph.get("viz_segments", []),
            "merged_from": G.graph.get("viz_merged_from_ids", []),
            "graph_fingerprint": graph_fingerprint,
            "shape_fingerprint": shape_fingerprint,
            "merged_from_fingerprints": G.graph.get("viz_merged_from_fingerprints", []),
            "merged_from_shapes": G.graph.get("viz_merged_from_shapes", []),
        })
        return cid
    except Exception:
        return 0


def viz_notify_nested_chatnet(parent_chatnet_id: int, parent_node: str, child_graph) -> int | None:
    """Register a nested definition once and link its Start to the expand node."""
    if not _viz_enabled:
        return None
    child_chatnet_id = viz_notify_graph_load(child_graph)
    if child_chatnet_id is None or child_chatnet_id == 0:
        return None
    link = (str(parent_chatnet_id), str(parent_node), str(child_chatnet_id), "expandIterable")
    with _viz_lock:
        if link in _chatnet_links:
            return child_chatnet_id
        _chatnet_links.add(link)
    viz_emit("graph_link", {
        "parent_chatnet_id": parent_chatnet_id,
        "parent_node_id": str(parent_node),
        "child_chatnet_id": child_chatnet_id,
        "child_node_id": "Start",
        "relation": "expandIterable",
    })
    return child_chatnet_id


def viz_notify_node_queued(chatnet_id: int, node_id: str, trace=None, run_id=None):
    viz_emit("node_queued", {"chatnet_id": chatnet_id, "node_id": str(node_id), "trace": list(trace or []), "run_id": run_id})


def viz_notify_node_start(chatnet_id: int, node_id: str, prompt: str = "", persona: str = "", fullPrompt: str = None, trace=None, run_id=None):
    viz_emit("node_start", {
        "chatnet_id": chatnet_id,
        "node_id": str(node_id),
        "prompt": (prompt or "")[:500],
        "persona": (persona or "")[:2000],
        "fullPrompt": (fullPrompt if fullPrompt is not None else prompt or "")[:5000],
        "trace": list(trace or []), "run_id": run_id,
    })


def viz_notify_node_complete(chatnet_id: int, node_id: str, response: str = "", cleaned: str = "", trace=None, run_id=None):
    viz_emit("node_complete", {
        "chatnet_id": chatnet_id,
        "node_id": str(node_id),
        "response": (response or "")[:500],
        "cleaned": (cleaned or "")[:500],
        "trace": list(trace or []), "run_id": run_id,
    })


def viz_notify_node_error(chatnet_id: int, node_id: str, error: str = "", trace=None, run_id=None):
    viz_emit("node_error", {
        "chatnet_id": chatnet_id,
        "node_id": str(node_id),
        "error": (error or "")[:500],
        "trace": list(trace or []), "run_id": run_id,
    })


def viz_notify_edge_evaluated(chatnet_id: int, edge_id: str, result: bool, fx: str | None = None):
    viz_emit("edge_evaluated", {
        "chatnet_id": chatnet_id,
        "edge_id": str(edge_id),
        "canonical": str(edge_id),
        "result": bool(result),
        "fx": str(fx) if fx is not None else None,
    })


def viz_notify_prompt_response(chatnet_id: int, node_id: str, prompt: str, response: str, cleaned: str = "", persona: str = "", fx: str = "", fullPrompt: str = None, trace=None, run_id=None):
    viz_emit("prompt_response", {
        "chatnet_id": chatnet_id,
        "node_id": str(node_id),
        "prompt": (prompt or "")[:500],
        "response": (response or "")[:500],
        "cleaned": (cleaned or "")[:500],
        "persona": (persona or "")[:2000],
        "fx": str(fx or ""),
        "fullPrompt": (fullPrompt if fullPrompt is not None else prompt or "")[:5000],
        "trace": list(trace or []), "run_id": run_id,
    })


def viz_notify_chatnet_complete(chatnet_id: int, success: bool = True, trace=None, run_id=None):
    viz_emit("chatnet_complete", {"chatnet_id": chatnet_id, "success": bool(success), "trace": list(trace or []), "run_id": run_id})


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
  #graphWrap svg{width:100%;height:100%;min-height:420px;display:block;cursor:grab}
  #graphWrap svg:active{cursor:grabbing}
  #d3Error{position:absolute;top:12px;left:50%;transform:translateX(-50%);background:#3d1a1a;border:1px solid #7a2a2a;color:#ffb4b4;padding:8px 12px;border-radius:8px;font-size:12px;display:none;z-index:10;max-width:80%;text-align:center}
  #tooltip{position:absolute;display:none;pointer-events:none;z-index:20;max-width:380px;background:rgba(17,26,38,.98);border:1px solid #2a3a4d;border-radius:8px;padding:8px 10px;font-size:11px;line-height:1.45;color:#e6eef6;box-shadow:0 8px 24px rgba(0,0,0,.45)}
  #tooltip .tt-title{font-weight:700;color:#9ec5ff;margin-bottom:4px;font-size:12px}
  #tooltip .tt-meta{color:#7aa0c6;font-size:10px;margin-bottom:6px}
  #tooltip .tt-body{white-space:pre-wrap;word-break:break-word;max-height:180px;overflow:auto;color:#cfe0f5}
  #tooltip .tt-row{margin:2px 0}
  #popupOverlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:30;align-items:center;justify-content:center;opacity:0;transition:opacity .18s ease-out}
  #popupOverlay.open{display:flex;opacity:1}
  #popup{width:min(820px,92vw);max-height:88vh;overflow:auto;background:#0f1419;border:1px solid #2a3a4d;border-radius:12px;box-shadow:0 16px 48px rgba(0,0,0,.6);padding:16px;scroll-behavior:smooth;transform:translateY(8px) scale(.98);transition:transform .18s ease-out}
  #popupOverlay.open #popup{transform:translateY(0) scale(1)}
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
  #panel{flex:1 1 auto;overflow-y:auto;overflow-x:hidden;padding:8px 10px;display:flex;flex-direction:column;gap:8px;min-height:0;align-items:stretch;scroll-behavior:smooth}
  .entry{border:1px solid #1e2d44;border-radius:8px;overflow:hidden;background:#111a26;flex-shrink:0;padding:2px;transition:border-color .12s ease-out,background .12s ease-out,transform .08s ease-out}
  .entry[data-clickable="1"]{cursor:pointer}
  .entry[data-clickable="1"]:hover{border-color:#3a7bd5;background:#152238}
  .entry[data-clickable="1"]:active{transform:translateY(1px)}
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
  <button id="resetBtn" title="Fit all visible workflows">Fit graph</button>
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
      <div class="row" style="margin-top:6px;border-top:1px solid #1e2d44;padding-top:6px;color:#7aa0c6">drag node to move · scroll to zoom · drag background to pan</div>
    </div>
    <div id="stats"></div>
  </div>
  <div id="side">
    <div id="sideHead"><span>Prompts &amp; Responses <span style="color:#5a7896">(first 500 chars)</span></span><span class="count" id="panelCount">0 / 20</span></div>
    <div id="panel"><div class="empty">No prompts yet. Run a chatNet with <code>walkChatNet</code> or <code>askChatQuestion</code>.</div></div>
  </div>
</div>
<div id="popupOverlay" onclick="if(event.target===this) closePopup()"><div id="popup"><button class="closeBtn" onclick="closePopup()">✕ Close</button><h3 id="popupTitle"></h3><div id="popupMeta" class="popup-meta"></div><div class="popup-section"><div class="popup-label">SYSTEM PROMPT (persona)</div><pre id="popupPersona"></pre></div><div class="popup-section"><div class="popup-label">INPUT PROMPT</div><pre id="popupPrompt"></pre></div><div class="popup-section"><div class="popup-label">OUTPUT (response)</div><pre id="popupContext"></pre></div></div></div>
<script>
const MAX_PANEL = 50;
const ANIM_DELAY = 0;

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
  const seg=d.segment?` · merged CDEN: ${esc(d.segment)}`:'';
  return `<div class="tt-title">${esc(d.label)}</div><div class="tt-meta">ChatNet: ${esc(title)} (id ${esc(d.chatnet_id)}) · state: ${esc(d.state)} · ${d.activeCount||0} active · ${d.completedCount||0} completed · fx: ${esc(d.cdenFx||'')}${seg}</div><div class="tt-body">${esc((d.prompt||'').slice(0,500))||'<i>no prompt</i>'}</div>`;
}
function edgeTip(d){
  const ch=chatnets.get(String(d.chatnet_id));
  const title=ch?ch.title:'';
  if(d.relation) return `<div class="tt-title">Nested CDEN: ${esc(d.label)}</div><div class="tt-meta">${esc(title)} → ${esc(chatnets.get(String(d.child_chatnet_id))?.title||'child CDEN')}</div><div class="tt-body">Dashed link connects the parent expandIterable node to the nested CDEN Start node.</div>`;
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
let gWrap = null, gEdges=null, gEdgeLabels=null, gNodes=null, gNodeLabels=null, gPulses=null;
if(svg){
  // diffuse glow via CSS drop-shadow (no SVG filter needed — kept minimal defs to avoid parse errors)
  svg.append("defs");
  gWrap = svg.append("g");
  gEdges = gWrap.append("g").attr("id","gEdges");
  gEdgeLabels = gWrap.append("g").attr("id","gEdgeLabels");
  gNodes = gWrap.append("g").attr("id","gNodes");
  gPulses = gWrap.append("g").attr("id","gActivityPulses").attr("pointer-events","none");
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
const chatnetAliases = new Map();
const chatnetByShape = new Map();
const graphDefinitions = new Map();
const activeGraphIds = new Set();
const nestedRelations = new Map();
const seenEvents = new Set();
let resolvingRelations = false;
// Source definitions remain available to the runtime, but welded composites
// replace them in the visible graph to avoid duplicate Habsburg-like clusters.
const hiddenChatnets = new Set();
const hiddenGraphFingerprints = new Set();
const hiddenGraphShapes = new Set();
// Base workflow identity stays separate from yellow/orange/green/red activity.
const palette = ["#a78bfa","#d8b4fe","#f0abdf","#c084fc","#e9c6ff","#d98acb","#b6a0e8","#e7acd0"];
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
function constrainViewport(transform, viewport, bounds){
  // D3's default constraint centers a graph smaller than the viewport, making
  // background dragging feel disabled. Only stop at the point where the graph
  // would disappear; otherwise preserve the user's translation at every scale.
  if(!bounds.flat().every(Number.isFinite)) return transform;
  const k=transform.k;
  const visibleX=Math.min(64,(bounds[1][0]-bounds[0][0])*k,
                          (viewport[1][0]-viewport[0][0])/2);
  const visibleY=Math.min(64,(bounds[1][1]-bounds[0][1])*k,
                          (viewport[1][1]-viewport[0][1])/2);
  const x=Math.max(viewport[0][0]+visibleX-bounds[1][0]*k,
                  Math.min(transform.x,viewport[1][0]-visibleX-bounds[0][0]*k));
  const y=Math.max(viewport[0][1]+visibleY-bounds[1][1]*k,
                  Math.min(transform.y,viewport[1][1]-visibleY-bounds[0][1]*k));
  return transform.translate((x-transform.x)/k,(y-transform.y)/k);
}
let zoom=null;
if(svg && typeof d3!=='undefined'){
  zoom = d3.zoom().scaleExtent([0.15,4])
    .extent(()=>[[0,0],[width,height]])
    .constrain(constrainViewport)
    .on("zoom", e=> gWrap.attr("transform", e.transform));
  svg.call(zoom);
}

// Bounds include node radii, without locking a small graph to the center.
function updateZoomBounds(){
  if(!zoom || !nodes.length) return;
  let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
  for(const node of nodes){
    if(Number.isFinite(node.x)){ minX=Math.min(minX,node.x); maxX=Math.max(maxX,node.x); }
    if(Number.isFinite(node.y)){ minY=Math.min(minY,node.y); maxY=Math.max(maxY,node.y); }
  }
  if(!Number.isFinite(minX) || !Number.isFinite(minY)) return;
  zoom.translateExtent([[minX-32,minY-32],[maxX+32,maxY+32]]);
}

function ensureChatnet(cid, title){
  cid = String(cid);
  if(hiddenChatnets.has(cid)) return null;
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
    chatnets.set(cid, {title: title||`ChatNet ${cid}`, color:c, center:{x:cx+Math.cos(ang)*R, y:cy+Math.sin(ang)*R}, animating:false, timer:null, pending:[], activeWalks:new Set()});
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
        .radius(d => nodeRadius(d) + 10)
        .strength(0.8)
    )
    // Per-chatnet clustering: pull every node toward its own chatnet's
    // assigned ring center. Without this, disconnected chatnets have
    // nothing tying them to the layout — link+charge alone let clusters
    // drift wherever the initial jitter throws them, and re-adding a
    // chatnet later doesn't pull the existing ones back into place.
    // Function accessors so a chatnet whose center gets recomputed by
    // ensureChatnet (e.g. after a new chatnet arrives and the ring is
    // re-spaced) is followed live, without needing to rebuild forces.
    .force("clusterX",
      d3.forceX(d => {
        const ch = chatnets.get(String(d.chatnet_id));
        return ch ? ch.center.x : width/2;
      }).strength(0.08)
    )
    .force("clusterY",
      d3.forceY(d => {
        const ch = chatnets.get(String(d.chatnet_id));
        return ch ? ch.center.y : height/2;
      }).strength(0.08)
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
  gPulses.selectAll("circle")
    .attr("cx",d=>d.x).attr("cy",d=>d.y);
  gNodeLabels.selectAll("text")
    .attr("x",d=>d.x).attr("y",d=>d.y + 4);

  // chatnet title badges — position from the LIVE cluster centroid, not
  // the assigned ring slot, so dragging nodes carries the label with them.
  // One pass over nodes tallies each chatnet's centroid and topmost y;
  // the label sits horizontally at the centroid, vertically just above
  // the highest node so it always clears the cluster.
  const centroids = new Map();
  for (const n of nodes){
    const cid = String(n.chatnet_id);
    let c = centroids.get(cid);
    if (!c){ c = {sx:0, sy:0, top:Infinity, count:0}; centroids.set(cid, c); }
    c.sx += n.x; c.sy += n.y; c.count++;
    if (n.y < c.top) c.top = n.y;
  }
  gWrap.selectAll("g.chatnetTitle")
    .attr("transform", d=>{
      const c = centroids.get(String(d.id));
      if (c && c.count > 0) return `translate(${c.sx/c.count},${c.top - 28})`;
      // no nodes yet for this chatnet — fall back to its ring slot
      return `translate(${d.center.x},${d.center.y - 90})`;
    });
}

// Refreshed once per updateGraph() call: the set of chatnet_ids that
// currently have at least one queued or processing node. colorForNode uses
// it to decide whether "completed" should still glow green (chatnet is
// mid-run and this is genuine progress) or fade back to the chatnet's
// base color (chatnet has settled; green would just be stale).
let activeChatnets = new Set();

function colorForNode(d){
  const ch = chatnets.get(String(d.chatnet_id));
  const base = ch? ch.color : palette[0];
  switch(d.state){
    case "queued": return "#f5c842";
    case "processing": return "#ff8c42";
    case "error": return "#ff4d4d";
    case "completed":
      // Only stays green while the chatnet is actively running. Once
      // everything settles, revert to base so an idle chatnet doesn't
      // look identical to one still finishing up.
      return activeChatnets.has(String(d.chatnet_id)) ? "#2ecc71" : base;
    default: return base;
  }
}
function colorForEdge(d){
  if(d.relation) return "#c084fc";
  if(d.state==="true") return "#3a7bd5";
  if(d.state==="false") return "#ff4d4d";
  return "#8a8f98";
}

// Start nodes are the entry points of each chatnet. They get a larger
// radius and a brightened fill so the eye finds them immediately when
// scanning any subgraph. Everything downstream (collision, drawing, and
// anything future that cares about node geometry) reads from nodeRadius()
// so a change here propagates in one place.
function isStart(d){ return String(d.label||"").toLowerCase() === "start"; }
function nodeRadius(d){ return isStart(d) ? 30 : 22; }
function segmentColor(name){ let h=0; for(const c of String(name)) h=(h*31+c.charCodeAt(0))|0; return palette[Math.abs(h)%palette.length]; }

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
    html += `<div style="margin:8px 0 4px;font-weight:600;color:${ch.color}">${ch.merged?'Merged CDEN · ':''}${escapeHtml(ch.title)}</div>`;
    const cn = nodes.filter(n=> String(n.chatnet_id)===String(cid));
    const ce = edges.filter(e=> String(e.chatnet_id)===String(cid));
    html += `<div>Nodes: ${cn.map(n=>`<span title="${n.segment?'Source CDEN: '+escapeHtml(n.segment):''}" style="display:inline-block;margin:2px 4px;padding:2px 6px;border-radius:999px;border:2px solid ${n.segment?segmentColor(n.segment):'transparent'};background:${colorForNode(n)};color:#0b1220;font-weight:600">${escapeHtml(n.label)}${n.segment?' · '+escapeHtml(n.segment):''}:${n.state}</span>`).join("")}</div>`;
    html += `<div style="margin-top:4px">Edges: ${ce.map(e=>`<span style="display:inline-block;margin:2px 4px;padding:2px 6px;border-radius:4px;border:1px solid ${colorForEdge(e)};color:${colorForEdge(e)}">${e.canonical}[${e.label}]:${e.state}</span>`).join("")}</div>`;
  }
  fb.innerHTML = html;
}
let renderPending = false, structuralPending = false;
function updateGraph(isStructural=false){
  structuralPending = structuralPending || isStructural;
  if(renderPending) return;
  renderPending = true;
  requestAnimationFrame(()=>{
    const structural=structuralPending;
    renderPending=false; structuralPending=false;
    renderGraph(structural);
  });
}
function renderGraph(isStructural=false){
  updateZoomBounds();
  if(!svg || typeof d3==='undefined'){
    renderFallbackList();
    document.getElementById("nodePill").textContent = `${nodes.length} nodes`;
    document.getElementById("graphPill").textContent = `${chatnets.size} chatNets`;
    return;
  }
  // Refresh the active-chatnets set that colorForNode reads for its
  // completed-node fallback. Recomputed here (not lazily inside
  // colorForNode) so it's an O(nodes) pass once per render, not O(nodes²)
  // across every per-node color lookup.
  activeChatnets = new Set();
  for(const nd of nodes){
    if(nd.state === "queued" || nd.state === "processing"){
      activeChatnets.add(String(nd.chatnet_id));
    }
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
    .attr("stroke-dasharray", d=> d.relation ? "4 5" : (d.state==="false" ? "6 4" : null))
    .on("mousemove", (evt,d)=>{ showTip(edgeTip(d), evt); })
    .on("mouseenter", (evt,d)=>{ showTip(edgeTip(d), evt); })
    .on("mouseleave", hideTip);
  link.exit().remove();

  const eLabel = gEdgeLabels.selectAll("text").data(edges, d=>d.id);
  eLabel.enter().append("text")
    .attr("text-anchor","middle").attr("font-size","10px").attr("fill","#e6eef6").attr("pointer-events","none")
    .attr("paint-order","stroke").attr("stroke","#0f1419").attr("stroke-width",2)
    .merge(eLabel).text(d=>d.relation ? "each item" : (d.label==="null" ? "" : d.label||""));
  eLabel.exit().remove();

  const n = gNodes.selectAll("circle").data(nodes, d=>d.id);
  const nEnter = n.enter().append("circle")
    .attr("r", d => nodeRadius(d)).attr("stroke","#0f1419").attr("stroke-width",1.5)
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
    .attr("r", d => nodeRadius(d))
    .attr("stroke", d=>d.segment?segmentColor(d.segment):"#0f1419")
    .attr("stroke-width", d=>d.segment?3:1.5)
    .classed("flash", d=> d._flash)
    .style("filter", d=>{
      const base = colorForNode(d);
      // Start nodes get a brightness lift on top of whatever state glow
      // the node currently carries, so they read as "entry point" at any
      // state without having to invent a Start-specific palette.
      const startBoost = isStart(d) ? "brightness(1.2) saturate(1.1) " : "";
      if(d.state==="processing") return startBoost + "drop-shadow(0 0 10px rgba(255,140,66,.95)) drop-shadow(0 0 18px rgba(255,140,66,.45))";
      if(d.state==="queued") return startBoost + "drop-shadow(0 0 9px rgba(245,200,66,.9)) drop-shadow(0 0 16px rgba(245,200,66,.5))";
      if(d.state==="completed") return startBoost + `drop-shadow(0 0 3px ${base}66)`;
      if(d.state==="error") return startBoost + "drop-shadow(0 0 8px rgba(255,77,77,.7))";
      return startBoost.trim() || null;
    });
  n.exit().remove();

  const pulses=gPulses.selectAll("circle").data(nodes,d=>d.id);
  pulses.enter().append("circle").attr("opacity",0).attr("stroke-width",3)
    .merge(pulses).attr("r",d=>nodeRadius(d)+3).attr("cx",d=>d.x).attr("cy",d=>d.y);
  pulses.exit().remove();

  const nl = gNodeLabels.selectAll("text").data(nodes, d=>d.id);
  nl.enter().append("text")
    .attr("text-anchor","middle").attr("font-size","11px").attr("font-weight","700").attr("fill","#ffffff").attr("stroke","#0b1220").attr("stroke-width",0.6).attr("paint-order","stroke").attr("pointer-events","none")
    .merge(nl).text(d=>d.label);
  nl.exit().remove();

  // chatnet titles — a colored pill badge per chatnet, positioned each
  // tick from the LIVE centroid of its nodes (see ticked()) so dragging
  // a cluster carries its label with it. The rect auto-sizes to the
  // text bbox on every render so async LLM-generated titles fit cleanly
  // when they arrive later.
  const titles = [...chatnets.entries()].map(([id,ch])=> ({id, ...ch}));
  const tSel = gWrap.selectAll("g.chatnetTitle").data(titles, d=>d.id);
  const tEnter = tSel.enter().append("g").attr("class","chatnetTitle")
    .attr("pointer-events","none");
  tEnter.append("rect").attr("class","titleBg").attr("rx",6).attr("ry",6);
  tEnter.append("text")
    .attr("text-anchor","middle").attr("dy","0.35em")
    .attr("font-size","11px").attr("font-weight","700")
    .attr("paint-order","stroke").attr("stroke","#0b1220").attr("stroke-width",2);
  tEnter.merge(tSel).each(function(d){
    const g = d3.select(this);
    g.select("text").attr("fill", d.color).text(d=>d.merged ? `Merged CDEN · ${d.title}` : d.title);
    const bbox = g.select("text").node().getBBox();
    const padX = 10, padY = 4;
    g.select("rect")
      .attr("fill", d.color).attr("fill-opacity", 0.14)
      .attr("stroke", d.color).attr("stroke-opacity", 0.55)
      .attr("stroke-width", 1)
      .attr("x", bbox.x - padX).attr("y", bbox.y - padY)
      .attr("width", bbox.width + padX*2).attr("height", bbox.height + padY*2);
  });
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
const panelEntryByNode = new Map();
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
  document.getElementById('popupMeta').textContent = 'ChatNet ' + cid + ' · fx: ' + (d.fx||'—') + ' · state: ' + (d.state||'') + (d.runCount>1 ? ' · latest of ' + d.runCount + ' runs' : '');
  document.getElementById('popupPrompt').textContent = d.fullPrompt || d.prompt || '';
  document.getElementById('popupPersona').textContent = d.persona || '(no system prompt)';
  document.getElementById('popupContext').textContent = d.response || '';
  const popup = document.getElementById('popup');
  const overlay = document.getElementById('popupOverlay');
  // Jump to top instantly BEFORE the fade-in so reopening on a different
  // node doesn't briefly flash the previous node's scroll position, then
  // rely on the CSS `scroll-behavior: smooth` for any subsequent scroll.
  popup.scrollTo({top: 0, behavior: 'instant'});
  overlay.classList.add('open');
  // Reset each pre section's own scroll too — same reason.
  ['popupPersona','popupPrompt','popupContext'].forEach(id=>{
    document.getElementById(id).scrollTo({top: 0, behavior: 'instant'});
  });
}
function closePopup(){ document.getElementById('popupOverlay').classList.remove('open'); }
document.addEventListener('keydown', e=>{ if(e.key==='Escape') closePopup(); });
function addPanelEntry(cid, nodeId, prompt, response, cleaned, extra={}){
  ensurePanelNotEmpty();
  const key = `${cid}::${nodeId}`;
  const prior=popupStore.get(key)||{};
  const runCount=(prior.runCount||0)+1;
  // store full data for popup (keep latest)
  popupStore.set(key, {prompt, response, cleaned, persona: extra.persona||'', fx: extra.fx||'', state: extra.state||'completed', fullPrompt: extra.fullPrompt||prompt, runCount});
  let div=panelEntryByNode.get(key);
  const isNew=!div;
  if(isNew){
    div = document.createElement("div");
    div.className="entry";
    div.dataset.clickable="1";
    div.style.cursor='pointer';
    div.title='Click for full prompt / system prompt / context';
    div.onclick = ()=> openPopupFor(cid, nodeId);
    panelEntryByNode.set(key,div);
  }
  const title = chatnets.get(String(cid))?.title || `ChatNet ${cid}`;
  div.innerHTML = `<div class="entryHead"><span>${escapeHtml(nodeId)}</span><span class="cid">${escapeHtml(title)}${runCount>1?' · ×'+runCount+' runs':''}</span></div>
    <div class="promptBox"><strong style="font-size:10px;letter-spacing:.04em;color:#6aa3d9">PROMPT</strong>\n${escapeHtml((prompt||"").slice(0,500))}</div>
    <div class="respBox"><strong style="font-size:10px;letter-spacing:.04em;color:#2ecc71">RESPONSE${cleaned? " → "+escapeHtml(cleaned.slice(0,120)):""}</strong>\n${escapeHtml((response||"").slice(0,500))}</div>`;
  if(isNew){
    panel.appendChild(div);
    panelEntries++;
  }
  if(isNew && panelEntries > MAX_PANEL){
    const removed=panel.firstElementChild;
    if(removed){
      for(const [entryKey,entry] of panelEntryByNode) if(entry===removed) panelEntryByNode.delete(entryKey);
      removed.remove(); panelEntries--;
    }
  }
  if(isNew) panel.scrollTop = panel.scrollHeight;
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

function hideMergedSource(cid){
  const original=String(cid), visible=canonicalChatnetId(original);
  hiddenChatnets.add(original);
  hiddenChatnets.add(visible);
  const old=chatnets.get(visible);
  if(old?.shapeFingerprint && chatnetByShape.get(old.shapeFingerprint)===visible) chatnetByShape.delete(old.shapeFingerprint);
  chatnets.delete(visible);
  for(const [key,node] of nodeById){
    if(String(node.chatnet_id)===visible) nodeById.delete(key);
  }
  nodes=nodes.filter(n=>String(n.chatnet_id)!==visible);
  edges=edges.filter(e=>String(e.chatnet_id)!==visible &&
    !hiddenChatnets.has(String(e.child_chatnet_id)));
  for(const [key,edge] of edgeByCanonical){
    if(String(edge.chatnet_id)===visible) edgeByCanonical.delete(key);
  }
}
function canonicalChatnetId(value){
  let id=String(value), next=id, seen=new Set();
  while(chatnetAliases.has(next) && !seen.has(next)){
    seen.add(next); const mapped=String(chatnetAliases.get(next));
    if(mapped===next) break;
    next=mapped;
  }
  return next;
}
function hideMergedFingerprint(fingerprint){
  if(!fingerprint) return;
  hiddenGraphFingerprints.add(String(fingerprint));
  const existing=[...chatnets.keys()].find(cid=>
    chatnets.get(cid)?.graphFingerprint===String(fingerprint));
  if(existing) hideMergedSource(existing);
}
function hideMergedShape(fingerprint){
  if(!fingerprint) return;
  fingerprint=String(fingerprint);
  hiddenGraphShapes.add(fingerprint);
  const existing=chatnetByShape.get(fingerprint);
  if(existing) hideMergedSource(existing);
}

function activateGraph(value){
  const cid=String(value);
  activeGraphIds.add(cid);
  const definition=graphDefinitions.get(cid);
  if(definition && !chatnets.has(canonicalChatnetId(cid)) && !hiddenChatnets.has(cid)){
    handleEvent({type:"graph_load",payload:{...definition,_activate:true}});
  }
  resolveNestedRelations();
}
function resolveNestedRelations(){
  if(resolvingRelations) return;
  resolvingRelations=true;
  let added=false;
  try{
    // Links can precede definitions or arrive in reverse nesting order.
    for(let pass=0;pass<=nestedRelations.size;pass++){
      let changed=false;
      for(const p of nestedRelations.values()){
        const parentCid=canonicalChatnetId(p.parent_chatnet_id);
        if(!chatnets.has(parentCid)) continue;
        activateGraph(p.child_chatnet_id);
        const childCid=canonicalChatnetId(p.child_chatnet_id);
        const from=nodeById.get(`${parentCid}::${p.parent_node_id}`);
        const to=nodeById.get(`${childCid}::${p.child_node_id||'Start'}`);
        const id=`nested::${parentCid}::${p.parent_node_id}::${childCid}`;
        if(from&&to&&!edges.some(e=>e.id===id)){
          edges.push({id,canonical:`${from.label} → ${to.label}`,source:from.id,target:to.id,
            _srcKey:from.id,_tgtKey:to.id,label:p.relation||'expandIterable',prompt:'',
            chatnet_id:parentCid,child_chatnet_id:childCid,state:'idle',relation:true});
          changed=true; added=true;
        }
      }
      if(!changed) break;
    }
  } finally { resolvingRelations=false; }
  if(added) updateGraph(true);
}
function eventRunKey(p){
  return String(p.run_id || JSON.stringify(p.trace||[]));
}
let pulseFramePending=false;
function signalActivity(node,kind){
  // Keep distinct pulses through short bursts without replaying a long visual
  // backlog. At high rates the brightest nodes flicker continuously.
  node.pulsePending=Math.min(12,(node.pulsePending||0)+1);
  node.pulseColor=kind==='node_error'?'#ff4d4d':'#2ecc71';
  if(gPulses && !pulseFramePending){
    pulseFramePending=true;
    requestAnimationFrame(activityTick);
  }
}
function advanceActivityPulse(node,now){
  if(node.pulsePending && now>=(node.pulseNextAt||0)){
    node.pulsePending--;
    node.pulseUntil=now+60;
    node.pulseNextAt=now+100;
    node.pulsesShown=(node.pulsesShown||0)+1;
  }
  return now<(node.pulseUntil||0)?1:0;
}
function activityTick(now){
  let more=false;
  for(const node of nodes){
    node.pulseOpacity=advanceActivityPulse(node,now);
    if(node.pulsePending || node.pulseOpacity) more=true;
  }
  if(gPulses) gPulses.selectAll('circle')
    .attr('fill',d=>d.pulseColor||'#2ecc71').attr('fill-opacity',0.85)
    .attr('stroke',d=>d.pulseColor||'#2ecc71')
    .attr('opacity',d=>d.pulseOpacity||0);
  pulseFramePending=more;
  if(more) requestAnimationFrame(activityTick);
}
function updateNodeActivity(node,p,type){
  const origin=String(p.chatnet_id);
  let activity=node.activity.get(origin);
  if(!activity){
    activity={active:new Set(),queued:new Set(),completed:0,failed:0};
    node.activity.set(origin,activity);
  }
  const run=eventRunKey(p);
  if(type==='node_snapshot'){
    activity={active:new Set(p.active||[]),queued:new Set(p.queued||[]),
              completed:p.completed||0,failed:p.failed||0};
    node.activity.set(origin,activity);
  } else if(type==='node_queued') activity.queued.add(run);
  else if(type==='node_start'){
    activity.queued.delete(run); activity.active.add(run);
  } else {
    activity.active.delete(run); activity.queued.delete(run);
    if(type==='node_complete') activity.completed++;
    else activity.failed++;
  }
  node.activeCount=0; node.queuedCount=0; node.completedCount=0; node.failedCount=0;
  for(const state of node.activity.values()){
    node.activeCount+=state.active.size; node.queuedCount+=state.queued.size;
    node.completedCount+=state.completed; node.failedCount+=state.failed;
  }
  node.state=node.activeCount?'processing':node.queuedCount?'queued':
             node.failedCount?'error':node.completedCount?'completed':'idle';
  if(!p._replay && (type==='node_complete' || type==='node_error')) signalActivity(node,type);
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
  if(msg.seq){
    if(seenEvents.has(msg.seq)) return;
    seenEvents.add(msg.seq);
    if(seenEvents.size>4096) seenEvents.delete(seenEvents.values().next().value);
  }
  const t = msg.type, p = msg.replay ? {...msg.payload,_replay:true} : msg.payload;
  if(t==="graph_load"){
    const definitionId=String(p.chatnet_id);
    graphDefinitions.set(definitionId,p);
    if(!p._activate && !activeGraphIds.has(definitionId)) return;
    if(hiddenChatnets.has(String(p.chatnet_id))) return;
    if(p.graph_fingerprint && hiddenGraphFingerprints.has(String(p.graph_fingerprint))) return;
    if(p.shape_fingerprint && hiddenGraphShapes.has(String(p.shape_fingerprint))) return;
    for(const sourceCid of (p.merged_from||[])) hideMergedSource(sourceCid);
    for(const fingerprint of (p.merged_from_fingerprints||[])) hideMergedFingerprint(fingerprint);
    for(const fingerprint of (p.merged_from_shapes||[])) hideMergedShape(fingerprint);
    const originalCid=String(p.chatnet_id);
    const existingShape=p.shape_fingerprint && chatnetByShape.get(String(p.shape_fingerprint));
    if(existingShape && existingShape!==originalCid && chatnets.has(existingShape)){
      chatnetAliases.set(originalCid, existingShape);
      return;
    }
    const chatnet=ensureChatnet(p.chatnet_id, p.title);
    if(!chatnet) return;
    chatnetAliases.set(originalCid, originalCid);
    chatnet.graphFingerprint=p.graph_fingerprint||'';
    chatnet.shapeFingerprint=p.shape_fingerprint||'';
    if(chatnet.shapeFingerprint) chatnetByShape.set(chatnet.shapeFingerprint, originalCid);
    chatnet.merged=(p.merged_from||[]).length>0 || (p.segments||[]).length>1;
    chatnet.segmentNames=(p.segments||[]).map(s=>s.name).filter(Boolean);
    // add nodes/edges (avoid duplicates if re-sent)
    for(const n of p.nodes){
      const key = `${p.chatnet_id}::${n.id}`;
      if(!nodeById.has(key)){
        const segment=(p.segments||[]).find(s=>(s.nodes||[]).map(String).includes(String(n.id)));
        const obj = {id: key, rawId:n.id, label:n.label, chatnet_id:String(p.chatnet_id), state:"idle", activity:new Map(), activeCount:0, queuedCount:0, completedCount:0, failedCount:0, _flash:false, prompt:n.prompt||"", cdenFx:n.fx||"", segment:segment?segment.name:'', trace:[], x: (chatnets.get(String(p.chatnet_id)).center.x + (Math.random()-0.5)*160), y: (chatnets.get(String(p.chatnet_id)).center.y + (Math.random()-0.5)*160)};
        nodes.push(obj); nodeById.set(key,obj);
      }
    }
    for(const e of p.edges){
      const scopedEdgeId=`${p.chatnet_id}::${e.id}`;
      if(!edges.find(x=>x.id===scopedEdgeId)){
        // source/target are raw ids like "Start"; map to keyed ids
        const sKey = `${p.chatnet_id}::${e.source}`;
        const tKey = `${p.chatnet_id}::${e.target}`;
        const sNode = nodeById.get(sKey), tNode = nodeById.get(tKey);
        if(sNode && tNode){
          const obj = {id:scopedEdgeId, canonical:e.canonical, source:sNode.id, target:tNode.id, _srcKey:sKey, _tgtKey:tKey, label:e.label, prompt:e.prompt||"", chatnet_id:String(p.chatnet_id), state:"idle"};
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
    for(const seg of (p.segments||[])) for(const id of (seg.nodes||[])) { const n=nodeById.get(`${p.chatnet_id}::${id}`); if(n && !n.segment)n.segment=seg.name||''; }
    updateGraph(true);
    resolveNestedRelations();
  } else if(t==="graph_active"){
    activateGraph(p.chatnet_id);
  } else if(t==="graph_link"){
    nestedRelations.set(`${p.parent_chatnet_id}::${p.parent_node_id}::${p.child_chatnet_id}`,p);
    resolveNestedRelations();
  } else if(t==="graph_title"){
    const ch = chatnets.get(canonicalChatnetId(p.chatnet_id));
    if(ch){ ch.title = p.title; updateGraph(); }
  } else if(t==="node_queued"){
    activateGraph(p.chatnet_id);
    const cid=canonicalChatnetId(p.chatnet_id);
    const ch=chatnets.get(cid); if(ch) ch.activeWalks.add(eventRunKey(p));
    withGate(cid, ()=>{
      const key = `${cid}::${p.node_id}`;
      const n = nodeById.get(key);
      if(n){ updateNodeActivity(n,p,t); n.trace=p.trace||[]; updateGraph(); }
    });
  } else if(t==="node_start"){
    activateGraph(p.chatnet_id);
    const cid=canonicalChatnetId(p.chatnet_id);
    withGate(cid, ()=>{
      const key = `${cid}::${p.node_id}`;
      const n = nodeById.get(key);
      if(n){ updateNodeActivity(n,p,t); n.trace=p.trace||[]; updateGraph(); }
      // store persona/prompt for popup even before response
      const pk = `${cid}::${p.node_id}`;
      const prev = popupStore.get(pk) || {};
      popupStore.set(pk, {...prev, prompt: p.prompt||prev.prompt||'', persona: p.persona||prev.persona||'', fx: n? n.cdenFx : '', fullPrompt: p.fullPrompt||p.prompt||''});
    });
  } else if(t==="node_complete"){
    const cid=canonicalChatnetId(p.chatnet_id);
    withGate(cid, ()=>{
      const key = `${cid}::${p.node_id}`;
      const n = nodeById.get(key);
      if(n){ updateNodeActivity(n,p,t); n.trace=p.trace||[]; updateGraph(); }
    });
  } else if(t==="node_error"){
    const cid=canonicalChatnetId(p.chatnet_id);
    withGate(cid, ()=>{
      const key = `${cid}::${p.node_id}`;
      const n = nodeById.get(key);
      if(n){ updateNodeActivity(n,p,t); n.trace=p.trace||[]; updateGraph(); }
    });
  } else if(t==="node_snapshot"){
    activateGraph(p.chatnet_id);
    const cid=canonicalChatnetId(p.chatnet_id), n=nodeById.get(`${cid}::${p.node_id}`);
    if(n){ updateNodeActivity(n,p,t); updateGraph(); }
  } else if(t==="edge_evaluated"){
    const cid=canonicalChatnetId(p.chatnet_id);
    withGate(cid, ()=>{
      // distinguish parallel edges by fx when provided
      let e = null;
      if(p.fx){
        e = edgeByCanonical.get(`${cid}::${p.canonical}::${p.fx}`);
      }
      if(!e){
        const key = `${cid}::${p.canonical}`;
        e = edgeByCanonical.get(key);
      }
      // fallback: first edge with that canonical
      if(!e) e = edges.find(x=> x.canonical===p.canonical && String(x.chatnet_id)===cid);
      // if fx provided and there are parallel edges, prefer the one with matching label
      if(!e && p.fx) e = edges.find(x=> x.canonical===p.canonical && x.label===p.fx && String(x.chatnet_id)===cid);
      if(e){ e.state = p.result ? "true":"false"; updateGraph(); }
      // if we emitted per-edge but there are parallel edges sharing canonical, ensure only the matching fx edge updated (above handles it)
    });
  } else if(t==="prompt_response"){
    // side panel always shows immediately (not gated); graph color already handled by node_complete
    addPanelEntry(canonicalChatnetId(p.chatnet_id), p.node_id, p.prompt, p.response, p.cleaned, {persona: p.persona||'', fx: p.fx||'', fullPrompt: p.fullPrompt||p.prompt, state: 'completed'});
  } else if(t==="standalone_prompt"){
    // store for pairing with response? just note
    window._lastStandalone = p;
  } else if(t==="standalone_response"){
    const lp = window._lastStandalone || {};
    addStandaloneEntry(lp.prompt||p.prompt, p.response, lp.persona);
    window._lastStandalone=null;
  } else if(t==="chatnet_complete"){
    const cid=canonicalChatnetId(p.chatnet_id);
    const ch = chatnets.get(cid);
    if(ch){
      ch.activeWalks.delete(eventRunKey(p));
      // if not yet animating, cancel timer and snap to completion instantly (no animation overhead)
      if(!ch.animating){
        clearTimeout(ch.timer);
        ch.animating = true;
        // flush all pending instantly to final states, then apply completion colors
        ch.pending.forEach(fn=>fn());
        ch.pending=[];
        updateGraph();
      }
    }
    // schedule reset after 4s
    if(!chatnets.get(cid)?.activeWalks.size) setTimeout(()=>{
      handleEvent({type:"chatnet_reset", payload:{chatnet_id:cid}});
    }, 4000);
  } else if(t==="chatnet_reset"){
    const cid=canonicalChatnetId(p.chatnet_id);
    if(chatnets.get(cid)?.activeWalks.size) return;
    // return colors to idle
    for(const n of nodes){ if(String(n.chatnet_id)===cid && !n.activeCount && !n.queuedCount && n.state!=="error") n.state="idle"; }
    for(const e of edges){ if(String(e.chatnet_id)===cid) e.state="idle"; }
    updateGraph();
  }
}

connectSSE();

document.getElementById("resetBtn").addEventListener("click", ()=>{
  fitView();
});
document.getElementById("clearBtn").addEventListener("click", ()=>{
  panel.innerHTML='<div class="empty">Cleared.</div>'; panelEntries=0; panelEntryByNode.clear();
  panelCount.textContent="0 / 20";
});
function fitView(){
  if(simulation) simulation.alpha(0.25).restart();
  if(!svg || !zoom) return;
  if(!nodes.length){ svg.transition().duration(350).call(zoom.transform,d3.zoomIdentity); return; }
  const margin=90, xs=nodes.map(n=>n.x).filter(Number.isFinite), ys=nodes.map(n=>n.y).filter(Number.isFinite);
  if(!xs.length || !ys.length) return;
  const minX=xs.reduce((a,b)=>Math.min(a,b),Infinity), maxX=xs.reduce((a,b)=>Math.max(a,b),-Infinity);
  const minY=ys.reduce((a,b)=>Math.min(a,b),Infinity), maxY=ys.reduce((a,b)=>Math.max(a,b),-Infinity);
  const cx=(minX+maxX)/2, cy=(minY+maxY)/2;
  const scale=Math.min(1.8,(width-margin*2)/Math.max(1,maxX-minX),(height-margin*2)/Math.max(1,maxY-minY));
  const transform=d3.zoomIdentity.translate(width/2-scale*cx,height/2-scale*cy).scale(scale);
  svg.transition().duration(350).call(zoom.transform,transform);
}
function updateStats(){
  const s = document.getElementById("stats");
  if(!s) return;
  const active=nodes.reduce((sum,n)=>sum+(n.activeCount||0),0);
  const completed=nodes.reduce((sum,n)=>sum+(n.completedCount||0),0);
  s.textContent = `${nodes.length} nodes · ${chatnets.size} CDENs · ${active} active node runs · ${completed} completed`;
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
            buffered = _replay_events()
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
                    if data is None:
                        break
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
