"""Local Python/JavaScript workbench. No external service or JS build required."""

import argparse
import ast
import asyncio
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import glob
import io
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import threading
import time
import traceback
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen
import webbrowser
import zipfile

import networkx as nx
import pandas as pd

from . import core as tb
from . import selfimprovement as improvement
from . import workbench_drafting as drafting
from . import workbench_atoms as atoms
from .runtime import WalkContext, WalkControl, walk_context

SCHEMA_VERSION = 1
STATIC = Path(__file__).with_name("web")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,100}$")
SETTINGS = {
    "workers": 2,
    "timeout": 1200,
    "assistant": "",
    "supervisor": "",
    "auto_review": False,
    "review_limit": 8,
    "draft_tokens": 4096,
    "model_params": {},
}
DEFAULT_ENDPOINTS = [
    {"kind": "ollama", "url": "http://127.0.0.1:11434"},
    {"kind": "lmstudio", "url": "http://127.0.0.1:1234"},
    {"kind": "litellm", "url": "http://127.0.0.1:4000", "key_env": "LITELLM_API_KEY"},
    {"kind": "owui", "url": "http://127.0.0.1:3000", "key_env": "OWUI_API_KEY"},
    {"kind": "owui", "url": "http://127.0.0.1:8080", "key_env": "OWUI_API_KEY"},
]
CLI_TEMPLATE = '''"""Editable project launcher; inference and execution stay in TabulAIrity."""
from pathlib import Path
from tabulairity.workbench import project_cli
if __name__ == '__main__':
    project_cli(Path(__file__).resolve().parent)
'''


def plain(value):
    if isinstance(value, pd.DataFrame):
        return plain(value.to_dict("records"))
    if isinstance(value, pd.Series):
        return plain(value.to_dict())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not pd.notna(value):
        return None
    if hasattr(value, "item"):
        return plain(value.item())
    if value is pd.NA:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(4) + ".tmp")
    temporary.write_text(
        json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_project_text(root, path, content):
    """Keep prior user files and only expose a complete replacement."""
    path = Path(path)
    if path.exists():
        backup = Path(root) / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") / path.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(4) + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def read_json(path, default=None):
    return (
        json.loads(Path(path).read_text(encoding="utf-8"))
        if Path(path).exists()
        else default
    )


def checked_name(name):
    if not isinstance(name, str) or not NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(
            "Use a short name containing letters, numbers, underscores, dots, or hyphens"
        )
    return name


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Path must remain inside the project")
    return path


def network_from_graph(graph):
    return {
        "nodes": [
            {"key": node, **plain(data)} for node, data in graph.nodes(data=True)
        ],
        "edges": [
            {"source": a, "target": b, **plain(data)}
            for a, b, data in graph.edges(data=True)
        ],
        "forced_dependencies": plain(graph.graph.get("forced_dependencies", [])),
    }


def graph_from_network(network):
    rows = [dict({"fx": "pass", "prompt": ""} | n, type="node") for n in network.get("nodes", [])]
    rows += [
        dict({"fx": "null", "prompt": ""} | e, type="edge", key=f"{e['source']}-{e['target']}")
        for e in network.get("edges", [])
    ]
    if not rows:
        raise ValueError("Add at least a Start node")
    graph = tb.buildChatNet(pd.DataFrame(rows))
    graph.graph["forced_dependencies"] = [
        tuple(pair) for pair in network.get("forced_dependencies", [])
    ]
    graph.graph["name"] = network.get("name", "")
    return graph


def validate_network(network, strict=True):
    if not isinstance(network, dict):
        raise ValueError("A CDEN must be an object with nodes and edges")
    nodes = network.get("nodes")
    if not isinstance(nodes, list) or any(not isinstance(n, dict) for n in nodes):
        raise ValueError("CDEN nodes must be a list of objects")
    edges = network.get("edges")
    if not isinstance(edges, list) or any(not isinstance(e, dict) for e in edges):
        raise ValueError("CDEN edges must be a list of objects")
    keys = [n.get("key") for n in nodes]
    if any(not isinstance(k, str) or not k for k in keys) or len(keys) != len(
        set(keys)
    ):
        raise ValueError("Node names must be unique, nonempty strings")
    if set(keys) & tb.RESERVED_NODE_NAMES:
        raise ValueError("success and errors are reserved node names")
    for node in nodes:
        if node.get("fx_params") is not None and not isinstance(
            node["fx_params"], dict
        ):
            raise ValueError("Node fx_params must be an object")
        if not str(node.get("fx", "pass")):
            raise ValueError("Each node needs a chatFX")
    if not strict:
        return
    pairs = set()
    for edge in network.get("edges", []):
        if not isinstance(edge, dict):
            raise ValueError("CDEN edges must be objects")
        pair = (edge.get("source"), edge.get("target"))
        if any(key not in keys for key in pair):
            raise ValueError(f"Connection {pair!r} refers to a missing node")
        if pair in pairs:
            raise ValueError(f"Duplicate connection: {pair[0]} → {pair[1]}")
        pairs.add(pair)
    for node in nodes:
        if node.get("persona") and not str(node.get("prompt", "")).strip():
            raise ValueError(f"{node['key']}: add a prompt")
        tokens = node.get("tokens", 2000)
        if isinstance(tokens, bool) or not isinstance(tokens, (int, float)) or tokens < 1:
            raise ValueError(f"{node['key']}: output tokens must be positive")
        if node.get("extra_params") is not None and not isinstance(node["extra_params"], dict):
            raise ValueError(f"{node['key']}: model parameters must be a JSON object")
    graph = graph_from_network(network)
    if "Start" not in graph:
        raise ValueError("Every CDEN needs a Start node")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("CDEN control edges must not contain cycles")
    unreachable = set(graph) - ({"Start"} | nx.descendants(graph, "Start"))
    if unreachable:
        raise ValueError(f"Connect these nodes to Start: {sorted(unreachable)}")
    overlay = tb.buildChatNetDependencies(graph, fxStore=atoms.RUNTIME_FX)
    if not nx.is_directed_acyclic_graph(overlay):
        raise ValueError("Prompt references create a dependency cycle")


def active_network_names(doc):
    selected, pending = set(), [doc["entry"]]
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        if name not in doc["nets"]:
            raise ValueError(f"Unknown nested CDEN {name!r}")
        selected.add(name)
        for node in doc["nets"][name].get("nodes", []):
            if node.get("fx") == "expandIterable":
                child = (node.get("fx_params") or {}).get("chatNet")
                if not isinstance(child, str) or child not in doc["nets"]:
                    raise ValueError(f"{name}/{node['key']}: select a nested CDEN")
                pending.append(child)
    return selected


def validate_project(doc, strict=False):
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported project schema version")
    if not isinstance(doc.get("nets"), dict) or doc.get("entry") not in doc["nets"]:
        raise ValueError("Select an entry CDEN")
    active = active_network_names(doc) if strict else set()
    for name, net in doc["nets"].items():
        checked_name(name)
        cden_path(".", name, net)
        validate_network(net, strict=strict and name in active)
        if strict and name in active:
            for node in net["nodes"]:
                if (
                    node.get("fx") == "expandIterable"
                    and node.get("fx_params", {}).get("chatNet") not in doc["nets"]
                ):
                    raise ValueError(
                        f"{name}/{node['key']}: select a nested CDEN in expansion settings"
                    )
    if strict:
        if doc.get("output_node") and doc["output_node"] not in {n["key"] for n in doc["nets"][doc["entry"]]["nodes"]}:
            raise ValueError("The selected output node is not in the entry CDEN; choose an output in Runs")
        references = nx.DiGraph()
        references.add_nodes_from(active)
        for name in active:
            net = doc["nets"][name]
            for node in net["nodes"]:
                if node.get("fx") == "expandIterable":
                    references.add_edge(name, node["fx_params"]["chatNet"])
        if not nx.is_directed_acyclic_graph(references):
            raise ValueError("Nested CDEN references contain recursion")
    settings = SETTINGS | doc.get("settings", {})
    for key in ("workers", "review_limit", "draft_tokens"):
        if (
            isinstance(settings[key], bool)
            or not isinstance(settings[key], int)
            or settings[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(settings["timeout"], (int, float)) or settings["timeout"] <= 0:
        raise ValueError("timeout must be positive")
    doc["routes"] = normalize_routes(doc.get("routes", []))
    for atom in doc.get("atoms", {}).values():
        atoms.validate_atom(atom)
    return doc


def backend_address(value, *, remote=False):
    """Match the core route contract without interpreting provider model names as URLs."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter a backend address, or remote for a provider-managed endpoint")
    value = value.strip()
    if remote and value.lower() == "remote":
        return "remote"
    if "://" not in value:
        value = "http://" + value
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or any(c.isspace() for c in value):
        raise ValueError("Use a backend address such as localhost:1234/v1, or remote for a provider")
    try:
        if parsed.port == 0:
            raise ValueError("zero port")
    except ValueError as exc:
        raise ValueError("The backend port must be a number from 1 to 65535") from exc
    return value


def normalize_routes(routes):
    if not isinstance(routes, list):
        raise ValueError("Routing rows must be a list")
    result = []
    for index, row in enumerate(routes, 1):
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k].strip() for k in ("model", "route", "ip")):
            raise ValueError(f"Routing row {index} needs model, route, and ip (use remote for Gemini or another provider)")
        item = dict(row)
        for field in ("model", "route", "key_env"):
            item[field] = str(item.get(field, "")).strip()
        item["ip"] = backend_address(item["ip"], remote=True)
        result.append(item)
    return result


def check_project(doc, root):
    """Workbench preflight; uses the existing core validators without inference."""
    errors, warnings = [], []
    try:
        validate_project(doc, strict=True)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append({"message": str(exc)})
    try:
        resolve_input_files(root, doc.get("input", ""))
    except (ValueError, FileNotFoundError) as exc:
        errors.append({"message": str(exc)})
    if not errors:
        available = set(tb.baseFx) | set(atoms.RUNTIME_FX)
        for names in fx_inventory(root).values():
            available.update(names)
        aliases = {route["model"] for route in doc.get("routes", [])}
        for name in sorted(active_network_names(doc)):
            network = doc["nets"][name]
            for node in network["nodes"]:
                if node.get("fx", "pass") not in available:
                    errors.append({"message": f"{name}/{node['key']}: missing function {node.get('fx')}"})
                if node.get("persona") and not str(node.get("prompt", "")).startswith("recall:"):
                    model = node.get("model") or doc.get("settings", {}).get("assistant", SETTINGS["assistant"])
                    model = {"[assistant_model]": doc.get("settings", {}).get("assistant"),
                             "[supervisor_model]": doc.get("settings", {}).get("supervisor")}.get(model, model)
                    if model and isinstance(model, str) and re.search(r"\[[^\]\n]+\]", model):
                        warnings.append({"message": f"{name}/{node['key']}: model is selected from a variable during the run; its value must match an imported or discovered model name."})
                    elif model not in aliases:
                        errors.append({"message": f"{name}/{node['key']}: choose a model in Models and discover its backend or import a routing table."})
            for edge in network.get("edges", []):
                if edge.get("fx", "null") not in available:
                    errors.append({"message": f"{name}: missing connection function {edge.get('fx')}"})
            # Bindings are deliberate variable reads. Check them against input
            # columns without interpreting literal bracket labels as variables.
            for group in network.get("atom_instances", []):
                produced = set().union(*(atoms._produced(doc["nets"][n]) for n in active_network_names(doc))) | {"assistant_model", "supervisor_model", "input_file", "input_row"}
                external = {b.get("variable") for b in group.get("bindings", {}).values()
                            if b.get("mode") == "variable" and b.get("variable") not in produced}
                if external:
                    for path in resolve_input_files(root, doc["input"]):
                        columns = set(pd.read_csv(path, nrows=0).columns) if path.suffix.lower() == ".csv" else set((load_inputs(path) or [{}])[0])
                        missing = external - columns
                        if missing:
                            errors.append({"message": f"{group['key']}: {path.name} has no input columns {', '.join(sorted(str(k) for k in missing))}. Select an input column or earlier step in the atom's settings."})
        if len(active_network_names(doc)) > 1:
            warnings.append({"message": "Nested inference calls depend on the number of items. Actual counts appear in Runs."})
    if not errors and not warnings:
        warnings.append({"message": "Graph and input files validated. Model availability is checked when the run starts."})
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def load_inputs(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return plain(pd.read_csv(path).to_dict("records"))
    if path.suffix.lower() == ".json":
        data = read_json(path)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError("JSON input must be a record object or list")
        return [v if isinstance(v, dict) else {"text": v} for v in data]
    if path.suffix.lower() == ".txt":
        return [{"text": path.read_text(encoding="utf-8-sig")}]
    raise ValueError("Input must be CSV, JSON, or TXT")


def resolve_input_files(root, pattern):
    """Resolve a project-relative glob in deterministic filename order."""
    if not pattern:
        raise ValueError("Select an input file or glob pattern")
    if "/" not in pattern:
        pattern = "inputs/" + pattern
    safe_path(root, pattern)
    files = sorted(p for p in Path(root).glob(pattern) if p.is_file())
    for path in files:
        safe_path(root, str(path.relative_to(root)))
        if path.suffix.lower() not in {".csv", ".json", ".txt"}:
            raise ValueError("A work batch can only contain CSV, JSON or TXT files")
    if not files:
        raise ValueError(f"No input files match {pattern!r}")
    return files


def iter_input_records(root, pattern):
    index = 0
    for path in resolve_input_files(root, pattern):
        relative = str(path.relative_to(root))
        if path.suffix.lower() == ".csv":
            chunks = pd.read_csv(path, chunksize=500)
            records = (
                plain(row) for chunk in chunks for row in chunk.to_dict("records")
            )
        else:
            records = iter(load_inputs(path))
        for file_row, record in enumerate(records):
            record = dict(record)
            record.setdefault("input_file", relative)
            record.setdefault("input_row", file_row)
            yield index, relative, file_row, record
            index += 1


def read_project(root):
    doc = read_json(Path(root) / "project.json")
    if doc is None:
        return None
    for name in doc["nets"]:
        path = cden_path(root, name, doc["nets"][name])
        if not path.exists():
            path = Path(root) / "cdens" / f"{name}.json"
        if path.exists():
            doc["nets"][name] = read_json(path)
    routes = Path(root) / "config/model_routes.csv"
    if routes.exists():
        doc["routes"] = plain(pd.read_csv(routes, dtype=str, keep_default_na=False).to_dict("records"))
    for name in doc.get("atoms", {}):
        path = Path(root) / "atoms" / (checked_name(name) + ".json")
        if path.exists():
            doc["atoms"][name] = read_json(path)
    return doc


def graph_statistics(network):
    graph = graph_from_network(network)
    degrees = [graph.out_degree(n) for n in graph]
    return {
        "nodes": len(graph),
        "edges": graph.number_of_edges(),
        "model_steps": sum(
            bool(n.get("persona"))
            and not str(n.get("prompt", "")).startswith("recall:")
            for _, n in graph.nodes(data=True)
        ),
        "branch_points": sum(degree > 1 for degree in degrees),
        "max_branching": max(degrees, default=0),
        "mean_branching": round(sum(degrees) / len(degrees), 2) if degrees else 0,
        "conditional_edges": sum(
            e.get("fx", "null") != "null" for _, _, e in graph.edges(data=True)
        ),
        "expansions": sum(
            n.get("fx") == "expandIterable" for _, n in graph.nodes(data=True)
        ),
    }


def write_routes(path, routes):
    columns = list(
        dict.fromkeys(
            ["model", "route", "ip", "key_env"] + [k for r in routes for k in r]
        )
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(routes)


def cden_path(root, name, network):
    pool = network.get("pool", "user")
    if pool not in {"internal", "samples", "user"}:
        raise ValueError("CDEN pool must be internal, samples, or user")
    return Path(root) / "cdens" / pool / (checked_name(name) + ".json")


def save_project(root, doc, backup=True):
    root = Path(root)
    validate_project(doc)
    previous = read_json(root / "project.json")
    if previous and backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        atomic_json(root / "history" / stamp / "project.json", previous)
    root.mkdir(parents=True, exist_ok=True)
    for name, net in doc["nets"].items():
        atomic_json(cden_path(root, name, net), net)
    for name, atom in doc.get("atoms", {}).items():
        atomic_json(root / "atoms" / (checked_name(name) + ".json"), atom)
    write_routes(root / "config/model_routes.csv", doc.get("routes", []))
    atomic_json(root / "project.json", doc)
    if not (root / "run.py").exists():
        (root / "run.py").write_text(CLI_TEMPLATE, encoding="utf-8")
    return doc


def default_project(title):
    doc = {
        "schema_version": 1,
        "title": title,
        "entry": "main",
        "input": "inputs/catalog.json",
        "output_node": "titles",
        "routes": [],
        "rankings": [],
        "settings": dict(SETTINGS),
        "nets": read_json(Path(__file__).with_name("bundled") / "samples.json"),
        "atoms": atoms.bundled_atoms(),
    }
    doc["workbench_workflows"] = {}
    for kind, network in improvement.getWorkbenchWorkflows().items():
        if kind == "cden":
            for node in network["nodes"]:
                node["tokens"] = 4096
        if kind == "review":
            for node in network["nodes"]:
                node["model"] = "[supervisor_model]"
        name = "workbench_" + kind
        network["pool"] = "internal"
        network["title"] = {
            "prompt": "Prompt improvement", "fx": "Python function drafting",
            "cden": "CDEN drafting", "review": "Extraction review",
        }[kind]
        network["description"] = {
            "prompt": "Infer intent, expected outputs and failure modes; draft, check and revise a prompt.",
            "fx": "Draft an importable Python chatFX module using the supplied request and validation feedback.",
            "cden": "Draft a CDEN definition using the project context and validation feedback.",
            "review": "Review bounded extraction evidence and propose flags or revisions.",
        }[kind]
        doc["nets"][name] = network
        doc["workbench_workflows"][kind] = name
    return doc


def discover_backends(endpoints):
    from concurrent.futures import ThreadPoolExecutor

    if not isinstance(endpoints, list) or len(endpoints) > 50 or any(not isinstance(e, dict) for e in endpoints):
        raise ValueError("Supply up to 50 backend addresses")

    def probe(endpoint):
        kind = endpoint["kind"]
        base = backend_address(endpoint["url"]).rstrip("/")
        if kind not in {"ollama", "lmstudio", "litellm", "owui"}:
            raise ValueError("Unknown backend kind")
        if kind == "ollama":
            url = base + "/api/tags"
            api_base = base
        elif kind == "owui":
            base = base.removesuffix("/api")
            url = base + "/api/models"
            api_base = base + "/api"
        else:
            base = base.removesuffix("/v1")
            url = base + "/v1/models"
            api_base = base + "/v1"
        key_env = endpoint.get("key_env", "")
        headers = {}
        if key_env and os.environ.get(key_env):
            headers["Authorization"] = "Bearer " + os.environ[key_env]
        try:
            with urlopen(Request(url, headers=headers), timeout=2) as response:
                data = json.load(response)
            models = (
                data.get("models", []) if kind == "ollama" else data.get("data", [])
            )
            routes = []
            for model in models:
                identifier = model.get("name") if kind == "ollama" else model.get("id")
                if not identifier:
                    continue
                routes.append(
                    {
                        "model": identifier,
                        "route": ("ollama_chat/" if kind == "ollama" else "openai/")
                        + identifier,
                        "ip": api_base,
                        "key_env": key_env,
                    }
                )
            return {"endpoint": endpoint, "status": "available", "routes": routes}
        except Exception as exc:
            return {
                "endpoint": endpoint,
                "status": "unavailable",
                "error": str(exc),
                "routes": [],
            }

    def safe_probe(endpoint):
        try:
            return probe(endpoint)
        except (ValueError, KeyError, TypeError) as exc:
            return {"endpoint": endpoint, "status": "unavailable", "error": str(exc), "routes": []}

    with ThreadPoolExecutor(max_workers=min(5, max(1, len(endpoints)))) as pool:
        probes = list(pool.map(safe_probe, endpoints))
    routes = [r for p in probes for r in p["routes"]]
    return {"probes": probes, "routes": routes}


def recommend_roles(routes, rankings):
    available = {r["model"] for r in routes}
    ranked = [
        r
        for r in rankings
        if isinstance(r, dict) and r.get("model") in available
        and isinstance(r.get("score"), (int, float))
        and not isinstance(r.get("score"), bool) and math.isfinite(r["score"])
        and r.get("source")
        and r.get("date")
    ]
    if ranked:
        best = max(ranked, key=lambda r: r["score"])
        return {
            "assistant": best["model"],
            "supervisor": best["model"],
            "basis": f"{best['source']} ({best['date']})",
        }
    if len(available) == 1:
        only = next(iter(available))
        return {
            "assistant": only,
            "supervisor": only,
            "basis": "Only model currently available",
        }
    return {
        "basis": "No verified ranking for these local variants; select both roles from the table"
    }


def fx_inventory(root):
    inventory = {}
    for path in sorted((Path(root) / "fx").glob("*.py")):
        try:
            inventory[path.name] = improvement.validateFxSource(
                path.read_text(encoding="utf-8")
            )
        except (ValueError, SyntaxError):
            inventory[path.name] = []
    return inventory


def load_fx(root, trusted):
    files = sorted((Path(root) / "fx").glob("*.py"))
    if not files:
        return {}
    if not trusted:
        raise ValueError(
            "This project contains Python chatFX. Enable Run project Python to execute them."
        )
    store = {}
    for path in files:
        improvement.validateFxSource(path.read_text(encoding="utf-8"))
        module_name = "_tabulairity_project_" + secrets.token_hex(8)
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        exported = getattr(module, "CHAT_FX", {})
        if not isinstance(exported, dict) or not all(
            isinstance(k, str) and callable(v) for k, v in exported.items()
        ):
            raise ValueError(f"{path.name}: CHAT_FX must map names to functions")
        collisions = set(exported) & (set(store) | set(tb.baseFx) | set(atoms.RUNTIME_FX))
        if collisions:
            raise ValueError(f"Duplicate or reserved chatFX: {sorted(collisions)}")
        store.update(exported)
    return store


def data_product(result, doc, input_index):
    node = doc.get("output_node")
    if node:
        value = result.get(node)
        if isinstance(value, pd.DataFrame):
            rows = value.to_dict("records")
        elif isinstance(value, list):
            rows = [v if isinstance(v, dict) else {node: v} for v in value]
        elif isinstance(value, dict):
            rows = [value]
        else:
            rows = [{node: value}]
    else:
        audit = {
            n["key"] + s
            for net in doc["nets"].values()
            for n in net["nodes"]
            for s in ("_raw", "_prompt")
        }
        audit |= {
            f"{e['source']}-{e['target']}"
            for net in doc["nets"].values()
            for e in net["edges"]
        }
        rows = [{k: v for k, v in result.items() if k not in audit}]
    return [
        plain(row) | {"_row_id": f"{input_index}:{i}", "_input_index": input_index}
        for i, row in enumerate(rows)
    ]


def iter_jsonl(path):
    if not Path(path).exists():
        return
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                break


def read_jsonl(path, offset=0, limit=None):
    from itertools import islice

    return list(
        islice(iter_jsonl(path), offset, None if limit is None else offset + limit)
    )


def csv_product_file(source, path):
    keys = list(
        dict.fromkeys(
            k
            for row in iter_jsonl(source)
            for k in row
            if k not in {"_row_id", "_input_index"}
        )
    )
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in iter_jsonl(source):
            writer.writerow(
                {
                    k: (
                        json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (dict, list))
                        else v
                    )
                    for k, v in row.items()
                    if k not in {"_row_id", "_input_index"}
                }
            )


def csv_product(rows, path):
    # Compatibility helper for a small already-materialized set of records.
    keys = list(
        dict.fromkeys(
            k for r in rows for k in r if k not in {"_row_id", "_input_index"}
        )
    )
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {
                k: (
                    json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else v
                )
                for k, v in r.items()
                if k not in {"_row_id", "_input_index"}
            }
            for r in rows
        )


class SharedSlots:
    def __init__(self, gate):
        self.gate = gate

    async def __aenter__(self):
        while not self.gate.acquire(False):
            await asyncio.sleep(0.025)
        return self

    async def __aexit__(self, *args):
        self.gate.release()


def _run_worker(project_root, run_root, operation, args, paused, stopped, gate):
    project_root = Path(project_root).resolve()
    run_root = Path(run_root).resolve()
    previous_directory = Path.cwd()
    previous_cache = tb.cacheDatabase
    previous_cache_reads = tb.useCache
    snapshot = run_root / "snapshot"
    doc = read_json(snapshot / "project.json")
    settings = SETTINGS | doc.get("settings", {})
    if operation == "assistant" and args.get("kind") == "cden" and args.get("draft_tokens") is not None:
        budget = args["draft_tokens"]
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("Constructor output limit must be a positive whole number")
        settings["draft_tokens"] = budget
    started = time.monotonic()
    input_calls = {}
    state = {
        "status": "running",
        "operation": operation,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "rows": 0,
        "inputs_done": 0,
        "nodes": 0,
        "errors": 0,
        "failed_inputs": 0,
        "elapsed": 0,
        "output_length": 0,
        "inference_calls": 0,
        "cache_hits": 0,
        "mean_calls_per_input": 0,
        "extraction_calls": 0,
        "workbench_calls": 0,
        "completed_input_calls": 0,
        "node_states": {},
    }
    emit_lock = threading.RLock()
    sequence = 0

    def emit(event):
        nonlocal sequence
        with emit_lock:
            sequence += 1
            event = plain(event) | {
                "seq": sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with (run_root / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def status():
        state["elapsed"] = round(time.monotonic() - started, 3)
        state["mean_calls_per_input"] = (
            round(state["completed_input_calls"] / state["inputs_done"], 2)
            if state["inputs_done"]
            else 0
        )
        atomic_json(run_root / "status.json", state)

    def observe(event):
        with emit_lock:
            if event["event"] == "cden_attempt":
                attempts = read_json(run_root / "constructor-attempts.json", {}).get("attempts", [])
                attempts.append(plain({k: v for k, v in event.items() if k != "event"}))
                atomic_json(run_root / "constructor-attempts.json", {
                    "schema_version": 1, "request": args.get("request", ""),
                    "model": settings["assistant"], "attempts": attempts,
                })
                if not event.get("valid"):
                    state["validation_failures"] = state.get("validation_failures", 0) + 1
            if event["event"] == "inference_start":
                state["inference_calls"] += 1
                state[
                    "extraction_calls"
                    if state.get("phase") == "extraction"
                    else "workbench_calls"
                ] += 1
                if "input_index" in event:
                    input_calls[event["input_index"]] = (
                        input_calls.get(event["input_index"], 0) + 1
                    )
            if event["event"] == "cache_hit":
                state["cache_hits"] += 1
            if event["event"] == "node_complete":
                state["nodes"] += 1
                state["output_length"] += event.get("output_length", 0)
            node_state = {"node_start": "running", "node_complete": "completed", "node_error": "failed"}.get(event["event"])
            if node_state and event.get("node"):
                state["node_states"].setdefault(event.get("network", ""), {})[str(event["node"])] = node_state
            if event["event"] == "node_error":
                state["errors"] += 1
            emit(event)
            status()

    def review_records(origin, row_ids=None):
        from itertools import islice

        flags = read_json(origin / "flags.json", {})
        wanted = set(row_ids or flags)
        rows = iter_jsonl(origin / "table.jsonl")
        if wanted:
            rows = (row for row in rows if row["_row_id"] in wanted)
        chosen = list(islice(rows, settings["review_limit"]))
        needed = {row["_input_index"] for row in chosen}
        raw_by_input = {}
        for raw in iter_jsonl(origin / "results.jsonl"):
            if raw["input_index"] in needed:
                raw_by_input[raw["input_index"]] = raw
            if len(raw_by_input) == len(needed):
                break
        records = []
        for row in chosen:
            raw = raw_by_input[row["_input_index"]]
            records.append(
                {
                    "row_id": row["_row_id"],
                    "input": raw["input"],
                    "output": row,
                    "raw": raw["result"],
                    "user_flag": flags.get(row["_row_id"]),
                }
            )
        return records

    try:
        # The core intentionally prioritises cwd/config. Give every workbench
        # run its own snapshot cwd so an unrelated launch-folder table cannot win.
        os.chdir(snapshot)
        # Cache within the project, partitioned by routing configuration. Moving
        # to a new backend must not replay another backend's cached answers.
        if tb.cacheConfig.get("backend") == "sqlite":
            route_identity = [{k: v for k, v in row.items() if k != "key"} for row in doc.get("routes", [])]
            digest = hashlib.sha256(json.dumps(route_identity, sort_keys=True).encode()).hexdigest()[:20]
            cache = project_root / "cache" / (digest + ".sqlite3")
            cache.parent.mkdir(parents=True, exist_ok=True)
            tb.cacheDatabase = str(cache)
            tb.initDbSQLite()
        tb.loadModelRoutes(str(snapshot / "config/model_routes.csv"))
        if not doc.get("routes"):
            tb._modelRoutes = None
        control = WalkControl(paused, stopped)
        workflow_vars = {
            "assistant_model": settings["assistant"],
            "supervisor_model": settings["supervisor"],
        }

        def workflow(kind):
            name = doc.get("workbench_workflows", {}).get(kind)
            selected = plain(doc["nets"][name] if name else improvement.getWorkbenchWorkflows()[kind])
            if kind == "cden":
                stock = improvement.getWorkbenchWorkflows()["cden"]
                # Old stock projects stored 4096; honour the chosen output
                # budget without overriding deliberately customised workflows.
                fields = ("key", "prompt", "persona", "fx", "model", "fx_params", "self_eval")
                if (len(selected.get("nodes", [])) == 1 and not selected.get("edges")
                        and selected.get("result_nodes", ["Start"]) == ["Start"]
                        and (name in {None, "workbench_cden"} or selected.get("bundled_source") == "workbench_cden")
                        and selected["nodes"][0].get("tokens") in {4096, 8192}
                        and all(selected["nodes"][0].get(k) == stock["nodes"][0].get(k) for k in fields)):
                    selected["nodes"][0]["tokens"] = settings["draft_tokens"]
            for node in selected.get("nodes", []):
                node["tokens"] = min(node.get("tokens", settings["draft_tokens"]), settings["draft_tokens"])
            if kind == "review":
                # Older projects have the generic one-step workflow pointing at
                # assistant_model. Correct the stock definition at execution;
                # retain deliberately customised workflows unchanged.
                candidate = plain(selected)
                for node in candidate.get("nodes", []):
                    if node.get("model") == "[supervisor_model]":
                        node["model"] = "[assistant_model]"
                stock = improvement.getWorkbenchWorkflows()["review"]
                for node in stock["nodes"]:
                    node["tokens"] = min(node["tokens"], settings["draft_tokens"])
                if all(candidate.get(k) == stock.get(k) for k in ("nodes", "edges", "result_nodes")):
                    selected = plain(selected)
                    for node in selected["nodes"]:
                        node["model"] = "[supervisor_model]"
            return selected | {"name": name or "workbench_" + kind}

        # An unused module must not block deterministic samples, route checks,
        # or built-in assistance. It is still never imported without opt-in.
        needed = []
        if operation in {"run", "atom_test"}:
            needed = [doc["nets"][n] for n in active_network_names(doc)]
            if settings["auto_review"]:
                needed.append(workflow("review") or {})
        elif operation in {"assistant", "review"}:
            kind = args["kind"] if operation == "assistant" else "review"
            name = doc.get("workbench_workflows", {}).get(kind)
            needed = [doc["nets"][n] for n in active_network_names(doc | {"entry": name})] if name else [workflow(kind)]
        builtins = set(tb.baseFx) | set(atoms.RUNTIME_FX)
        requires_python = (operation == "fx_test" and args.get("fx") not in builtins) or any(
            item.get("fx", "null") not in builtins
            for net in needed for item in [*net.get("nodes", []), *net.get("edges", [])]
        )
        fx = atoms.RUNTIME_FX | (load_fx(snapshot, True) if args.get("trust_fx", False) else {})
        if requires_python and not args.get("trust_fx", False):
            raise ValueError("This operation uses project Python chatFX. Enable Run project Python to execute them.")
        workflow_registry = {}
        for name, network in doc["nets"].items():
            try:
                workflow_registry[name] = graph_from_network(network | {"name": name})
            except (ValueError, KeyError, TypeError):
                pass  # unrelated unfinished graphs do not block the selected workflow
        workflow_token = walk_context.set(
            WalkContext(
                SharedSlots(gate),
                workflow_registry,
                fx | tb.baseFx,
                settings["workers"],
                observe,
                control,
            )
        )
        if operation in {"run", "atom_test"}:
            state["phase"] = "extraction"
            validate_project(doc, strict=True)
            nets = {
                name: graph_from_network(net | {"name": name})
                for name, net in doc["nets"].items()
                if name in active_network_names(doc)
            }
            for name, graph in nets.items():
                for node in graph:
                    if graph.nodes[node].get("fx", "null") not in (
                        set(fx) | set(tb.baseFx)
                    ):
                        raise ValueError(
                            f"{name}/{node}: missing chatFX {graph.nodes[node].get('fx')}"
                        )
                for _, _, edge in graph.edges(data=True):
                    if edge.get("fx", "null") not in (set(fx) | set(tb.baseFx)):
                        raise ValueError(
                            f"{name}: missing edge chatFX {edge.get('fx')}"
                        )
            files = resolve_input_files(snapshot, doc["input"])
            state["files_total"] = len(files)
            state["graph"] = graph_statistics(doc["nets"][doc["entry"]])
            status()

            async def run():
                token = walk_context.set(
                    WalkContext(
                        SharedSlots(gate),
                        nets,
                        fx | tb.baseFx,
                        workers=settings["workers"],
                        observer=observe,
                        control=control,
                    )
                )
                try:
                    for index, input_file, input_row, record in iter_input_records(
                        snapshot, doc["input"]
                    ):
                        await control.checkpoint()
                        emit(
                            {
                                "event": "input_start",
                                "input_index": index,
                                "input_file": input_file,
                                "input_row": input_row,
                            }
                        )

                        def scoped_observe(event):
                            observe(event | {"input_index": index})

                        result = await tb.walkChatNetAsync(
                            nets[doc["entry"]].copy(),
                            fx,
                            workflow_vars | record,
                            0,
                            numWorkers=settings["workers"],
                            tolerant=True,
                            timeout=settings["timeout"],
                            chatNets=nets,
                            observer=scoped_observe,
                            control=control,
                        )
                        payload = {
                            "input_index": index,
                            "input_file": input_file,
                            "input_row": input_row,
                            "input": record,
                            "result": plain(result),
                        }
                        if not result.get("success"):
                            state["failed_inputs"] += 1
                        with (run_root / "results.jsonl").open(
                            "a", encoding="utf-8"
                        ) as f:
                            f.write(
                                json.dumps(payload, ensure_ascii=False, allow_nan=False)
                                + "\n"
                            )
                        rows = data_product(result, doc, index)
                        with (run_root / "table.jsonl").open(
                            "a", encoding="utf-8"
                        ) as f:
                            for row in rows:
                                f.write(
                                    json.dumps(row, ensure_ascii=False, allow_nan=False)
                                    + "\n"
                                )
                                emit({"event": "row", "row": row})
                        state["completed_input_calls"] += input_calls.get(index, 0)
                        state["inputs_done"] += 1
                        state["rows"] += len(rows)
                        status()
                finally:
                    walk_context.reset(token)

            asyncio.run(run())
            csv_product_file(run_root / "table.jsonl", run_root / "output.csv")
            if operation == "atom_test" and state["failed_inputs"]:
                raise ValueError("The atom trial failed. Inspect the failed steps in the transcript; no validated result was produced.")
            if settings["auto_review"] and not stopped.is_set():
                state["phase"] = "review"
                try:
                    draft = improvement.reviewExtractions(
                        review_records(run_root),
                        doc["nets"][doc["entry"]],
                        model=settings["supervisor"],
                        max_records=settings["review_limit"],
                        tokens=settings["draft_tokens"],
                        extra_params=settings["model_params"],
                        workflow=workflow("review"),
                        workflow_vars=workflow_vars,
                        fxStore=fx,
                        observer=observe,
                    )
                    draft["run_id"] = run_root.name
                    draft["target_network"] = doc["entry"]
                    atomic_json(run_root / "draft.json", draft)
                    atomic_json(
                        run_root / "flags.json",
                        merge_review_flags(
                            read_json(run_root / "flags.json", {}), draft["flags"]
                        ),
                    )
                    emit({"event": "review_ready", "draft": draft})
                except Exception as exc:
                    state["review_error"] = str(exc)
                    emit({"event": "review_error", "error": str(exc)})
        elif operation in {"assistant", "review", "route_test"}:
            if stopped.is_set():
                raise asyncio.CancelledError()
            if operation == "assistant":
                network = doc["nets"][args.get("network", doc["entry"])]
                options = dict(
                    model=settings["assistant"], tokens=settings["draft_tokens"],
                    extra_params=settings["model_params"], workflow=workflow(args["kind"]),
                    workflow_vars=workflow_vars, fxStore=fx, observer=observe,
                )
                if args["kind"] == "cden":
                    # This is a dedicated worker process. A new construction
                    # must not replay an earlier malformed cached response.
                    tb.useCache = False
                    prepared = drafting.prepare_context(
                        doc, args.get("network", doc["entry"]), args["request"], snapshot, fx,
                    )
                    atomic_json(run_root / "authoring-context.json", prepared)
                    emit({"event": "authoring_context", **prepared["summary"]})
                    draft = drafting.draft_cden(prepared, doc, **options)
                else:
                    draft = improvement.draftArtifact(
                        args["kind"], args["request"], network, node=args.get("node"),
                        validate_network=validate_network, **options,
                    )
            elif operation == "review":
                origin = safe_path(project_root, "runs/" + checked_name(args["run_id"]))
                original = read_json(origin / "snapshot/project.json")
                draft = improvement.reviewExtractions(
                    review_records(origin, args.get("row_ids")),
                    original["nets"][original["entry"]],
                    model=settings["supervisor"],
                    max_records=settings["review_limit"],
                    tokens=settings["draft_tokens"],
                    extra_params=settings["model_params"],
                    workflow=workflow("review"),
                    workflow_vars=workflow_vars,
                    fxStore=fx,
                    observer=observe,
                )
                draft["run_id"] = args["run_id"]
                existing = read_json(origin / "flags.json", {})
                existing = merge_review_flags(existing, draft["flags"])
                atomic_json(origin / "flags.json", existing)
            else:
                draft = {
                    "working": tb.testRoutes(
                        models=args.get("models"),
                        query="Return the word ready.",
                        persona="Follow the instruction.",
                    )
                }
            draft["target_network"] = (
                original["entry"]
                if operation == "review"
                else args.get("network", doc["entry"])
            )
            atomic_json(run_root / "draft.json", draft)
            emit({"event": "draft_ready", "draft": draft})
        elif operation == "fx_test":
            store = fx
            graph = graph_from_network(
                {
                    "nodes": [
                        {
                            "key": "Start",
                            "fx": args["fx"],
                            "prompt": "recall: " + args.get("text", ""),
                            "persona": "",
                            "tokens": 1,
                        }
                    ],
                    "edges": [],
                }
            )
            draft = tb.walkChatNet(
                graph,
                fxStore=store,
                varStore=args.get("variables", {}),
                verbosity=0,
                observer=observe,
            )
            if not draft.get("success"):
                raise ValueError("Function test failed: " + str(draft.get("errors", [])))
            atomic_json(run_root / "draft.json", plain(draft))
            (run_root / "table.jsonl").write_text(json.dumps(plain({"result": draft.get("Start"), "_row_id": "0:0", "_input_index": 0}), ensure_ascii=False) + "\n", encoding="utf-8")
            csv_product_file(run_root / "table.jsonl", run_root / "output.csv")
            state["rows"] = state["inputs_done"] = 1
            emit({"event": "draft_ready", "draft": plain(draft)})
        else:
            raise ValueError("Unknown operation")
        state["status"] = "completed"
    except (asyncio.CancelledError, KeyboardInterrupt):
        state["status"] = "stopped"
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        emit(
            {
                "event": "run_error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        tb.useCache = previous_cache_reads
        tb.cacheDatabase = previous_cache
        os.chdir(previous_directory)
        if "workflow_token" in locals():
            walk_context.reset(workflow_token)
        status()
        emit({"event": "run_complete", "status": state["status"]})


def merge_review_flags(existing, proposed):
    merged = dict(existing)
    for flag in proposed:
        key = str(flag["row_id"])
        if key in merged and merged[key].get("author") == "user":
            merged[key] = dict(merged[key]) | {"supervisor_feedback": flag}
        else:
            merged[key] = dict(flag) | {"author": "supervisor"}
    return merged


def snapshot_project(root, run_root):
    target = Path(run_root) / "snapshot"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(root) / "project.json", target / "project.json")
    for name in ("inputs", "cdens", "config", "fx", "atoms"):
        source = Path(root) / name
        if source.exists():
            shutil.copytree(source, target / name)


def draft_baseline(root, run_id, draft):
    """Revision previews must use the source snapshot, never the edited project."""
    source = Path(root) / "runs" / checked_name((draft or {}).get("run_id", run_id)) / "snapshot"
    doc = read_json(source / "project.json", {})
    target = (draft or {}).get("target_network", doc.get("entry"))
    network = doc.get("nets", {}).get(target, {})
    files = {}
    for revision in (draft or {}).get("revisions", [draft or {}]):
        if revision.get("code") and revision.get("filename"):
            name = checked_name(revision["filename"])
            path = source / "fx" / name
            if path.is_file():
                files[name] = path.read_text(encoding="utf-8")
    return {"network": target, "nodes": {n["key"]: {k: n.get(k, "") for k in ("prompt", "persona")}
                                           for n in network.get("nodes", [])}, "files": files}


class Workbench:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.processes = {}
        self.lock = threading.RLock()
        self.context = mp.get_context("spawn")
        self.gate = None

    def project(self, identifier):
        return safe_path(self.root, checked_name(identifier))

    def create(self, title):
        identifier = (
            re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "project"
        )
        identifier += "-" + secrets.token_hex(3)
        root = self.project(identifier)
        doc = default_project(title)
        save_project(root, doc, backup=False)
        (root / "inputs").mkdir(exist_ok=True)
        for path in (Path(__file__).with_name("bundled") / "inputs").iterdir():
            shutil.copy2(path, root / "inputs" / path.name)
        return {"id": identifier, "project": doc}

    def starter(self):
        """Give a fresh installation a usable example workspace, once."""
        with self.lock:
            projects = sorted(p for p in self.root.iterdir() if (p / "project.json").is_file())
            if projects:
                return {"id": projects[0].name}
            return self.create("Examples and internal CDENs")

    def add_bundled(self, identifier):
        """Fill missing defaults; retain existing CDENs, inputs, routes and settings."""
        with self.lock:
            root = self.project(identifier)
            doc = read_project(root)
            if doc is None:
                raise FileNotFoundError("Project not found")
            defaults = default_project("Bundled CDENs")
            mapping = {}
            for name, net in defaults["nets"].items():
                chosen = name
                index = 1
                while chosen in doc["nets"] and (
                    doc["nets"][chosen].get("pool", "user") != net["pool"]
                    or (chosen != name and doc["nets"][chosen].get("bundled_source") != name)
                ):
                    chosen = "bundled_" + name + ("_" + str(index) if index > 1 else "")
                    index += 1
                mapping[name] = chosen
            inputs = {}
            for source in (Path(__file__).with_name("bundled") / "inputs").iterdir():
                target = root / "inputs" / source.name
                index = 1
                while target.exists() and target.read_bytes() != source.read_bytes():
                    suffix = "_" + str(index) if index > 1 else ""
                    target = root / "inputs" / ("bundled_" + source.stem + suffix + source.suffix)
                    index += 1
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
                inputs["inputs/" + source.name] = target.relative_to(root).as_posix()
            added = []
            for name, net in defaults["nets"].items():
                chosen = mapping[name]
                net = json.loads(json.dumps(net))
                net["bundled_source"] = name
                for node in net["nodes"]:
                    params = node.get("fx_params") or {}
                    if params.get("chatNet") in mapping:
                        params["chatNet"] = mapping[params["chatNet"]]
                if "example_input" in net:
                    net["example_input"] = inputs[net["example_input"]]
                if chosen not in doc["nets"]:
                    doc["nets"][chosen] = net
                    added.append(chosen)
                else:
                    # Metadata can be added to dev1/dev2 projects without changing their graph.
                    for key in ("title", "description", "example_input", "example_output_node"):
                        if key in net:
                            doc["nets"][chosen].setdefault(key, net[key])
            workflows = doc.setdefault("workbench_workflows", {})
            for name, atom in defaults["atoms"].items():
                doc.setdefault("atoms", {}).setdefault(name, atom)
            for kind, name in defaults["workbench_workflows"].items():
                workflows.setdefault(kind, mapping[name])
            save_project(root, doc)
            return self.describe(identifier) | {"added": added}

    def describe(self, identifier):
        root = self.project(identifier)
        doc = read_project(root)
        if doc is None:
            raise FileNotFoundError("Project not found")
        validate_project(doc)
        files = [
            str(p.relative_to(root))
            for d in ("inputs", "cdens", "fx", "atoms")
            for p in (root / d).rglob("*")
            if p.is_file()
        ]
        return {
            "id": identifier,
            "project": doc,
            "files": files,
            "fx_inventory": fx_inventory(root),
            "graph_stats": {
                name: graph_statistics(net)
                for name, net in doc["nets"].items()
                if net.get("nodes")
            },
        }

    def start(self, identifier, operation, args):
        if operation not in {"run", "atom_test", "assistant", "review", "route_test", "fx_test"}:
            raise ValueError("Unknown operation")
        if operation == "assistant" and args.get("kind") == "cden" and args.get("draft_tokens") is not None:
            budget = args["draft_tokens"]
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
                raise ValueError("Constructor output limit must be a positive whole number")
        root = self.project(identifier)
        doc = read_project(root)
        if doc is None:
            raise FileNotFoundError("Project not found")
        if operation == "run":
            report = check_project(doc, root)
            if not report["valid"]:
                raise ValueError("; ".join(issue["message"] for issue in report["errors"]))
        if operation == "atom_test":
            atom = doc.get("atoms", {}).get(args.get("atom"))
            if atom is None:
                raise ValueError("Choose an atom to try")
            values = args.get("values", {})
            if not isinstance(values, dict) or set(values) != set(atom["inputs"]):
                raise ValueError("Supply every atom input")
            candidate = doc | {"entry": "atom_test", "output_node": "result",
                "input": "inputs/_atom_test.json", "nets": {"atom_test": {
                    "pool": "user", "nodes": [{"key": "Start", "fx": "pass", "prompt": "recall: ready", "persona": "", "tokens": 1}], "edges": []}}}
            doc = atoms.insert_atom(candidate, "atom_test", atom, "Start", "result",
                {name: {"mode": "literal", "value": value} for name, value in values.items()})["project"]
            # A trial is its own saved run, never a replacement for the user's graph.
            doc["settings"] = doc["settings"] | {"auto_review": False}
            validate_project(doc, strict=True)
        else:
            save_project(root, doc, backup=False)
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + secrets.token_hex(3)
        )
        run_root = root / "runs" / run_id
        run_root.mkdir(parents=True)
        atomic_json(run_root / "operation.json", {"operation": operation, "args": args})
        snapshot_project(root, run_root)
        if operation == "atom_test":
            save_project(run_root / "snapshot", doc, backup=False)
            atomic_json(run_root / "snapshot/inputs/_atom_test.json", [{}])
            report = check_project(doc, run_root / "snapshot")
            if not report["valid"]:
                shutil.rmtree(run_root)
                raise ValueError("; ".join(issue["message"] for issue in report["errors"]))
        paused = self.context.Event()
        stopped = self.context.Event()
        with self.lock:
            for key, (finished, _, _) in list(self.processes.items()):
                if not finished.is_alive():
                    finished.join()
                    finished.close()
                    del self.processes[key]
            active = [p for p, _, _ in self.processes.values() if p.is_alive()]
            if not active:
                self.gate = self.context.BoundedSemaphore(
                    doc.get("settings", {}).get("workers", 2)
                )
            process = self.context.Process(
                target=_run_worker,
                args=(
                    str(root),
                    str(run_root),
                    operation,
                    args,
                    paused,
                    stopped,
                    self.gate,
                ),
            )
            atomic_json(
                run_root / "status.json",
                {"status": "starting", "operation": operation, "rows": 0},
            )
            self.processes[identifier + "/" + run_id] = (process, paused, stopped)
            process.start()
        return {"run_id": run_id}

    def run_status(self, identifier, run_id):
        root = self.project(identifier) / "runs" / checked_name(run_id)
        state = read_json(root / "status.json", {})
        handle = self.processes.get(identifier + "/" + run_id)
        if handle:
            process, paused, stopped = handle
            if process.is_alive() and state.get("status") not in {
                "completed",
                "failed",
                "stopped",
            }:
                state["status"] = (
                    "stopping"
                    if stopped.is_set()
                    else "paused"
                    if paused.is_set()
                    else state.get("status", "running")
                )
            elif not process.is_alive() and state.get("status") in {
                "starting",
                "running",
                "paused",
                "stopping",
            }:
                state["status"] = "interrupted"
        elif state.get("status") in {"starting", "running", "paused", "stopping"}:
            state["status"] = "interrupted"
        return state

    def control(self, identifier, run_id, action):
        handle = self.processes.get(identifier + "/" + run_id)
        if handle is None or not handle[0].is_alive():
            raise ValueError("This run is no longer active")
        process, paused, stopped = handle
        if action == "pause":
            paused.set()
        elif action == "resume":
            paused.clear()
        elif action == "stop":
            stopped.set()
            paused.clear()
        else:
            raise ValueError("Unknown control action")
        return {"status": self.run_status(identifier, run_id)}

    def upload(self, identifier, name, kind, content):
        root = self.project(identifier)
        name = checked_name(name)
        doc = read_project(root)
        if kind == "input":
            if Path(name).suffix.lower() not in {".csv", ".json", ".txt"}:
                raise ValueError("Use CSV, JSON, or TXT")
            path = root / "inputs" / name
            # Parse before replacing an existing input file.
            if path.suffix.lower() == ".csv":
                pd.read_csv(io.StringIO(content))
            elif path.suffix.lower() == ".json":
                data = json.loads(content)
                if not isinstance(data, (dict, list)):
                    raise ValueError("JSON input must be a record object or list")
            doc["input"] = "inputs/" + name
            validate_project(doc)
            write_project_text(root, path, content)
        elif kind == "cden":
            if name.endswith(".csv"):
                graph = tb.buildChatNet(pd.read_csv(io.StringIO(content)))
                net = network_from_graph(graph)
            else:
                net = json.loads(content)
                if isinstance(net, list):
                    net = network_from_graph(tb.buildChatNet(pd.DataFrame(net)))
            validate_network(net, strict=False)
            net["pool"] = "user"
            network_name = checked_name(Path(name).stem)
            original, suffix = network_name, 2
            while network_name in doc["nets"]:
                network_name = original[:90] + "_" + str(suffix)
                suffix += 1
            doc["nets"][network_name] = net
            doc["entry"] = network_name
            doc["output_node"] = ""
        elif kind == "fx":
            if not name.endswith(".py"):
                raise ValueError("chatFX modules use .py")
            improvement.validateFxSource(content)
            path = root / "fx" / name
            write_project_text(root, path, content)
        elif kind == "routes":
            routes = plain(
                pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False).to_dict("records")
            )
            doc["routes"] = normalize_routes(routes)
            available = {route["model"] for route in doc["routes"]}
            for role in ("assistant", "supervisor"):
                if doc["settings"].get(role) not in available:
                    doc["settings"][role] = next(iter(available)) if len(available) == 1 else ""
        elif kind == "atom":
            atom = json.loads(content)
            atoms.validate_atom(atom)
            atom_name = checked_name(Path(name).stem)
            original, suffix = atom_name, 2
            while atom_name in doc.setdefault("atoms", {}):
                atom_name = original[:90] + "_" + str(suffix)
                suffix += 1
            doc["atoms"][atom_name] = atom
        elif kind == "rankings":
            rankings = json.loads(content)
            if not isinstance(rankings, list):
                raise ValueError("Rankings must be a JSON list")
            for item in rankings:
                if not isinstance(item, dict) or not all(item.get(k) for k in ("model", "source", "date")):
                    raise ValueError("Each score needs model, score, source URL and date")
                if isinstance(item.get("score"), bool) or not isinstance(item.get("score"), (int, float)) or not math.isfinite(item["score"]):
                    raise ValueError("Benchmark scores must be finite numbers")
                if urlsplit(item["source"]).scheme not in {"http", "https"}:
                    raise ValueError("Benchmark source must be an http:// or https:// URL")
                datetime.fromisoformat(item["date"])
            doc["rankings"] = rankings
        else:
            raise ValueError("Unknown file kind")
        save_project(root, doc)
        return self.describe(identifier)

    def weld(self, identifier, upstream, downstream, node, start_key=None):
        root = self.project(identifier)
        doc = read_project(root)
        up = graph_from_network(doc['nets'][upstream])
        down = graph_from_network(doc['nets'][downstream])
        dependencies = active_network_names(doc | {'entry': downstream}) - {downstream}
        reserved = set(doc['nets'])
        aliases = {}
        for name in sorted(dependencies):
            base = f'{upstream}_{downstream}_{name}'[:90]
            alias, index = base, 2
            while alias in reserved:
                alias = f'{base}_{index}'
                index += 1
            reserved.add(alias)
            aliases[name] = alias

        def registry_references(graph):
            for _, data in graph.nodes(data=True):
                parameters = data.get('fx_params') or {}
                nested = parameters.get('chatNet')
                if data.get('fx') == 'expandIterable' and nested in aliases:
                    data['fx_params'] = parameters | {'chatNet': aliases[nested]}
            return graph

        registry_references(down)
        down.graph['chatNets'] = {
            aliases[name]: registry_references(graph_from_network(doc['nets'][name]))
            for name in sorted(dependencies)
        }
        for name, alias in aliases.items():
            down.graph['chatNets'][alias].graph['workbench_source'] = name
        graph = tb.nestChatNets(up, down, node, startKey=start_key or None)
        combined = doc['nets'][upstream] | network_from_graph(graph)
        # Keep the source CDENs and their children independently editable. Each
        # composed child is saved under its own project registry alias.
        for name, child in graph.graph.get('chatNets', {}).items():
            if name in doc['nets']:
                raise ValueError(f'Joined child CDEN name {name!r} already exists; rename the source CDEN before joining')
            original = child.graph.get('workbench_source')
            metadata = doc['nets'][original] if original is not None else {}
            doc['nets'][name] = metadata | network_from_graph(child) | {'pool': 'user'}
        doc['nets'][upstream] = combined
        save_project(root, doc)
        return self.describe(identifier) | {'rename_map': graph.graph['nested_node_map']}

    def apply_draft(self, identifier, run_id, index=None, network=None):
        with self.lock:
            return self._apply_draft(identifier, run_id, index, network)

    def _apply_draft(self, identifier, run_id, index=None, network=None):
        root = self.project(identifier)
        ledger_path = root / "runs" / checked_name(run_id) / "applications.json"
        ledger = read_json(ledger_path, {})
        key = str(index) if index is not None else "artifact"
        if key in ledger:
            return self.describe(identifier) | {"already_applied": True, **ledger[key]}
        doc = read_project(root)
        draft = read_json(root / "runs" / checked_name(run_id) / "draft.json")
        if draft is None:
            raise ValueError("No draft is available")
        metadata = draft
        if index is not None:
            draft = draft["revisions"][int(index)]
        if "network" in draft:
            name = checked_name(draft.get("name", "draft"))
            validate_network(draft["network"])
            original_name = name
            suffix = 2
            while name in doc["nets"]:
                name = checked_name(original_name[:90] + "_" + str(suffix))
                suffix += 1
            draft["network"]["pool"] = "user"
            doc["nets"][name] = draft["network"]
            doc["entry"] = name
            doc["output_node"] = draft.get("output_node") or ""
            # Validate nested references before saving an assistant-created graph.
            validate_project(doc, strict=True)
        elif "code" in draft:
            name = checked_name(draft.get("filename", "draft.py"))
            source_run = metadata.get("run_id", run_id)
            original_path = root / "runs" / checked_name(source_run) / "snapshot/fx" / name
            current_path = root / "fx" / name
            if current_path.exists() and (not original_path.exists() or current_path.read_bytes() != original_path.read_bytes()):
                raise ValueError("This Python module changed after the draft was created; review the current source before applying")
            result = self.upload(identifier, name, "fx", draft["code"])
            ledger[key] = {"applied_file": "fx/" + name}
            atomic_json(ledger_path, ledger)
            return result
        elif "prompt" in draft:
            chosen = network or metadata.get("target_network") or doc["entry"]
            node = next((n for n in doc["nets"].get(chosen, {}).get("nodes", []) if n["key"] == draft["node"]), None)
            if node is None:
                raise ValueError("The draft's target node no longer exists; regenerate against the current CDEN")
            origin = read_json(
                root / "runs" / checked_name(run_id) / "snapshot/project.json"
            )
            if metadata.get("run_id"):
                origin = read_json(
                    root
                    / "runs"
                    / checked_name(metadata["run_id"])
                    / "snapshot/project.json"
                )
            old = next(
                (
                    n
                    for n in (origin or {}).get("nets", {}).get(chosen, {}).get("nodes", [])
                    if n["key"] == draft["node"]
                ),
                None,
            )
            if old is not None and any(
                node.get(k, "") != old.get(k, "") for k in ("prompt", "persona")
            ):
                raise ValueError(
                    "This node changed after the draft was created. Review the newer prompt before applying a revision."
                )
            node.update(prompt=draft["prompt"], persona=draft["persona"])
        else:
            raise ValueError("This draft cannot be applied as a project revision")
        save_project(root, doc)
        result = self.describe(identifier)
        if "network" in draft:
            result["applied_network"] = name
        ledger[key] = {"applied_at": datetime.now(timezone.utc).isoformat()}
        if "network" in draft:
            ledger[key]["applied_network"] = name
        atomic_json(ledger_path, ledger)
        return result

    def export(self, identifier):
        root = self.project(identifier)
        destination = self.root / "_exports"
        destination.mkdir(exist_ok=True)
        output = destination / (identifier + ".zip")
        # Project exports intentionally omit API key literals and unrelated files.
        doc = read_project(root)
        doc = plain(doc)
        for route in doc["routes"]:
            route.pop("key", None)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "project.json", json.dumps(doc, ensure_ascii=False, indent=2)
            )
            archive.writestr("run.py", CLI_TEMPLATE)
            archive.writestr(
                "README.txt",
                "Install TabulAIrity 1.4.0.dev19, then python run.py. Edit project.json, cdens/*.json and fx/*.py. Use --trust-fx for your own Python modules. Credentials belong in environment variables.\n",
            )
            for dirname in ("inputs", "cdens", "fx", "atoms", "runs", "history"):
                for path in (root / dirname).rglob("*"):
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or "__pycache__" in path.parts
                        or path.name.endswith(".tmp")
                    ):
                        continue
                    relative = path.relative_to(root)
                    if path.name == "project.json":
                        saved = read_json(path)
                        for route in saved.get("routes", []):
                            route.pop("key", None)
                        archive.writestr(
                            str(relative),
                            json.dumps(saved, ensure_ascii=False, indent=2),
                        )
                    elif path.name == "model_routes.csv":
                        routes = pd.read_csv(path).drop(
                            columns=["key"], errors="ignore"
                        )
                        archive.writestr(str(relative), routes.to_csv(index=False))
                    else:
                        archive.write(path, str(relative))
            stream = io.StringIO()
            writer = csv.DictWriter(
                stream, fieldnames=["model", "route", "ip", "key_env"]
            )
            writer.writeheader()
            writer.writerows(
                {k: r.get(k, "") for k in writer.fieldnames} for r in doc["routes"]
            )
            archive.writestr("config/model_routes.csv", stream.getvalue())
        return output


class Handler(BaseHTTPRequestHandler):
    server_version = "TabulAIrityWorkbench"

    def log_message(self, *args):
        pass

    def send(self, value, status=200):
        payload = json.dumps(plain(value), ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def file(self, path, download=False):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("File not found")
        types = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
            ".csv": "text/csv",
            ".zip": "application/zip",
        }
        self.send_response(200)
        self.send_header(
            "Content-Type", types.get(path.suffix, "text/plain") + "; charset=utf-8"
        )
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("X-Content-Type-Options", "nosniff")
        if path.suffix in {".html", ".js", ".css"}:
            self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header(
                "Content-Disposition", 'attachment; filename="' + path.name + '"'
            )
        self.end_headers()
        with path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch()

    def do_PUT(self):
        self.dispatch()

    def do_DELETE(self):
        self.dispatch()

    def dispatch(self):
        app = self.server.app
        try:
            if self.headers.get("Host", "").split(":")[0] not in {
                "localhost",
                "127.0.0.1",
            }:
                return self.send({"error": "Loopback Host required"}, 403)
            url = urlsplit(self.path)
            parts = [p for p in url.path.split("/") if p]
            query = parse_qs(url.query)
            if not parts or parts[0] != "api":
                if self.command != "GET":
                    raise ValueError("Static files are read only")
                return self.file(
                    safe_path(STATIC, url.path.lstrip("/") or "index.html")
                )
            # Loopback service: same-origin write header blocks cross-site forms.
            if (
                self.command != "GET"
                and self.headers.get("X-Tabulairity") != "workbench"
            ):
                return self.send(
                    {"error": "Same-origin workbench header required"}, 403
                )
            body = {}
            if self.command != "GET":
                length = int(self.headers.get("Content-Length", "0"))
                if length > 16 * 1024 * 1024:
                    raise ValueError(
                        "Upload exceeds 16 MiB; place a larger file in the project inputs folder"
                    )
                body = json.loads(self.rfile.read(length) or "{}")
            if parts == ["api", "authoring-guide"]:
                if self.command != "GET":
                    return self.send({"error": "GET required"}, 405)
                return self.send({"guide": drafting.authoring_guide()})
            if parts == ["api", "projects"]:
                if self.command == "GET":
                    return self.send(
                        [
                            {
                                "id": p.name,
                                "title": read_json(p / "project.json").get(
                                    "title", p.name
                                ),
                            }
                            for p in sorted(app.root.iterdir())
                            if (p / "project.json").is_file()
                        ]
                    )
                return self.send(app.create(body.get("title", "New project")))
            if parts == ["api", "starter"]:
                if self.command != "POST":
                    return self.send({"error": "POST required"}, 405)
                return self.send(app.starter())
            if parts == ["api", "endpoints"]:
                return self.send(DEFAULT_ENDPOINTS)
            if len(parts) < 3 or parts[1] != "projects":
                raise FileNotFoundError("Endpoint not found")
            identifier = checked_name(parts[2])
            root = app.project(identifier)
            if len(parts) == 3:
                if self.command == "GET":
                    return self.send(app.describe(identifier))
                if self.command == "PUT":
                    return self.send(save_project(root, body["project"]))
            action = parts[3]
            if (
                action in {"upload", "discover", "weld", "start", "apply", "bundled", "validate", "atom-insert", "atom-save"}
                and self.command != "POST"
            ):
                return self.send({"error": "POST required"}, 405)
            if action == "bundled":
                return self.send(app.add_bundled(identifier))
            if action == "validate":
                return self.send(check_project(body.get("project") or read_project(root), root))
            if action in {"atom-insert", "atom-save"}:
                doc = body.get("project") or read_project(root)
                validate_project(doc)
                if action == "atom-save":
                    atom = atoms.from_network(doc, body["network"], body["title"], body["output"])
                    return self.send({"atom": atom})
                atom = doc.get("atoms", {}).get(body["atom"])
                if atom is None:
                    raise ValueError("Choose an atom")
                return self.send(atoms.insert_atom(doc, body["network"], atom, body["after"], body["instance"], body["bindings"]))
            if action == "input":
                doc = read_project(root)
                files = resolve_input_files(root, doc["input"])
                preview = []
                for _, file_name, file_row, record in iter_input_records(
                    root, doc["input"]
                ):
                    preview.append(record)
                    if len(preview) >= 20:
                        break
                return self.send(
                    {
                        "files": [str(p.relative_to(root)) for p in files],
                        "rows": preview,
                    }
                )
            if action == "upload":
                return self.send(
                    app.upload(identifier, body["name"], body["kind"], body["content"])
                )
            if action == "file":
                relative = query.get("name", [""])[0]
                if Path(relative).parts[0] not in {"inputs", "fx", "cdens", "atoms"}:
                    raise ValueError(
                        "Only project inputs, CDENs and chatFX can be opened here"
                    )
                return self.send(
                    {"content": safe_path(root, relative).read_text(encoding="utf-8")}
                )
            if action == "discover":
                found = discover_backends(body.get("endpoints", DEFAULT_ENDPOINTS))
                doc = read_project(root)
                found["recommendation"] = recommend_roles(
                    doc.get("routes", []) + found["routes"], doc.get("rankings", [])
                )
                return self.send(found)
            if action == "recommend-roles":
                return self.send(recommend_roles(body.get("routes", []), body.get("rankings", [])))
            if action == "weld":
                return self.send(
                    app.weld(identifier, body['upstream'], body['downstream'],
                             body['node'], body.get('start_key'))
                )
            if action == "start":
                return self.send(
                    app.start(identifier, body.get("operation", "run"), body)
                )
            if action == "apply":
                return self.send(
                    app.apply_draft(
                        identifier,
                        body["run_id"],
                        body.get("index"),
                        body.get("network"),
                    )
                )
            if action == "export":
                return self.file(app.export(identifier), download=True)
            if action == "runs":
                if len(parts) == 4:
                    return self.send(
                        [
                            {"run_id": p.name, **app.run_status(identifier, p.name)}
                            for p in sorted((root / "runs").glob("*"), reverse=True)
                            if p.is_dir()
                        ]
                    )
                run_id = checked_name(parts[4])
                run_root = root / "runs" / run_id
                if len(parts) > 5:
                    verb = parts[5]
                    if verb in {"control", "flag"} and self.command != "POST":
                        return self.send({"error": "POST required"}, 405)
                    if verb == "control":
                        return self.send(
                            app.control(identifier, run_id, body["action"])
                        )
                    if verb == "flag":
                        with app.lock:
                            flags = read_json(run_root / "flags.json", {})
                            if body.get("remove"):
                                flags.pop(body["row_id"], None)
                            else:
                                flags[body["row_id"]] = flags.get(
                                    body["row_id"], {}
                                ) | {
                                    "row_id": body["row_id"],
                                    "reason": body.get("reason", "Flagged by user"),
                                    "node": body.get("node", ""),
                                    "author": "user",
                                }
                            atomic_json(run_root / "flags.json", flags)
                        return self.send(flags)
                    if verb == "draft":
                        saved_draft = read_json(run_root / "draft.json")
                        return self.send(
                            {
                                "draft": saved_draft,
                                "baseline": draft_baseline(root, run_id, saved_draft) if saved_draft else None,
                                "applications": read_json(run_root / "applications.json", {}),
                                "status": app.run_status(identifier, run_id),
                            }
                        )
                    if verb == "authoring-context":
                        if self.command != "GET":
                            return self.send({"error": "GET required"}, 405)
                        path = run_root / "authoring-context.json"
                        if not path.is_file():
                            raise FileNotFoundError("No CDEN authoring context for this run")
                        return self.file(path, download=True)
                    if verb == "constructor":
                        if self.command != "GET":
                            return self.send({"error": "GET required"}, 405)
                        return self.file(run_root / "constructor-attempts.json", download=True)
                    if verb == "transcript":
                        if self.command != "GET":
                            return self.send({"error": "GET required"}, 405)
                        return self.file(run_root / "events.jsonl", download=True)
                    if verb == "csv":
                        csv_product_file(
                            run_root / "table.jsonl", run_root / "output.csv"
                        )
                        return self.file(run_root / "output.csv", download=True)
                after = max(0, int(query.get("after", ["0"])[0]))
                events = []
                has_more = False
                cursor = after
                if (run_root / "events.jsonl").exists():
                    with (run_root / "events.jsonl").open("rb") as f:
                        f.seek(after)
                        for _ in range(60):
                            line = f.readline()
                            if not line:
                                break
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                break
                            cursor = f.tell()
                            event["_cursor"] = cursor
                            events.append(event)
                        has_more = bool(f.read(1))
                definition = read_json(run_root / "snapshot/project.json", {})
                entry = definition.get("entry")
                offset = max(0, int(query.get("offset", ["0"])[0]))
                return self.send(
                    {
                        "status": app.run_status(identifier, run_id),
                        "events": events,
                        "cursor": cursor,
                        "has_more_events": has_more,
                        "definition": {"entry": entry, "output_node": definition.get("output_node", ""),
                                       "nodes": [n["key"] for n in definition.get("nets", {}).get(entry, {}).get("nodes", [])]},
                        "request": read_json(run_root / "operation.json", {}).get("args"),
                        "rows": read_jsonl(run_root / "table.jsonl", offset, 100),
                        "flags": read_json(run_root / "flags.json", {}),
                        "draft": read_json(run_root / "draft.json"),
                    }
                )
            raise FileNotFoundError("Endpoint not found")
        except (KeyError, ValueError, TypeError, SyntaxError, StopIteration) as exc:
            self.send({"error": str(exc)}, 400)
        except FileNotFoundError as exc:
            self.send({"error": str(exc)}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.send({"error": str(exc)}, 500)


def serve(root, host="127.0.0.1", port=8765, open_browser=True):
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Workbench is a local application; use a loopback host")
    app = Workbench(root)
    server = ThreadingHTTPServer((host, port), Handler)
    server.app = app
    url = f"http://{host}:{server.server_port}/"
    print("TabulAIrity Workbench: " + url, flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for process, paused, stopped in app.processes.values():
            stopped.set()
            paused.clear()
        server.server_close()


def project_cli(project_root, argv=None):
    parser = argparse.ArgumentParser(description="Run an editable TabulAIrity project")
    parser.add_argument("--input")
    parser.add_argument("--trust-fx", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(project_root).resolve()
    doc = read_project(root)
    # CDEN JSON files are authoritative for external project edits.
    for name in doc["nets"]:
        path = root / "cdens" / f"{name}.json"
        if path.exists():
            doc["nets"][name] = read_json(path)
    if args.workers:
        doc["settings"]["workers"] = args.workers
    if args.input:
        sources = sorted(
            Path(p).resolve() for p in glob.glob(args.input) if Path(p).is_file()
        )
        if not sources:
            raise ValueError(f"No files match {args.input!r}")
        for source in sources:
            target = root / "inputs" / source.name
            target.parent.mkdir(exist_ok=True)
            if source != target:
                shutil.copy2(source, target)
        doc["input"] = "inputs/" + Path(args.input).name
    save_project(root, doc)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-"
        + secrets.token_hex(3)
    )
    run_root = root / "runs" / run_id
    run_root.mkdir(parents=True)
    snapshot_project(root, run_root)
    context = mp.get_context("spawn")
    _run_worker(
        str(root),
        str(run_root),
        "run",
        vars(args),
        context.Event(),
        context.Event(),
        context.BoundedSemaphore(doc["settings"]["workers"]),
    )
    state = read_json(run_root / "status.json")
    print(json.dumps(state, indent=2))
    if args.output and (run_root / "output.csv").exists():
        shutil.copy2(run_root / "output.csv", args.output)
    if state["status"] != "completed":
        raise SystemExit(1)
    print(f"Output: {run_root / 'output.csv'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local TabulAIrity workbench")
    parser.add_argument("command", choices=["gui", "run"], nargs="?", default="gui")
    parser.add_argument("project", nargs="?")
    parser.add_argument("--projects", default=str(Path.home() / "TabulAIrityProjects"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args, extra = parser.parse_known_args(argv)
    if args.command == "gui":
        serve(args.projects, port=args.port, open_browser=not args.no_browser)
    else:
        if not args.project:
            parser.error("run requires a project directory")
        project_cli(args.project, extra)


if __name__ == "__main__":
    main()
