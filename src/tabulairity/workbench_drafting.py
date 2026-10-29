"""Workbench CDEN context and validation; core inference/execution stays unchanged."""

from copy import deepcopy
import hashlib
from itertools import islice
import json
from pathlib import Path
import re

from . import core as tb
from . import selfimprovement as improvement
from . import workbench_constructor as constructor

BUNDLED = Path(__file__).with_name("bundled")
MAX_PROMPT_CHARS = 20000
MAX_REQUEST_CHARS = 6000


def authoring_guide():
    return (BUNDLED / "CDEN-AUTHORING.md").read_text(encoding="utf-8")


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def definition(network):
    """Preserve executable example content, omitting display-only metadata."""
    return {k: deepcopy(network[k]) for k in ("nodes", "edges", "forced_dependencies") if k in network}


def input_contract(root, doc):
    from .workbench import iter_input_records

    fields = {}
    try:
        for _, _, _, record in islice(iter_input_records(root, doc.get("input", "")), 3):
            for key, value in record.items():
                if len(fields) >= 64 and key not in fields:
                    continue
                fields.setdefault(key, set()).add(type(value).__name__)
        return {"source": doc.get("input"), "fields": {k: sorted(v) for k, v in fields.items()},
                "scope": "Types observed in up to three current input rows; values omitted. The requested new task may need different inputs."}
    except (ValueError, OSError, TypeError) as exc:
        return {"source": doc.get("input"), "fields": {}, "unavailable": str(exc)[:240]}


def prepare_context(doc, network_name, request, root, fx_names=()):
    """Use real examples on every call; add project examples only within a budget."""
    from .workbench import graph_from_network

    if not isinstance(request, str) or not request.strip():
        raise ValueError("Describe the CDEN task")
    if len(request) > MAX_REQUEST_CHARS:
        raise ValueError(f"CDEN requests can contain at most {MAX_REQUEST_CHARS} characters; shorten the task description")
    guide = authoring_guide()
    samples = json.loads((BUNDLED / "samples.json").read_text(encoding="utf-8"))
    patterns = json.loads((BUNDLED / "authoring-patterns.json").read_text(encoding="utf-8"))
    for pattern in patterns.values():
        for node in pattern.get("network", {}).get("nodes", []):
            if node.get("persona"):
                node["model"] = doc.get("settings", {}).get("assistant", improvement.DEFAULT_MODEL)
    model_step = deepcopy(improvement.SCHEMA_EXAMPLE)
    for node in model_step["nodes"]:
        if node.get("persona"):
            node["model"] = doc.get("settings", {}).get("assistant", improvement.DEFAULT_MODEL)
    context = {
        "request": request,
        "input_contract": input_contract(root, doc),
        "model_aliases": sorted({r["model"] for r in doc.get("routes", []) if r.get("model")}),
        "preferred_model": doc.get("settings", {}).get("assistant", improvement.DEFAULT_MODEL),
        "available_functions": sorted(set(tb.baseFx) | set(fx_names)),
        "examples": [{"source": "bundled/samples.json", "networks": {n: definition(v) for n, v in samples.items()}},
                     {"source": "model-step abstraction", "network": model_step}],
        "provided_cden_abstractions": patterns,
        "available_project_cdens": [],
        "current_network_name": network_name,
        "current_network": definition(doc["nets"][network_name]),
        "context_notes": [],
    }
    if len(compact(context["current_network"])) > 3500:
        current = context["current_network"]
        context["current_network"] = {
            "abstraction": True,
            "nodes": [{k: (v[:160] if isinstance(v, str) else v) for k, v in n.items()
                       if k in {"key", "fx", "prompt", "persona"}} for n in current["nodes"][:20]],
            "edges": current.get("edges", [])[:24],
            "total_nodes": len(current["nodes"]), "total_edges": len(current.get("edges", [])),
        }
        context["context_notes"].append("Current graph abstracted for context budget; do not assume omitted details.")
    words = set(re.findall(r"[a-z0-9]+", request.lower()))
    names = sorted(doc["nets"], key=lambda n: (
        n != network_name,
        -len(words & set(re.findall(r"[a-z0-9]+", n.lower() + " " + doc["nets"][n].get("description", "")))), n))
    graphs = {}
    for name in names:
        try:
            graphs[name] = graph_from_network(doc["nets"][name])
        except (ValueError, TypeError, KeyError):
            pass
    for name in names[:32]:
        net = doc["nets"][name]
        try:
            inputs = tb.chatNetInputKeys(graphs.get(name), chatNets=graphs)
        except (ValueError, TypeError, KeyError):
            inputs = None
        context["available_project_cdens"].append({
            "name": name, "pool": net.get("pool", "user"),
            "inputs": sorted(inputs) if inputs is not None else "Not statically known",
            "outputs": [n["key"] for n in net.get("nodes", [])][:20],
        })
    if len(names) > 32:
        context["context_notes"].append(f"Showing 32 of {len(names)} project CDEN contracts, prioritised by current selection and request terms.")
    # Preserve full project sample definitions, including their nested children,
    # as space permits. Bundled examples above remain available in older projects.
    project_examples = {}
    sources = ["bundled:" + name for name in samples] + ["abstraction:model-step"] + ["provided:" + p["source"] for p in patterns.values()]
    for name in names:
        net = doc["nets"][name]
        if net.get("pool") != "samples" or (name in samples and definition(net) == definition(samples[name])):
            continue
        family, pending = {}, [name]
        while pending:
            child = pending.pop()
            if child in family or child not in doc["nets"]:
                continue
            family[child] = definition(doc["nets"][child])
            pending.extend(n.get("fx_params", {}).get("chatNet") for n in family[child]["nodes"]
                           if n.get("fx") == "expandIterable" and isinstance((n.get("fx_params") or {}).get("chatNet"), str))
        proposed = project_examples | family
        if len(compact(proposed)) > 3500:
            context["context_notes"].append(f"Project sample {name} omitted: its complete child family exceeds the sample budget.")
            continue
        project_examples = proposed
    if project_examples:
        context["examples"].append({"source": "project sample pool", "networks": project_examples})
        sources.extend("project:" + name for name in project_examples)
    prefix = (
        "Draft a cden artifact for this task. Learn the CDEN contract from the guide and supplied examples. "
        "Adapt their structures to the requested domain; do not guess what a CDEN means. "
        "The current graph and examples are reference material; the request defines the new task. "
        "Only names under available_project_cdens are reusable nested graphs. "
        "Return the single JSON object specified by the guide. Do not repeat the context or examples. "
        "Escape double quotes and line breaks within prompt strings; use \\n for a line break.\n\n"
    )
    prompt = prefix + guide + "\n\nCDEN CONTEXT (JSON):\n" + compact(context)
    if len(prompt) > MAX_PROMPT_CHARS:
        # Keep the explicit task, whole guide, and complete bundled examples.
        context["examples"] = context["examples"][:2]
        sources = [s for s in sources if not s.startswith("project:")]
        context["available_project_cdens"] = context["available_project_cdens"][:12]
        context["context_notes"].append("Optional project samples and registry entries reduced to fit the prompt budget.")
        prompt = prefix + guide + "\n\nCDEN CONTEXT (JSON):\n" + compact(context)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("CDEN context exceeds its prompt budget; select a smaller reference graph or shorten the request")
    return {"prompt": prompt, "context": context, "summary": {
        "guide": "CDEN-AUTHORING.md", "guide_sha256": hashlib.sha256(guide.encode()).hexdigest(),
        "examples": sources, "prompt_chars": len(prompt), "prompt_char_limit": MAX_PROMPT_CHARS,
        "notes": context["context_notes"],
    }}


def draft_cden(prepared, doc, *, model, fxStore=None, **kwargs):
    """Validate against the project before exposing a draft as ready for review."""
    from .workbench import checked_name, validate_project

    workflow = kwargs.get("workflow")
    if workflow and not any("request" in tb.extractChatVars(n.get("prompt", ""))
                            for n in workflow.get("nodes", [])):
        raise ValueError("The CDEN drafting workflow must consume [request] in a node prompt so it receives the authoring guide and examples")

    def validate(draft):
        name = checked_name(draft["name"])
        if not isinstance(draft.get("rationale"), str):
            raise ValueError("Include a brief rationale with input assumptions and the final output node")
        candidate = deepcopy(doc)
        # The applied graph receives an unused name too; do not mask an existing
        # child with a draft that happens to propose the same name.
        original, suffix = name, 2
        while name in candidate["nets"]:
            name = original[:90] + "_" + str(suffix)
            suffix += 1
        candidate["nets"][name] = deepcopy(draft["network"]) | {"pool": "user"}
        candidate["entry"] = name
        candidate["output_node"] = draft.get("output_node") or ""
        if not isinstance(candidate["output_node"], str):
            raise ValueError("output_node must be a node key or an empty string for all outputs")
        validate_project(candidate, strict=True)
        known = set(tb.baseFx) | set(fxStore or {})
        from .workbench import active_network_names
        for active in active_network_names(candidate):
            network = candidate["nets"][active]
            for item in [*network.get("nodes", []), *network.get("edges", [])]:
                if item.get("fx", "null") not in known:
                    raise ValueError(f"Unknown or disabled chatFX {item.get('fx')!r}; use an available function")
        aliases = {r["model"] for r in doc.get("routes", []) if r.get("model")}
        for node in draft["network"]["nodes"]:
            if node.get("persona") and not str(node.get("prompt", "")).startswith("recall:"):
                alias = node.get("model", tb.modelName)
                alias = {"[assistant_model]": doc.get("settings", {}).get("assistant"),
                         "[supervisor_model]": doc.get("settings", {}).get("supervisor")}.get(alias, alias)
                if alias not in aliases:
                    raise ValueError(f"{node['key']}: select an available model alias")

    draft = constructor.run_constructor(prepared["prompt"], model, validate, fxStore=fxStore, **kwargs)
    draft["authoring_context"] = prepared["summary"]
    return draft
