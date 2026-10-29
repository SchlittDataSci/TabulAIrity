"""Reusable CDEN interfaces, compiled to ordinary core graphs, with no inference here."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from . import core as tb

ROLE_INPUTS = {"assistant_model", "supervisor_model"}
SUFFIXES = ("", "_raw", "_prompt", "_item", "_expansion_errors")
KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,59}$")


@tb.parallelFx
def atom_input(response, chatVars, *, mode, variable=None, value=None):
    if mode == "literal":
        return deepcopy(value)
    if mode != "variable" or variable not in chatVars:
        raise ValueError(f"Atom input {variable!r} is missing; bind it to an input column or an earlier step")
    return deepcopy(chatVars[variable])


@tb.parallelFx
def atom_output(response, chatVars, *, outputs, fields=None, nonempty=None):
    missing = [name for name, source in outputs.items() if source not in chatVars]
    if missing:
        raise ValueError("Atom did not produce required outputs: " + ", ".join(missing))
    values = {name: deepcopy(chatVars[source]) for name, source in outputs.items()}
    for name in nonempty or []:
        if not isinstance(values[name], str) or not values[name].strip():
            raise ValueError(f"Atom output {name!r} must contain text")
    for name, selected in (fields or {}).items():
        value = values[name]
        if isinstance(value, pd.DataFrame):
            values[name] = value.reindex(columns=selected)
        elif isinstance(value, list):
            values[name] = [{key: row.get(key) for key in selected} if isinstance(row, dict) else row for row in value]
        elif isinstance(value, dict):
            values[name] = {key: value.get(key) for key in selected}
        else:
            raise ValueError(f"Atom output {name!r} needs structured data for its selected fields")
    return next(iter(values.values())) if len(values) == 1 else values


@tb.parallelFx
def atom_json(response, chatVars, *, fields):
    value = tb.insertJson(response)
    types = {"text": str, "list": list, "object": dict}
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object for this atom step")
    for name, kind in fields.items():
        if not isinstance(value.get(name), types[kind]):
            raise ValueError(f"Atom response needs {name!r} as {kind}")
    return value


RUNTIME_FX = {"atom_input": atom_input, "atom_output": atom_output, "atom_json": atom_json}


def bundled_atoms():
    return json.loads((Path(__file__).with_name("bundled") / "atoms.json").read_text(encoding="utf-8"))


def _graphs(atom):
    from .workbench import graph_from_network
    return {name: graph_from_network(net) for name, net in atom["networks"].items()}


def validate_atom(atom):
    from .workbench import validate_network, checked_name
    if not isinstance(atom, dict) or atom.get("schema_version") != 1 or atom.get("kind") != "cden_atom":
        raise ValueError("Use an atom JSON file with kind cden_atom and schema_version 1")
    if not isinstance(atom.get("title"), str) or not atom["title"].strip():
        raise ValueError("Give the atom a title")
    nets = atom.get("networks")
    if not isinstance(nets, dict) or atom.get("entry") not in nets:
        raise ValueError("Select the atom's entry CDEN")
    inputs, outputs = atom.get("inputs"), atom.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict) or not outputs:
        raise ValueError("Atoms need named inputs and at least one named output")
    for port, spec in inputs.items():
        if not isinstance(port, str) or not port.strip() or any(c in port for c in "[]\n") or port in ROLE_INPUTS | tb.RESERVED_NODE_NAMES or not isinstance(spec, dict):
            raise ValueError("Input names must be static variable names; model roles are configured separately")
    for name, net in nets.items():
        checked_name(name)
        validate_network(net)
        symbols = [str(n["key"]) + suffix for n in net["nodes"] for suffix in SUFFIXES]
        symbols += [f"{e['source']}-{e['target']}" for e in net["edges"]]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Atom step names collide with generated response or edge variables; rename those steps")
        if set(inputs) & set(symbols):
            raise ValueError("An atom input must have a different name from its steps and generated response variables")
        for item in [*net["nodes"], *net["edges"]]:
            if item.get("fx") == "atom_json":
                fields = (item.get("fx_params") or {}).get("fields")
                if not isinstance(fields, dict) or any(kind not in {"text", "list", "object"} for kind in fields.values()):
                    raise ValueError("Atom JSON fields must declare text, list or object")
    graphs = _graphs(atom)
    root_symbols = [str(n["key"]) + suffix for n in nets[atom["entry"]]["nodes"] for suffix in SUFFIXES]
    root_symbols += [port + suffix for port in inputs for suffix in SUFFIXES]
    if len(root_symbols) != len(set(root_symbols)):
        raise ValueError("Atom inputs collide with generated response variables; rename the inputs")
    refs = tb.chatNetInputKeys(graphs[atom["entry"]], fxStore=RUNTIME_FX, chatNets=graphs)
    if refs is None:
        raise ValueError("This CDEN cannot yet be isolated as an atom: use static references and built-in response handlers; variable-writing functions and dynamic references need an explicit isolation adapter")
    extra = refs - set(inputs) - ROLE_INPUTS
    if extra:
        raise ValueError("Declare these atom inputs: " + ", ".join(sorted(extra)))
    keys = set(graphs[atom["entry"]])
    for port, source in outputs.items():
        if not isinstance(port, str) or not port.strip() or not isinstance(source, str) or source not in keys:
            raise ValueError("Each named output must select a step in the entry CDEN")
    fields = atom.get("output_fields", {})
    if not isinstance(fields, dict) or any(port not in outputs or not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names) for port, names in fields.items()):
        raise ValueError("Output fields must select named fields for an existing atom output")
    if not isinstance(atom.get("output_nonempty", []), list) or any(name not in outputs for name in atom.get("output_nonempty", [])):
        raise ValueError("Nonempty output checks must name existing atom outputs")
    return atom


def from_network(doc, name, title, output):
    """Capture the complete static CDEN family, independent of future source edits."""
    from .workbench import active_network_names
    selected = active_network_names(doc | {"entry": name})
    nets = {key: deepcopy(doc["nets"][key]) for key in sorted(selected)}
    atom = {"schema_version": 1, "kind": "cden_atom", "title": title,
            "description": doc["nets"][name].get("description", ""), "entry": name,
            "networks": nets, "inputs": {}, "outputs": {"result": output}, "version": 1}
    graphs = _graphs(atom)
    refs = tb.chatNetInputKeys(graphs[name], fxStore=RUNTIME_FX, chatNets=graphs)
    if refs is None:
        raise ValueError("This CDEN uses dynamic references or variable-writing functions and cannot yet be isolated as an atom")
    atom["inputs"] = {key: {"label": key.replace("_", " "), "description": ""} for key in sorted(refs - ROLE_INPUTS)}
    def selected_fields(net_name, key, seen=frozenset()):
        if (net_name, key) in seen:
            return None
        node = next((n for n in nets[net_name]["nodes"] if n["key"] == key), {})
        params = node.get("fx_params") or {}
        selected = params.get("output_keys")
        if node.get("fx") != "expandIterable" or not selected:
            return None
        if not params.get("flatten"):
            return list(selected)
        # output_keys selects the pre-flatten child record. The final fields
        # come from that record plus the rows returned by its flatten step.
        nested = selected_fields(params["chatNet"], params["flatten"], seen | {(net_name, key)})
        return list(dict.fromkeys([key for key in selected if key != params["flatten"]] + nested)) if nested else None
    fields = selected_fields(name, output)
    if fields:
        atom["output_fields"] = {"result": fields}
    return validate_atom(atom)


def _produced(net):
    return {str(n["key"]) + suffix for n in net["nodes"] for suffix in SUFFIXES} | {
        f"{e['source']}-{e['target']}" for e in net["edges"]}


def _rewrite(text, mapping):
    if not isinstance(text, str):
        return text
    return re.sub(r"\[([^\[\]\n]+)\]", lambda m: "[" + mapping.get(m[1], m[1]) + "]", text)


def insert_atom(doc, network, atom, after, instance, bindings):
    """Return a copy; namespace the full inserted graph and freeze its provenance."""
    from .workbench import graph_from_network, network_from_graph, validate_network
    validate_atom(atom)
    if not isinstance(instance, str) or not KEY.fullmatch(instance):
        raise ValueError("Name the atom with a letter followed by letters, numbers or underscores (up to 60 characters)")
    if network not in doc["nets"]:
        raise ValueError("Choose the CDEN to receive the atom")
    if not isinstance(bindings, dict) or set(bindings) != set(atom["inputs"]):
        raise ValueError("Bind every atom input")
    source = doc["nets"][network]
    if after not in {n["key"] for n in source["nodes"]}:
        raise ValueError("Choose the step before this atom")
    up = graph_from_network(source)
    down = _graphs(atom)[atom["entry"]]
    for port in atom["inputs"]:
        down.add_node(port, fx="atom_input", prompt="", persona="", tokens=1)
    occupied = _produced(source) | ROLE_INPUTS | tb.RESERVED_NODE_NAMES
    for binding in bindings.values():
        if not isinstance(binding, dict) or binding.get("mode") not in {"variable", "literal"}:
            raise ValueError("Each input needs an earlier variable or a fixed value")
        if binding["mode"] == "variable":
            key = binding.get("variable")
            if not isinstance(key, str) or not key or any(c in key for c in "[]\n"):
                raise ValueError("Choose an input column or step name")
            occupied.add(key)
    base, count = instance, 2
    # Reserve edges too: an existing step named Start-block__payload must not
    # be overwritten by the new Start → block__payload edge result.
    while any(key == instance or key.startswith(instance + "__") or ("-" + instance + "__") in key or key.endswith("-" + instance) or key in {instance + s for s in SUFFIXES} for key in occupied) or any(key.startswith(instance + "__") for key in doc["nets"]):
        instance = base[:54] + "_" + str(count)
        count += 1
    prefix = instance + "__"
    aliases = {name: (prefix + name if len(prefix + name) <= 100 else (prefix + name)[:90] + "_" + hashlib.sha256(name.encode()).hexdigest()[:8])
               for name in atom["networks"] if name != atom["entry"]}
    # The child Start remains structural inside its private registry scope.
    def alias_children(graph):
        for _, data in graph.nodes(data=True):
            params = data.get("fx_params") or {}
            if data.get("fx") == "expandIterable" and params.get("chatNet") in aliases:
                data["fx_params"] = params | {"chatNet": aliases[params["chatNet"]]}
        return graph
    alias_children(down)
    down.graph["chatNets"] = {aliases[name]: alias_children(graph_from_network(net))
                              for name, net in atom["networks"].items() if name in aliases}
    combined = tb.nestChatNets(up, down, after, startKey=prefix + "Start", prefix=prefix)
    mapping = combined.graph["nested_node_map"]
    refs = {old + suffix: new + suffix for old, new in mapping.items() for suffix in SUFFIXES}
    refs.update({f"{a}-{b}": f"{mapping[a]}-{mapping[b]}" for a, b in down.edges()})
    for old, key in mapping.items():
        data = combined.nodes[key]
        data["model"] = _rewrite(data.get("model"), refs)
        params = deepcopy(data.get("fx_params") or {})
        if data.get("fx") == "atom_input" and params.get("mode") == "variable":
            params["variable"] = refs.get(params.get("variable"), params.get("variable"))
        if data.get("fx") == "atom_output":
            params["outputs"] = {name: refs.get(value, value) for name, value in params["outputs"].items()}
        data["fx_params"] = params
    for child in combined.graph.get("chatNets", {}).values():
        local = {str(node) + suffix for node in child for suffix in SUFFIXES} | {f"{a}-{b}" for a, b in child.edges()}
        inherited = {old: new for old, new in refs.items() if old not in local}
        for _, data in child.nodes(data=True):
            data["model"] = _rewrite(data.get("model"), inherited)
            params = deepcopy(data.get("fx_params") or {})
            if data.get("fx") == "atom_input" and params.get("mode") == "variable":
                params["variable"] = inherited.get(params.get("variable"), params.get("variable"))
            if data.get("fx") == "atom_output":
                params["outputs"] = {name: inherited.get(value, value) for name, value in params["outputs"].items()}
            data["fx_params"] = params
    for port, binding in bindings.items():
        key = mapping[port]
        combined.nodes[key].update(fx_params=deepcopy(binding),
                                   prompt="recall: [" + binding["variable"] + "]" if binding["mode"] == "variable" else "recall:")
        combined.add_edge(after, key, fx="null", prompt="")
        combined.graph["forced_dependencies"].append((key, mapping["Start"]))
    outputs = {name: mapping[key] for name, key in atom["outputs"].items()}
    combined.add_node(instance, fx="atom_output", persona="", tokens=1,
                      prompt="recall: " + " ".join("[" + key + "]" for key in outputs.values()),
                      fx_params={"outputs": outputs, "fields": deepcopy(atom.get("output_fields", {})), "nonempty": deepcopy(atom.get("output_nonempty", []))})
    for key in set(outputs.values()):
        combined.add_edge(key, instance, fx="null", prompt="")
        combined.graph["forced_dependencies"].append((key, instance))
    result = deepcopy(doc)
    for name, child in combined.graph.get("chatNets", {}).items():
        if name in result["nets"]:
            raise ValueError("Nested atom name collision; choose another atom name")
        result["nets"][name] = network_from_graph(child) | {"pool": "user", "atom_owner": instance}
    root = source | network_from_graph(combined)
    # Welding must not erase the parent's coordinates or editor metadata.
    originals = {node["key"]: node for node in source["nodes"]}
    root["nodes"] = [deepcopy(originals.get(node["key"], node)) for node in root["nodes"]]
    original_edges = {(edge["source"], edge["target"]): edge for edge in source["edges"]}
    root["edges"] = [deepcopy(original_edges.get((edge["source"], edge["target"]), edge)) for edge in root["edges"]]
    new_symbols = [node["key"] + suffix for node in root["nodes"] if node["key"] not in originals for suffix in SUFFIXES]
    new_symbols += [f"{edge['source']}-{edge['target']}" for edge in root["edges"] if (edge["source"], edge["target"]) not in original_edges]
    if len(new_symbols) != len(set(new_symbols)) or set(new_symbols) & occupied:
        raise ValueError("This atom's generated response names collide; rename its steps or choose a different instance name")
    members = list(mapping.values()) + [instance]
    # Positioned near its parent; internal steps are laid out when expanded.
    origin = next(n for n in source["nodes"] if n["key"] == after)
    for index, node in enumerate(root["nodes"]):
        if node["key"] in members:
            node.update(x=origin.get("x", 300) + 210 + (index % 3) * 190,
                        y=origin.get("y", 240) + (index // 3) * 95)
    fingerprint = hashlib.sha256(json.dumps(atom, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    group = {"key": instance, "title": atom["title"], "version": atom.get("version", 1),
             "definition_sha256": fingerprint, "members": members, "node_map": mapping,
             "networks": list(combined.graph.get("chatNets", {})), "bindings": deepcopy(bindings),
             "input_labels": {name: spec.get("label", name) for name, spec in atom["inputs"].items()},
             "outputs": outputs, "collapsed": True, "model_steps": sum(
                 bool(n.get("persona")) and not str(n.get("prompt", "")).startswith("recall:")
                 for n in root["nodes"] if n["key"] in members)}
    root["atom_instances"] = deepcopy(source.get("atom_instances", [])) + [group]
    validate_network(root)
    result["nets"][network] = root
    return {"project": result, "instance": instance}
