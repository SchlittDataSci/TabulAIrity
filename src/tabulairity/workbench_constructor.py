"""CDEN response decoding and bounded repair, outside the unchanged core parser."""

import ast
from copy import deepcopy
import json
import re

from . import core as tb


class CdenSyntaxError(ValueError):
    def __init__(self, message, *, kind="syntax", line=None, column=None):
        super().__init__(message)
        self.kind, self.line, self.column = kind, line, column


def _without_trailing_commas(text):
    out, quoted, escaped = [], False, False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "," and re.match(r"\s*[}\]]", text[index + 1:]):
            continue
        out.append(char)
    return "".join(out)


def _decode(text):
    errors = []
    for candidate, notes in [(text, []), (_without_trailing_commas(text), ["Removed trailing JSON commas"] )]:
        try:
            value = json.loads(candidate, strict=False, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON number {value}")))
            if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
                value = json.loads(value, strict=False)
                notes = notes + ["Decoded a JSON-encoded response"]
            if not isinstance(value, (dict, list)):
                raise ValueError("Return a CDEN object or a list of CDEN rows")
            # Re-serialisation escapes literal controls without editing prompt text.
            json.dumps(value, allow_nan=False)
            if any(char in text for char in ("\n", "\t", "\r")):
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    notes = notes + ["Escaped literal control characters in JSON strings"]
            return value, notes
        except (ValueError, TypeError) as exc:
            errors.append(exc)
    try:
        value = ast.literal_eval(text)
        if not isinstance(value, (dict, list)):
            raise ValueError("Return a CDEN object or row list")
        # Refuse sets, tuples, non-text keys and other Python-only data recursively.
        def json_data(item):
            if isinstance(item, dict):
                return all(isinstance(k, str) and json_data(v) for k, v in item.items())
            if isinstance(item, list):
                return all(json_data(v) for v in item)
            return item is None or isinstance(item, (str, int, float, bool))
        if not json_data(value):
            raise ValueError("CDEN data must use JSON-compatible objects, lists and values")
        json.dumps(value, allow_nan=False)
        return value, ["Converted a Python literal to JSON; no Python code was executed"]
    except (ValueError, SyntaxError, TypeError):
        error = next((e for e in errors if isinstance(e, json.JSONDecodeError)), errors[0])
        raise CdenSyntaxError(str(error), line=getattr(error, "lineno", None), column=getattr(error, "colno", None)) from error


def _outer_container(text):
    """Keep the first complete outer container; never salvage a truncated interior."""
    match = re.search(r"\{|\[\s*(?:\{|\[|[\"']|\])", text)
    if not match:
        raise CdenSyntaxError("The response contains no CDEN object or row list")
    start, stack, quote, escaped = match.start(), [], None, False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack.pop() != ("]" == char and "[" or "{"):
                raise CdenSyntaxError("The response has mismatched JSON brackets")
            if not stack:
                remainder = text[index + 1:].strip()
                if re.search(r"(?:^|\n)\s*[\[{]", remainder):
                    raise CdenSyntaxError("The response contains more than one container; return one CDEN", kind="ambiguous")
                return text[start:index + 1]
    raise CdenSyntaxError("The response stopped before its CDEN was complete. Raise the constructor output limit or request a smaller graph.", kind="incomplete")


def parse_response(response):
    """Decode complete model output with only lossless syntax normalisation."""
    if isinstance(response, (dict, list)):
        return deepcopy(response), []
    if not isinstance(response, str) or not response.strip():
        raise CdenSyntaxError("The constructor returned no CDEN text", kind="empty")
    text = response.lstrip("\ufeff").strip()
    try:
        return _decode(text)
    except CdenSyntaxError:
        pass
    # The core helper preserves reasoning tags when they are inside quoted data.
    text = tb._stripChatReasoning(text).strip()
    fence = re.match(r"^```(?:json|python)?\s*\n?", text, re.IGNORECASE)
    if fence:
        text = text[fence.end():]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    candidate = _outer_container(text)
    value, notes = _decode(candidate)
    return value, (["Removed response framing"] if candidate != response.strip() else []) + notes


def normalize_draft(value):
    """Accept the workbench envelope and the core's explicit CDEN formats."""
    from .workbench import network_from_graph
    import pandas as pd

    notes = []
    if isinstance(value, list):
        draft, network = {}, deepcopy(value)
    elif isinstance(value, dict) and "network" in value:
        draft, network = deepcopy(value), deepcopy(value["network"])
    elif isinstance(value, dict) and "nodes" in value and "edges" in value:
        draft = {k: deepcopy(value[k]) for k in ("name", "rationale", "output_node") if k in value}
        network = {k: deepcopy(v) for k, v in value.items() if k not in {"name", "rationale", "output_node"}}
        notes.append("Wrapped a bare CDEN definition for workbench review")
    else:
        raise ValueError("Return a network with nodes and edges, or explicit node/edge rows")
    if isinstance(network, list):
        if not network or any(not isinstance(row, dict) or row.get("type") not in {"node", "edge"} for row in network):
            raise ValueError("Each CDEN row needs type node or edge")
        for row in network:
            if row["type"] == "node" and row.get("persona") and not row.get("model"):
                row["model"] = "[assistant_model]"
        network = network_from_graph(tb.buildChatNet(pd.DataFrame(network)))
        notes.append("Converted explicit CDEN rows to nodes and edges")
    if not isinstance(network, dict):
        raise ValueError("network must be a CDEN object or row list")
    if isinstance(network.get("nodes"), list):
        for node in network["nodes"]:
            if isinstance(node, dict) and node.get("persona") and not node.get("model"):
                node["model"] = "[assistant_model]"
                notes.append("Used the project assistant for an unspecified model")
    draft["network"] = network
    draft.setdefault("name", "draft_cden")
    # Rationale and a display name are review metadata, not CDEN syntax.
    draft.setdefault("rationale", "Review the generated CDEN and its input and output fields before running it.")
    return draft, list(dict.fromkeys(notes))


def _feedback(raw, error):
    # Keep both ends of a large response; do not replace a failed parse with None.
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    if len(text) > 6000:
        line, column = getattr(error, "line", None), getattr(error, "column", None)
        location = ""
        if line and column:
            position = sum(len(value) for value in text.splitlines(keepends=True)[:line - 1]) + column - 1
            location = "\nText near the syntax error:\n" + text[max(0, position - 750):position + 750]
        text = text[:2500] + location + "\n[Other text omitted from repair context; full response retained in run evidence.]\n" + text[-1500:]
    return ("\nYour last draft failed validation: " + str(error) +
            "\nRepair the previous draft below. Preserve the requested behaviour. Return one complete JSON object only; escape quotes and newlines inside strings. Do not repeat examples, code, or analysis.\nPrevious draft:\n" + text)


def run_constructor(prompt, model, validate, *, tokens=4096, attempts=2, extra_params=None,
                    workflow=None, workflow_vars=None, fxStore=None, observer=None):
    """Execute normal CDEN model nodes, retain failed text, and attempt one repair."""
    from .workbench import graph_from_network
    if isinstance(attempts, bool) or attempts not in {1, 2}:
        raise ValueError("The constructor supports one generation and at most one repair")
    selected = deepcopy(workflow or {
        "nodes": [{"key": "Start", "prompt": "[request]\n[validation_feedback]", "persona": "Return the requested CDEN JSON object only.",
                   "fx": "pass", "tokens": tokens, "model": "[assistant_model]"}],
        "edges": [], "result_nodes": ["Start"]})
    outputs = selected.get("result_nodes", ["Start"])
    if not isinstance(outputs, list) or not outputs or any(not isinstance(key, str) for key in outputs):
        raise ValueError("Choose the constructor's return node")
    feedback, evidence = "", []
    for attempt in range(1, attempts + 1):
        if observer:
            observer({"event": "workflow_attempt", "attempt": attempt, "attempts": attempts})
        graph = graph_from_network(selected)
        if any(key not in graph for key in outputs):
            raise ValueError("The constructor's return node is missing from its workflow")
        # A one-step constructor need not parse inside the walker. Decode at the
        # boundary where syntax errors can be repaired without losing raw text.
        if len(graph) == 1 and graph.number_of_edges() == 0:
            only = next(iter(graph))
            if only in outputs and graph.nodes[only].get("fx") == "insertJson":
                graph.nodes[only]["fx"] = "pass"
        for _, node in graph.nodes(data=True):
            if node.get("extra_params") is None:
                node["extra_params"] = extra_params
        variables = {"request": prompt, "validation_feedback": feedback,
                     "assistant_model": model, "supervisor_model": model} | (workflow_vars or {})
        walked = tb.walkChatNet(graph, fxStore=fxStore, varStore=variables,
                               runAsync=True, numWorkers=2, verbosity=0, observer=observer)
        key = next((key for key in outputs if key in walked or key + "_raw" in walked), None)
        result = walked.get(key) if key in walked else None
        raw = walked.get(str(key) + "_raw", result)
        record = {"attempt": attempt, "node": key, "raw": raw,
                  "output_limits": {key: value.get("tokens") for key, value in graph.nodes(data=True) if value.get("persona")}}
        try:
            if not walked.get("success"):
                raise ValueError(f"The constructor workflow failed: {walked.get('errors')}")
            checks = [walked[k] for k in selected.get("check_nodes", []) if k in walked]
            if checks and tb.getYN(checks[-1]) != "yes":
                raise ValueError("The CDEN alignment check rejected this draft")
            parsed, notes = parse_response(result)
            parsed, normalised = normalize_draft(parsed)
            validate(parsed)
            record.update(valid=True, normalisation=notes + normalised)
            evidence.append(record)
            if observer:
                observer({"event": "cden_attempt", **record})
            parsed["constructor"] = {"attempts": attempt, "normalisation": notes + normalised}
            return parsed
        except (ValueError, TypeError, KeyError, SyntaxError) as exc:
            record.update(valid=False, kind=getattr(exc, "kind", "validation"), error=str(exc),
                          line=getattr(exc, "line", None), column=getattr(exc, "column", None))
            evidence.append(record)
            if observer:
                observer({"event": "cden_attempt", **record})
            feedback = _feedback(raw, exc)
    final = evidence[-1]
    error = CdenSyntaxError(f"No valid CDEN after {attempts} attempts: {final['error']}. The full responses are available in constructor diagnostics.",
                            kind=final["kind"], line=final["line"], column=final["column"])
    error.attempts = evidence
    raise error
