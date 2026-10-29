"""Bounded iterable expansion through TabulAIrity's own dependency walker."""

import asyncio
from collections.abc import Iterable, Mapping
from copy import deepcopy
import math

import networkx as nx
import pandas as pd

from .runtime import WalkContext, walk_context


def chatNetInputKeys(net, fxStore=None, chatNets=None, _visiting=None):
    """External reads of a statically understood, non-mutating CDEN tree.

    None means dependencies cannot be proven (dynamic templates, variable
    writers, unknown callbacks or recursive/unresolved graph definitions).
    Marked parallelFx callbacks must declare their reads in templates.
    """
    import re
    from .core import _resolveChatTemplate, _chatVarProducer, _parallelChatFxNames

    visiting = set() if _visiting is None else _visiting
    if not isinstance(net, nx.MultiDiGraph) or id(net) in visiting:
        return None
    visiting = visiting | {id(net)}
    known = {'null', 'pass', 'getYN', 'isYes', 'isNo', 'insertJson',
             'expandIterable'} | _parallelChatFxNames(fxStore)
    refs = set()
    for data in [*dict(net.nodes(data=True)).values(),
                 *(data for _, _, data in net.edges(data=True))]:
        if data.get('fx', 'null') not in known:
            return None
        for field in ('prompt', 'persona', 'model'):
            text = data.get(field, '')
            if isinstance(text, str) and re.search(r'\[[^\]\n]*\[', text):
                return None
            refs.update(_resolveChatTemplate(text, {})[1])
    for node, data in net.nodes(data=True):
        if data.get('fx') != 'expandIterable':
            continue
        params = data.get('fx_params') or {}
        # Runtime callback/registry overrides need their own scope analysis;
        # keep them on the conservative path rather than assuming root purity.
        if params.get('fxStore') is not None or params.get('chatNets') is not None:
            return None
        child = params.get('chatNet')
        if isinstance(child, str):
            child = (chatNets or {}).get(child)
        child_refs = chatNetInputKeys(child, fxStore, chatNets, visiting)
        if child_refs is None:
            return None
        refs.update(child_refs - {str(node) + '_item'})
    return {key for key in refs if _chatVarProducer(key, net) is None}


def _joined_child(net, node, fx, nets):
    """Validate the deliberately narrow opt-in final-result join contract."""
    params = net.nodes[node].get('fx_params') or {}
    child = params.get('chatNet')
    if isinstance(child, str):
        child = nets.get(child)
    if (net.out_degree(node) or params.get('flatten') is not None
            or params.get('output_keys') is not None or params.get('as_dataframe')):
        raise ValueError('inherit_parent=join requires a leaf expansion returning full records')
    refs = chatNetInputKeys(child, fx, nets)
    if refs is None or chatNetInputKeys(net, fx, nets) is None:
        raise ValueError('inherit_parent=join requires static templates and pure chatFX')
    return child, refs - {str(node) + '_item'}


def _join_parent_results(net, values, fx, nets, _plans=None):
    """Enrich completed leaf rows only, never mutate inputs of running walks.

    Each child scope owns its node keys even when that producer was pruned or
    failed. Ancestor status, diagnostics, audit fields and expansion containers
    are not inherited. Recursing propagates late article fields to host rows.
    """
    from .core import _chatVarProducer

    plans = {} if _plans is None else _plans

    expansions = {n for n, d in net.nodes(data=True) if d.get('fx') == 'expandIterable'}
    excluded = {'success', 'errors'} | expansions
    excluded |= {str(n) + '_expansion_errors' for n in expansions}
    excluded |= {str(n) + suffix for n in net for suffix in ('_raw', '_prompt')}
    excluded |= {f'{a}-{b}' for a, b in net.edges()}
    parent = {k: v for k, v in values.items() if k not in excluded}
    for node in expansions:
        params = net.nodes[node].get('fx_params') or {}
        if params.get('inherit_parent', 'snapshot') != 'join' or node not in values:
            continue
        identity = (id(net), node)
        if identity not in plans:
            plans[identity] = _joined_child(net, node, fx, nets)
        child, _ = plans[identity]
        inherited = {k: v for k, v in parent.items()
                     if _chatVarProducer(k, child) is None}
        rows = []
        for row in values[node]:
            # Replace stale inherited snapshots, but preserve every child output.
            merged = deepcopy(inherited)
            merged.update({k: v for k, v in row.items() if k not in inherited})
            _join_parent_results(child, merged, fx, nets, plans)
            rows.append(merged)
        values[node] = rows


def _missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "null", "none", "nan", "n/a"}
    if isinstance(value, (Mapping, list, tuple, set)):
        return False
    try:
        missing = pd.isna(value)
        return bool(missing) if not hasattr(missing, "__len__") else False
    except (TypeError, ValueError):
        return False


def cleanIterable(value, *, dropna=True, items_key=None, max_items=None):
    """Repair a serialized container; never turn plain text into characters.

    A JSON object is one item. Use items_key for a wrapper such as {items: [...]}.
    Zero and False are retained. Arbitrary malformed scalar text is an error.
    """
    from .core import insertJson, _stripChatReasoning

    if max_items is not None and (
        isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1
    ):
        raise ValueError("max_items must be a positive integer or None")
    if _missing(value):
        return []
    if isinstance(value, str):
        text = _stripChatReasoning(value).strip()
        try:
            value = insertJson(text)
        except (ValueError, SyntaxError):
            from json_repair import loads

            if "[" not in text and "{" not in text:
                raise ValueError(
                    "expandIterable expects a list or object, not plain text"
                )
            value = loads(text)
    if items_key is not None:
        if not isinstance(value, Mapping) or items_key not in value:
            raise ValueError(f"Iterable wrapper has no {items_key!r} key")
        value = value[items_key]
    if isinstance(value, pd.DataFrame):
        value = value.to_dict("records")
    elif isinstance(value, Mapping):
        value = [dict(value)] if value else []
    elif isinstance(value, set):
        value = sorted(value, key=repr)
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError("expandIterable expects an iterable container")
    items = []
    for item in value:
        if dropna and _missing(item):
            continue
        items.append(item)
        if max_items is not None and len(items) > max_items:
            raise ValueError(
                f"Iterable exceeds max_items={max_items}; nothing was silently truncated"
            )
    return items


def _resolve_net(chatNet, context):
    from .core import buildChatNet

    if isinstance(chatNet, str):
        if chatNet not in context.nets:
            raise ValueError(f"Unknown nested chatNet {chatNet!r}")
        chatNet = context.nets[chatNet]
    if isinstance(chatNet, pd.DataFrame):
        chatNet = buildChatNet(chatNet)
    if not isinstance(chatNet, nx.MultiDiGraph) or "Start" not in chatNet:
        raise ValueError("A nested chatNet must be a built graph with a Start node")
    if len(context.path) >= 16:
        raise ValueError(
            "Nested CDEN depth exceeds 16; check for recursive chatNet references"
        )
    return chatNet


def _parallel_expansion_safe(net, context, fx, visiting=None):
    """Only change serial defaults for built-ins or explicitly pure callbacks.

    Inspect nested definitions too: a stateful grandchild callback must not be
    overlapped by automatically running multiple copies of its parent graph.
    Unknown definitions remain conservative and fail normally when executed.
    """
    from .core import baseFx, buildChatNet, _parallelChatFxNames

    visiting = set() if visiting is None else visiting
    if id(net) in visiting:
        return False
    visiting = visiting | {id(net)}
    known = set(baseFx) | _parallelChatFxNames(fx)
    for _, _, data in net.edges(data=True):
        if data.get("fx", "null") not in known:
            return False
    for _, data in net.nodes(data=True):
        name = data.get("fx", "null")
        if name not in known:
            return False
        if name == "expandIterable":
            nested = (data.get("fx_params") or {}).get("chatNet")
            if isinstance(nested, str):
                nested = context.nets.get(nested)
            if isinstance(nested, pd.DataFrame):
                nested = buildChatNet(nested)
            if not isinstance(nested, nx.MultiDiGraph):
                return False
            if not _parallel_expansion_safe(nested, context, fx, visiting):
                return False
    return True


async def expandIterableAsync(
    response,
    chatVars=None,
    *,
    chatNet,
    source,
    dropna=True,
    on_error="drop",
    concurrency=None,
    as_dataframe=False,
    output_keys=None,
    required=None,
    flatten=None,
    items_key=None,
    max_items=None,
    fxStore=None,
    chatNets=None,
    timeout=None,
    inherit_parent="snapshot",
):
    """Return ordered rows. Each child receives an isolated {source}_item.

    Failed children are dropped by default, kept with success/errors on request,
    or raised in strict mode. Diagnostics are always written to
    chatVars['{source}_expansion_errors']. Cancellation is never treated as an
    item failure. Nested model calls share the outer walk's inference slots.
    Omitted concurrency divides the outer worker budget across nesting levels;
    stateful callbacks and strict failure mode retain sequential defaults.
    inherit_parent='join' opts a leaf expansion into early execution and a
    final parent-result join by the enclosing walker. It requires static pure
    graphs and full records; the default 'snapshot' retains ancestor waits.
    """
    from .core import baseFx, walkChatNetAsync

    if inherit_parent not in {'snapshot', 'join'}:
        raise ValueError('inherit_parent must be snapshot or join')
    if inherit_parent == 'join' and walk_context.get() is None:
        raise ValueError('inherit_parent=join requires a parent walk')

    if on_error not in {"drop", "keep", "raise"}:
        raise ValueError("on_error must be drop, keep, or raise")
    if not isinstance(source, str) or not source:
        raise ValueError("source must be the source node name")
    if concurrency is not None and (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < 1
    ):
        raise ValueError("concurrency must be a positive integer or None")
    for name, value in [("output_keys", output_keys), ("required", required)]:
        if value is not None and (
            isinstance(value, str) or not all(isinstance(k, str) for k in value)
        ):
            raise ValueError(f"{name} must be a list of field names")
    parent = dict(chatVars or {})
    context = walk_context.get()
    if context is None:
        context = WalkContext(
            asyncio.Semaphore(concurrency or 1),
            chatNets or {},
            (fxStore or {}) | baseFx,
            workers=concurrency or 1,
        )
    net = _resolve_net(chatNet, context)
    # Register the child graph and its parent relation once, before any item
    # walks begin. This keeps nested executions in the same live viz namespace.
    if context.graph is not None:
        try:
            from . import core as _core
            if _core._viz_enabled:
                parent_cid = _core._viz_cid_for_graph(context.graph)
                if parent_cid is not None:
                    from .visualization import viz_notify_nested_chatnet
                    viz_notify_nested_chatnet(parent_cid, source, net)
        except Exception:
            pass
    items = cleanIterable(
        response, dropna=dropna, items_key=items_key, max_items=max_items
    )
    automatic = concurrency is None
    siblings = sum(data.get("fx") == "expandIterable"
                   for _, data in context.graph.nodes(data=True)) if context.graph is not None else 1
    siblings = max(1, siblings)
    inherited_fanout = context.expansion_fanout * siblings
    if automatic:
        concurrency = 1
        effective_fx = context.fx if fxStore is None else fxStore
        if on_error != "raise" and _parallel_expansion_safe(net, context, effective_fx):
            concurrency = max(1, context.workers // inherited_fanout)
    concurrency = min(concurrency, max(1, len(items)))
    context.emit("expansion_start", node=source, total=len(items),
                 concurrency=concurrency, automatic=automatic,
                 fanout=inherited_fanout * concurrency)
    item_key = source + "_item"
    if item_key in net:
        raise ValueError(
            f"Nested node {item_key!r} collides with the iteration variable"
        )
    # Child output keys from an older/parent walk must never masquerade as
    # completed child producers when that child branch is pruned.
    child_keys = set(net) | {"success", "errors"}
    child_keys |= {str(n) + suffix for n in net for suffix in ("_raw", "_prompt")}
    child_keys |= {f"{a}-{b}" for a, b in net.edges()}
    parent = {k: v for k, v in parent.items() if k not in child_keys}
    parent_audit = set()
    if context.graph is not None:
        parent_audit |= {
            str(n) + suffix for n in context.graph for suffix in ("_raw", "_prompt")
        }
        parent_audit |= {f"{a}-{b}" for a, b in context.graph.edges()}
    rows, failures = [None] * len(items), [None] * len(items)
    indices = iter(range(len(items)))

    def project_record(result, index):
        if output_keys is None:
            audit = {str(n) + suffix for n in net for suffix in ("_raw", "_prompt")}
            audit |= {f"{a}-{b}" for a, b in net.edges()}
            record = {k: v for k, v in result.items() if k not in audit | parent_audit}
        else:
            record = {k: result.get(k) for k in output_keys}
        record[item_key] = result.get(item_key, items[index])
        if not result.get("success", False):
            record.update(success=False, errors=result.get("errors", []))
        return record

    async def worker():
        for index in indices:
            await context.checkpoint()
            variables = dict(parent)
            variables[item_key] = items[index]
            child_context = context.child(source, index)
            child_context.expansion_fanout = inherited_fanout * concurrency
            token = walk_context.set(child_context)
            result = variables
            try:
                variables = deepcopy(variables)
                result = variables
                result = await walkChatNetAsync(
                    net.copy(),
                    context.fx if fxStore is None else fxStore,
                    variables,
                    0,
                    numWorkers=context.workers,
                    tolerant=True,
                    timeout=timeout,
                )
                absent = [
                    key
                    for key in required or []
                    if key not in result or _missing(result[key])
                ]
                if absent:
                    result["success"] = False
                    result["errors"].append({"required": absent, "type": "node"})
                if not result.get("success", False):
                    raise ValueError("Nested walk failed")
                record = project_record(result, index)
                if flatten is not None:
                    children = result.get(flatten)
                    if isinstance(children, pd.DataFrame):
                        children = children.to_dict("records")
                    if not isinstance(children, list) or not all(
                        isinstance(row, dict) for row in children
                    ):
                        raise ValueError(
                            f"flatten field {flatten!r} must contain a list of dicts"
                        )
                    record.pop(flatten, None)
                    rows[index] = [record | child for child in children]
                else:
                    rows[index] = [record]
            except Exception as exc:
                result = dict(result)
                result["success"] = False
                if not result.get("errors"):
                    result["errors"] = [{"expandIterable": str(exc), "type": "node"}]
                failures[index] = {
                    "index": index,
                    "item": items[index],
                    "errors": result["errors"],
                }
                child_context.emit("expansion_failed", node=source, **failures[index])
                if on_error == "raise":
                    raise ValueError(
                        f"Expansion {source}[{index}] failed: {result['errors']}"
                    ) from exc
                if on_error == "keep":
                    rows[index] = [project_record(result, index)]
            finally:
                walk_context.reset(token)

    tasks = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(items)))]
    try:
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if chatVars is not None:
            chatVars[source + "_expansion_errors"] = [
                f for f in failures if f is not None
            ]
    records = [record for group in rows if group is not None for record in group]
    return pd.DataFrame(records) if as_dataframe else records


def expandIterable(response, chatVars=None, **kwargs):
    """Synchronous API; async callers should await expandIterableAsync."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(expandIterableAsync(response, chatVars, **kwargs))
    raise RuntimeError("Use await expandIterableAsync(...) inside an active event loop")
