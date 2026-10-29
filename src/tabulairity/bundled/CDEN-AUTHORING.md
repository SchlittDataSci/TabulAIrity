# CDEN authoring guide

A CDEN is a directed extraction network: named nodes run prompts or deterministic
transformations, and directed edges decide which nodes may run next. The workbench
stores a CDEN as JSON with `nodes`, `edges`, and optional `forced_dependencies`.
The core builds and executes it; a CDEN is not a Python program or an agent plan.

## Design from the task and examples

1. Identify input fields, required outputs, filtering rules, and missing-data behavior.
2. Adapt the supplied samples. Use the model-step abstraction for inference,
   `article_triage` for yes/no branches, `main` with `book` for per-item work,
   and `nested_groups` with its two helpers for two levels of expansion.
   The provided research CDEN abstractions additionally teach conditional answer
   checking, parallel report synthesis, scoped evidence extraction and typed
   branches. Their source filenames and hashes identify what was abstracted.
   Transfer structure and contracts; do not copy application-specific cleaners,
   clinical rubrics or old model/context settings into an unrelated task.
3. Give each operation a short, meaningful node key. Keep distinct operations
   visible; do not hide model calls inside custom Python functions.
4. Connect every node to `Start`. Reference earlier results explicitly in prompts.
   Describe output shape and evidence rules in each model prompt.
5. Check branches, dependencies, function names, child contracts, and inference
   cost before returning the JSON. State any input assumptions in the rationale.

## Nodes, prompts, and variables

- Every graph needs a node whose key is exactly `Start`. Keys are unique;
  `success` and `errors` are reserved. Node names must not shadow input fields.
- A model node has a nonempty `persona` (system text), `prompt`, available `model`
  alias, positive `tokens` output ceiling, and `fx` response transformation.
- A prompt starting with `recall:` bypasses inference. For example,
  `recall: [text]` copies the input text; `recall: ready` is a deterministic Start.
  Use an empty persona and `tokens: 1` for these deterministic examples.
- `[text]` reads an input field; `[extract]` reads the transformed output of node
  `extract`. `[extract_raw]` reads its raw response, `[extract_prompt]` its prompt.
  Bracket variables are exact names, not Python expressions. `insertJson` returns
  a parsed value stored under the node key; it does not create top-level fields.
- References create data dependencies in async mode, but control edges still
  govern reachability. Keep a DAG: no control, data, or nested-reference cycles.
  Do not put a required producer only on a mutually exclusive branch.
- `extra_params` contains model parameters. `fx_params` contains transformation
  parameters. They are different dictionaries. Do not invent function names,
  model aliases, APIs, or input columns. Use the project's actual contracts;
  if the task needs different inputs, say so explicitly in the rationale.

## Functions and edges

- Node `fx: pass` preserves text; `insertJson` parses JSON; `getYN` normalizes a
  yes/no response. `mergeJson` also writes keys into the variable store: prefer
  explicit node outputs unless that mutation is required.
- An edge is `{"source":"Start","target":"extract","fx":"null"}`.
  `null` opens unconditionally. `isYes` and `isNo` evaluate the source's RAW
  response, not its transformed output. A branch producer must return yes/no.
- Custom functions must already exist in the enabled project CHAT_FX registry.
  They accept `(response, chatVars, **parameters)` as appropriate. A CDEN draft
  cannot create an unprovided Python module simply by naming its function.

## Nested CDENs

`expandIterable` transforms a list into ordered child results. Its parameters
include `chatNet` (an existing project CDEN name), optional `output_keys`,
`flatten` (a child output containing rows), and `concurrency`.
The child receives `<expansion_node_key>_item`; for a parent node `titles`, the
child reads `[titles_item]`. The parameter is NOT the child CDEN's name.
Use the exact parent node key expected by a reused child. Children have isolated
node outputs and inherit a snapshot of parent inputs/completed variables.
Nested inference shares the outer worker limit; list sizes multiply total calls.
Examples are teaching material, not automatically registered project children.
Only reuse names listed as available project CDENs. This drafter returns ONE
new graph; create a new child separately before drafting its parent.

## Return and review contract

Return one JSON object: `{"name":"short_name","network":{"nodes":[],
"edges":[]},"output_node":"final_node_key","rationale":"brief design and input assumptions"}`. Fill in the
actual nodes and edges. No markdown fences, executable code, or unrelated keys.
Use the requested task rather than copying the samples' catalog domain.
Set `output_node` to an existing node key for tabular export, or an empty string
to keep all extracted values (useful for terminal branches). Name its shape in
the rationale. The workbench keeps complete run evidence as well.
A structurally valid draft is still a proposal: inspect it,
apply explicitly, then test with known inputs before using its model outputs.

Escape double quotes and line breaks inside prompt strings: write `\"` for
a quote and `\n` for a line break. Return only the CDEN; do not reprint examples.
The constructor permits at most two attempts. A repair receives the failed
response and validation error, alongside the guide and example context. The
construction dialog exposes the output limit for that request; custom workflows
retain their per-step ceilings. A response cut off before its outer container
closes is rejected, never silently completed. Backend context and reasoning
settings remain under user control. Context is bounded by characters, not a
model-specific tokenizer; allow room for input and output in the serving model.
