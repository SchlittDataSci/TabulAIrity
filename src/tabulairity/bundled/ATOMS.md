# Reusable CDEN atoms

An atom packages a small workflow behind named inputs and outputs. It behaves as one step in the editor and expands into ordinary CDEN nodes when executed. The underlying Tabulairity execution libraries are unchanged.

## Try the prompt improver

1. In **Models**, discover your backend by address and port, or import your routing CSV. Choose the assistant model. A new project starts with empty routes and no model preference.
2. Open **Atoms → Improve a prompt → Try atom**. Enter the prompt you want to improve and run the trial.
3. In **Runs**, read the revised prompt and inspect the transcript. It records the analysis, draft, critique, final revision, requests, cache hits and errors.

This atom has four model steps: analyse intent, inputs, output structure, constraints and likely problems; draft; critique against those measures and propose mitigations; implement the justified mitigations. Its definition retains the exact original authoring request. It instructs the model to preserve explicit requirements and bracket placeholders, distinguish assumptions from requirements, and return the final prompt without a preface. Those instructions still need evaluation on your chosen model.

Trials use separate run snapshots. They do not replace your graph, change its inputs, or apply their outputs to your project automatically. Malformed structured responses and absent final outputs fail visibly.

## Use an atom in a larger workflow

Select the preceding step, then **Add atom**. Give the inserted step a name, choose when it runs, and bind each input to an input column, an earlier step, or fixed text. For example, bind `user_prompt` to a CSV column called `prompt`. An earlier step's structured value is passed without converting its type. Fixed text keeps literal brackets.

Connect the inserted step to the next step as needed. Use `[your_step_name]` in later prompts. A single output returns its value directly; multiple outputs return an object with the declared output names. Existing conditional branches remain in place when an atom is inserted. A missing required input or output is an error.

The editor initially shows one block per insertion. **Show internal steps** reveals its CDEN. You can inspect or change model prompts there. The block's settings control input bindings and its model. Use **Ungroup into ordinary steps** before deleting individual internal steps. Undo restores insertion, removal, grouping and binding edits within the editor session.

Each insertion is a frozen independent copy. Editing the saved atom later does not update existing copies. The block records the definition version and SHA-256 fingerprint; inspect the run snapshot to establish what actually ran, including any later edits to the inserted copy.

## Save your own atom

In the CDEN editor, choose **More → Save as atom**. Name the atom, give it a title, and choose the output step. The workbench captures all referenced nested CDENs and identifies static external input references. Use **Export atom** to share the JSON definition; **Import atom** adds a copy to another project's library of atoms. Existing names receive a separate suffix.

The supported isolation contract includes built-in pure response handlers, explicit model nodes, static bracket references, conditional edges, nested iterable CDENs, and the workbench's pure input/output adapters. It rejects dynamic variable references, `mergeJson`, unknown callbacks, callback registry overrides, recursive children, and ambiguous generated variable names. These need an explicit isolation adapter; accepting them without knowing their reads and writes would make collision protection unreliable. Literal bracket labels in an authored CDEN can be interpreted as external inputs; inspect the inferred input interface when saving.

Namespaces cover every inserted root step, its `_raw`, `_prompt`, `_item`, and `_expansion_errors` variables, edge-result names, prompt/persona/model references, and forced dependencies. Nested helpers receive private registry aliases, and inherited references are rewritten with scope shadowing. A child retains its structural `Start` inside its isolated scope. Instance names receive numeric suffixes when needed. Collisions that cannot be safely resolved are rejected before the project is modified.

For nested expansions with selected output fields, the saved atom preserves that public field selection. Full child execution records remain in the run evidence. Item counts and branches can multiply model calls; displayed model-step counts describe graph structure, not a guaranteed number of HTTP requests.

## Portable composition

The installed workbench layer also exposes composition without opening the GUI:

```python
from tabulairity.workbench import read_project, save_project
from tabulairity.workbench_atoms import insert_atom

project = read_project("my_project")
result = insert_atom(
    project, "main", project["atoms"]["improve_prompt"],
    after="Start", instance="improve",
    bindings={"user_prompt": {"mode": "variable", "variable": "prompt"}},
)
save_project("my_project", result["project"])
```

Use `{"mode": "literal", "value": ...}` to bind fixed JSON data. `insert_atom` returns a new project and the allocated `instance` name; it does not mutate its inputs. Exported projects include their definitions and compiled graphs and run with `python run.py`. Direct core callers executing a compiled graph must supply `workbench_atoms.RUNTIME_FX` as their function store, plus the project's nested graph registry. The normal workbench and exported launcher do this automatically. These adapters perform data binding, output selection and shape checks; they make no model requests.

For public-health and life-science tasks, establish expected outputs on known examples before adopting an atom. A completed trial verifies execution, not the accuracy or suitability of model-generated scientific conclusions.
