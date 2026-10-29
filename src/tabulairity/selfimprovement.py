import re
import pandas as pd
from random import randint
from datetime import datetime

from . import core as tb


verbosity = 1


def getSeedParams(randomize, model):
    """Returns seed text and string depending on model"""
    seedSupported = not model.startswith("gemini")
    if not randomize:
        return None, ""
    if seedSupported:
        seed = randint(0, 9999)
        seedText = ""
    else:
        seed = None
        seedText = f"(random seed = {randint(0, 9999)})"
    return seed, seedText


def getEvaluatorNet(supervisor="gemma4:26b", script=None):
    """Local default evaluator; optionally pass your own CDEN DataFrame."""
    if script is None:
        script = pd.DataFrame(
            [
                {
                    "type": "node",
                    "key": "Start",
                    "fx": "getYN",
                    "tokens": 256,
                    "persona": "Evaluate the proposed answer against the supplied instructions and evidence.",
                    "prompt": "Does this answer obey the prompt and avoid unsupported information? Return exactly yes or no.\nPrompt: [preppedPrompt]\nAnswer: [answer]",
                },
                {"type": "edge", "key": "Start-Explain error", "fx": "isNo"},
                {
                    "type": "node",
                    "key": "Explain error",
                    "fx": "pass",
                    "tokens": 2000,
                    "persona": "Identify only defects visible in the supplied prompt and answer.",
                    "prompt": "Explain the observable error without inventing evidence.\nPrompt: [preppedPrompt]\nAnswer: [answer]",
                },
            ]
        )
    else:
        script = script.copy()
    script["model"] = supervisor
    return tb.buildChatNet(script)


def rewritePrompt(
    prompt,
    persona,
    model,
    rewritePersona=False,
    intentPrompt=None,
    errorSummary=None,
    randomize=True,
):
    """ "Takes a given prompt and persona and returns an LLM improved version of each"""
    maxTries = 20

    if intentPrompt is None:
        intentPrompt = ""
    else:
        intentPrompt = " " + intentPrompt
    if errorSummary is None:
        errorSummary = ""
    else:
        errorSummary = " " + errorSummary

    seed, seedText = getSeedParams(randomize, model)
    keptAllVars = False
    preservedIntent = False
    originalVars = tb.extractChatVars(prompt)
    tries = 0

    while not (keptAllVars and preservedIntent) and tries != maxTries:
        seed, seedText = getSeedParams(randomize, model)
        questionPrompt = f"""Rewrite the following prompt text to maximize its performance according to the criteria below.{intentPrompt}{errorSummary}

Objectives:

* Expand and optimize the prompt so that it consistently produces clean, structured output containing only the required data.
* Ensure the rewritten prompt strictly forbids inclusion of any markdown, explanations, commentary, variable placeholders, or descriptive text.
* Explicitly state in the rewritten prompt what must be included and what must not be included in the model’s output.
* Preserve and correctly utilize all flag values indicated in square brackets [like_this].
* Preserve and implement the full intent of the prompt.
* Do not invent or modify any flag values.
* Do not add or include meta-instructions, comments, or contextual discussion in the final rewritten text.

The prompt being improved operates under the system prompt:

{persona}

Task:
Rewrite the following prompt text (provided after this instruction block) so that it performs optimally under the criteria above.
You absolutely must not provide descriptions or commentary.
{seedText}
Prompt to improve:

{prompt}"""

        aiPrompt = tb.askChatQuestion(
            questionPrompt, "You are a skilled prompt engineer.", model=model, seed=seed
        )
        aiVars = tb.extractChatVars(aiPrompt)
        keptAllVars = aiVars == originalVars
        preservedIntent = validatePromptIntent(aiPrompt, intentPrompt.strip(), model)
        tries += 1

    if tries == maxTries and not keptAllVars:
        raise ValueError(
            f"Rewritten prompts failed to retain chatvars after {maxTries} attempts."
        )
    elif tries == maxTries and not preservedIntent:
        raise ValueError(
            f"Rewritten prompts failed to preserve intent after {maxTries} attempts."
        )

    if not rewritePersona:
        return aiPrompt, persona

    personaPrompt = f"""Please consider the following prompt and revise our LLM system text to maximize prompt efficiency if needed.
The current system text is "{persona}". {intentPrompt}{errorSummary}If you decide to replace it, your replacement should be optimized for use with LLM APIs.
You must only return the recommended system text for our prompt.
You absolutely must not provide descriptions or commentary.
{seedText}
{prompt}"""

    aiPersona = tb.askChatQuestion(
        personaPrompt, "You are a skilled prompt engineer.", model=model, seed=seed
    )

    return aiPrompt, aiPersona


def summarizeErrors(errors, intent, model):
    persona = "You are a skilled evaluator."
    seed, seedText = getSeedParams(True, model)

    errorPrompt = f"""You will be given a bulleted list of previously logged prompt errors.

Your task is to:
1. Write a 1–2 sentence summary describing the general nature or cause of these errors.  
   - Your summary must begin with the phrase:
     "This prompt has had issues with"
2. Follow the summary with a single sentence suggesting how the prompt could be improved to prevent these issues.
3. This sentence should not contradict the original intent of the prompt, stated as follows:
### INTENT STATEMENT:
{intent}

Example:
If the errors involve missing context or ambiguous instructions, your output might be:
"This prompt has had issues with unclear context and ambiguous task phrasing.  
It could be improved by explicitly defining the input format and desired output behavior."

Now produce your summary and improvement suggestion based on the provided error list.
{seedText}
### ERROR LIST - BULLETED:
{errors}"""
    errorSummary = tb.askChatQuestion(
        errorPrompt, "You are a skilled evaluator.", model=model, seed=seed
    )
    while "\n\n" in errorSummary:
        errorSummary = errorSummary.replace("\n\n", "\n")
    errorSummary = errorSummary.replace(".\n", ". ")
    return errorSummary


def evaluateAnswer(originalPrompt, persona, varsIn, model, evaluatorNet):
    """Evaluates the y/n correctness of an answer, returning an explanation of the error if found"""

    duration = 0.0
    try:
        tStart = datetime.now()
        preppedPrompt = tb.insertChatVars(originalPrompt, varsIn)
        answer = tb.askChatQuestion(preppedPrompt, persona, model, tokens=4000)
        tFinish = datetime.now()
        duration = (tFinish - tStart).total_seconds()

        answerVars = {"preppedPrompt": preppedPrompt, "answer": answer}
        varsOut = dict(varsIn) | answerVars
        evaluation = tb.walkChatNet(evaluatorNet, varStore=varsOut, verbosity=verbosity)
        if tb.ynToBool(evaluation["Start"]):
            answeredCorrectly = True
            explanation = None
        else:
            answeredCorrectly = False
            explanation = evaluation["Explain error"]
    except Exception as e:
        print("Error:", e)
        answeredCorrectly = False
        explanation = None

    return answeredCorrectly, explanation, duration


def extractIntent(prompt, model):
    """Summarizes the intent of a prompt"""
    evaluationPersona = "You are an expert evaluator."
    intentPrompt = f"""You are an expert at analyzing instructions and identifying their underlying purpose.  
You will be given a **Prompt**, and your task is to write **two or three sentences** that clearly describe its intent, define the underlying context for its use, and the domain-specific considerations needed to adhere to each.

---

## Instructions

1. Read the provided **Prompt** carefully.  
2. Determine what the user is ultimately asking the model to do and what needs must be met for its given context.  
3. Write the first sentence beginning with:  
"The intent of this prompt is to ..."
4. Do **not** include any other commentary, formatting, or analysis — only a description of the intent of the prompt.
5. Think deeply on this one to understand the purpose of the prompt, do not simply return data formatting requirements.

---

## Input Format
"Prompt: {prompt}"

---

## Output Format
"The intent of this prompt is to [describe the task]."""
    intent = tb.askChatQuestion(
        intentPrompt, evaluationPersona, model, autoformatPersona=False
    )
    return intent


def validatePromptIntent(prompt, intent, model="gemma4:26b"):
    evaluatorPrompt = """You are an intent–prompt alignment classifier. You will receive:

1. **Intent statement** – a description of the user's desired goal, constraints, boundaries, or outcomes.
2. **Prompt statement** – the actual prompt the user intends to provide to an AI system.

Your task is to determine whether the prompt statement **faithfully follows, reflects, and remains consistent with the full intent expressed**. This includes both explicit instructions and implicit meaning.

### Interpretation rules
- Consider the **entire** intent, not just surface keywords.
- Identify the user’s **goal**, **tone**, **scope**, **format requirements**, and any **limitations or exclusions** expressed in the intent.
- Determine whether the prompt:
  - Accurately attempts to achieve the described goal.
  - Remains within the boundaries and constraints of the intent.
  - Maintains the correct domain, topic, or purpose.
  - Does not introduce major deviations, contradictions, or irrelevant objectives.
  - Does not omit any *critical* requirement from the intent.
- Minor differences in wording or style are acceptable **as long as** the core intent is preserved.

### Output rules
- Respond **only** with `"yes"` if the prompt broadly aligns with the user’s intent as a whole.
- Respond **only** with `"no"` if the prompt does not align, includes contradictions, or misses essential parts of the intent.
- Provide **no explanations, no reasoning, and no additional text** of any kind.

**Inputs:**  
Intent: [intent]  
Prompt: [prompt]

**Output:**  
"yes" or "no"\n"""
    evaluatorPrompt = tb.insertChatVars(
        evaluatorPrompt, {"intent": intent, "prompt": prompt}
    )
    intentCheck = tb.askChatQuestion(
        evaluatorPrompt, "You are a classifier.", model=model
    )

    result = {"yes": True, "no": False}[tb.getYN(intentCheck)]
    return result


def iteratePrompt(
    bestPrompt, bestPersona, testDfIn, model, depth=20, intent=None, supervisor=None
):
    """Processes a tabulairity extraction for a single prompt and iteratively improves it"""
    if supervisor is None:
        supervisor = model

    if len(testDfIn) == 0:
        raise ValueError("Prompt iteration requires at least one evaluation row")
    evaluatorNet = getEvaluatorNet(supervisor)

    if intent is None:
        intent = extractIntent(bestPrompt, supervisor)
        print("Inferred intent:", intent)

    testDf = testDfIn.copy(deep=True)
    varsInData = set(testDf.columns)
    varsInPrompt = set(tb.extractChatVars(bestPrompt))
    missingVars = varsInPrompt.difference(varsInData)
    # print("DEBOOO1",varsInPrompt)
    # print("DEBOOO2",varsInData)
    if len(missingVars) != 0:
        print(
            f"Warning: some expected prompt vars were not found in the passed DataFrame, is this intentional?\n\t{missingVars}"
        )

    # Evaluate initial prompt
    bestErrors = []
    bestScores = []
    bestTimes = []
    for _, row in testDf.iterrows():
        result, error, duration = evaluateAnswer(
            bestPrompt, bestPersona, row, model, evaluatorNet
        )
        bestScores.append(result)
        bestTimes.append(duration)
        if error is not None:
            bestErrors.append(error)

    bestScore = float(sum(bestScores) / len(bestScores))
    bestTime = float(sum(bestTimes) / len(bestTimes))
    initialScore = bestScore
    history = [
        {
            "prompt": bestPrompt,
            "intent": intent,
            "persona": bestPersona,
            "score": bestScore,
            "time": bestTime,
            "iteration": 0,
        }
    ]

    if bestScore == 1:
        print(
            "All responses flagged as correct, returning finalized prompt and persona."
        )
        return pd.DataFrame(history)

    if bestErrors == []:
        errorReport = "No errors were logged."
    else:
        errorReport = "\n".join([f"* {e}" for e in set(bestErrors)])

    # Begin iterative improvements
    for i in range(depth):
        print(f"\nModel: {model}\nIteration: {i}\nbest prompt: {bestPrompt}\n")
        newPrompt, newPersona = rewritePrompt(
            bestPrompt,
            bestPersona,
            supervisor,
            intentPrompt=intent,
            errorSummary=errorReport,
        )

        newErrors = []
        newScores = []
        newTimes = []
        for _, row in testDf.iterrows():
            result, error, duration = evaluateAnswer(
                newPrompt, newPersona, row, model, evaluatorNet
            )
            newScores.append(result)
            newTimes.append(duration)
            if error is not None:
                newErrors.append(error)

        newScore = float(sum(newScores) / len(newScores))
        newTime = float(sum(newTimes) / len(newTimes))
        print(f"Model: {model}     Best score: {bestScore}    New score: {newScore}")

        if newScore > bestScore:
            bestScore = newScore
            bestPrompt = newPrompt
            bestPersona = newPersona
            bestErrors = newErrors
            bestTime = newTime

        history.append(
            {
                "prompt": bestPrompt,
                "intent": intent,
                "persona": bestPersona,
                "score": bestScore,
                "time": bestTime,
                "iteration": i,
            }
        )

        if bestScore == 1:
            print(
                "All responses flagged as correct, returning finalized prompt and persona."
            )
            return pd.DataFrame(history)

        errorList = "\n".join([f"* {iError}" for iError in set(bestErrors)])
        errorReport = summarizeErrors(errorList, intent, supervisor)
        errorReport = f"The current prompt yields {round(bestScore, 2) * 100}% accuracy. {errorReport}"
        print(errorReport)

    print(
        f"Iterations complete, initial score: {initialScore}   final score: {bestScore}"
    )
    return pd.DataFrame(history)


# Local workbench helpers. Every inference call goes through TabulAIrity.
import ast
import json

DEFAULT_MODEL = "gemma4:26b"

# Synthetic, deliberately unrelated to the user's private research pipelines.
SCHEMA_EXAMPLE = {
    "nodes": [
        {
            "key": "Start",
            "prompt": "recall: ready",
            "persona": "",
            "fx": "pass",
            "tokens": 1,
        },
        {
            "key": "category",
            "prompt": "Classify this library book description: [text]. Return one category.",
            "persona": "You classify books from supplied evidence.",
            "fx": "pass",
            "tokens": 128,
        },
    ],
    "edges": [{"source": "Start", "target": "category", "fx": "null"}],
}


def _structured(
    prompt,
    model,
    validate,
    *,
    tokens=8192,
    attempts=2,
    extra_params=None,
    workflow=None,
    workflow_vars=None,
    fxStore=None,
    observer=None,
):
    """Bounded repair iteration; include observable validation feedback."""
    feedback = ""
    for attempt in range(attempts):
        if observer is not None:
            observer(
                {
                    "event": "workflow_attempt",
                    "attempt": attempt + 1,
                    "attempts": attempts,
                }
            )
        from .workbench import graph_from_network

        selected = workflow or {
            "nodes": [
                {
                    "key": "Start",
                    "prompt": "[request]\n[validation_feedback]",
                    "persona": "You develop small evidence-grounded projects. Return only the requested JSON object.",
                    "fx": "insertJson",
                    "tokens": tokens,
                    "model": "[assistant_model]",
                }
            ],
            "edges": [],
            "result_nodes": ["Start"],
        }
        graph = graph_from_network(selected)
        for _, data in graph.nodes(data=True):
            if data.get("extra_params") is None:
                data["extra_params"] = extra_params
        variables = {
            "request": prompt,
            "validation_feedback": feedback,
            "assistant_model": model,
            "supervisor_model": model,
        } | (workflow_vars or {})
        walked = tb.walkChatNet(
            graph,
            fxStore=fxStore,
            varStore=variables,
            runAsync=True,
            numWorkers=2,
            verbosity=0,
            observer=observer,
        )
        result = next(
            (walked[k] for k in selected.get("result_nodes", ["Start"]) if k in walked),
            None,
        )
        try:
            if not walked.get("success"):
                raise ValueError(f"Workbench CDEN failed: {walked.get('errors')}")
            checks = [walked[k] for k in selected.get("check_nodes", []) if k in walked]
            if checks and tb.getYN(checks[-1]) != "yes":
                raise ValueError("The CDEN alignment check rejected this draft")
            parsed = result if isinstance(result, dict) else tb.insertJson(result)
            if not isinstance(parsed, dict):
                raise ValueError("The answer must be a JSON object")
            validate(parsed)
            return parsed
        except (ValueError, TypeError, KeyError, SyntaxError) as exc:
            feedback = (
                "\nYour last draft failed validation: "
                + str(exc)
                + "\nRevise the draft. Previous draft:\n"
                + str(result)[-12000:]
            )
    raise ValueError(f"No valid draft after {attempts} attempts: {feedback[:500]}")


def draftArtifact(
    kind,
    request,
    network,
    *,
    model=DEFAULT_MODEL,
    node=None,
    validate_network=None,
    examples=None,
    tokens=8192,
    extra_params=None,
    workflow=None,
    workflow_vars=None,
    fxStore=None,
    observer=None,
):
    """Draft a prompt, importable Python FX module, or editable CDEN structure."""
    if kind not in {"prompt", "fx", "cden"}:
        raise ValueError("kind must be prompt, fx, or cden")
    schemas = {
        "prompt": {
            "node": node,
            "prompt": "new text",
            "persona": "system text",
            "rationale": "reason",
        },
        "fx": {
            "filename": "cleaner.py",
            "code": 'def clean(response, chatVars):\n    return response.strip()\n\nCHAT_FX = {"clean": clean}\n',
            "rationale": "reason",
        },
        "cden": {
            "name": "new_network",
            "network": SCHEMA_EXAMPLE,
            "rationale": "reason",
        },
    }
    current_node = next((n for n in network.get("nodes", []) if n["key"] == node), None)

    def validate(draft):
        if kind == "fx":
            validateFxSource(draft["code"])
        elif kind == "prompt":
            if current_node is None or draft["node"] != node:
                raise ValueError(
                    "Prompt revisions must name the requested existing node"
                )
            if not isinstance(draft["prompt"], str) or not isinstance(
                draft["persona"], str
            ):
                raise ValueError("Prompt and persona must be strings")
            old = set(tb.extractChatVars(current_node.get("prompt", ""))) | set(
                tb.extractChatVars(current_node.get("persona", ""))
            )
            new = set(tb.extractChatVars(draft["prompt"])) | set(
                tb.extractChatVars(draft["persona"])
            )
            if not old <= new:
                raise ValueError(f"Revision lost variables: {sorted(old - new)}")
        elif validate_network:
            validate_network(draft["network"])
        elif not isinstance(draft.get("network"), dict):
            raise ValueError("network must be an object")

    context = {
        "request": request,
        "current_network": network,
        "schema": schemas[kind],
        "synthetic_example": SCHEMA_EXAMPLE,
        "examples": examples or [],
    }
    prompt = (
        "Draft a " + kind + " artifact using the following context. "
        "Keep the project simple and preserve its purpose and bracket variables. "
        "Nodes require unique keys and a Start node. Edges use explicit source/target and "
        "null/isYes/isNo or an existing custom predicate. Built-ins include pass, insertJson, mergeJson, "
        "getYN and expandIterable. Expansion uses fx_params.chatNet for a registered network name; "
        "its child input variable is sourceNode_item. Python FX use (response, chatVars) and a CHAT_FX dictionary. "
        "Use TabulAIrity modules for any model inference. No credentials, deployment code, invented APIs, "
        "or unrequested dependencies. Supply the exact JSON schema, no markdown.\n"
        + json.dumps(context, ensure_ascii=False)
    )
    return _structured(
        prompt,
        model,
        validate,
        tokens=tokens,
        extra_params=extra_params,
        workflow=workflow or getWorkbenchWorkflows()[kind],
        workflow_vars=workflow_vars,
        fxStore=fxStore,
        observer=observer,
    )


def validateFxSource(code):
    """Inspect an explicit export map and callable signatures without execution."""
    if not isinstance(code, str) or not code.strip():
        raise ValueError("A Python module requires source code")
    tree = ast.parse(code)
    definitions = {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    export = next(
        (
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CHAT_FX" for t in n.targets)
        ),
        None,
    )
    if not isinstance(export, ast.Dict) or not export.keys:
        raise ValueError(
            'Export functions with a literal CHAT_FX = {"name": function} dictionary'
        )
    names = []
    for key, value in zip(export.keys, export.values):
        if (
            not isinstance(key, ast.Constant)
            or not isinstance(key.value, str)
            or not key.value
        ):
            raise ValueError("CHAT_FX keys must be nonempty strings")
        names.append(key.value)
        function = (
            definitions.get(value.id)
            if isinstance(value, ast.Name)
            else value
            if isinstance(value, ast.Lambda)
            else None
        )
        if function is not None:
            args = function.args
            if len(args.posonlyargs) + len(args.args) < 2 and args.vararg is None:
                raise ValueError(f"{key.value} must accept response and chatVars")
    if len(names) != len(set(names)):
        raise ValueError("CHAT_FX export names must be unique")
    compile(tree, "<chatFX draft>", "exec")
    return names


def reviewExtractions(
    records,
    network,
    *,
    model=DEFAULT_MODEL,
    max_records=8,
    max_chars=24000,
    tokens=8192,
    extra_params=None,
    workflow=None,
    workflow_vars=None,
    fxStore=None,
    observer=None,
):
    """Review bounded evidence; record hypotheses separately from observations.

    Revisions are proposals, never applied here. LLM judgements are not labels
    or accuracy measurements and should be checked against held-out examples.
    """
    selected = []
    for record in records[:max_records]:

        def bound(value):
            text = json.dumps(value, ensure_ascii=False, default=str)
            return (
                value if len(text) <= 2400 else text[:2400] + " [TRUNCATED FOR REVIEW]"
            )

        item = {k: bound(v) for k, v in record.items()}
        if len(json.dumps(selected + [item], ensure_ascii=False)) > max_chars:
            break
        selected.append(item)
    if not selected:
        raise ValueError("No review records fit within the review budget")
    # The node definition used for this run must accompany its evidence.
    context = {
        "records": selected,
        "nodes": network.get("nodes", []),
        "edges": network.get("edges", []),
    }
    context_text = json.dumps(context, ensure_ascii=False)
    if len(context_text) > max_chars + 12000:
        context["nodes"] = [
            {
                k: (
                    v[:1500] + " [TRUNCATED FOR REVIEW]"
                    if isinstance(v, str) and len(v) > 1500
                    else v
                )
                for k, v in n.items()
            }
            for n in context["nodes"]
        ]
        context_text = json.dumps(context, ensure_ascii=False)
    if len(context_text) > max_chars + 12000:
        raise ValueError(
            "Network context exceeds review budget; review a smaller network"
        )
    prompt = (
        """Review these extraction records against their supplied inputs and prompts.
Flag only observable failures: missing evidence, unsupported values, formatting failures,
variable errors, or failure to follow explicit constraints. Separate evidence from hypotheses.
Do not treat your world knowledge as source evidence. A truncated source cannot establish absence.
Propose prompt or Python FX revisions where useful, without changing project purpose or bracket variables.
Use TabulAIrity for any inference. Return JSON:
{"flags":[{"row_id":"id from record", "node":"node key", "reason":"observable failure", "evidence":"source excerpt"}],
 "failure_modes":[{"observation":"...", "hypothesis":"...", "correction":"..."}],
 "revisions":[{"kind":"prompt", "node":"existing key", "prompt":"revised", "persona":"revised", "rationale":"..."},
              {"kind":"fx", "filename":"cleaner.py", "code":"module with CHAT_FX", "rationale":"..."}]}
If evidence is insufficient, return empty lists. No markdown.
"""
        + context_text
    )
    row_ids = {str(r.get("row_id")) for r in selected}
    nodes = {n["key"]: n for n in network.get("nodes", [])}

    def validate(draft):
        for key in ("flags", "failure_modes", "revisions"):
            if not isinstance(draft.get(key), list):
                raise ValueError(f"{key} must be a list")
        for flag in draft["flags"]:
            if str(flag["row_id"]) not in row_ids or flag["node"] not in nodes:
                raise ValueError("A flag names a row or node outside this review")
        for revision in draft["revisions"]:
            if revision["kind"] == "fx":
                validateFxSource(revision["code"])
            elif revision["kind"] == "prompt":
                original = nodes[revision["node"]]
                old = set(tb.extractChatVars(original.get("prompt", ""))) | set(
                    tb.extractChatVars(original.get("persona", ""))
                )
                new = set(tb.extractChatVars(revision["prompt"])) | set(
                    tb.extractChatVars(revision["persona"])
                )
                if not old <= new:
                    raise ValueError("A supervisor revision removed bracket variables")
            else:
                raise ValueError(
                    "Only prompt and fx revisions are accepted from reviews"
                )

    result = _structured(
        prompt,
        model,
        validate,
        tokens=tokens,
        extra_params=extra_params,
        workflow=workflow or getWorkbenchWorkflows()["review"],
        workflow_vars=workflow_vars,
        fxStore=fxStore,
        observer=observer,
    )
    result["reviewed_records"] = len(selected)
    return result


def getWorkbenchWorkflows():
    """Synthetic, editable CDEN defaults for the local workbench."""

    def step(key, prompt, fx="pass", tokens=2000, model="[assistant_model]"):
        return {
            "key": key,
            "prompt": prompt,
            "persona": "You develop simple, evidence-grounded data extraction workflows.",
            "fx": fx,
            "tokens": tokens,
            "model": model,
        }

    def edge(a, b, fx="null"):
        return {"source": a, "target": b, "fx": fx}

    prompt_net = {
        "nodes": [
            {
                "key": "Start",
                "prompt": "recall: ready",
                "persona": "",
                "fx": "pass",
                "tokens": 1,
            },
            step(
                "intent",
                "Infer the likely intent of this prompt revision request. Distinguish the explicit intent from your guesses and name ambiguities.\n[request]\n[validation_feedback]",
            ),
            step(
                "expected_output",
                "Identify the expected output schema and required evidence from this request. Preserve explicit constraints.\n[request]\nIntent analysis:\n[intent]",
            ),
            step(
                "failure_modes",
                "Identify plausible, testable failure modes in the current prompt. Separate observed defects from hypotheses.\n[request]\nIntent: [intent]\nOutput: [expected_output]",
            ),
            step(
                "draft_prompt",
                "[request]\nIntent: [intent]\nExpected output: [expected_output]\nPotential failures: [failure_modes]\n[validation_feedback]\nReturn only the requested prompt revision JSON object.",
                fx="insertJson",
                tokens=8192,
            ),
            step(
                "alignment",
                "Does the revision preserve the explicit intent, output requirements and bracket variables? Judge against supplied text only. Return exactly yes or no.\nRequest: [request]\nIntent: [intent]\nExpected output: [expected_output]\nDraft: [draft_prompt_raw]",
                fx="getYN",
                tokens=256,
                model="[supervisor_model]",
            ),
            step(
                "revised_prompt",
                "[request]\nRevise the prior draft because its alignment check failed. Preserve bracket variables and correct the concrete mismatch against the supplied request.\nIntent: [intent]\nExpected output: [expected_output]\nPrior draft: [draft_prompt_raw]\nFailure analysis: [failure_modes]\nReturn only the requested prompt revision JSON object.",
                fx="insertJson",
                tokens=8192,
            ),
            step(
                "revised_alignment",
                "Does this revised draft align with the explicit intent, output requirements and bracket variables? Return exactly yes or no.\nRequest: [request]\nIntent: [intent]\nExpected output: [expected_output]\nRevision: [revised_prompt_raw]",
                fx="getYN",
                tokens=256,
                model="[supervisor_model]",
            ),
        ],
        "edges": [
            edge("Start", "intent"),
            edge("intent", "expected_output"),
            edge("expected_output", "failure_modes"),
            edge("failure_modes", "draft_prompt"),
            edge("draft_prompt", "alignment"),
            edge("alignment", "revised_prompt", "isNo"),
            edge("revised_prompt", "revised_alignment"),
        ],
        "result_nodes": ["revised_prompt", "draft_prompt"],
        "check_nodes": ["alignment", "revised_alignment"],
    }

    def single():
        return {
            "nodes": [
                step(
                    "Start",
                    "[request]\n[validation_feedback]",
                    fx="insertJson",
                    tokens=8192,
                )
            ],
            "edges": [],
            "result_nodes": ["Start"],
        }

    return {"prompt": prompt_net, "fx": single(), "cden": single(), "review": single()}
