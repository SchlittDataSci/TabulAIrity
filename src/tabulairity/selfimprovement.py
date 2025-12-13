import re
import pandas as pd
from random import randint
from datetime import datetime

import tabulairity as tb
import gsheetconnector as gs


verbosity = 1


def getSeedParams(randomize, model):
    """Returns seed text and string depending on model"""
    seedSupported = not model.startswith('gemini')
    if not randomize:
        seed = None
        seedText = ''
    if seedSupported:
        seed = randint(0,9999)
        seedText = ''
    else:
        seed = None
        seedText = f'(random seed = {randint(0,9999)})'
    return seed, seedText



def getEvaluatorNet(supervisor='gemma3:27b'):
    """Pulls and preps true/false evaluation net"""
    evaluatorNetDf = gs.gSheetToDf('Google Alerts Trackers',
                                   'Answer Check Network',
                                   tb.config)
    evaluatorNetDf.loc[:,'model'] = supervisor
    evaluatorNet = tb.buildChatNet(evaluatorNetDf)

    return evaluatorNet



def rewritePrompt(prompt,
                  persona,
                  model,
                  rewritePersona = False,
                  intentPrompt = None,
                  errorSummary = None,
                  randomize = True):
    """"Takes a given prompt and persona and returns an LLM improved version of each"""
    maxTries = 20
    
    if intentPrompt is None:
        intentPrompt = ''
    else:
        intentPrompt = ' ' + intentPrompt
    if errorSummary is None:
        errorSummary = ''
    else:
        errorSummary = ' ' + errorSummary

    seed, seedText = getSeedParams(randomize,model)
    keptAllVars = False
    preservedIntent = False
    originalVars = tb.extractChatVars(prompt)
    tries = 0
    
    while not (keptAllVars and preservedIntent) and tries != maxTries:
        seed, seedText = getSeedParams(randomize,model)
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
    
        aiPrompt = tb.askChatQuestion(questionPrompt,
                                      'You are a skilled prompt engineer.',
                                      model = model,
                                      seed = seed)
        aiVars = tb.extractChatVars(prompt)
        keptAllVars = aiVars == originalVars
        preservedIntent = validatePromptIntent(aiPrompt,
                                               intentPrompt.strip(),
                                               model)
        tries += 1

    if tries == maxTries and not keptAllVars:
        raise ValueError(f"Rewritten prompts failed to retain chatvars after {maxTries} attempts.")
    elif tries == maxTries and not preservedIntent:
        raise ValueError(f"Rewritten prompts failed to preserve intent after {maxTries} attempts.")

    if not rewritePersona:
        return aiPrompt, persona

    personaPrompt = f"""Please consider the following prompt and revise our LLM system text to maximize prompt efficiency if needed.
The current system text is "{persona}". {intentPrompt}{errorSummary}If you decide to replace it, your replacement should be optimized for use with LLM APIs.
You must only return the recommended system text for our prompt.
You absolutely must not provide descriptions or commentary.
{seedText}
{prompt}"""

    aiPersona = tb.askChatQuestion(personaPrompt,
                                   'You are a skilled prompt engineer.',
                                   model = model,
                                   seed = seed)
    
    return aiPrompt, aiPersona

def summarizeErrors(errors,
                    intent,
                    model):
    persona = "You are a skilled evaluator."
    seed, seedText = getSeedParams(True,model)

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
    errorSummary = tb.askChatQuestion(errorPrompt,
                                      'You are a skilled evaluator.',
                                      model = model,
                                      seed = seed)
    while '\n\n' in errorSummary:
        errorSummary = errorSummary.replace('\n\n','\n')
    errorSummary = errorSummary.replace('.\n','. ')
    return errorSummary



def evaluateAnswer(originalPrompt,
                   persona,
                   varsIn,
                   model,
                   evaluatorNet):
    """Evaluates the y/n correctness of an answer, returning an explanation of the error if found"""
    
    try:
        tStart = datetime.now()
        preppedPrompt = tb.insertChatVars(originalPrompt,varsIn)
        answer = tb.askChatQuestion(preppedPrompt,
                                    persona,
                                    model,
                                    tokens = 4000)
        tFinish = datetime.now()
        duration = (tFinish - tStart).total_seconds()
        
        answerVars = {'preppedPrompt':preppedPrompt,
                      'answer':answer}
        varsOut = dict(varsIn) | answerVars
        evaluation = tb.walkChatNet(evaluatorNet,
                                    varStore = varsOut,
                                    verbosity = verbosity)
        if tb.ynToBool(evaluation['Start']):
            answeredCorrectly = True
            explanation = None
        else:
            answeredCorrectly = False
            explanation = evaluation['Explain error']
    except Exception as e:
        print("Error:",e)
        answeredCorrectly = False
        explanation = None
        
    return answeredCorrectly, explanation, duration


def extractIntent(prompt,
                  model):
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
    intent = tb.askChatQuestion(intentPrompt,
                                evaluationPersona,
                                model,
                                autoformatPersona = False)
    return intent


                                
def validatePromptIntent(prompt,
                         intent,
                         model = 'gemma3:27b'):
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
    evaluatorPrompt = tb.insertChatVars(evaluatorPrompt,{'intent':intent,
                                                         'prompt':prompt})
    intentCheck = tb.askChatQuestion(evaluatorPrompt,
                                     'You are a classifier.',
                                     model = model)

    result = {'yes':True,'no':False}[tb.getYN(intentCheck)]
    return result



def iteratePrompt(bestPrompt,
                  bestPersona,
                  testDfIn,
                  model,
                  depth=20,
                  intent=None,
                  supervisor=None):
    """Processes a tabulairity extraction for a single prompt and iteratively improves it"""
    if supervisor is None:
        supervisor = model

    evaluatorNet = getEvaluatorNet(supervisor)

    if intent is None:
        intent = extractIntent(bestPrompt, supervisor)
        print("Inferred intent:",intent)

    testDf = testDfIn.copy(deep=True)
    varsInData = set(testDf.columns)
    varsInPrompt = set(tb.extractChatVars(bestPrompt))
    missingVars = varsInPrompt.difference(varsInData)
    #print("DEBOOO1",varsInPrompt)
    #print("DEBOOO2",varsInData)
    if len(missingVars) != 0:
        print(f"Warning: some expected prompt vars were not found in the passed DataFrame, is this intentional?\n\t{missingVars}")

    # Evaluate initial prompt
    bestErrors = []
    bestScores = []
    bestTimes = []
    for _, row in testDf.iterrows():
        result, error, duration = evaluateAnswer(bestPrompt,
                                                 bestPersona,
                                                 row,
                                                 model,
                                                 evaluatorNet)
        bestScores.append(result)
        bestTimes.append(duration)
        if error is not None:
            bestErrors.append(error)

    bestScore = float(sum(bestScores) / len(bestScores))
    bestTime = float(sum(bestTimes) / len(bestTimes))
    initialScore = bestScore
    history = [{'prompt':bestPrompt,
                'intent':intent,
                'persona':bestPersona,
                'score':bestScore,
                'time':bestTime,
                'iteration':0}]

    if bestScore == 1:
        print("All responses flagged as correct, returning finalized prompt and persona.")
        return pd.DataFrame(history)

    if bestErrors == []:
        errorReport = "No errors were logged."
    else:
        errorReport = '\n'.join([f'* {e}' for e in set(bestErrors)])

    # Begin iterative improvements
    for i in range(depth):
        print(f'\nModel: {model}\nIteration: {i}\nbest prompt: {bestPrompt}\n')
        newPrompt, newPersona = rewritePrompt(bestPrompt,
                                              bestPersona,
                                              supervisor,
                                              intentPrompt = intent,
                                              errorSummary = errorReport)

        newErrors = []
        newScores = []
        newTimes = []
        for _, row in testDf.iterrows():
            result, error, duration = evaluateAnswer(newPrompt,
                                                     newPersona,
                                                     row,
                                                     model,
                                                     evaluatorNet)
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
            
        history.append({'prompt':bestPrompt,
                        'intent':intent,
                        'persona':bestPersona,
                        'score':bestScore,
                        'time':bestTime,
                        'iteration':i})

        if bestScore == 1:
            print("All responses flagged as correct, returning finalized prompt and persona.")
            return pd.DataFrame(history)


        errorList = '\n'.join([f'* {iError}' for iError in set(bestErrors)])
        errorReport = summarizeErrors(errorList,
                                      intent,
                                      supervisor)
        errorReport = f'The current prompt yields {round(bestScore,2)*100}% accuracy. {errorReport}'
        print(errorReport)


    print(f"Iterations complete, initial score: {initialScore}   final score: {bestScore}")
    return pd.DataFrame(history)
