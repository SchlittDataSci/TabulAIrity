import re
import pandas as pd
from random import randint

import tabulairity as tb
import gsheetconnector as gs





def getEvaluatorNet(supervisor='gemma3:27b'):
    """Pulls and preps true/false evaluation net"""
    evaluatorNetDf = gs.gSheetToDf('Google Alerts Trackers',
                                   'Answer Check Network',
                                   tb.config)
    evaluatorNetDf.loc[:,'model'] = supervisor
    evaluatorNet = tb.buildChatNet(evaluatorNetDf)

    return evaluatorNet

evaluatorNet = getEvaluatorNet()



def rewritePrompt(prompt,
                  persona,
                  model,
                  rewritePersona = False,
                  intentPrompt = None,
                  errorSummary = None,
                  randomize = True):
    """"Takes a given prompt and persona and returns an LLM improved version of each"""
    if intentPrompt is None:
        intentPrompt = ''
    else:
        intentPrompt = ' ' + intentPrompt
    if errorSummary is None:
        errorSummary = ''
    else:
        errorSummary = ' ' + errorSummary

    if randomize:
        seed = randint(0,10000)
    else:
        seed = None

    keptAllVars = False
    originalVars = tb.extractChatVars(prompt)
    tries = 0
    
    while not keptAllVars and tries != 5:
        questionPrompt = f"""Rewrite the following prompt text to maximize its performance according to the criteria below.{intentPrompt}{errorSummary}

Objectives:

* Expand and optimize the prompt so that it consistently produces clean, structured output containing only the required data.
* Ensure the rewritten prompt strictly forbids inclusion of any markdown, explanations, commentary, variable placeholders, or descriptive text.
* Explicitly state in the rewritten prompt what must be included and what must not be included in the model’s output.
* Preserve and correctly utilize all flag values indicated in square brackets [like_this].
* Do not invent or modify any flag values.
* Do not add or include meta-instructions, comments, or contextual discussion in the final rewritten text.

The prompt being improved operates under the system prompt:

{persona}

Task:
Rewrite the following prompt text (provided after this instruction block) so that it performs optimally under the criteria above.

Prompt to improve:

{prompt}"""
    
        aiPrompt = tb.askChatQuestion(questionPrompt,
                                      'You are a skilled prompt engineer.',
                                      model = model,
                                      seed = seed)
        aiVars = tb.extractChatVars(prompt)
        keptAllVars = aiVars == originalVars
        tries += 1

    if tries == 5 and not keptVallVars:
        raise ValueError("Rewritten prompts failed to retain chatvars after 5 attempts.")

    if not rewritePersona:
        return aiPrompt, persona

    personaPrompt = f"""Please consider the following prompt and revise our LLM system text to maximize prompt efficiency if needed.
The current system text is "{persona}". {intentPrompt}{errorSummary}If you decide to replace it, your replacement should be optimized for use with LLM APIs.
You must only return the recommended system text for our prompt.
You absolutely must not provide descriptions or commentary.

{prompt}"""

    aiPersona = tb.askChatQuestion(personaPrompt,
                                   'You are a skilled prompt engineer.',
                                   model = model,
                                   seed = seed)
    
    return aiPrompt, aiPersona



def summarizeErrors(errors,
                    model):
    """Summarizes a bulleted list of errors"""
    persona = "You are a skilled evaluator."
    errorPrompt = f"""You will be given a bulleted list of previously logged prompt errors.

Your task is to:
1. Write a 1–2 sentence summary describing the general nature or cause of these errors.  
   - Your summary must begin with the phrase:
     "This prompt has had issues with"
2. Follow the summary with a single sentence suggesting how the prompt could be improved to prevent these issues.

Example:
If the errors involve missing context or ambiguous instructions, your output might be:
"This prompt has had issues with unclear context and ambiguous task phrasing.  
It could be improved by explicitly defining the input format and desired output behavior."

Now produce your summary and improvement suggestion based on the provided error list.

[ERROR LIST - BULLETED]
{errors}"""
    errorSummary = tb.askChatQuestion(errorPrompt,
                                      'You are a skilled evaluator.',
                                      model = model)
    return errorSummary



def evaluateAnswer(originalPrompt,
                   persona,
                   varsIn,
                   model,
                   supervisor):
    """Evaluates the y/n correctness of an answer, returning an explanation of the error if found"""
    
    chatFx = {'isYes':lambda x,y:tb.ynToBool(x),
              'isNo':lambda x,y:not tb.ynToBool(x),
              'getYN':lambda x,y:tb.getYN(x)}
    
    if True:
        preppedPrompt = tb.insertChatVars(originalPrompt,varsIn)
        answer = tb.askChatQuestion(preppedPrompt,
                                    persona,
                                    model)
        
        #print("DEBOO1")
        answerVars = {'preppedPrompt':preppedPrompt,
                      'answer':answer}
        #print(answerVars)
        varsOut = dict(varsIn) | answerVars
        evaluation = tb.walkChatNet(evaluatorNet,
                                    varStore = varsOut,
                                    fxStore = chatFx,
                                    verbosity = 0)
        if tb.ynToBool(evaluation['Start']):
            answeredCorrectly = True
            explanation = None
        else:
            answeredCorrectly = False
            explanation = evaluation['Explain error']
    else:
        answeredCorrectly = False
        explantion = None
        
    return answeredCorrectly, explanation



def extractIntent(prompt,
                  model):
    """Summarizes the intent of a prompt"""
    evaluationPersona = "You are an expert evaluator."
    intentPrompt = f"""You are an expert at analyzing instructions and identifying their underlying purpose.  
You will be given a **Prompt**, and your task is to write **one single sentence** that clearly describes its intent.

---

## Instructions

1. Read the provided **Prompt** carefully.  
2. Determine what the user is ultimately asking the model to do.  
3. Write exactly **one sentence** beginning with:  
"The intent of this prompt is to ..."
4. Do **not** include any other commentary, formatting, or analysis — only that one sentence.

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

    if intent is None:
        intent = extractIntent(bestPrompt, supervisor)

    testDf = testDfIn.copy(deep=True)

    # Evaluate initial prompt
    bestErrors = []
    bestScores = []
    for _, row in testDf.iterrows():
        result, error = evaluateAnswer(bestPrompt,
                                       bestPersona,
                                       row,
                                       model,
                                       supervisor)
        bestScores.append(result)
        if error is not None:
            bestErrors.append(error)

    bestScore = float(sum(bestScores) / len(bestScores))
    initialScore = bestScore

    if bestScore == 1:
        print("All responses flagged as correct, returning finalized prompt and persona.")
        return bestPrompt, bestPersona

    errorReport = '\n'.join([f'* {e}' for e in set(bestErrors)])

    # Begin iterative improvements
    for i in range(depth):
        print(f'\nIteration: {i}\nbest prompt: {bestPrompt}\n')
        newPrompt, newPersona = rewritePrompt(bestPrompt,
                                              bestPersona,
                                              supervisor,
                                              errorSummary=errorReport)

        newErrors = []
        newScores = []
        for _, row in testDf.iterrows():
            result, error = evaluateAnswer(newPrompt,
                                           newPersona,
                                           row,
                                           model,
                                           supervisor)
            newScores.append(result)
            if error is not None:
                newErrors.append(error)

        newScore = float(sum(newScores) / len(newScores))

        if newScore > bestScore:
            bestScore = newScore
            bestPrompt = newPrompt
            bestPersona = newPersona
            bestErrors = newErrors

        if bestScore == 1:
            print("All responses flagged as correct, returning finalized prompt and persona.")
            return bestPrompt, bestPersona

        print(f"Best score: {bestScore}    New score: {newScore}")
        errorList = '\n'.join([f'* {iError}' for iError in set(bestErrors)])
        errorReport = summarizeErrors(errorList, supervisor)
        errorReport = f'The current prompt yields {round(bestScore,2)*100}% accuracy. {errorReport}'
        print(errorReport)


    print(f"Iterations complete, initial score: {initialScore}   final score: {bestScore}")
    return bestPrompt, bestPersona
