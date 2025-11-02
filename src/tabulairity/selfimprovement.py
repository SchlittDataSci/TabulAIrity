import re
import tabulairity as tb



def evaluateAnswer(originalPrompt,
                   originalPersona,
                   model,
                   supervisor):
    try:
        answer = tb.askChatQuestion(originalPrompt,
                                    originalPersona,
                                    model,
                                    autoformatPersona = False)

        evaluationPersona = "You are an expert evaluator."
        evaluationPrompt = f""""You will be given:  
- **Prompt (the question asked)**  
- **Answer (the model’s response)**  

Your task is to determine whether the **Answer correctly and fully responds to the Prompt**.

#### Instructions:
1. Read both the **Prompt** and **Answer** carefully.  
2. Consider whether the **Answer**:
   - Accurately addresses the question asked.  
   - Contains no major factual or logical errors.  
   - Provides a relevant and direct response (not evasive, off-topic, or incomplete).  
3. If the answer meets all criteria above, respond **exactly** with:
"yes"

Otherwise, respond **exactly** with:
"no"

---

## Input Format
Prompt: {originalPrompt}
Answer: {answer}

---

## Output Format
"yes"
or
"no" """
        evaluation = tb.askChatQuestion(evaluationPrompt,
                                        evaluationPersona,
                                        supervisor,
                                        autoformatPersona = False)
        answeredCorrectly = tb.ynToBool(evaluation)
    except:
        answeredCorrectly = False
        
    return answeredCorrectly




def extractIntent(prompt,
                  model):
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