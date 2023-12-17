from openai import OpenAI
from time import sleep
from bs4 import BeautifulSoup

import os
import dbm
import json
import requests
import osmnx




#########################################
#                                       #
#     OPEN AI CACHING FUNCTIONS         #
#                                       #
#########################################



def getClient():
    """Loads Open AI api key from local file"""
    with open('login/openai.txt') as credentials:
        key = credentials.read()
    os.environ['OPENAI_API_KEY'] = key
    client = OpenAI(api_key = key)
    
    return client



def queryToCache(query,
                 maxAttempts = 3,
                 tolerant = False,
                 delay=.05):
    """Attempts to eval a given str query, pulling from cache if found or caching if new and successful"""
    with dbm.open('tempDB','c') as db:
        try:
            result = json.loads(db[query])
        except:
            sleep(delay)
            gotResults = False
            attempts = 0
            while not gotResults and attempts < maxAttempts:
                if tolerant:
                    try:
                        result = eval(query)
                        gotResults = True
                    except:
                        attempts += 1
                        sleep(5)
                else:
                    result = eval(query)
                    attempts = maxAttempts
            queryKey = bytes(query,'utf-8')
            db[queryKey] = json.dumps(result)
    return result



def scrapePage(url):
    """Pulls a beautifulsoup representation of a page"""
    response = requests.get(url)
    sleep(2)
    statusCode = response.status_code
    if statusCode == 200:
        return response.text
    else:
        raise ValueError(f"Page returned unscrapable status code {statusCode}")
    


def cachePage(url):
    """Attempts to scrape a page, pulling from cache if found or caching if new and successful"""
    result = queryToCache(f"scrapePage('{url}')")
    return result


def cacheGeocode(locText):
    """Attempts to geocode a location via osmnx"""
    try:
        result = queryToCache(f"osmnx.geocode('{locText}')")
    except:
        return None
    if type(result) not in (list,tuple):
        return None
    return list(result)

    

client = getClient()



#########################################
#                                       #
#     CHAT QUERIES                      #
#                                       #
#########################################



def getChatContent(messages,tokens):
    """Wrapper to load chat content from OpenAI"""
    content = client.chat.completions.create(model='gpt-4',
                                             max_tokens=int(tokens),
                                             messages=messages)
    cleaned = content.choices[0].message.content.strip()
    return cleaned



def askChatQuestion(prompt,persona,tokens=200):
    messages = [{'role':'system',
                 'content':f'You are {persona}. You must answer questions as {persona}.'},
                {'role':'user',
                 'content':prompt[:350000]}]

    query = f"getChatContent({messages},{tokens})"
    result = queryToCache(query)
    return result



def getYN(text):
    messages = [{'role':'system',
                 'content':'You are a python API that standardizes yes or no answers. You may only return one word in lowercase or None as appropriate.'},
                {'role':'user',
                 'content':f'Please return a value for the following text, coding the ouput as "yes" for any affirmative response, "no" for any negative response: {text}'}]

    query = f"getChatContent({messages},3)"
    result = queryToCache(query)
    result = result.lower().replace('"','')
    return result



def ynToBool(evaluation):
    textAnswer = getYN(evaluation)
    result = {'y':True,
              'n':False}[textAnswer[0].lower()]
    return result



def evaluateAnswer(question, response):
    messages = [{'role':'system',
                 'content':'You are a debate moderator skilled at identifying the presence of answer in long statements'},
                {'role':'user',
                 'content':f'Please answer in one short sentence, does the following answer provide any useable answer for the provided question?\nquestion: {question}\nanswer: {response}'}]

    query = f"getChatContent({messages},100)"
    result = queryToCache(query)

    return result



def evaluateAnswerOld(question, response):
    messages = [{'role':'system',
                 'content':'You are a skilled moderator of debates. You must answer questions as a skilled moderator of debates.'},
                {'role':'user',
                 'content':f'Please answer in one short sentence, does the following answer provide any useable answer for the provided question?\nquestion: {question}\nanswer: {response}'}]

    query = f"getChatContent({messages},100)"
    result = queryToCache(query)

    return result



def evaluateAuthor(response):
    messages = [{'role':'user',
                 'content':f'Please answer in one short sentence, does the author of the following answer include any text specically identifying itself as an AI?\nanswer: {response}'}]

    query = f"getChatContent({messages},100)"
    result = queryToCache(query)

    return result



def isUseful(question,response):
    answerEval = evaluateAnswer(question,response)
    authorEval = evaluateAuthor(response)
    answerYN = getYN(answerEval)
    authorYN = getYN(authorEval)
    print(f'is answer:{answerYN}\tis AI: {authorYN}')

    result = answerYN == 'yes' and authorYN == 'no'

    return result



def getColor(text):
    messages = [{'role':'system',
                 'content':'You are a python API that returns the first named color found in a sample of text. You may only return one word in lowercase or None if no color is found.'},
                {'role':'user',
                 'content':f'Please return a value for the following text: {text}'}]

    query = f"getChatContent({messages},3)"
    result = queryToCache(query)

    return result