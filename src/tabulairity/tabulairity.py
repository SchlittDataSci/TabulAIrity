import networkx as nx
import pandas as pd
import numpy as np

import scrapertools as st

from datetime import datetime
from copy import deepcopy
from matplotlib import pyplot as plt
from time import sleep
from bs4 import BeautifulSoup
from litellm import completion
from langdetect import detect
from random import uniform, randint

import os
import re
import json
import requests
import osmnx
import pickle
import hashlib
import pycountry



#########################################
#                                       #
#     ENVIRONMENT PREP.                 #
#                                       #
#########################################



cachePath = 'TabulAIrityCache'
config = dict()

if not os.path.exists(cachePath):
    os.mkdir(cachePath)

modelName = "gemma3:12b"
maxTranslateTokens = 8000
promptDelay = 0.0
targetLanguage = 'en'
translationModel = "gemma3:27b"


def prepEnvironment():
    """Loads Open AI api key from local file"""
    credentialsRef = 'config/environment_args.txt'
    if os.path.exists(credentialsRef):
        with open(credentialsRef) as credentials:
            lines = credentials.readlines()
        for line in lines:
            if ' = ' in line:
                [arg,value] = [i.strip() for i in line.split(' = ')][:2]
                os.environ[arg] = value
    else:
        print("Environment args not found. If this is intentional, you must manually add LLM credentials to your system environment in order to use TabulAIrity. Otherwise, TabulAIrity will assume a local ollama installation. Please refer to: https://docs.litellm.ai/docs/.")

    configRef = 'config/config.txt'
    if os.path.exists(configRef):
        with open(configRef) as configs:
            lines = configs.readlines()
        for line in lines:
            if ' = ' in line:
                [arg,value] = [i.strip() for i in line.split(' = ')][:2]
                config[arg] = value

    routesRef = 'config/model_routes.csv'
    if os.path.exists(routesRef):
        routes = pd.read_csv(routesRef)
        routes = routes.replace({np.nan: None, 'remote': None})
        routes['last used'] = datetime.utcnow()
    else:
        routes = {modelName:{'route':modelName,
                             'ip':'http://localhost:11434'}}
        routes = pd.DataFrame(routes).T

    return routes


modelRoutes = prepEnvironment()



def getModelRoute(name):
    """Model route accessor with LRU (least recently used) logic and user reminders"""
    global modelRoutes  # assume global persistence

    for col in ['model', 'route', 'ip', 'last used']:
        if col not in modelRoutes.columns:
            modelRoutes[col] = None

    matches = modelRoutes.loc[modelRoutes['model'] == name]

    if matches.empty:
        print(f"{name} not found in model routes (config/model_routes.csv), adding manually...")
        if 'ollama/' in name:
            defaultIP = 'http://localhost:11434'
        else:
            defaultIP = None

        modelRoutes.loc[len(modelRoutes)] = {'model': name,
                                             'route': name,
                                             'ip': defaultIP,
                                             'last used': datetime.utcnow()}
        return name, defaultIP

    if len(matches) > 1:
        temp = matches.copy()
        temp['last used'] = pd.to_datetime(temp['last used'], errors='coerce')
        if temp['last used'].notnull().any():
            chosenIdx = temp['last used'].idxmin()  # this is the actual index in modelRoutes
        else:
            chosenIdx = temp.index[0]
        routeRow = modelRoutes.loc[chosenIdx]
    else:
        chosenIdx = matches.index[0]
        routeRow = modelRoutes.loc[chosenIdx]

    modelRoutes.at[chosenIdx, 'last used'] = datetime.utcnow()

    return routeRow['route'], routeRow['ip']


    

#########################################
#                                       #
#     TEXT INTEROGATION FUNCTIONS       #
#                                       #
#########################################


def validRun(var,prompt):
    """returns true if the persona is valid or disused"""
    return isValid(var) or str(prompt).startswith('recall:')



def isValid(var):
    """returns true if var is displayable"""
    return var == var and var not in {None,''}



def showIfValid(var):
    """prints a var if it is printable"""
    if isValid(var):
        print(var)

        
def mapEdgeColor(fx):
    if fx == 'null':
        return 'black'
    elif fx == 'isYes':
        return 'blue'
    elif fx == 'isNo':
        return 'red'
    else:
        return 'green'

    
def buildChatNet(script,show=False):
    """Builds a chat network from a formatted csv"""
    script['fx'] = script['fx'].fillna('null')
    script['prompt'] = script['prompt'].fillna('')
    script['self_eval'] = script['self_eval'].fillna(False)
    
    if 'model' not in script.columns:
        script.loc[:,'model'] = modelName
    
    chatEdges = script[script.type == 'edge']
    chatNodes = script[script.type == 'node']
    G = nx.MultiDiGraph()

    nodesParsed = [(row['key'],
                  {'prompt':row['prompt'],
                   'fx':row['fx'],
                   'persona':row['persona'],
                   'tokens':row['tokens'],
                   'self_eval':row['self_eval'],
                   'model':row['model']}) for index,row in chatNodes.T.items()]

    G.add_nodes_from(nodesParsed)

    splitEdge = lambda x: x['key'].split('-')
    edgesParsed = {tuple(splitEdge(row)+[row['fx']]):{'prompt':row['prompt'],'fx':row['fx'],} for index,row in chatEdges.T.items()}
    G.add_edges_from(edgesParsed)
    nx.set_edge_attributes(G, edgesParsed)
    connected = nx.is_weakly_connected(G)
    
    if not connected:
        print(f"Warning, the chat graph has one or more stray components.")
    
    if show:
        pos = nx.kamada_kawai_layout(G)
        pos = nx.spring_layout(G,
                               pos=pos,
                               iterations=10)
        colors = [mapEdgeColor(i[2]) for i in G.edges]

        fig,ax = plt.subplots(figsize=(10,10))
        nx.draw_networkx_edges(G,
                               pos = pos,
                               edge_color = colors,
                               connectionstyle="arc3,rad=0.1",
                               alpha = .6)
        
        nx.draw_networkx_nodes(G,
                               pos = pos,
                               alpha = .6)
        
        nx.draw_networkx_labels(G,
                               pos = pos)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.savefig('lastplot.png')
        
                
    return G



def insertChatVars(text,varStore):
    """Adds vars from the varstore into a chat prompt"""
    for key,value in varStore.items():
        toReplace = f'[{key}]'
        text = text.replace(toReplace,str(value))
    return text



def extractChatVars(text): 
    """Returns the set of chatvars from a prompt"""
    matches = set(re.findall(r"\[([^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*)\]", text))
    matches = [match for match in matches if "\n" not in match and "," not in match]
    matches = [match for match in matches if match != '']
    return matches


baseFx = {'isYes':lambda x,y: ynToBool(x),
          'isNo':lambda x,y: not ynToBool(x),
          'getYN':lambda x,y: getYN(x),
          'null':lambda x,y: True,
          'pass':lambda x,y: x}

    
def walkChatNet(G,
                fxStore=dict(),
                varStore=dict(),
                verbosity=1):
    """Walks the chat network, interrogating data through all available paths"""
    toAsk = ['Start']
    fxStore = fxStore | baseFx
    chatVars = deepcopy(varStore)
    while toAsk != []:
        nextQ = toAsk.pop()
        nodeVars = G.nodes[nextQ]
        prompt = insertChatVars(nodeVars['prompt'],chatVars)
        if verbosity == 2:
            print()
            print(prompt)

        tokens = nodeVars['tokens']
        persona = nodeVars['persona']
        rowModel = nodeVars['model']
        
        failed = False

        if validRun(persona,prompt):
            if not str(prompt).startswith('recall:'):
                chatResponse = askChatQuestion(prompt,
                                               persona,
                                               model=rowModel,
                                               tokens=tokens)
            else:
                chatResponse = prompt[7:].strip()
                if verbosity > 0:
                    print(chatResponse)
                
            chatVars[nextQ+'_prompt'] = prompt
            chatVars[nextQ+'_raw'] = chatResponse
            selfEval = nodeVars['self_eval']

            if selfEval:
                worthUsing = isUseful(prompt,chatResponse)
            else:
                worthUsing = True


            if worthUsing:
                try:
                    cleanedResponse = fxStore[nodeVars['fx']](chatResponse,chatVars)
                except:
                    cleanedResponse = chatResponse
                chatVars[nextQ] = cleanedResponse
                if verbosity > 0:
                    print(f'\t-{persona}: {chatResponse} ({cleanedResponse})')
            else:
                if verbosity > 0:
                    print(f'\t*FAILS: {chatResponse}')
                failed = True

        if not failed:
            edgesFromQ = G.out_edges([nextQ],data=True)
            nextNodes = []

            for start,end,edgeData in edgesFromQ:
                edgeResult = fxStore[edgeData['fx']](chatResponse,chatVars)
                chatVars[f'{start}-{end}'] = edgeResult
                if edgeResult in {True,'true','True'}:
                    nextNodes.append(end)
                    prompt = insertChatVars(edgeData['prompt'],chatVars)
                    showIfValid(prompt)
                elif edgeResult in {False,'false','False'}:
                    pass
            nextNodes.sort(reverse=True)
            toAsk += nextNodes

    return chatVars





#########################################
#                                       #
#     QUERY CACHING FUNCTIONS           #
#                                       #
#########################################



def getHash(query):
    """For a given query, returns a hash str"""
    hasher = hashlib.md5()
    encoded = str(query).encode('utf-8')
    hasher.update(encoded)
    result = hasher.hexdigest()
    return result



def queryToCache(query,
                 maxAttempts = 3,
                 tolerant = False,
                 delay=.05):
    """Attempts to eval a given str query, pulling from cache if found or caching if new and successful"""
    fileRef = f'{cachePath}/{getHash(query)}.json'
    
    
        
    if os.path.exists(fileRef):
        with open(fileRef,'r') as cacheIn:    
            result = json.load(cacheIn)['response']
        
    else:
        sleep(promptDelay)
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
                
        jsonOut = {'query':query,
                   'response':result}
             
        with open(fileRef,'w') as cacheOut:
            json.dump(jsonOut,cacheOut)
        #queryKey = bytes(query,'utf-8')
        #db[queryKey] = json.dumps(result)
    return result



def scrapePage(url):
    """Pulls a beautifulsoup representation of a page"""
    response = requests.get(url)
    statusCode = response.status_code
    if statusCode == 200:
        return response.text
    else:
        raise ValueError(f"Page returned unscrapable status code {statusCode}")
    


def cachePage(url):
    """Attempts to scrape a page, pulling from cache if found or caching if new and successful"""
    result = queryToCache(f"st.scrapePageText('{url}')")
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



#########################################
#                                       #
#     LANGUAGE HANDLING                 #
#                                       #
#########################################


def getLanguageName(code):
    """Returns language name from language codes, defaulting to english"""
    lang = pycountry.languages.get(alpha_2=code)
    return lang.name if lang else "English"



def translateOne(text):
    """Global arg-heavy text translation function"""
    languageName = getLanguageName(targetLanguage)
    translationPersona = f"You are a highly accurate and fluent {languageName} translator."
    translationPrompt = f"Translate the following text to {languageName}. Output only the translated text. Do not include any markdown, explanations, commentary, variable placeholders, or descriptive text.\n\n{text.strip()}"
    
    translation = askChatQuestion(translationPrompt,
                                  translationPersona,
                                  tokens = maxTranslateTokens,
                                  model = translationModel)
    return translation


def getLanguage(text):
    """Checks language of text"""
    if text in {'',None,np.nan}:
        language = "unidentified"
    else:
        try:
            language = detect(text)
        except:
            language = "unidentified"
    return language


def autoTranslate(dfIn,
                  column,
                  targetLanguage = 'en',
                  model = modelName):
    """Automatically translates all values in one column that are not in the target language"""
    df = dfIn.copy(deep=True)
    langOut = f'{column}_language'
    textOut = f'{column}_translated'
    df.loc[:,langOut] = df[column].apply(getLanguage)
    df.loc[:,textOut] = df[column]
    df.loc[df[langOut] != targetLanguage,textOut] = df.loc[df[langOut] != targetLanguage,textOut].apply(translateOne)

    return df



#########################################
#                                       #
#     CHAT QUERIES                      #
#                                       #
#########################################



def testRoutes(query = 'How many Rs are there in strawberry?',
               persona = 'an AI assistant',
               autoformatPersona = True):
    """Quick method to test models from config/model_routes.py"""
    working = []
    for model in tb.modelRoutes.index.sort_values():
        try:
            response = tb.askChatQuestion(query,
                                          persona,
                                          autoformatPersona = autoformatPersona,
                                          model = model)
            print(f'{model} ~ {response}\n')
            working.append(model)
        except:
            print(f'{model} ~ FAILS\n')
    return working



def getChatContent(messages,
                   tokens,
                   modelName,
                   temperature = None,
                   seed = None):
    """Wrapper to load chat content from OpenAI"""
    modelRoute, ip = getModelRoute(modelName)
    content = completion(model = modelRoute,
                         max_tokens = int(tokens),
                         messages = messages,
                         api_base = ip,
                         seed = seed,
                         temperature = temperature)
    cleaned = content.choices[0].message.content.strip() if content.choices[0].message.content else ''
    return cleaned



def askChatQuestion(prompt,
                    persona,
                    model = modelName,
                    autoformatPersona = None,
                    tokens = 2000,
                    temperature = None,
                    seed = None):
    """Simple method to ask a single chat question"""
    
    if autoformatPersona is True and persona.strip()[-1] != '.':
        personaText = f'You are {persona}. You must answer questions as {persona}.'
    else:
        personaText = persona
    
    messages = [{'role':'system',
                 'content':personaText},
                {'role':'user',
                 'content':prompt[:350000]}]

    query = f"getChatContent({messages},{tokens},'{model}',{temperature},{seed})"
    result = queryToCache(query)
    return result



def getYN(text):
    """Input cleaner to standardize yes or no answers"""
    messages = [{'role':'system',
                 'content':'You are an API that standardizes yes or no answers. You may only return a one word answer in lowercase or "None" as appropriate.'},
                {'role':'user',
                 'content':f'Please return a value for the following text, coding the ouput as "yes" for any affirmative response, "no" for any negative response: {text}'}]

    query = f"getChatContent({messages},3,'gemma3:12b')"
    result = queryToCache(query)
    result = result.lower().replace('"','')
    return result



def ynToBool(evaluation):
    """Input cleaner to convert yes or no answers to boolean"""
    textAnswer = getYN(evaluation)
    textAnswer = ''.join(i for i in textAnswer if i.isalnum())
    result = {'y':True,
              'n':False}[textAnswer[0].lower()]
    return result



def evaluateAnswer(question, response):
    messages = [{'role':'system',
                 'content':'You are a debate moderator skilled at identifying the presence of answer in long statements'},
                {'role':'user',
                 'content':f'Please answer in one short sentence, does the following answer provide any useable answer for the provided question?\nquestion: {question}\nanswer: {response}'}]

    query = f"getChatContent({messages},100,'{modelName}')"
    result = queryToCache(query)

    return result



def evaluateAuthor(response):
    messages = [{'role':'user',
                 'content':f'Please answer in one short sentence, does the author of the following answer include any text specically identifying itself as an AI?\nanswer: {response}'}]

    query = f"getChatContent({messages},100,'{modelName}')"
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

    query = f"getChatContent({messages},3,'{modelName}')"
    result = queryToCache(query)

    return result
