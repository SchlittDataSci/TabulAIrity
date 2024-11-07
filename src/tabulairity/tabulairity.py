import networkx as nx
import pandas as pd

from copy import deepcopy
from matplotlib import pyplot as plt
from time import sleep
from bs4 import BeautifulSoup
from litellm import completion

import os
import json
import requests
import osmnx
import pickle
import hashlib



#########################################
#                                       #
#     ENVIRONMENT PREP.                 #
#                                       #
#########################################



cachePath = 'TabulAIrityCache'

if not os.path.exists(cachePath):
    os.mkdir(cachePath)


def prepEnvironment():
    """Loads Open AI api key from local file"""
    credentialsRef = 'login/credentials.txt'
    if os.path.exists(credentialsRef):
        with open('login/credentials.txt') as credentials:
            lines = credentials.readlines()
        for line in lines:
            if ' = ' in line:
                [arg,value] = [i.strip() for i in line.split(' = ')][:2]
                os.environ[arg] = value
    else:
        print("Credentials file not found. If this is intentional, you must manually add LLM credentials to your system environment in order to use TabulAIrity. Please refer to: https://docs.litellm.ai/docs/.")
    
    
client = prepEnvironment()
modelName = "gpt-4o-mini"




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
    script.fx.fillna('null', inplace = True)
    script.prompt.fillna('', inplace = True)
    script.self_eval.fillna(False, inplace = True)
    chatEdges = script[script.type == 'edge']
    chatNodes = script[script.type == 'node']
    G = nx.MultiDiGraph()

    nodesParsed = [(row['key'],
                  {'prompt':row['prompt'],
                   'fx':row['fx'],
                   'persona':row['persona'],
                   'tokens':row['tokens'],
                   'self_eval':row['self_eval']}) for index,row in chatNodes.T.items()]

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
        #pos = nx.fruchterman_reingold_layout(G)
        #pos['start'] = (0,0)
        pos = nx.spring_layout(G,
                               pos=pos,
                               iterations=10)
        #pos['start'] = (0,0)
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



def walkChatNet(G,
                fxStore=dict(),
                varStore=dict(),
                verbosity=1):
    """Walks the chat network, interrogating data through all available paths"""
    toAsk = ['Start']
    fxStore['null'] =  lambda x,y:True
    fxStore['pass'] =  lambda x,y:x
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
        failed = False

        if validRun(persona,prompt):
            if not str(prompt).startswith('recall:'):
                chatResponse = askChatQuestion(prompt,
                                                   persona,
                                                   tokens)
            else:
                chatResponse = prompt[7:].strip()
                if verbosity > 0:
                    print(prompt)
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
#     OPEN AI CACHING FUNCTIONS         #
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




#########################################
#                                       #
#     CHAT QUERIES                      #
#                                       #
#########################################




def getChatContent(messages,tokens,modelName):
    """Wrapper to load chat content from OpenAI"""
    content = completion(model=modelName,
                         max_tokens=int(tokens),
                         messages=messages)
    cleaned = content.choices[0].message.content.strip() if content.choices[0].message.content else ''
    return cleaned



def askChatQuestion(prompt,persona,tokens=200):
    messages = [{'role':'system',
                 'content':f'You are {persona}. You must answer questions as {persona}.'},
                {'role':'user',
                 'content':prompt[:350000]}]

    query = f"getChatContent({messages},{tokens},'{modelName}')"
    result = queryToCache(query)
    return result



def getYN(text):
    messages = [{'role':'system',
                 'content':'You are an API that standardizes yes or no answers. You may only return a one word answer in lowercase or "None" as appropriate.'},
                {'role':'user',
                 'content':f'Please return a value for the following text, coding the ouput as "yes" for any affirmative response, "no" for any negative response: {text}'}]

    query = f"getChatContent({messages},3,'{modelName}')"
    result = queryToCache(query)
    result = result.lower().replace('"','')
    return result



def ynToBool(evaluation):
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



def evaluateAnswerOld(question, response):
    messages = [{'role':'system',
                 'content':'You are a skilled moderator of debates. You must answer questions as a skilled moderator of debates.'},
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
