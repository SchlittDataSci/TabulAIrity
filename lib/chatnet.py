import networkx as nx
import pandas as pd
import chatprompts as ncp

from copy import deepcopy
from matplotlib import pyplot as plt




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
                chatResponse = ncp.askChatQuestion(prompt,
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
                worthUsing = ncp.isUseful(prompt,chatResponse)
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