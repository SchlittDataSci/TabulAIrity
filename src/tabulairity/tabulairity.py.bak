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
import sqlite3
import asyncio
import traceback
import sys

#########################################
#                                       #
#      POSTGRESQL CACHE BACKEND         #
#                                       #
#########################################

try:
    import psycopg2
    from psycopg2 import pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("[Warning] psycopg2 not found. Install with: pip install psycopg2-binary")
    print("[Warning] Falling back to SQLite cache (not multi-instance safe)")

# Cache configuration
cacheConfig = {
    'backend': 'postgres',  # 'postgres' or 'sqlite'
    'host': 'localhost',
    'port': 5433,
    'database': 'tabulairity_cache',
    'user': None,  # Will use OS user if None
    'password': None,
    'minConnections': 2,
    'maxConnections': 20
}

# Global connection pool
_connectionPool = None
useCache = True
config = dict()

#########################################
#                                       #
#      CACHE BACKEND FUNCTIONS          #
#                                       #
#########################################

def initCachePool(cacheConfigOverride=None):
    """Initialize PostgreSQL connection pool"""
    global _connectionPool, cacheConfig
    
    if not POSTGRES_AVAILABLE:
        print("[Cache] PostgreSQL not available, using SQLite fallback")
        cacheConfig['backend'] = 'sqlite'
        return initDbSQLite()
    
    if cacheConfigOverride:
        cacheConfig.update(cacheConfigOverride)
    
    # Auto-detect backend
    if cacheConfig['backend'] == 'postgres':
        # Use OS user if not specified
        if cacheConfig['user'] is None:
            cacheConfig['user'] = os.environ.get('USER', 'postgres')
        
        try:
            _connectionPool = psycopg2.pool.ThreadedConnectionPool(
                cacheConfig['minConnections'],
                cacheConfig['maxConnections'],
                host=cacheConfig['host'],
                port=cacheConfig['port'],
                database=cacheConfig['database'],
                user=cacheConfig['user'],
                password=cacheConfig['password']
            )
            print(f"[Cache] PostgreSQL initialized: {cacheConfig['database']}@{cacheConfig['host']}:{cacheConfig['port']}")
            return True
        except Exception as e:
            print(f"[Cache] PostgreSQL connection failed: {e}")
            print("[Cache] Falling back to SQLite")
            cacheConfig['backend'] = 'sqlite'
            return initDbSQLite()
    else:
        return initDbSQLite()


def getConnection():
    """Get connection from pool"""
    global _connectionPool
    
    if cacheConfig['backend'] == 'postgres':
        if _connectionPool is None:
            if not initCachePool():
                raise Exception("Cache pool not initialized")
        return _connectionPool.getconn()
    else:
        # SQLite connection
        return sqlite3.connect(cacheDatabase, timeout=120)


def returnConnection(conn):
    """Return connection to pool"""
    global _connectionPool
    
    if cacheConfig['backend'] == 'postgres':
        if _connectionPool:
            _connectionPool.putconn(conn)
    else:
        # SQLite connection
        if conn:
            conn.close()


def cacheGet(queryHash):
    """Retrieve cached result by hash"""
    conn = None
    try:
        conn = getConnection()
        cursor = conn.cursor()
        
        if cacheConfig['backend'] == 'postgres':
            cursor.execute(
                "SELECT response FROM cache WHERE hash = %s",
                (queryHash,)
            )
        else:
            cursor.execute(
                "SELECT response FROM cache WHERE hash = ?",
                (queryHash,)
            )
        
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            return json.loads(row[0])
        return None
        
    except Exception as e:
        return None
    finally:
        if conn:
            returnConnection(conn)


def cacheSet(queryHash, query, result):
    """Store query result in cache"""
    conn = None
    try:
        conn = getConnection()
        cursor = conn.cursor()
        
        if cacheConfig['backend'] == 'postgres':
            cursor.execute("""
                INSERT INTO cache (hash, query, response, timestamp)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (hash) 
                DO UPDATE SET 
                    response = EXCLUDED.response,
                    timestamp = NOW()
            """, (queryHash, str(query), json.dumps(result)))
        else:
            cursor.execute(
                "INSERT OR REPLACE INTO cache (hash, query, response) VALUES (?, ?, ?)",
                (queryHash, str(query), json.dumps(result))
            )
        
        conn.commit()
        cursor.close()
        return True
        
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return False
    finally:
        if conn:
            returnConnection(conn)


def purgeOldCache(days=14):
    """Delete cache entries older than specified days"""
    conn = None
    try:
        conn = getConnection()
        cursor = conn.cursor()
        
        if cacheConfig['backend'] == 'postgres':
            cursor.execute("""
                DELETE FROM cache
                WHERE timestamp < NOW() - INTERVAL '%s days'
            """, (days,))
        else:
            cursor.execute(
                "DELETE FROM cache WHERE timestamp < datetime('now', '-' || ? || ' days')",
                (days,)
            )
        
        deleted = cursor.rowcount
        conn.commit()
        
        # Vacuum to reclaim space
        if cacheConfig['backend'] == 'postgres':
            cursor.execute("VACUUM ANALYZE cache")
        else:
            try:
                conn.execute("VACUUM")
            except:
                pass
        
        cursor.close()
        
        if deleted > 0:
            print(f"[Cache] Purged {deleted} entries older than {days} days")
        return deleted
        
    except Exception as e:
        print(f"[Cache] Purge error: {e}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return 0
    finally:
        if conn:
            returnConnection(conn)


def cacheStats():
    """Get cache statistics"""
    conn = None
    try:
        conn = getConnection()
        cursor = conn.cursor()
        
        # Total entries
        cursor.execute("SELECT COUNT(*) FROM cache")
        total = cursor.fetchone()[0]
        
        # Entries from last 24 hours
        if cacheConfig['backend'] == 'postgres':
            cursor.execute("""
                SELECT COUNT(*) FROM cache
                WHERE timestamp > NOW() - INTERVAL '1 day'
            """)
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM cache
                WHERE timestamp > datetime('now', '-1 day')
            """)
        recent = cursor.fetchone()[0]
        
        # Oldest entry
        cursor.execute("SELECT MIN(timestamp) FROM cache")
        oldest = cursor.fetchone()[0]
        
        cursor.close()
        
        return {
            'total': total,
            'last24h': recent,
            'oldest': oldest,
            'backend': cacheConfig['backend']
        }
        
    except Exception as e:
        print(f"[Cache] Stats error: {e}")
        return None
    finally:
        if conn:
            returnConnection(conn)


#########################################
#                                       #
#      SQLITE FALLBACK                  #
#                                       #
#########################################

cacheDatabase = 'TabulAIrityCache.db'

def initDbSQLite():
    """Initialize SQLite database as fallback"""
    try:
        with sqlite3.connect(cacheDatabase, timeout=120) as conn:
            cursor = conn.cursor()
            
            # WAL mode for concurrent reads
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.execute("PRAGMA temp_store = MEMORY")
            cursor.execute("PRAGMA mmap_size = 30000000000")
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    hash TEXT PRIMARY KEY,
                    query TEXT,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache(hash)
            ''')
            
            conn.commit()
        
        purgeOldCache(days=14)
        print(f"[Cache] SQLite initialized: {cacheDatabase}")
        return True
        
    except Exception as e:
        print(f"[Cache] SQLite initialization error: {e}")
        return False


# Initialize cache on import
if POSTGRES_AVAILABLE and cacheConfig['backend'] == 'postgres':
    initCachePool()
else:
    initDbSQLite()


#########################################
#                                       #
#      ENVIRONMENT PREP                 #
#                                       #
#########################################

modelName = "gemma3:12b"
maxTranslateTokens = 8000
promptDelay = 0.0
targetLanguage = 'en'
translationModel = "gemma3:27b"


def prepEnvironment(routesRef='config/model_routes.csv'):
    """Load environment args and model routes"""
    credentialsRef = 'config/environment_args.txt'
    if os.path.exists(credentialsRef):
        with open(credentialsRef) as credentials:
            lines = credentials.readlines()
        for line in lines:
            if ' = ' in line:
                [arg, value] = [i.strip() for i in line.split(' = ')][:2]
                os.environ[arg] = value
    else:
        print("[Config] Environment args not found. Using defaults.")

    configRef = 'config/config.txt'
    if os.path.exists(configRef):
        with open(configRef) as configs:
            lines = configs.readlines()
        for line in lines:
            if ' = ' in line:
                [arg, value] = [i.strip() for i in line.split(' = ')][:2]
                config[arg] = value

    if os.path.exists(routesRef):
        routes = pd.read_csv(routesRef)
        routes = routes.replace({np.nan: None, 'remote': None})
        routes['last used'] = datetime.utcnow()
    else:
        routes = {modelName: {'route': modelName,
                              'ip': 'http://localhost:11434'}}
        routes = pd.DataFrame(routes).T

    return routes


modelRoutes = prepEnvironment()


def getModelRoute(name):
    """Model route accessor with LRU"""
    global modelRoutes

    for col in ['model', 'route', 'ip', 'last used']:
        if col not in modelRoutes.columns:
            modelRoutes[col] = None

    matches = modelRoutes.loc[modelRoutes['model'] == name]

    if matches.empty:
        print(f"[Config] {name} not found in routes, adding...")
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
            chosenIdx = temp['last used'].idxmin()
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
#      TEXT INTERROGATION FUNCTIONS     #
#                                       #
#########################################


def validRun(var, prompt):
    return isValid(var) or str(prompt).startswith('recall:')


def isValid(var):
    return var == var and var not in {None, ''}


def showIfValid(var):
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


def buildChatNet(script, show=False):
    script['fx'] = script['fx'].fillna('null')
    script['prompt'] = script['prompt'].fillna('')
    script['self_eval'] = script['self_eval'].fillna(False)

    if 'model' not in script.columns:
        script.loc[:, 'model'] = modelName

    chatEdges = script[script.type == 'edge']
    chatNodes = script[script.type == 'node']
    G = nx.MultiDiGraph()

    nodesParsed = [(row['key'],
                    {'prompt': row['prompt'],
                     'fx': row['fx'],
                     'persona': row['persona'],
                     'tokens': row['tokens'],
                     'self_eval': row['self_eval'],
                     'model': row['model']}) for index, row in chatNodes.T.items()]

    G.add_nodes_from(nodesParsed)

    splitEdge = lambda x: x['key'].split('-')
    edgesParsed = {tuple(splitEdge(row) + [row['fx']]): {'prompt': row['prompt'], 'fx': row['fx'], } for index, row in
                   chatEdges.T.items()}
    G.add_edges_from(edgesParsed)
    nx.set_edge_attributes(G, edgesParsed)
    connected = nx.is_weakly_connected(G)

    if not connected:
        print(f"[Warning] Chat graph has disconnected components.")

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


def insertChatVars(text, varStore):
    for key, value in varStore.items():
        toReplace = f'[{key}]'
        text = text.replace(toReplace, str(value))
    return text


def extractChatVars(text):
    matches = set(re.findall(r"\[([^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*)\]", text))
    matches = [match for match in matches if "\n" not in match and "," not in match]
    matches = [match for match in matches if match != '']
    return matches


baseFx = {'isYes': lambda x, y: ynToBool(x),
          'isNo': lambda x, y: not ynToBool(x),
          'getYN': lambda x, y: getYN(x),
          'null': lambda x, y: True,
          'pass': lambda x, y: x}


def processNodeStep(currentNode, G, chatVars, fxStore, verbosity):
    """Process a single node - FAIL FAST on errors to prevent garbage data propagation"""
    nodeVars = G.nodes[currentNode]

    # --- BLOCK 1: PREPARATION ---
    try:
        prompt = insertChatVars(nodeVars['prompt'], chatVars)
        tokens = nodeVars['tokens']
        persona = nodeVars['persona']
        rowModel = nodeVars['model']
        selfEval = nodeVars['self_eval']
    except Exception:
        print(f"\n[ERROR] Node '{currentNode}' preparation failed")
        traceback.print_exc()
        return []

    failed = False
    chatResponse = ""

    # --- BLOCK 2: EXTERNAL I/O (Fail Fast - No Retries) ---
    try:
        if validRun(persona, prompt):
            if not str(prompt).startswith('recall:'):
                if verbosity == 2:
                    print()
                    print(prompt)
                elif verbosity > 0:
                    print(f"   >>> Processing '{currentNode}' (Model: {rowModel})...")

                chatResponse = askChatQuestion(prompt,
                                                persona,
                                                model=rowModel,
                                                tokens=tokens)

                if verbosity > 0:
                    print(f"   <<< Finished '{currentNode}': {chatResponse[:100]}...")

            else:
                chatResponse = prompt[7:].strip()

            chatVars[currentNode + '_prompt'] = prompt
            chatVars[currentNode + '_raw'] = chatResponse

    except Exception as e:
        # FAIL FAST: Any error stops the graph
        print(f"\n[FATAL] Node '{currentNode}' failed - stopping graph to prevent garbage data")
        print(f"Error: {str(e)[:200]}")
        traceback.print_exc()
        raise  # Re-raise to stop entire graph

    # --- BLOCK 3: POST-PROCESSING ---
    try:
        if selfEval:
            worthUsing = isUseful(prompt, chatResponse)
        else:
            worthUsing = True

        if worthUsing:
            try:
                cleanedResponse = fxStore[nodeVars['fx']](chatResponse, chatVars)
            except Exception as fxErr:
                print(f"[Warning] Cleaning function {nodeVars['fx']} failed: {fxErr}")
                cleanedResponse = chatResponse

            chatVars[currentNode] = cleanedResponse
            if verbosity > 0:
                print(f'\t-{persona}: {cleanedResponse}')
        else:
            if verbosity > 0:
                print(f'\t*FAILS: {chatResponse[:50]}...')
            failed = True

        nextNodes = []
        if not failed:
            edgesFromQ = G.out_edges([currentNode], data=True)
            for start, end, edgeData in edgesFromQ:
                edgeResult = fxStore[edgeData['fx']](chatResponse, chatVars)
                chatVars[f'{start}-{end}'] = edgeResult

                if str(edgeResult).lower() == 'true':
                    nextNodes.append(end)
                    edgePrompt = insertChatVars(edgeData['prompt'], chatVars)
                    showIfValid(edgePrompt)

        nextNodes.sort(reverse=True)
        return nextNodes

    except Exception:
        print(f"\n[FATAL] Node '{currentNode}' edge evaluation failed")
        traceback.print_exc()
        raise  # Re-raise to stop entire graph


async def process_one_node(node, G, chatVars, fxStore, verbosity, semaphore, workerID=0):
    """Process single node and return its children"""
    startTime = datetime.utcnow()
    
    try:
        if verbosity >= 2:
            print(f"[Worker-{workerID}] Started: {node}")
        
        async with semaphore:
            try:
                nextNodes = await asyncio.wait_for(
                    asyncio.to_thread(
                        processNodeStep,
                        node,
                        G,
                        chatVars,
                        fxStore,
                        verbosity
                    ),
                    timeout=1500  # 15 minute max per node
                )
            except asyncio.TimeoutError:
                elapsed = (datetime.utcnow() - startTime).total_seconds()
                print(f"\n[TIMEOUT] Node '{node}' exceeded 15 minutes ({elapsed:.0f}s)")
                raise Exception(f"Node '{node}' timed out after {elapsed:.0f}s")
        
        if verbosity >= 2:
            elapsed = (datetime.utcnow() - startTime).total_seconds()
            print(f"[Worker-{workerID}] Completed: {node} ({elapsed:.1f}s)")
        
        return nextNodes
        
    except Exception as e:
        elapsed = (datetime.utcnow() - startTime).total_seconds()
        print(f"\n[FATAL] Node '{node}' failed after {elapsed:.1f}s")
        raise


async def walkChatNetAsync(G, fxStore, varStore, verbosity, numWorkers=4):
    """Async graph traversal with wave-based processing"""
    chatVars = deepcopy(varStore)
    fxStore = fxStore | baseFx
    semaphore = asyncio.Semaphore(numWorkers)
    
    currentWave = ['Start']
    waveNumber = 0
    
    try:
        while currentWave:
            waveNumber += 1
            if verbosity > 0:
                if len(currentWave) <= 10:
                    print(f"\n[Wave {waveNumber}] Processing {len(currentWave)} nodes: {currentWave}")
                else:
                    print(f"\n[Wave {waveNumber}] Processing {len(currentWave)} nodes: {currentWave[:10]} ... and {len(currentWave) - 10} more")
            
            # Create tasks for all nodes in current wave
            tasks = []
            for idx, node in enumerate(currentWave):
                task = asyncio.create_task(
                    process_one_node(node, G, chatVars, fxStore, verbosity, semaphore, workerID=idx % numWorkers)
                )
                tasks.append((node, task))
            
            # Wait for ALL nodes in this wave to complete
            nextWave = []
            for node, task in tasks:
                try:
                    childNodes = await task
                    nextWave.extend(childNodes)
                except Exception as e:
                    print(f"\n[FATAL] Wave {waveNumber} failed on node '{node}'")
                    raise
            
            # Remove duplicates, sort for next wave
            currentWave = sorted(set(nextWave), reverse=True)
        
        if verbosity > 0:
            print(f"\n[Complete] Processed {waveNumber} waves")
    
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
        raise
    except Exception as e:
        print(f"\n[STOPPED] Graph execution stopped: {e}")
        raise
    
    return chatVars


def walkChatNet(G,
                fxStore=dict(),
                varStore=dict(),
                verbosity=1,
                runAsync=False,
                numWorkers=4):
    """Main entry point for graph traversal"""
    global useCache

    try:
        if runAsync:
            try:
                # Handle Jupyter notebook event loop
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()

                result = asyncio.run(walkChatNetAsync(G, fxStore, varStore, verbosity, numWorkers))
                return result

            except ImportError:
                print("[Error] Install 'nest_asyncio' for Jupyter async support")
                return varStore
        else:
            # Synchronous execution
            toAsk = ['Start']
            fxStore = fxStore | baseFx
            chatVars = deepcopy(varStore)

            while toAsk != []:
                nextQ = toAsk.pop()
                nextNodes = processNodeStep(nextQ, G, chatVars, fxStore, verbosity)
                toAsk += nextNodes

            return chatVars

    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
        return varStore


#########################################
#                                       #
#      QUERY CACHING FUNCTIONS          #
#                                       #
#########################################


def getHash(query):
    """Generate MD5 hash of query"""
    hasher = hashlib.md5()
    encoded = str(query).encode('utf-8')
    hasher.update(encoded)
    result = hasher.hexdigest()
    return result


def queryToCache(query,
                 maxAttempts=3,
                 tolerant=False,
                 delay=.05):
    """Execute query with caching"""
    global useCache

    queryHash = getHash(query)

    # --- READ FROM CACHE ---
    if useCache:
        cached = cacheGet(queryHash)
        if cached is not None:
            return cached

    # --- EXECUTE QUERY ---
    sleep(promptDelay)
    gotResults = False
    attempts = 0
    result = None

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
            gotResults = True
            attempts = maxAttempts

    # --- WRITE TO CACHE ---
    if gotResults:
        cacheSet(queryHash, query, result)

    return result


def scrapePage(url):
    """Fetch webpage content"""
    response = requests.get(url)
    statusCode = response.status_code
    if statusCode == 200:
        return response.text
    else:
        raise ValueError(f"Page returned status {statusCode}")


def cachePage(url, maxLen = 100000):
    """Cached page scraping"""
    result = queryToCache(f"st.scrapePageText('{url}',maxLen={maxLen})")
    return result


def cacheGeocode(locText):
    """Cached geocoding with validation"""
    if locText is None or pd.isna(locText) or str(locText).strip() == "":
        return None

    safeLoc = repr(locText)
    query = f"osmnx.geocode({safeLoc})"

    try:
        result = queryToCache(query)
    except Exception as e:
        print(f"[Geocode] Failed on {locText}: {e}")
        return None

    if type(result) not in (list, tuple):
        return None

    return list(result)


#########################################
#                                       #
#      LANGUAGE HANDLING                #
#                                       #
#########################################


def getLanguageName(code):
    """Convert language code to name"""
    lang = pycountry.languages.get(alpha_2=code)
    return lang.name if lang else "English"


def translateOne(text):
    """Translate text to target language"""
    languageName = getLanguageName(targetLanguage)
    translationPersona = f"You are a highly accurate and fluent {languageName} translator."
    translationPrompt = f"Translate the following text to {languageName}. Output only the translated text. Do not include any markdown, explanations, commentary, variable placeholders, or descriptive text.\n\n{text.strip()}"

    translation = askChatQuestion(translationPrompt,
                                  translationPersona,
                                  tokens=maxTranslateTokens,
                                  model=translationModel)
    return translation


def getLanguage(text):
    """Detect language of text"""
    if text in {'', None, np.nan}:
        language = "unidentified"
    else:
        try:
            language = detect(text)
        except:
            language = "unidentified"
    return language


def autoTranslate(dfIn,
                  column,
                  targetLanguage='en',
                  model=modelName):
    """Auto-translate DataFrame column"""
    df = dfIn.copy(deep=True)
    langOut = f'{column}_language'
    textOut = f'{column}_translated'
    df.loc[:, langOut] = df[column].apply(getLanguage)
    df.loc[:, textOut] = df[column]
    df.loc[df[langOut] != targetLanguage, textOut] = df.loc[df[langOut] != targetLanguage, textOut].apply(translateOne)

    return df


#########################################
#                                       #
#      CHAT QUERIES                     #
#                                       #
#########################################


def testRoutes(query='How many Rs are there in strawberry?',
               persona='an AI assistant',
               autoformatPersona=True):
    """Test all model routes"""
    working = []
    for model in modelRoutes.index.sort_values():
        try:
            response = askChatQuestion(query,
                                       persona,
                                       autoformatPersona=autoformatPersona,
                                       model=model)
            print(f'{model} ~ {response}\n')
            working.append(model)
        except:
            print(f'{model} ~ FAILS\n')
    return working


def getChatContent(messages,
                   tokens,
                   modelName,
                   temperature=None,
                   seed=None,
                   timeout=600):
    """Get completion from LLM with timeout - FAIL FAST on errors"""
    modelRoute, ip = getModelRoute(modelName)
    
    try:
        content = completion(
            model=modelRoute,
            max_tokens=int(tokens),
            messages=messages,
            api_base=ip,
            seed=seed,
            temperature=temperature,
            timeout=timeout
        )
        cleaned = content.choices[0].message.content.strip() if content.choices[0].message.content else ''
        return cleaned
    except Exception as e:
        # Fail immediately - don't retry with garbage data downstream
        raise e


def askChatQuestion(prompt,
                    persona,
                    model=modelName,
                    autoformatPersona=None,
                    tokens=2000,
                    temperature=None,
                    seed=None):
    """Ask a question to the chat model"""

    if autoformatPersona is True and persona.strip()[-1] != '.':
        personaText = f'You are {persona}. You must answer questions as {persona}.'
    else:
        personaText = persona

    messages = [
        {'role': 'system', 'content': personaText},
        {'role': 'user', 'content': prompt[:350000]}
    ]

    query = f"getChatContent({messages},{tokens},'{model}',{temperature},{seed}, timeout=600)"
    result = queryToCache(query, tolerant=False)
    return result


def getYN(text):
    """Standardize yes/no answers"""
    messages = [
        {'role': 'system',
         'content': 'You are an API that standardizes yes or no answers. You may only return a one word answer in lowercase or "None" as appropriate.'},
        {'role': 'user',
         'content': f'Please return a value for the following text, coding the ouput as "yes" for any affirmative response, "no" for any negative response: {text}'}
    ]

    query = f"getChatContent({messages},3,'gemma3:12b')"
    result = queryToCache(query)
    if result:
        result = result.lower().replace('"', '')
    else:
        result = "no"
    return result


def ynToBool(evaluation):
    """Convert yes/no text to boolean"""
    textAnswer = getYN(evaluation)
    textAnswer = ''.join(i for i in textAnswer if i.isalnum())
    if not textAnswer: return False
    result = {'y': True, 'n': False}.get(textAnswer[0].lower(), False)
    return result


def evaluateAnswer(question, response):
    """Evaluate if response answers question"""
    messages = [
        {'role': 'system',
         'content': 'You are a debate moderator skilled at identifying the presence of answer in long statements'},
        {'role': 'user',
         'content': f'Please answer in one short sentence, does the following answer provide any useable answer for the provided question?\nquestion: {question}\nanswer: {response}'}
    ]

    query = f"getChatContent({messages},100,'{modelName}')"
    result = queryToCache(query)
    return result


def evaluateAuthor(response):
    """Check if response identifies as AI"""
    messages = [
        {'role': 'user',
         'content': f'Please answer in one short sentence, does the author of the following answer include any text specically identifying itself as an AI?\nanswer: {response}'}
    ]

    query = f"getChatContent({messages},100,'{modelName}')"
    result = queryToCache(query)
    return result


def isUseful(question, response):
    """Determine if response is useful"""
    answerEval = evaluateAnswer(question, response)
    authorEval = evaluateAuthor(response)
    answerYN = getYN(answerEval)
    authorYN = getYN(authorEval)
    print(f'is answer:{answerYN}\tis AI: {authorYN}')

    result = answerYN == 'yes' and authorYN == 'no'
    return result


def getColor(text):
    """Extract color from text"""
    messages = [
        {'role': 'system',
         'content': 'You are a python API that returns the first named color found in a sample of text. You may only return one word in lowercase or None if no color is found.'},
        {'role': 'user', 'content': f'Please return a value for the following text: {text}'}
    ]

    query = f"getChatContent({messages},3,'{modelName}')"
    result = queryToCache(query)
    return result