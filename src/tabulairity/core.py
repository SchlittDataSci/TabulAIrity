import networkx as nx
import pandas as pd
import numpy as np

from . import scrapertools as st

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
from geopy.geocoders import Nominatim
import pickle
import hashlib
from uuid import uuid4
import pycountry
import sqlite3
import asyncio
import traceback
import sys
import argparse
import ast
import math

# ---------------------------------------------------------------------------
# Visualization hook flag — flipped by visualization.vizOn/vizOff.
# Core checks this cheaply before attempting any viz work.
# ---------------------------------------------------------------------------
_viz_enabled = False
_viz_walk_active = 0  # depth counter: >0 when inside walkChatNet/Async, used to suppress duplicate standalone viz


def _viz_emit(event_type: str, payload: dict):
    """Lazy viz emit — no-op when viz is off, never raises."""
    if not _viz_enabled:
        return
    try:
        from .visualization import viz_emit as _ve
        _ve(event_type, payload)
    except Exception:
        pass


def _viz_cid_for_graph(G) -> int | None:
    """Return viz chatnet_id for G, or None."""
    try:
        cid = G.graph.get("viz_id")  # type: ignore
        if cid is not None:
            return int(cid)
    except Exception:
        pass
    return None

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
promptTimeout = 1200
targetLanguage = 'en'
translationModel = "gemma4:26b"
getYNModel = "gemma4:26b"
translationExtraParams = {"reasoning_effort": "none"}
getYNExtraParams = {"reasoning_effort": "none"}


def getPromptTimeout(timeout=None):
    """Resolve a positive timeout: explicit argument, CLI flag, then default.

    The CLI flags are read with parse_known_args so importing the library does
    not consume an application's arguments. Applications with their own parser
    should also register --prompt-timeout and pass its value to walkChatNet.
    """
    if timeout is None:
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument('--prompt-timeout', '--promptTimeout', type=float)
        args, _ = parser.parse_known_args()
        timeout = args.prompt_timeout if args.prompt_timeout is not None else promptTimeout
    try:
        value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError('Prompt timeout must be a positive number of seconds') from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Prompt timeout must be a positive number of seconds')
    return value


def _parseEnvLines(lines):
    """Parse key = value lines from a config file.

    Handles:
      - Plain strings:  FOO = bar
      - JSON values:    ROUTE_KEYS = {"host:port": "KEY_NAME"}
      - Inline comments stripped after # (only for non-JSON values)
      - Blank lines and comment-only lines ignored
    """
    result = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ' = ' not in line:
            continue
        arg, _, raw = line.partition(' = ')
        arg = arg.strip()
        raw = raw.strip()
        # Strip inline comment only when value is not JSON
        if not raw.startswith(('{', '[')):
            raw = raw.split('#')[0].strip()
        result[arg] = raw
    return result


def prepEnvironment():
    """Load environment args and config.

    environment_args.txt values are stored in os.environ as raw strings.
    JSON dict/list values (ROUTE_KEYS etc.) are stored as-is and parsed
    at point of use via json.loads().
    """
    credentialsRef = 'config/environment_args.txt'
    if os.path.exists(credentialsRef):
        with open(credentialsRef) as f:
            parsed = _parseEnvLines(f.readlines())
        for arg, value in parsed.items():
            os.environ[arg] = value
    else:
        print("[Config] Environment args not found. Using defaults.")

    configRef = 'config/config.txt'
    if os.path.exists(configRef):
        with open(configRef) as f:
            parsed = _parseEnvLines(f.readlines())
        for arg, value in parsed.items():
            config[arg] = value


# prepEnvironment must run before endpoint is assigned so LITELLM_URL is available
prepEnvironment()

endpoint = os.environ.get('LITELLM_URL', 'http://localhost:4000/v1')


#########################################
#                                       #
#      MODEL ROUTING TABLE              #
#                                       #
#########################################

_modelRoutes = None

def loadModelRoutes(csvPath='config/model_routes.csv'):
    """Load model routing table from CSV.

    Expected columns: model, route, ip
    Optional column:  key  (API key override per row; falls back to env vars if absent)

    Can be called again at runtime to hot-reload the table, e.g.:
        import core; core.loadModelRoutes('config/model_routes_dev.csv')
    """
    global _modelRoutes
    if os.path.exists(csvPath):
        _modelRoutes = pd.read_csv(csvPath).set_index('model')
        print(f"[Routes] Loaded {len(_modelRoutes)} model routes from {csvPath}")
    else:
        _modelRoutes = None
        print(f"[Routes] No routing table found at {csvPath}, using LITELLM_URL fallback")


loadModelRoutes()


def getModelRoute(name):
    """Return (litellm model string, base URL, api key) for a given model name.

    All models must be defined in model_routes.csv. Unrecognised models raise
    a KeyError immediately rather than silently falling back to a wrong endpoint.

    Namespace prefixes (e.g. 'owui/') are stripped before lookup — the routing
    table's 'route' column holds the correct litellm-facing prefix.
    """
    # Strip any caller-side namespace prefix
    if '/' in name:
        name = name.split('/', 1)[1]

    if _modelRoutes is None:
        raise RuntimeError("[Routes] No routing table loaded. Check config/model_routes.csv exists.")

    if name not in _modelRoutes.index:
        raise KeyError(f"[Routes] Model '{name}' not found in model_routes.csv. Add it before use.")

    row = _modelRoutes.loc[name]
    ip = row['ip']
    route = row['route']
    routeKeys = json.loads(os.environ.get('ROUTE_KEYS', '{}'))
    host = ip.split('://')[-1].split('/')[0]  # extract host:port
    keyVar = routeKeys.get(host, 'OPENAI_API_KEY')
    return route, ip, os.environ.get(keyVar, 'dummy')


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


# Node names reserved for walkChatNet() debugging output. Nodes may not use these
# keys because walkChatNet() writes 'success' and 'errors' into the returned
# chatVars dict, and a colliding node name would silently overwrite (or be
# overwritten by) that debugging metadata.
RESERVED_NODE_NAMES = {'success', 'errors'}


def _validateReservedNodeNames(chatNodes):
    """Raise if the script defines a node named 'success' or 'errors'."""
    conflicts = set(chatNodes['key']) & RESERVED_NODE_NAMES
    if conflicts:
        raise ValueError(
            f"Node name(s) {sorted(conflicts)} are reserved and cannot be used. "
            f"'success' and 'errors' are reserved keys that walkChatNet() adds to "
            f"its returned dict."
        )


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
    # Normalize a private copy. Callers often retain/reuse the Sheet-backed
    # DataFrame, so graph construction must not mutate it in place.
    script = script.copy()
    viz_script = script.copy(deep=True)
    script['fx'] = script['fx'].fillna('null')
    script['prompt'] = script['prompt'].fillna('')
    if 'self_eval' not in script.columns:
        script['self_eval'] = False
    script['self_eval'] = script['self_eval'].fillna(False)

    if 'model' not in script.columns:
        script.loc[:, 'model'] = modelName

    # Parse extra_params column: expects JSON object or empty/null → None
    if 'extra_params' in script.columns:
        def parseExtraParams(val):
            if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == '':
                return None
            try:
                parsed = json.loads(val)
                return parsed if isinstance(parsed, dict) else None
            except (json.JSONDecodeError, TypeError):
                print(f"[Warning] Could not parse extra_params value: {val!r}")
                return None
        script['extra_params'] = script['extra_params'].apply(parseExtraParams)
    else:
        script['extra_params'] = None

    chatEdges = script[script.type == 'edge']
    chatNodes = script[script.type == 'node']
    _validateReservedNodeNames(chatNodes)
    G = nx.MultiDiGraph()

    nodesParsed = [(row['key'],
                    {'prompt': row['prompt'],
                     'fx': row['fx'],
                     'persona': row['persona'],
                     'tokens': row['tokens'],
                     'self_eval': row['self_eval'],
                     'model': row['model'],
                     'extra_params': row['extra_params']}) for index, row in chatNodes.T.items()]

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

    # ---- viz hook: announce graph load (assigns viz_id) ----
    if _viz_enabled:
        try:
            from .visualization import viz_notify_graph_load
            viz_notify_graph_load(G, chatnet=viz_script)
        except Exception:
            pass
    return G


def insertChatVars(text, varStore):
    """Substitute [key] placeholders from varStore into text.

    Non-string input (NaN from an empty sheet cell, None) is returned
    unchanged so callers can still apply their own validity checks.
    Placeholders with no matching key are left alone, so literal bracket
    text - output labels like [yes]/[no], or illustrative examples - passes
    through untouched.
    """
    if not isinstance(text, str):
        return text
    return _resolveChatTemplate(text, varStore)[0]


def _resolveChatTemplate(text, varStore):
    """Resolve original bracket expressions from the inside out.

    Return (text, referenced keys, unresolved dynamic keys). Values inserted
    from source material are not themselves interpreted as templates.
    """
    if not isinstance(text, str):
        return text, set(), set()
    refs, unresolved = set(), set()

    def resolve(fragment):
        output = []
        cursor = 0
        while cursor < len(fragment):
            start = fragment.find('[', cursor)
            if start < 0:
                output.append(fragment[cursor:])
                break
            output.append(fragment[cursor:start])
            depth, end = 1, start + 1
            while end < len(fragment) and depth:
                depth += (fragment[end] == '[') - (fragment[end] == ']')
                end += 1
            if depth:
                output.append(fragment[start:])
                break
            inner = fragment[start + 1:end - 1]
            nested = '[' in inner
            key = resolve(inner) if nested else inner
            if key and '\n' not in key and ',' not in key:
                # An unresolved inner expression is not yet a variable name.
                if '[' not in key and ']' not in key:
                    refs.add(key)
                    if key in varStore:
                        output.append(str(varStore[key]))
                    else:
                        output.append('[' + key + ']')
                        if nested:
                            unresolved.add(key)
                else:
                    output.append('[' + key + ']')
                    if nested:
                        unresolved.add(key)
            else:
                output.append('[' + key + ']')
            cursor = end
        return ''.join(output)

    return resolve(text), refs, unresolved


def extractChatVars(text):
    if not isinstance(text, str):
        return []
    matches = set(re.findall(r"\[([^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*)\]", text))
    matches = [match for match in matches if "\n" not in match and "," not in match]
    matches = [match for match in matches if match != '']
    return matches


def _stripChatReasoning(text):
    """Drop reasoning blocks outside containers, preserving quoted data."""
    pattern = re.compile(r'<(think|analysis)\b[^>]*>.*?</\1>', re.DOTALL | re.IGNORECASE)
    parts, stack = [], []
    quote, escaped, cursor, kept = None, False, 0, 0
    pairs = {']': '[', '}': '{', ')': '('}
    while cursor < len(text):
        char = text[cursor]
        if not stack and char == '<':
            match = pattern.match(text, cursor)
            if match:
                parts.append(text[kept:cursor])
                cursor = kept = match.end()
                continue
        if quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
        elif stack and char in ('"', "'"):
            quote = char
        elif char in '[{(':
            stack.append(char)
        elif char in pairs and stack:
            if stack[-1] == pairs[char]:
                stack.pop()
        cursor += 1
    parts.append(text[kept:])
    return ''.join(parts)


def insertJson(response, chatVars=None):
    """Extract a complete JSON/container value without modifying its contents.

    Accept already-decoded containers, Markdown fences, surrounding prose,
    and Python literal containers. Invalid or truncated containers raise;
    nested fragments are never salvaged from a malformed outer container.
    """
    containers = (dict, list, tuple, set)
    if isinstance(response, containers):
        return response
    if isinstance(response, bytes):
        response = response.decode('utf-8')
    if not isinstance(response, str):
        raise ValueError('Expected JSON or an iterable container')
    text = response.strip()

    def decode(candidate):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            parsed = ast.literal_eval(candidate)
        if not isinstance(parsed, containers):
            raise ValueError('Expected a JSON object, array, or literal container')
        return parsed

    try:
        return decode(text)
    except (ValueError, SyntaxError):
        pass
    text = _stripChatReasoning(text).strip()
    fences = re.findall(r'```(?:json|python)?\s*\n?(.*?)```', text,
                        flags=re.DOTALL | re.IGNORECASE)
    candidates = fences or [text]
    for candidate in candidates:
        try:
            return decode(candidate.strip())
        except (ValueError, SyntaxError):
            pass
        cursor = 0
        while cursor < len(candidate):
            match = re.search(r'[\[{]', candidate[cursor:])
            if not match:
                break
            start = cursor + match.start()
            stack, quote, escaped = [], None, False
            end = start
            for end in range(start, len(candidate)):
                char = candidate[end]
                if quote:
                    if escaped:
                        escaped = False
                    elif char == '\\':
                        escaped = True
                    elif char == quote:
                        quote = None
                    continue
                if char in ('"', "'"):
                    quote = char
                elif char in '[{':
                    stack.append(char)
                elif char in ']}':
                    if not stack or stack.pop() != ( '[' if char == ']' else '{'):
                        break
                    if not stack:
                        break
            if stack or quote:
                # Do not extract a seemingly valid inner object from a
                # truncated outer response.
                break
            try:
                return decode(candidate[start:end + 1])
            except (ValueError, SyntaxError):
                cursor = end + 1
    raise ValueError('No complete JSON or literal container found in response')


def mergeJson(response, chatVars):
    """Extract a complete JSON object and overwrite its keys in chatVars."""
    parsed = insertJson(response)
    if not isinstance(parsed, dict):
        raise ValueError('mergeJson requires a JSON object')
    chatVars.update(parsed)
    return parsed


baseFx = {'isYes': lambda x, y: ynToBool(x),
          'isNo': lambda x, y: not ynToBool(x),
          'getYN': lambda x, y: getYN(x),
          'null': lambda x, y: True,
          'pass': lambda x, y: x,
          'mergeJson': mergeJson,
          'insertJson': insertJson}


def processNodeStep(currentNode, G, chatVars, fxStore, verbosity, timeout=None):
    """Process a single node.

    Returns (nextNodes, nodeErrors). nodeErrors is a list of structured error
    records - {<node/edge name>: <error message>, 'type': 'node' | 'edge'} -
    for every node or edge fx failure encountered while processing this node.
    Node/edge fx failures are never raised or silently swallowed here; they are
    always recorded so callers (walkChatNet / walkChatNetAsync) can decide,
    based on the `tolerant` flag, whether to stop the whole traversal or
    continue with partial completion. If this node fails, it does not
    activate any outgoing edges (nextNodes will be empty for that failure)."""
    # viz: resolve chatnet id once
    _viz_cid = _viz_cid_for_graph(G) if _viz_enabled else None
    nodeErrors = []
    nodeVars = G.nodes[currentNode]

    # --- BLOCK 1: PREPARATION ---
    try:
        prompt = insertChatVars(nodeVars['prompt'], chatVars)
        tokens = nodeVars['tokens']
        # The persona is substituted on the same terms as the prompt. It was
        # previously passed through raw, which meant any [variable] written
        # into a persona reached the model as literal text - a silent failure,
        # since nothing raises on an unresolved placeholder. Personas are the
        # natural home for role and criteria, so they need the same treatment.
        persona = insertChatVars(nodeVars['persona'], chatVars)
        rowModel = nodeVars['model']
        selfEval = nodeVars.get('self_eval', False)
        extraParams = nodeVars.get('extra_params', None)
    except Exception as e:
        print(f"\n[ERROR] Node '{currentNode}' preparation failed")
        traceback.print_exc()
        nodeErrors.append({currentNode: f"Preparation failed: {e}", 'type': 'node'})
        return [], nodeErrors

    failed = False
    chatResponse = ""

    # viz: node entering processing (after prompt resolved)
    if _viz_enabled and _viz_cid is not None:
        try:
            from .visualization import viz_notify_node_start
            _fullP = str(prompt) if isinstance(prompt, str) else ""
            _fullPersona = str(persona) if isinstance(persona, str) else ""
            viz_notify_node_start(_viz_cid, currentNode, prompt=_fullP, persona=_fullPersona, fullPrompt=_fullP)
        except Exception:
            pass

    # --- BLOCK 2: EXTERNAL I/O ---
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
                                                tokens=tokens,
                                                extra_params=extraParams,
                                                timeout=timeout)

                if verbosity > 0:
                    print(f"   <<< Finished '{currentNode}': {chatResponse[:100]}...")

            else:
                chatResponse = prompt[7:].strip()

            chatVars[currentNode + '_prompt'] = prompt
            chatVars[currentNode + '_raw'] = chatResponse

    except Exception as e:
        print(f"\n[ERROR] Node '{currentNode}' failed during execution")
        print(f"Error: {str(e)[:200]}")
        traceback.print_exc()
        nodeErrors.append({currentNode: str(e), 'type': 'node'})
        if _viz_enabled and _viz_cid is not None:
            try:
                from .visualization import viz_notify_node_error
                viz_notify_node_error(_viz_cid, currentNode, str(e)[:500])
            except Exception:
                pass
        return [], nodeErrors

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
                print(f"[ERROR] Node fx '{nodeVars['fx']}' failed on node '{currentNode}': {fxErr}")
                nodeErrors.append({currentNode: f"Node fx '{nodeVars['fx']}' failed: {fxErr}", 'type': 'node'})
                if _viz_enabled and _viz_cid is not None:
                    try:
                        from .visualization import viz_notify_node_error
                        viz_notify_node_error(_viz_cid, currentNode, f"fx {nodeVars['fx']}: {fxErr}"[:500])
                    except Exception:
                        pass
                return [], nodeErrors

            chatVars[currentNode] = cleanedResponse
            if verbosity > 0:
                print(f'\t-{persona}: {cleanedResponse}')
            # viz: side panel + completed color (send full prompt/persona/fx for popup)
            if _viz_enabled and _viz_cid is not None:
                try:
                    from .visualization import viz_notify_prompt_response, viz_notify_node_complete
                    viz_notify_prompt_response(_viz_cid, currentNode, str(prompt), str(chatResponse), str(cleanedResponse), persona=str(persona), fx=str(nodeVars.get('fx','')), fullPrompt=str(prompt))
                    viz_notify_node_complete(_viz_cid, currentNode, str(chatResponse), str(cleanedResponse))
                except Exception:
                    pass
        else:
            if verbosity > 0:
                print(f'\t*FAILS: {chatResponse[:50]}...')
            failed = True
            if _viz_enabled and _viz_cid is not None:
                try:
                    from .visualization import viz_notify_prompt_response, viz_notify_node_error
                    viz_notify_prompt_response(_viz_cid, currentNode, str(prompt), str(chatResponse), "", persona=str(persona), fx=str(nodeVars.get('fx','')), fullPrompt=str(prompt))
                    viz_notify_node_error(_viz_cid, currentNode, "self_eval failed")
                except Exception:
                    pass

        nextNodes = []
        if not failed:
            edgesFromQ = G.out_edges([currentNode], data=True)
            for start, end, edgeData in edgesFromQ:
                edgeName = f'{start}-{end}'
                edgeFx = edgeData.get('fx', '') if isinstance(edgeData, dict) else ''
                try:
                    edgeResult = fxStore[edgeData['fx']](chatResponse, chatVars)
                except Exception as edgeErr:
                    print(f"[ERROR] Edge fx '{edgeData['fx']}' failed on edge '{edgeName}': {edgeErr}")
                    nodeErrors.append({edgeName: f"Edge fx '{edgeData['fx']}' failed: {edgeErr}", 'type': 'edge'})
                    if _viz_enabled and _viz_cid is not None:
                        try:
                            from .visualization import viz_notify_edge_evaluated
                            viz_notify_edge_evaluated(_viz_cid, edgeName, False, fx=edgeFx)
                        except Exception:
                            pass
                    continue  # skip this edge; other edges from this node may still succeed

                chatVars[edgeName] = edgeResult
                # viz: edge evaluated color — include fx to disambiguate parallel edges
                if _viz_enabled and _viz_cid is not None:
                    try:
                        from .visualization import viz_notify_edge_evaluated
                        viz_notify_edge_evaluated(_viz_cid, edgeName, str(edgeResult).lower() == 'true', fx=edgeFx)
                    except Exception:
                        pass

                if str(edgeResult).lower() == 'true':
                    nextNodes.append(end)
                    edgePrompt = insertChatVars(edgeData['prompt'], chatVars)
                    showIfValid(edgePrompt)

        nextNodes.sort(reverse=True)
        return nextNodes, nodeErrors

    except Exception as e:
        print(f"\n[ERROR] Node '{currentNode}' edge evaluation failed")
        traceback.print_exc()
        nodeErrors.append({currentNode: str(e), 'type': 'node'})
        return [], nodeErrors


class _NodeVariables(dict):
    """A node's isolated top-level writes, committed only after it settles."""
    def __init__(self, values, isolateNested=False):
        # Pure built-ins only assign top-level values. Custom callables may
        # mutate existing containers in place, so isolate those snapshots too.
        snapshots = dict(values)
        mutableTypes = (dict, list, set, tuple, pd.Series, pd.DataFrame, np.ndarray)
        if isolateNested:
            snapshots = {key: deepcopy(value) if isinstance(value, mutableTypes) else value
                         for key, value in snapshots.items()}
        super().__init__(snapshots)
        self.nestedOriginal = {}
        if isolateNested:
            for key, value in self.items():
                if isinstance(value, mutableTypes):
                    self.nestedOriginal[key] = pickle.dumps(value)
        self.writes = {}
        self.deleted = set()
        self.timedOut = False

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.writes[key] = value
        self.deleted.discard(key)

    def __delitem__(self, key):
        super().__delitem__(key)
        self.writes.pop(key, None)
        self.deleted.add(key)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, *default):
        if key in self:
            value = self[key]
            del self[key]
            return value
        return super().pop(key, *default)

    def popitem(self):
        if not self:
            raise KeyError('popitem(): dictionary is empty')
        key = next(reversed(self))
        return key, self.pop(key)

    def clear(self):
        for key in list(self):
            del self[key]

    def __ior__(self, other):
        self.update(other)
        return self

    def commit(self, target):
        if not self.timedOut:
            for key, before in self.nestedOriginal.items():
                if key in self and key not in self.writes and pickle.dumps(self[key]) != before:
                    self.writes[key] = self[key]
            for key in self.deleted:
                target.pop(key, None)
            target.update(self.writes)


def _chatVarProducer(key, G):
    if key in G:
        return key
    for suffix in ('_raw', '_prompt'):
        if key.endswith(suffix) and key[:-len(suffix)] in G:
            return key[:-len(suffix)]
    for source, target in G.edges():
        if key == f'{source}-{target}':
            return source
    return None


def parallelFx(function):
    """Opt in a pure custom cleaner to parallel scheduling.

    The cleaner must not mutate chatVars or shared state. Any chatVars it
    reads must also appear as references in the node's prompt or persona.
    Unknown callables remain conservative unless explicitly marked.
    """
    function._tabulairity_parallel_safe = True
    return function


def _parallelChatFxNames(fxStore=None):
    return {name for name, function in (fxStore or {}).items()
            if getattr(function, '_tabulairity_parallel_safe', False)}


def _hasCustomChatFx(node, G, fxStore=None):
    known = set(baseFx) | _parallelChatFxNames(fxStore)
    return G.nodes[node].get('fx', 'null') not in known or any(
        edge.get('fx', 'null') not in known
        for _, _, edge in G.out_edges(node, data=True))


def buildChatNetDependencies(G, varStore=None, fxStore=None):
    """Overlay data dependencies and pruning barriers on a chat network.

    Unconditional control edges define potential reachability, not execution
    order. Every conditional edge's source gates the target and all of its
    descendants. Dynamic templates and unknown/custom variable writers gate
    their descendants too. Rebuild with settled variables to discover dynamic
    targets; the original control graph is never changed.
    """
    values = varStore if varStore is not None else {}
    overlay = nx.DiGraph()
    overlay.add_nodes_from(G.nodes())
    pureFx = {'null', 'pass', 'getYN', 'isYes', 'isNo', 'insertJson'} | _parallelChatFxNames(fxStore)

    def depend(source, target, kind, variable=None):
        if source == target:
            return
        if not overlay.has_edge(source, target):
            overlay.add_edge(source, target, kinds=set(), variables=set())
        data = overlay[source][target]
        data['kinds'].add(kind)
        if variable is not None:
            data['variables'].add(variable)

    for node, data in G.nodes(data=True):
        templates = (data.get('prompt', ''), data.get('persona', ''))
        dynamic = any(isinstance(text, str) and re.search(r'\[[^\]\n]*\[', text)
                      for text in templates)
        overlay.nodes[node]['dynamic'] = bool(dynamic)
        for text in templates:
            _, refs, _ = _resolveChatTemplate(text, values)
            for key in refs:
                producer = _chatVarProducer(key, G)
                if producer is not None:
                    depend(producer, node, 'data', key)
        fx = data.get('fx', 'null')
        opaque = fx not in pureFx and fx != 'mergeJson'
        opaqueEdges = any(edge.get('fx', 'null') not in pureFx
                          for _, _, edge in G.out_edges(node, data=True))
        if opaque or opaqueEdges:
            # Custom callables may read chatVars without bracket references.
            for ancestor in nx.ancestors(G, node):
                depend(ancestor, node, 'order')
        # self_eval may reject the node after inference, before any edges open.
        # All descendants must await that decision even on unconditional paths.
        if data.get('self_eval', False) or dynamic or fx == 'mergeJson' or opaque or opaqueEdges:
            for descendant in nx.descendants(G, node):
                depend(node, descendant, 'barrier')

    if 'Start' in G:
        for descendant in nx.descendants(G, 'Start'):
            depend('Start', descendant, 'barrier')
    for source, target, data in G.edges(data=True):
        if data.get('fx', 'null') != 'null':
            for descendant in {target} | nx.descendants(G, target):
                depend(source, descendant, 'barrier')
    return overlay


def _possibleChatNodes(G, settled, children):
    """Reachability with unresolved edges retained and rejected edges removed."""
    active, pending = set(), ['Start'] if 'Start' in G else []
    while pending:
        node = pending.pop()
        if node in active or node not in G:
            continue
        active.add(node)
        targets = children.get(node, ()) if node in settled else G.successors(node)
        pending.extend(targets)
    return active


async def process_one_node(node, G, chatVars, fxStore, verbosity, semaphore, workerID=0,
                           timeout=None):
    """Process single node and return (nextNodes, nodeErrors).

    Errors (including timeouts and unexpected exceptions) are recorded as
    structured entries rather than raised, so a single node failure can't
    take down the entire async traversal - walkChatNetAsync decides whether
    to stop based on the `tolerant` flag."""
    startTime = datetime.utcnow()

    try:
        if verbosity >= 2:
            print(f"[Worker-{workerID}] Started: {node}")
        
        async with semaphore:
            try:
                # self_eval performs four additional model calls after the
                # node's answer. Those helpers retain their default/CLI timeout.
                # The node watchdog must leave room for that existing workflow.
                nodeBudget = getPromptTimeout(timeout)
                if G.nodes[node].get('self_eval', False):
                    nodeBudget += 4 * getPromptTimeout()
                nextNodes, nodeErrors = await asyncio.wait_for(
                    asyncio.to_thread(
                        processNodeStep,
                        node,
                        G,
                        chatVars,
                        fxStore,
                        verbosity,
                        timeout
                    ),
                    timeout=max(1500, nodeBudget + 300)
                )
            except asyncio.TimeoutError:
                if isinstance(chatVars, _NodeVariables):
                    chatVars.timedOut = True
                elapsed = (datetime.utcnow() - startTime).total_seconds()
                msg = f"Node '{node}' timed out after {elapsed:.0f}s"
                print(f"\n[ERROR] {msg}")
                return [], [{node: msg, 'type': 'node'}]
        
        if verbosity >= 2:
            elapsed = (datetime.utcnow() - startTime).total_seconds()
            print(f"[Worker-{workerID}] Completed: {node} ({elapsed:.1f}s)")
        
        return nextNodes, nodeErrors
        
    except Exception as e:
        elapsed = (datetime.utcnow() - startTime).total_seconds()
        msg = f"Node '{node}' failed unexpectedly after {elapsed:.1f}s: {e}"
        print(f"\n[ERROR] {msg}")
        traceback.print_exc()
        return [], [{node: msg, 'type': 'node'}]


async def walkChatNetAsync(G, fxStore, varStore, verbosity, numWorkers=4, tolerant=False,
                           timeout=None):
    """Schedule ready nodes on each completion, with inferred dependencies.

    Conditional edges and self_eval retain pruning across every descendant.
    Dynamic templates and unknown variable writers are conservative barriers.
    Nodes execute at most once. Failures stop new dispatches by default; running
    nodes are drained. Tolerant mode continues independent reachable branches.

    Returns chatVars with two additional keys:
      - 'success': True only if every reachable node ran without error
      - 'errors': list of {<node/edge name>: <message>, 'type': 'node'|'edge'}
    """
    global _viz_walk_active
    _viz_walk_active += 1
    try:
        _viz_cid = _viz_cid_for_graph(G) if _viz_enabled else None
        if _viz_enabled and _viz_cid is None:
            try:
                from .visualization import viz_notify_graph_load
                _viz_cid = viz_notify_graph_load(G)
            except Exception:
                pass
        if not isinstance(numWorkers, int) or numWorkers < 1:
            raise ValueError('numWorkers must be a positive integer')
        timeout = getPromptTimeout(timeout)
        if isinstance(varStore, pd.Series):
            chatVars = varStore.to_dict()
        else:
            chatVars = dict(varStore) if varStore is not None else {}
        fxStore = fxStore | baseFx
        semaphore = asyncio.Semaphore(numWorkers)
    
        settled, failed, children, running = set(), set(), {}, {}
        dispatched = 0
        errors = []
        stopped = False
        if 'Start' not in G:
            errors.append({'walkChatNet': "Chat graph has no 'Start' node", 'type': 'node'})
            stopped = True

        try:
            while True:
                active = _possibleChatNodes(G, settled, children)
                overlay = buildChatNetDependencies(G, chatVars, fxStore)
                G.graph['dependency_graph'] = overlay
                inFlight = {node for node, _ in running.values()}
                customRunning = any(_hasCustomChatFx(node, G, fxStore) for node in inFlight)
                pending = [node for node in G if node in active and node not in settled | inFlight]
                waiting = {}
                refreshNeeded = False
                if not stopped:
                    for node in pending:
                        blockers = []
                        for producer, _, dep in overlay.in_edges(node, data=True):
                            if 'data' in dep['kinds']:
                                if producer not in settled or producer in failed or producer not in active:
                                    blockers.append(producer)
                            elif producer in active and producer not in settled:
                                blockers.append(producer)
                        unresolved = set()
                        for text in (G.nodes[node].get('prompt', ''), G.nodes[node].get('persona', '')):
                            _, _, missing = _resolveChatTemplate(text, chatVars)
                            unresolved.update(missing)
                        if blockers or unresolved:
                            waiting[node] = (blockers, unresolved)
                            continue
                        if len(running) >= numWorkers:
                            continue
                        customFx = _hasCustomChatFx(node, G, fxStore)
                        # Hidden in-place mutations cannot be merged reliably
                        # across snapshots. Unknown callbacks take turns; ordinary
                        # inference/known cleaners can still run in parallel.
                        if customFx and customRunning:
                            continue
                        try:
                            localVars = _NodeVariables(chatVars, isolateNested=customFx)
                        except Exception as exc:
                            errors.append({node: f'Variable snapshot failed: {exc}', 'type': 'node'})
                            settled.add(node)
                            failed.add(node)
                            children[node] = set()
                            stopped = not tolerant
                            refreshNeeded = True
                            break
                        if _viz_enabled and _viz_cid is not None:
                            try:
                                from .visualization import viz_notify_node_queued
                                viz_notify_node_queued(_viz_cid, node)
                            except Exception:
                                pass
                        task = asyncio.create_task(process_one_node(
                            node, G, localVars, fxStore, verbosity, semaphore,
                            workerID=dispatched % numWorkers, timeout=timeout))
                        running[task] = (node, localVars)
                        customRunning = customRunning or customFx
                        dispatched += 1
                if refreshNeeded and not stopped:
                    continue
                if not running:
                    if pending and not stopped:
                        for node in pending:
                            blockers, unresolved = waiting.get(node, ([], set()))
                            message = (f'Unresolved dependencies: {blockers}; '
                                       f'dynamic variables: {sorted(unresolved)}. '
                                       'A producer failed, was pruned, is absent, or the graph has a dependency cycle.')
                            errors.append({node: message, 'type': 'node'})
                    break
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                # Process every completed result before considering any new work.
                for task in sorted(done, key=lambda item: str(running[item][0])):
                    node, localVars = running.pop(task)
                    childNodes, nodeErrors = await task
                    try:
                        localVars.commit(chatVars)
                    except Exception as exc:
                        childNodes = []
                        nodeErrors.append({node: f'Variable writeback failed: {exc}', 'type': 'node'})
                    settled.add(node)
                    children[node] = set(childNodes)
                    if any(error.get('type') == 'node' for error in nodeErrors):
                        failed.add(node)
                    if nodeErrors:
                        errors.extend(nodeErrors)
                        if not tolerant:
                            stopped = True
            if verbosity > 0:
                print(f"\n[Complete] Processed {len(settled)} nodes" +
                      ('; stopped after an error' if stopped else ''))
    
        except KeyboardInterrupt:
            print("\n[!] Execution interrupted by user.")
            errors.append({'walkChatNet': 'Execution interrupted by user (KeyboardInterrupt)', 'type': 'node'})
            chatVars['success'] = False
            chatVars['errors'] = errors
            raise
        finally:
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)

        chatVars['success'] = len(errors) == 0
        chatVars['errors'] = errors
        if _viz_enabled and _viz_cid is not None:
            try:
                from .visualization import viz_notify_chatnet_complete
                viz_notify_chatnet_complete(_viz_cid, chatVars['success'])
            except Exception:
                pass
        return chatVars
    finally:
        _viz_walk_active -= 1


def walkChatNet(G,
                fxStore=dict(),
                varStore=dict(),
                verbosity=1,
                runAsync=False,
                numWorkers=4,
                tolerant=False,
                timeout=None):
    """Main entry point for graph traversal.

    Node/edge fx failures are never allowed to crash unpredictably or vanish
    silently. Every failure is recorded in the returned dict's 'errors' list.

    If tolerant is False (default), the traversal stops as soon as an error is
    hit (no further nodes are processed). If tolerant is True, the traversal
    continues past errors, skipping only the branch(es) affected, to allow
    partial completion.

    The returned dict always includes:
      - 'success': True only if every reachable node ran without error
      - 'errors': list of {<node/edge name>: <message>, 'type': 'node'|'edge'}
    """
    global useCache
    timeout = getPromptTimeout(timeout)

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

                result = asyncio.run(walkChatNetAsync(G, fxStore, varStore, verbosity, numWorkers, tolerant, timeout))
                return result

            except ImportError:
                print("[Error] Install 'nest_asyncio' for Jupyter async support")
                result = dict(varStore) if not isinstance(varStore, dict) else deepcopy(varStore)
                result['success'] = False
                result['errors'] = [{
                    'walkChatNet': "Missing 'nest_asyncio' dependency required for async execution inside an active event loop",
                    'type': 'node',
                }]
                return result
        else:
            # Synchronous execution
            global _viz_walk_active
            _viz_walk_active += 1
            try:
                _viz_cid = _viz_cid_for_graph(G) if _viz_enabled else None
                # If viz was enabled after G was built, G may have no viz_id yet — allocate now
                if _viz_enabled and _viz_cid is None:
                    try:
                        from .visualization import viz_notify_graph_load
                        _viz_cid = viz_notify_graph_load(G)
                    except Exception:
                        pass
                toAsk = ['Start']
                fxStore = fxStore | baseFx
                chatVars = deepcopy(varStore)
                errors = []

                # viz: initial queued
                if _viz_enabled and _viz_cid is not None:
                    try:
                        from .visualization import viz_notify_node_queued
                        viz_notify_node_queued(_viz_cid, 'Start')
                    except Exception:
                        pass

                while toAsk != []:
                    nextQ = toAsk.pop()
                    nextNodes, nodeErrors = processNodeStep(nextQ, G, chatVars, fxStore, verbosity, timeout)
                    if nodeErrors:
                        errors.extend(nodeErrors)
                        if not tolerant:
                            if verbosity > 0:
                                print(f"\n[STOPPED] Error hit on '{nextQ}'; halting traversal (tolerant=False)")
                            break
                    # viz: nodes queued
                    if _viz_enabled and _viz_cid is not None and nextNodes:
                        try:
                            from .visualization import viz_notify_node_queued
                            for _n in nextNodes:
                                viz_notify_node_queued(_viz_cid, _n)
                        except Exception:
                            pass
                    toAsk += nextNodes

                chatVars['success'] = len(errors) == 0
                chatVars['errors'] = errors
                # viz: chatnet complete + reset (frontend schedules reset after delay, but emit both)
                if _viz_enabled and _viz_cid is not None:
                    try:
                        from .visualization import viz_notify_chatnet_complete
                        viz_notify_chatnet_complete(_viz_cid, chatVars['success'])
                    except Exception:
                        pass
                return chatVars
            finally:
                _viz_walk_active -= 1

    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
        result = dict(varStore) if not isinstance(varStore, dict) else deepcopy(varStore)
        result['success'] = False
        result['errors'] = [{'walkChatNet': 'Execution interrupted by user (KeyboardInterrupt)', 'type': 'node'}]
        return result


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


import re as _re
_GARBAGE_TOKEN_RE = _re.compile(
    r'<(?:unused|unk|unknown|reserved|extra_id)_?\d+>',
    _re.IGNORECASE
)


def _contains_garbage_tokens(value) -> bool:
    """True if a model response contains reserved/special-vocab tokens that
    indicate a corrupted or degenerate generation (e.g. <unused56>,
    <unknown12>, <unk3>). These must never be cached or returned as real
    output - treating them the same as a failed call means the next cycle
    gets a fresh attempt rather than replaying the corruption forever."""
    if isinstance(value, str):
        return bool(_GARBAGE_TOKEN_RE.search(value))
    if isinstance(value, (list, dict)):
        return bool(_GARBAGE_TOKEN_RE.search(str(value)))
    return False


def queryToCache(cacheKey,
                 fn,
                 args=(),
                 kwargs=None,
                 maxAttempts=3,
                 tolerant=False,
                 delay=.05):
    """Execute a callable with caching.

    Parameters
    ----------
    cacheKey : str
        Stable string used to derive the cache hash (replaces the old eval
        query string – keep the same value the caller used to build before so
        existing cache entries are still hit).
    fn : callable
        The function to call when no cached result is found.
    args : tuple
        Positional arguments forwarded to *fn*.
    kwargs : dict | None
        Keyword arguments forwarded to *fn*.
    """
    global useCache

    if kwargs is None:
        kwargs = {}

    queryHash = getHash(cacheKey)

    # --- READ FROM CACHE ---
    if useCache:
        cached = cacheGet(queryHash)
        if cached is not None:
            # Never replay cached empty model responses from earlier versions.
            if cacheKey.startswith(('getChatContent(', 'getYN(')) and (
                    not isinstance(cached, str) or not cached.strip()):
                print('[Cache] Ignoring empty cached inference')
            else:
                return cached

    # --- EXECUTE QUERY ---
    sleep(promptDelay)
    gotResults = False
    attempts = 0
    result = None

    while not gotResults and attempts < maxAttempts:
        if tolerant:
            try:
                result = fn(*args, **kwargs)
                gotResults = True
            except Exception:
                attempts += 1
                sleep(5)
        else:
            result = fn(*args, **kwargs)
            gotResults = True
            attempts = maxAttempts

    # --- WRITE TO CACHE ---
    if gotResults:
        if _contains_garbage_tokens(result):
            print(f"[queryToCache] garbage token(s) detected in model response "
                  f"for key {cacheKey!r:.80s} - discarding, not caching")
            return None
        cacheSet(queryHash, cacheKey, result)

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
    cacheKey = f"st.scrapePageText('{url}',maxLen={maxLen})"
    result = queryToCache(cacheKey, st.scrapePageText, args=(url,), kwargs={'maxLen': maxLen})
    return result


_geocoder = Nominatim(
    user_agent="tabulairity",
    timeout=10
)


def _geocodeCoordinates(locText):
    """Geocode text and return only JSON-serializable coordinates.

    New Nominatim queries are rate-limited to 1 req/s per usage policy;
    cached lookups bypass this via queryToCache and incur no delay.
    """
    sleep(1)
    location = _geocoder.geocode(locText)

    if location is None or location.point is None:
        return None

    return [location.point.latitude, location.point.longitude]


def cacheGeocode(locText):
    """Cached geocoding with validation.

    Nominatim returns a geopy Location object, but the generic cache stores
    results as JSON. Convert the Location to the historical [lat, lon]
    representation before it enters the cache.

    Failed requests are retried by queryToCache() with a 5-second delay
    between attempts.
    """
    if locText is None or pd.isna(locText) or str(locText).strip() == "":
        return None

    safeLoc = repr(locText)

    # Use a new namespace so historical OSMnx coordinate lists cannot
    # be returned as though they were geopy Location objects.
    cacheKey = f"geopy.nominatim.geocode({safeLoc})"

    try:
        return queryToCache(
            cacheKey,
            _geocodeCoordinates,
            args=(locText,),
            tolerant=True
        )
    except Exception as e:
        print(f"[Geocode] Failed on {locText}: {e}")
        return None


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
                                  model=translationModel,
                                  extra_params=translationExtraParams)
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
               autoformatPersona=True,
               models=None):
    """Test one or more model routes via the litellm proxy.

    Parameters
    ----------
    models : list[str] | None
        Model names to test. Defaults to [modelName] when not provided.
    """
    if models is None:
        models = [modelName]
    working = []
    for model in sorted(models):
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
                   timeout=None,
                   extra_params=None):
    """Get completion from LLM with timeout - FAIL FAST on errors"""
    timeout = getPromptTimeout(timeout)
    modelRoute, ip, apiKey = getModelRoute(modelName)

    # CDEN extra_params and standalone calls use the same LiteLLM interface.
    # Move standard LiteLLM parameters out of extra_body rather than sending
    # incompatible provider-native fields alongside them.
    provider_params = dict(extra_params or {})
    reasoning_effort = provider_params.pop('reasoning_effort', None)
    if provider_params.get('reasoning') == 'off':
        provider_params.pop('reasoning')
        if reasoning_effort is None:
            reasoning_effort = 'none'

    try:
        content = completion(
            model=modelRoute,
            max_tokens=int(tokens),
            messages=messages,
            api_base=ip,
            api_key=apiKey,
            seed=seed,
            temperature=temperature,
            timeout=timeout,
            **({'reasoning_effort': reasoning_effort,
                'allowed_openai_params': ['reasoning_effort']}
               if reasoning_effort is not None else {}),
            **({'extra_body': provider_params} if provider_params else {})
        )
        choice = content.choices[0]
        cleaned = (choice.message.content or '').strip()
        if not cleaned:
            raise ValueError(
                f"Model {modelName!r} returned empty content "
                f"(finish_reason={choice.finish_reason!r}); inference not cached"
            )
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
                    seed=None,
                    extra_params=None,
                    timeout=None):
    """Ask a question to the chat model"""
    timeout = getPromptTimeout(timeout)
    # Correlate standalone viz events across concurrent calls.
    viz_request_id = uuid4().hex if _viz_enabled and _viz_walk_active == 0 else None
    if viz_request_id:
        try:
            from .visualization import viz_notify_standalone_prompt
            viz_notify_standalone_prompt(
                str(prompt)[:500], str(persona)[:200], str(model),
                request_id=viz_request_id)
        except Exception:
            pass

    if autoformatPersona is True and persona.strip()[-1] != '.':
        personaText = f'You are {persona}. You must answer questions as {persona}.'
    else:
        personaText = persona

    messages = [
        {'role': 'system', 'content': personaText},
        {'role': 'user', 'content': prompt[:350000]}
    ]

    # Keep the legacy cache identity. A transport timeout does not change the
    # answer, and changing this token would invalidate existing prompt caches.
    cacheKey = f"getChatContent({messages},{tokens},'{model}',{temperature},{seed},timeout=600,extra_params={repr(extra_params)})"
    result = queryToCache(
        cacheKey,
        getChatContent,
        args=(messages, tokens, model),
        kwargs={'temperature': temperature, 'seed': seed, 'timeout': timeout, 'extra_params': extra_params},
        tolerant=False,
    )
    if viz_request_id:
        try:
            from .visualization import viz_notify_standalone_response
            viz_notify_standalone_response(
                str(prompt)[:500], str(result)[:500],
                request_id=viz_request_id)
        except Exception:
            pass
    return result


def getYN(text):
    """Standardize yes/no answers without converting inference errors to 'no'."""
    messages = [
        {'role': 'system',
         'content': 'You are an API that standardizes yes or no answers. You may only return a one word answer in lowercase or "None" as appropriate.'},
        {'role': 'user',
         'content': f'Please return a value for the following text, coding the ouput as "yes" for any affirmative response, "no" for any negative response: {text}'}
    ]

    def classify():
        answer = getChatContent(messages, 16, getYNModel,
                                extra_params=getYNExtraParams)
        answer = answer.strip().lower().replace('"', '')
        if answer not in {'yes', 'no', 'none'}:
            raise ValueError(f'Invalid getYN response: {answer!r}')
        return answer

    cacheKey = f"getYN({messages!r},model={getYNModel!r},extra_params={getYNExtraParams!r})"
    result = queryToCache(cacheKey, classify)
    if result not in {'yes', 'no', 'none'}:
        raise ValueError(f'Invalid cached getYN response: {result!r}')
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

    cacheKey = f"getChatContent({messages},100,'{modelName}')"
    result = queryToCache(cacheKey, getChatContent, args=(messages, 100, modelName))
    return result


def evaluateAuthor(response):
    """Check if response identifies as AI"""
    messages = [
        {'role': 'user',
         'content': f'Please answer in one short sentence, does the author of the following answer include any text specically identifying itself as an AI?\nanswer: {response}'}
    ]

    cacheKey = f"getChatContent({messages},100,'{modelName}')"
    result = queryToCache(cacheKey, getChatContent, args=(messages, 100, modelName))
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

    cacheKey = f"getChatContent({messages},3,'{modelName}')"
    result = queryToCache(cacheKey, getChatContent, args=(messages, 3, modelName))
    return result
