import networkx as nx
import pandas as pd
import numpy as np

from . import scrapertools as st

from datetime import datetime
from copy import deepcopy
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
import pycountry
import sqlite3
import asyncio
import traceback
import sys
import argparse
import ast
import math
import threading
from uuid import uuid4
import inspect
from dataclasses import replace
from contextvars import copy_context
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from contextlib import nullcontext
from urllib.parse import urlsplit

from .runtime import WalkContext, WalkControl, walk_context
from .expansion import (expandIterable, expandIterableAsync, cleanIterable,
                        chatNetInputKeys, _joined_child, _join_parent_results)

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


def _viz_dispatch(name, *args, **kwargs):
    """The legacy visualization remains optional and cannot fail a walk."""
    try:
        from . import visualization
        getattr(visualization, name)(*args, **kwargs)
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
    print("[Cache] Using the local SQLite fallback")

# Cache configuration
cacheConfig = {
    'backend': 'postgres',  # 'postgres' or 'sqlite'
    'host': 'localhost',
    'port': 5433,
    'database': 'tabulairity_cache',
    'user': None,  # Will use OS user if None
    'password': None,
    'minConnections': 2,
    'maxConnections': 20,
    'retentionDays': float(os.environ.get('TABULAIRITY_CACHE_RETENTION_DAYS', '14'))
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
    
    if cacheConfigOverride:
        cacheConfig.update(cacheConfigOverride)
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
            purgeOldCache(days=cacheConfig.get('retentionDays', 14))
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
        if cacheConfig['backend'] != 'postgres':
            return sqlite3.connect(cacheDatabase, timeout=120)
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


def purgeOldCache(days=14, vacuum=False):
    """Expire cache entries in either backend; optional VACUUM is outside a transaction."""
    if isinstance(days, bool) or not isinstance(days, (int, float)) or not math.isfinite(days) or days < 0:
        raise ValueError('days must be a finite nonnegative number')
    conn = cursor = None
    deleted = 0
    try:
        conn = getConnection()
        cursor = conn.cursor()
        if cacheConfig['backend'] == 'postgres':
            cursor.execute("DELETE FROM cache WHERE timestamp < NOW() - (%s * INTERVAL '1 day')", (days,))
        else:
            cursor.execute("DELETE FROM cache WHERE timestamp < datetime('now', '-' || ? || ' days')", (days,))
        deleted = cursor.rowcount
        conn.commit()
        # PostgreSQL autovacuum normally handles reclamation. Explicit vacuum
        # must use autocommit, restored before returning this pooled connection.
        if vacuum:
            try:
                if cacheConfig['backend'] == 'postgres':
                    previous = conn.autocommit
                    try:
                        conn.autocommit = True
                        cursor.execute('VACUUM ANALYZE cache')
                    finally:
                        conn.autocommit = previous
                else:
                    conn.execute('VACUUM')
            except Exception as exc:
                conn.rollback()
                print(f'[Cache] Entries expired; optional vacuum failed: {exc}')
        if deleted:
            print(f'[Cache] Purged {deleted} entries older than {days} days')
        return deleted
    except Exception as exc:
        print(f'[Cache] Purge error: {exc}')
        if conn is not None:
            conn.rollback()
        return 0
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
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
    cacheConfig['backend'] = 'sqlite'
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
        
        purgeOldCache(days=cacheConfig.get('retentionDays', 14))
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

modelName = "gemma4:26b"
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
_routeLock = threading.Lock()
_routeLastUsed = {}
_routeSequence = 0
_localModelRoutes = None
_localRoutesStamp = None


def _readModelRoutes(path):
    """Read a complete route table without combining fields from other sources."""
    try:
        routes = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read model routing table {path}: {exc}') from exc
    required = {'model', 'route', 'ip'}
    missing = required.difference(routes.columns)
    if missing:
        raise ValueError(f'Model routing table {path} is missing columns: {", ".join(sorted(missing))}')
    for column in required:
        routes[column] = routes[column].str.strip()
        if routes[column].eq('').any():
            raise ValueError(f'Model routing table {path} has an empty {column!r} value')
    normalized_bases = []
    for base in routes['ip']:
        if base.casefold() == 'remote':
            # Preserve the established sentinel for provider-default endpoints.
            normalized_bases.append(base)
            continue
        # Older local route files often used host:port without a URL scheme.
        # Treat those as local HTTP endpoints; keep explicit non-HTTP schemes invalid.
        if '://' not in base:
            base = 'http://' + base.lstrip('/')
        parsed = urlsplit(base)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            raise ValueError(f'Model routing table {path}: ip must use http:// or https://')
        normalized_bases.append(base)
    routes['ip'] = normalized_bases
    return routes.set_index('model')


def _refreshLocalModelRoutes():
    """Refresh the working-directory table when it changes; caller holds the lock."""
    global _localModelRoutes, _localRoutesStamp
    path = os.path.abspath('config/model_routes.csv')
    try:
        stat = os.stat(path)
        stamp = (path, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except FileNotFoundError:
        stamp = (path, None)
    if stamp != _localRoutesStamp:
        routes = _readModelRoutes(path) if stamp[1] is not None else None
        _localModelRoutes, _localRoutesStamp = routes, stamp
        _routeLastUsed.clear()
        if routes is not None:
            print(f'[Routes] Loaded {len(routes)} local model routes from {path} (highest priority)')
    return _localModelRoutes

def loadModelRoutes(csvPath='config/model_routes.csv'):
    """Load CSV routes, giving local config/model_routes.csv highest priority.

    Expected columns: model, route, ip
    Optional columns: key, key_env (API key or environment-variable name).
    An ip value of 'remote' delegates endpoint and default credentials to LiteLLM.

    Another CSV can supply aliases absent from the working-directory table.
    For aliases defined locally, only local rows participate in LRU selection.
    HTTP and HTTPS backend URLs are both supported. No backend is auto-selected.
    """
    global _modelRoutes, _localRoutesStamp
    path = os.path.abspath(os.fspath(csvPath))
    localPath = os.path.abspath('config/model_routes.csv')
    with _routeLock:
        if path == localPath:
            _modelRoutes = None
            _localRoutesStamp = None  # Explicit reload, even if file metadata is unchanged.
        else:
            _modelRoutes = _readModelRoutes(path) if os.path.exists(path) else None
            if _modelRoutes is not None:
                print(f'[Routes] Loaded {len(_modelRoutes)} additional model routes from {path}; local aliases take priority')
        local = _refreshLocalModelRoutes()
        _routeLastUsed.clear()
        if local is None and _modelRoutes is None:
            print(f'[Routes] No routing table at {localPath}; configure routes before model inference')


loadModelRoutes()


def getModelRoute(name):
    """Resolve local aliases first, retaining LRU across their duplicate deployments."""
    global _routeSequence
    with _routeLock:
        local = _refreshLocalModelRoutes()
        if local is None and _modelRoutes is None:
            raise RuntimeError(f'No routing table loaded. Configure {os.path.abspath("config/model_routes.csv")} first.')
        aliases = [name, name.split('/', 1)[1]] if '/' in name else [name]
        matches = None
        for routes in (local, _modelRoutes):
            if routes is None:
                continue
            for alias in aliases:
                if alias in routes.index:
                    name = alias
                    matches = routes.loc[[alias]]
                    break
            if matches is not None:
                break
        if matches is None:
            raise KeyError(f'Model {name!r} is not in local config/model_routes.csv or the additional routing table')
        offset = min(range(len(matches)), key=lambda i: _routeLastUsed.get((name, i), -1))
        row = matches.iloc[offset]
        _routeSequence += 1
        _routeLastUsed[name, offset] = _routeSequence
        ip, route = row['ip'], row['route']
        remote = isinstance(ip, str) and ip.casefold() == 'remote'
        if remote:
            ip = None  # Let LiteLLM use the provider's standard endpoint.
            host = 'remote'
        else:
            host = ip.split('://')[-1].split('/')[0]
        routeKeys = json.loads(os.environ.get('ROUTE_KEYS', '{}'))
        keyVar = row.get('key_env')
        if not isinstance(keyVar, str) or not keyVar.strip():
            keyVar = routeKeys.get(host, None if remote else 'OPENAI_API_KEY')
        key = row.get('key')
        if not isinstance(key, str) or not key.strip():
            key = os.environ.get(keyVar) if keyVar else None
            if key is None and not remote:
                key = os.environ.get('OPENAI_API_KEY', 'dummy')
        return route, ip, key


#########################################
#                                       #
#      TEXT INTERROGATION FUNCTIONS     #
#                                       #
#########################################


def validRun(var, prompt):
    return isValid(var) or str(prompt).startswith('recall:')


def isValid(var):
    return not (var is None or (isinstance(var, str) and not var) or
                (isinstance(var, (float, np.floating)) and np.isnan(var)))


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


def nestChatNets(upstream, downstream, upstreamNode, *, startKey=None, prefix=None, edgeFx='null'):
    """Weld a copy of downstream to upstreamNode, retaining dependency order.

    Rename the downstream Start and any colliding node names. Rewrite static
    bracket references in prompts/personas/edge prompts, including _raw,
    _prompt and edge result variables. Neither input graph is mutated.
    The rename map is exposed as result.graph['nested_node_map'].
    """
    if not isinstance(upstream, nx.MultiDiGraph) or not isinstance(downstream, nx.MultiDiGraph):
        raise TypeError('nestChatNets takes two built chat networks')
    if upstreamNode not in upstream or 'Start' not in upstream or 'Start' not in downstream:
        raise ValueError('Both graphs need Start and upstreamNode must exist in upstream')
    if edgeFx == 'expandIterable':
        raise ValueError('Use a predicate or null for the connecting edge')
    used = set(upstream) | (set(downstream) - {'Start'}) | RESERVED_NODE_NAMES
    mapping = {}
    def unique(candidate):
        if not isinstance(candidate, str) or not candidate:
            raise ValueError('Renamed keys must be nonempty strings')
        base, number = candidate, 2
        while candidate in used:
            candidate = f'{base}_{number}'
            number += 1
        used.add(candidate)
        return candidate
    if startKey is not None and startKey in used:
        raise ValueError(f'Requested startKey {startKey!r} is already in use')
    mapping['Start'] = unique(startKey or f'{upstreamNode}_Start')
    for node in downstream:
        if node != 'Start':
            mapping[node] = unique(f'{prefix}{node}' if prefix else f'{upstreamNode}_{node}') \
                if prefix or node in upstream else node
    refs = dict(mapping)
    refs.update({str(old)+suffix: str(new)+suffix for old,new in mapping.items()
                 for suffix in ('_raw', '_prompt', '_item', '_expansion_errors')})
    refs.update({f'{a}-{b}': f'{mapping[a]}-{mapping[b]}' for a,b in downstream.edges()})
    def rewrite(text, names=refs):
        if not isinstance(text, str):
            return text
        return re.sub(r'\[([^\[\]\n]+)\]', lambda m: '['+names.get(m[1], m[1])+']', text)
    result = deepcopy(upstream)
    child = nx.relabel_nodes(deepcopy(downstream), mapping, copy=True)
    for _, data in child.nodes(data=True):
        for field in ('prompt', 'persona'):
            data[field] = rewrite(data.get(field, ''))
    for _, _, data in child.edges(data=True):
        data['prompt'] = rewrite(data.get('prompt', ''))
    result.add_nodes_from(child.nodes(data=True))
    for a,b,data in child.edges(data=True):
        result.add_edge(a,b,**data)
    renamed_start = mapping['Start']
    result.add_edge(upstreamNode, renamed_start, fx=edgeFx, prompt='', fx_params=None)
    forced = list(upstream.graph.get('forced_dependencies', []))
    forced += [(mapping.get(a,a),mapping.get(b,b)) for a,b in downstream.graph.get('forced_dependencies', [])]
    # A copied root retains its own Start barrier even after it is renamed.
    forced += [(upstreamNode, renamed_start)]
    forced += [(renamed_start, mapping[n]) for n in nx.descendants(downstream, 'Start')]
    result.graph['forced_dependencies'] = forced
    result.graph['nested_node_map'] = mapping
    def source_segments(graph, fallback, renames=None):
        renames = renames or {}
        existing = graph.graph.get('viz_segments')
        if not existing:
            existing = [{
                'name': graph.graph.get('name') or fallback,
                'nodes': [str(node) for node in graph.nodes],
            }]
        return [{
            'name': str(segment.get('name') or fallback),
            'nodes': [str(renames.get(node, node)) for node in segment.get('nodes', [])],
        } for segment in existing]
    result.graph['viz_segments'] = (
        source_segments(upstream, 'Upstream CDEN')
        + source_segments(downstream, 'Merged CDEN', mapping)
    )
    # The live viewer should render the welded result as the visible network,
    # not retain extra standalone copies of its inputs beside it. Preserve
    # transitive source ids so repeated merges collapse earlier composites too.
    merged_from = []
    merged_fingerprints = []
    merged_shapes = []
    from .visualization import viz_graph_fingerprint
    from .visualization import viz_graph_shape_fingerprint
    for source_graph in (upstream, downstream):
        merged_from.extend(source_graph.graph.get('viz_merged_from_ids', []))
        merged_fingerprints.extend(source_graph.graph.get('viz_merged_from_fingerprints', []))
        merged_shapes.extend(source_graph.graph.get('viz_merged_from_shapes', []))
        merged_fingerprints.append(viz_graph_fingerprint(source_graph))
        merged_shapes.append(viz_graph_shape_fingerprint(source_graph))
        source_viz_id = source_graph.graph.get('viz_id')
        if source_viz_id is not None:
            merged_from.append(source_viz_id)
    result.graph['viz_merged_from_ids'] = list(dict.fromkeys(merged_from))
    result.graph['viz_merged_from_fingerprints'] = list(dict.fromkeys(merged_fingerprints))
    result.graph['viz_merged_from_shapes'] = list(dict.fromkeys(merged_shapes))
    result.graph['chatNets'] = deepcopy(upstream.graph.get('chatNets', {}))
    registry = downstream.graph.get('chatNets', {})
    collisions = registry.keys() & result.graph['chatNets'].keys()
    if collisions:
        raise ValueError(f'Nested network registry collision: {sorted(collisions)!r}; rename it before welding')
    reserved_names = set(registry) | set(result.graph['chatNets'])
    rewritten = {}
    copies = {}

    def copy_nested(graph, inherited, stack):
        # Child nodes retain their own keys. Only references inherited from
        # the renamed parent change; [Start] in a child remains its own Start.
        local = set(graph) | {'success', 'errors'}
        local |= {str(n)+suffix for n in graph for suffix in ('_raw', '_prompt')}
        local |= {f'{a}-{b}' for a,b in graph.edges()}
        names = {old:new for old,new in inherited.items() if old not in local}
        copied = deepcopy(graph)
        for source, data in copied.nodes(data=True):
            for field in ('prompt', 'persona'):
                data[field] = rewrite(data.get(field, ''), names)
            if data.get('fx') == 'expandIterable':
                parameters = data.get('fx_params') or {}
                nested = parameters.get('chatNet')
                child_names = {old:new for old,new in names.items()
                               if old not in {str(source)+'_item', str(source)+'_expansion_errors'}}
                if isinstance(nested, str) and nested in registry:
                    data['fx_params'] = parameters | {'chatNet': copy_registered(nested, child_names, stack)}
                elif isinstance(nested, nx.MultiDiGraph):
                    data['fx_params'] = parameters | {'chatNet': copy_nested(nested, child_names, stack)}
        for _,_,data in copied.edges(data=True):
            data['prompt'] = rewrite(data.get('prompt', ''), names)
        return copied

    def copy_registered(name, names, stack):
        if name in stack:
            raise ValueError('Nested CDEN references contain recursion')
        key = (name, tuple(sorted(names.items())))
        if key in copies:
            return copies[key]
        alias, index = name, 2
        if alias in rewritten:
            base = str(name)[:80]
            alias = f'{base}_scope_{index}'
            while alias in reserved_names:
                index += 1
                alias = f'{base}_scope_{index}'
        reserved_names.add(alias)
        copies[key] = alias
        rewritten[alias] = None
        rewritten[alias] = copy_nested(registry[name], names, stack + (name,))
        return alias

    for node in downstream:
        data = result.nodes[mapping[node]]
        if data.get('fx') != 'expandIterable':
            continue
        parameters = data.get('fx_params') or {}
        nested = parameters.get('chatNet')
        if isinstance(nested, str) and nested in registry:
            data['fx_params'] = parameters | {'chatNet': copy_registered(nested, refs, ())}
        elif isinstance(nested, nx.MultiDiGraph):
            data['fx_params'] = parameters | {'chatNet': copy_nested(nested, refs, ())}
    for name in registry:
        if name not in rewritten:
            copy_registered(name, refs, ())
    result.graph['chatNets'].update(rewritten)
    result.graph.pop('dependency_graph', None)
    result.graph.pop('viz_id', None)
    return result


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

    def parseParams(val):
        if isinstance(val, dict):
            return dict(val)
        if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == '':
            return None
        parsed = json.loads(val)
        if not isinstance(parsed, dict):
            raise ValueError('extra_params and fx_params must be JSON objects')
        return parsed
    for column in ('extra_params', 'fx_params'):
        script[column] = script[column].apply(parseParams) if column in script else None
    for column, default in [('persona', ''), ('tokens', 2000)]:
        if column not in script:
            script[column] = default
    script['self_eval'] = script['self_eval'].apply(
        lambda value: value is True or str(value).strip().lower() in {'true', 'yes', '1'})

    script['model'] = script['model'].fillna(modelName)
    script['tokens'] = script['tokens'].fillna(2000)
    script['persona'] = script['persona'].fillna('')
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
                     'extra_params': row['extra_params'],
                     'fx_params': row['fx_params']}) for index, row in chatNodes.T.items()]

    G.add_nodes_from(nodesParsed)

    for _, row in chatEdges.iterrows():
        source, target = row.get('source'), row.get('target')
        if not isinstance(source, str) or not isinstance(target, str):
            parts = row['key'].split('-')
            if len(parts) != 2:
                raise ValueError('Use source/target columns for node names containing hyphens')
            source, target = parts
        if source not in G or target not in G:
            raise ValueError(f'Edge {source!r} -> {target!r} refers to an absent node')
        if row['fx'] == 'expandIterable':
            raise ValueError('expandIterable is a node chatFX; use a predicate on edges')
        G.add_edge(source, target, prompt=row['prompt'], fx=row['fx'], fx_params=row['fx_params'])
    connected = nx.is_weakly_connected(G)

    if not connected:
        print(f"[Warning] Chat graph has disconnected components.")

    if show:
        from matplotlib import pyplot as plt
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
                        value = varStore[key]
                        if isinstance(value, pd.DataFrame):
                            output.append(value.to_json(orient='records', force_ascii=False))
                        elif isinstance(value, pd.Series):
                            output.append(value.to_json(force_ascii=False))
                        else:
                            output.append(str(value))
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
          'insertJson': insertJson,
          'expandIterable': expandIterable}


async def _callChatFx(name, response, variables, params, context, source=None):
    params = dict(params or {})
    if name == 'expandIterable':
        if source is None:
            raise ValueError('expandIterable can only be attached to nodes')
        params['source'] = source
        return await expandIterableAsync(response, variables, **params)
    fn = context.fx[name]
    if inspect.iscoroutinefunction(fn):
        return await fn(response, variables, **params)
    result = await asyncio.to_thread(fn, response, variables, **params)
    return await result if inspect.isawaitable(result) else result


async def _limitedInference(context, function, *args, budget=None, **kwargs):
    async with context.slots:
        await context.checkpoint()
        token = walk_context.set(replace(context, in_slot=True))
        try:
            copied = copy_context()
            call = asyncio.get_running_loop().run_in_executor(context.executor, copied.run, partial(function, *args, **kwargs))
            return await asyncio.wait_for(call, budget) if budget is not None else await call
        finally:
            walk_context.reset(token)


async def _executeNode(currentNode, G, chatVars, fxStore, verbosity, semaphore, timeout=None):
    context = walk_context.get() or WalkContext(semaphore, {}, fxStore)
    data = G.nodes[currentNode]
    context = replace(context, node=currentNode, graph=G,
                      model=insertChatVars(data.get('model', modelName), chatVars))
    token = walk_context.set(context)
    started = perf_counter()
    errors, children = [], []
    cid = _viz_cid_for_graph(G) if _viz_enabled else None
    response = ''
    try:
        await context.checkpoint()
        prompt = insertChatVars(data.get('prompt', ''), chatVars)
        persona = insertChatVars(data.get('persona', ''), chatVars)
        context.emit('node_start', node=currentNode, prompt=prompt, persona=persona,
                     model=context.model, fx=data.get('fx', 'null'))
        if cid is not None:
            _viz_dispatch('viz_notify_node_start', cid, currentNode,
                          prompt=str(prompt), persona=str(persona), fullPrompt=str(prompt),
                          trace=list(context.path), run_id=context.execution_id)
        if validRun(persona, prompt):
            if str(prompt).startswith('recall:'):
                response = prompt[7:].strip()
            else:
                # Only inference holds a slot. Expansion awaits children after
                # releasing it, so recursive walks cannot starve their children.
                response = await _limitedInference(context, askChatQuestion,
                    prompt, persona, model=context.model, tokens=data.get('tokens', 2000),
                    extra_params=data.get('extra_params'), timeout=timeout,
                    budget=max(1500, getPromptTimeout(timeout) + 300))
            chatVars[currentNode + '_prompt'] = prompt
            chatVars[currentNode + '_raw'] = response
        if data.get('self_eval', False):
            useful = await _limitedInference(context, isUseful, prompt, response)
            if not useful:
                context.emit('node_rejected', node=currentNode, elapsed=perf_counter()-started)
                if cid is not None:
                    _viz_dispatch('viz_notify_node_error', cid, currentNode, 'self_eval rejected response', trace=list(context.path), run_id=context.execution_id)
                return [], []
        await context.checkpoint()
        cleaned = await _callChatFx(data.get('fx', 'null'), response, chatVars,
                                    data.get('fx_params'), context, source=currentNode)
        chatVars[currentNode] = cleaned
        for source, target, edge in G.out_edges(currentNode, data=True):
            edgeName = f'{source}-{target}'
            try:
                opened = await _callChatFx(edge.get('fx', 'null'), response, chatVars,
                                           edge.get('fx_params'), context)
                chatVars[edgeName] = opened
                if cid is not None:
                    _viz_dispatch('viz_notify_edge_evaluated', cid, edgeName, str(opened).lower() == 'true', fx=edge.get('fx', 'null'))
                context.emit('edge', node=currentNode, source=source, target=target,
                             fx=edge.get('fx', 'null'), opened=str(opened).lower() == 'true')
                if str(opened).lower() == 'true':
                    children.append(target)
                    if verbosity:
                        showIfValid(insertChatVars(edge.get('prompt', ''), chatVars))
            except Exception as exc:
                errors.append({edgeName: str(exc), 'type': 'edge'})
        context.emit('node_complete', node=currentNode, prompt=prompt, persona=persona,
                     raw=response, output=cleaned, elapsed=perf_counter()-started,
                     output_length=len(str(response)), fx=data.get('fx', 'null'))
        if cid is not None:
            _viz_dispatch('viz_notify_prompt_response', cid, currentNode, str(prompt), str(response), str(cleaned),
                persona=str(persona), fx=str(data.get('fx','null')), fullPrompt=str(prompt), trace=list(context.path), run_id=context.execution_id)
            _viz_dispatch('viz_notify_node_complete', cid, currentNode, str(response), str(cleaned), trace=list(context.path), run_id=context.execution_id)
        if verbosity:
            print(f"   <<< {currentNode}: {str(cleaned)[:150]}")
        return sorted(set(children), reverse=True), errors
    except asyncio.CancelledError:
        if isinstance(chatVars, _NodeVariables):
            chatVars.timedOut = True
        raise
    except Exception as exc:
        errors.append({currentNode: str(exc), 'type': 'node'})
        context.emit('node_error', node=currentNode, error=str(exc), raw=response,
                     elapsed=perf_counter()-started)
        if cid is not None:
            try:
                from .visualization import viz_notify_node_error
                viz_notify_node_error(cid, currentNode, str(exc), trace=list(context.path), run_id=context.execution_id)
            except Exception:
                pass
        return [], errors
    finally:
        walk_context.reset(token)


def processNodeStep(currentNode, G, chatVars, fxStore, verbosity, timeout=None):
    """Compatibility entry point for a single synchronous node step."""
    async def run():
        slots = asyncio.Semaphore(1)
        return await _executeNode(currentNode, G, chatVars, fxStore | baseFx, verbosity, slots, timeout)
    return asyncio.run(run())


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
    for suffix in ('_raw', '_prompt', '_expansion_errors'):
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


def buildChatNetDependencies(G, varStore=None, fxStore=None, *, _joinedPlans=None):
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
        templates = (data.get('prompt', ''), data.get('persona', ''), data.get('model', ''))
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
        joined = fx == 'expandIterable' and (data.get('fx_params') or {}).get('inherit_parent') == 'join'
        if joined:
            context = walk_context.get()
            nets = context.nets if context else G.graph.get('chatNets', {})
            _, inheritedRefs = (_joinedPlans[node] if _joinedPlans is not None
                                else _joined_child(G, node, fxStore, nets))
            for key in inheritedRefs:
                producer = _chatVarProducer(key, G)
                if producer is not None:
                    depend(producer, node, 'data', key)
        # Snapshot expansions retain ancestor ordering. Joined leaf expansions
        # wait for actual child reads instead and inherit late parent values
        # only after the parent walk settles. Neither is an opaque writer.
        opaque = fx not in pureFx and fx not in {'mergeJson', 'expandIterable'}
        opaqueEdges = any(edge.get('fx', 'null') not in pureFx
                          for _, _, edge in G.out_edges(node, data=True))
        if opaque or opaqueEdges or (fx == 'expandIterable' and not joined):
            # Custom callables may read chatVars without bracket references.
            for ancestor in nx.ancestors(G, node):
                depend(ancestor, node, 'order')
        # self_eval may reject the node after inference, before any edges open.
        # All descendants must await that decision even on unconditional paths.
        if data.get('self_eval', False) or dynamic or fx == 'mergeJson' or opaque or opaqueEdges:
            for descendant in nx.descendants(G, node):
                depend(node, descendant, 'barrier')

    for source, target in G.graph.get('forced_dependencies', []):
        if source not in G or target not in G:
            raise ValueError('A forced dependency refers to an absent node')
        depend(source, target, 'order')
        for descendant in nx.descendants(G, target):
            depend(source, descendant, 'barrier')
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
    return await _executeNode(node, G, chatVars, fxStore, verbosity, semaphore, timeout)


async def walkChatNetAsync(G, fxStore=None, varStore=None, verbosity=1, numWorkers=4, tolerant=False,
                           timeout=None, *, chatNets=None, observer=None, control=None):
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
    contextToken = None
    ownedExecutor = None
    try:
        _viz_cid = _viz_cid_for_graph(G) if _viz_enabled else None
        if _viz_enabled and _viz_cid is None:
            try:
                from .visualization import viz_notify_graph_load
                _viz_cid = viz_notify_graph_load(G)
            except Exception:
                pass
        if isinstance(numWorkers, bool) or not isinstance(numWorkers, int) or numWorkers < 1:
            raise ValueError('numWorkers must be a positive integer')
        timeout = getPromptTimeout(timeout)
        if isinstance(varStore, pd.Series):
            chatVars = varStore.to_dict()
        else:
            chatVars = dict(varStore) if varStore is not None else {}
        fxStore = (fxStore or {}) | baseFx
        inherited = walk_context.get()
        executor = inherited.executor if inherited is not None else None
        if executor is None:
            executor = ownedExecutor = ThreadPoolExecutor(max_workers=numWorkers, thread_name_prefix='tabulairity-inference')
        semaphore = inherited.slots if inherited is not None else asyncio.Semaphore(numWorkers)
        context = WalkContext(semaphore,
            dict(inherited.nets if inherited else G.graph.get('chatNets', {})) | (chatNets or {}),
            fxStore, numWorkers, observer if observer is not None else (inherited.observer if inherited else None),
            control if control is not None else (inherited.control if inherited else None),
            inherited.path if inherited else (), graph=G, loop=asyncio.get_running_loop(), executor=executor,
            expansion_fanout=inherited.expansion_fanout if inherited else 1,
            execution_id=uuid4().hex)
        contextToken = walk_context.set(context)
    
        settled, failed, children, running = set(), set(), {}, {}
        dispatched = 0
        errors = []
        stopped = False
        if 'Start' not in G:
            errors.append({'walkChatNet': "Chat graph has no 'Start' node", 'type': 'node'})
            stopped = True

        # Joined definitions are static by contract. Scan their templates once
        # per walk rather than repeatedly traversing large nested prompt trees
        # on every node completion.
        joinedPlans = {node: _joined_child(G, node, fxStore, context.nets)
                       for node, data in G.nodes(data=True)
                       if data.get('fx') == 'expandIterable' and
                       (data.get('fx_params') or {}).get('inherit_parent') == 'join'}

        try:
            while True:
                await context.checkpoint()
                active = _possibleChatNodes(G, settled, children)
                overlay = buildChatNetDependencies(G, chatVars, fxStore, _joinedPlans=joinedPlans)
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
                        for text in (G.nodes[node].get('prompt', ''), G.nodes[node].get('persona', ''), G.nodes[node].get('model', '')):
                            _, _, missing = _resolveChatTemplate(text, chatVars)
                            unresolved.update(missing)
                        if blockers or unresolved:
                            waiting[node] = (blockers, unresolved)
                            continue
                        # A long-lived expansion orchestrates child walks; it
                        # must not consume the local dispatch positions needed
                        # by independent inference. Keep both queues bounded,
                        # with the shared semaphore enforcing the actual model
                        # limit across all graph depths. One-worker mode stays
                        # sequential for synchronous callers.
                        expanding = sum(G.nodes[name].get('fx') == 'expandIterable'
                                        for name, _ in running.values())
                        isExpansion = G.nodes[node].get('fx') == 'expandIterable'
                        occupied = expanding if isExpansion else len(running) - expanding
                        if occupied >= numWorkers or (numWorkers == 1 and running):
                            continue
                        customFx = _hasCustomChatFx(node, G, fxStore)
                        # Hidden in-place mutations cannot be merged reliably
                        # across snapshots. Unknown callbacks take turns; ordinary
                        # inference/known cleaners can still run in parallel.
                        if customFx and customRunning:
                            continue
                        try:
                            # Expansion deep-copies each child's inputs itself;
                            # duplicating/pickling the entire parent snapshot
                            # here adds cost without additional isolation.
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
                                viz_notify_node_queued(_viz_cid, node, trace=list(context.path), run_id=context.execution_id)
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
        _join_parent_results(G, chatVars, fxStore, context.nets)
        if _viz_enabled and _viz_cid is not None:
            try:
                from .visualization import viz_notify_chatnet_complete
                viz_notify_chatnet_complete(_viz_cid, chatVars['success'], trace=list(context.path), run_id=context.execution_id)
            except Exception:
                pass
        return chatVars
    finally:
        if ownedExecutor is not None:
            ownedExecutor.shutdown(wait=False, cancel_futures=True)
        if contextToken is not None:
            walk_context.reset(contextToken)
        _viz_walk_active -= 1


def walkChatNet(G, fxStore=None, varStore=None, verbosity=1, runAsync=False,
                numWorkers=4, tolerant=False, timeout=None, *, chatNets=None,
                observer=None, control=None):
    """Walk a CDEN using the same dependency semantics in both modes.

    runAsync=False uses one node worker. In an async application prefer
    await walkChatNetAsync(...) over this synchronous compatibility wrapper.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        import nest_asyncio
        nest_asyncio.apply(loop)
    try:
        return asyncio.run(walkChatNetAsync(G, fxStore or {}, {} if varStore is None else varStore, verbosity,
            numWorkers if runAsync else 1, tolerant, timeout,
            chatNets=chatNets, observer=observer, control=control))
    except KeyboardInterrupt:
        return dict({} if varStore is None else varStore) | {'success': False,
            'errors': [{'walkChatNet': 'Execution interrupted by user', 'type': 'node'}]}


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
                context = walk_context.get()
                if context:
                    context.emit('cache_hit')
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


def _getChatContentTransport(messages,
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
    # A provider body must not override the destination/model/key selected by CSV.
    for field in ('model', 'api_base', 'base_url', 'api_key', 'custom_llm_provider'):
        provider_params.pop(field, None)
    reasoning_effort = provider_params.pop('reasoning_effort', None)
    allowed = provider_params.pop('allowed_openai_params', [])
    if not isinstance(allowed, list) or not all(isinstance(name, str) for name in allowed):
        raise ValueError('allowed_openai_params must be a list of parameter names')
    allowed = list(allowed)
    if provider_params.get('reasoning') == 'off':
        provider_params.pop('reasoning')
        if reasoning_effort is None:
            reasoning_effort = 'none'

    if reasoning_effort is not None and 'reasoning_effort' not in allowed:
        allowed.append('reasoning_effort')
    standard_params = {'allowed_openai_params': allowed} if allowed else {}
    if reasoning_effort is not None:
        standard_params['reasoning_effort'] = reasoning_effort

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
            **standard_params,
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


def getChatContent(messages, tokens, modelName, temperature=None, seed=None, timeout=None, extra_params=None):
    """Instrument the transport boundary, including inference inside chatFX."""
    context = walk_context.get()
    slot = context.inference_slot() if context else nullcontext()
    with slot:
        start = perf_counter()
        if context:
            context.emit('inference_start', model=modelName, tokens=tokens, messages=messages)
        try:
            result = _getChatContentTransport(messages, tokens, modelName, temperature, seed, timeout, extra_params)
        except Exception as exc:
            if context: context.emit('inference_error', model=modelName, error=str(exc), elapsed=perf_counter()-start)
            raise
        if context:
            context.emit('inference_complete', model=modelName, elapsed=perf_counter()-start,
                         output_length=len(str(result)), raw=result)
        return result


def askChatQuestion(prompt,
                    persona,
                    model=None,
                    autoformatPersona=None,
                    tokens=2000,
                    temperature=None,
                    seed=None,
                    extra_params=None,
                    timeout=None):
    """Ask a question to the chat model"""
    if model is None:
        model = modelName
    timeout = getPromptTimeout(timeout)
    # viz: standalone prompt — suppress duplicate entries during graph walks
    if _viz_enabled and _viz_walk_active == 0:
        try:
            from .visualization import viz_notify_standalone_prompt
            viz_notify_standalone_prompt(str(prompt)[:500], str(persona)[:200], str(model))
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
    if _viz_enabled and _viz_walk_active == 0:
        try:
            from .visualization import viz_notify_standalone_response
            viz_notify_standalone_response(str(prompt)[:500], str(result)[:500])
        except Exception:
            pass
    return result


def getYN(text):
    """Standardize yes/no answers without converting inference errors to 'no'."""
    label = _stripChatReasoning(str(text)).strip().lower().strip('"\' .!')
    known = {'yes': 'yes', '[yes]': 'yes', 'true': 'yes', 'no': 'no', '[no]': 'no', 'false': 'no'}
    if label in known:
        return known[label]
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
    context = walk_context.get()
    helperModel = context.model if context and context.model else modelName
    messages = [
        {'role': 'system',
         'content': 'You are a debate moderator skilled at identifying the presence of answer in long statements'},
        {'role': 'user',
         'content': f'Please answer in one short sentence, does the following answer provide any useable answer for the provided question?\nquestion: {question}\nanswer: {response}'}
    ]

    cacheKey = f"getChatContent({messages},100,'{helperModel}')"
    result = queryToCache(cacheKey, getChatContent, args=(messages, 100, helperModel))
    return result


def evaluateAuthor(response):
    """Check if response identifies as AI"""
    context = walk_context.get()
    helperModel = context.model if context and context.model else modelName
    messages = [
        {'role': 'user',
         'content': f'Please answer in one short sentence, does the author of the following answer include any text specically identifying itself as an AI?\nanswer: {response}'}
    ]

    cacheKey = f"getChatContent({messages},100,'{helperModel}')"
    result = queryToCache(cacheKey, getChatContent, args=(messages, 100, helperModel))
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
