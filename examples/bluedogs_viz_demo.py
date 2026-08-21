"""
bluedogs_viz_demo.py — Simple script version of BlueDogsDemo.ipynb for debugging the live viz.

- Caching disabled (useCache = False) so every run hits the viz pipeline and is easy to debug.
- Starts viz, runs 1-2 chatNets, keeps server alive for manual inspection.

Usage:
    PYTHONPATH=src python examples/bluedogs_viz_demo.py            # real LLM (needs config/model_routes.csv + keys)
    PYTHONPATH=src python examples/bluedogs_viz_demo.py --mock     # no LLM, instant fake responses, good for viz smoke test
    PYTHONPATH=src python examples/bluedogs_viz_demo.py --two      # load two chatNets side-by-side to test force separation
"""
import argparse
import sys
import time
sys.path.insert(0, "src")

import pandas as pd
import tabulairity as tb

parser = argparse.ArgumentParser()
parser.add_argument("--mock", action="store_true", help="Use fake LLM responses (no network, instant).")
parser.add_argument("--two", action="store_true", help="Load two chatNets side-by-side.")
parser.add_argument("--delay", type=float, default=0.0, help="Artificial per-node delay in seconds (to see animations).")
parser.add_argument("--keep-eval", action="store_true", help="Keep self_eval / isUseful checks (default OFF for viz debugging).")
args = parser.parse_args()

# --- disable cache for debugging ---
tb.core.useCache = False
# also disable any prompt delay that would hide timing
tb.core.promptDelay = 0.0
# try to purge to guarantee no stale hits
try:
    tb.core.purgeOldCache(days=0)
except Exception:
    pass
print(f"[Demo] useCache = {tb.core.useCache} (cache bypassed, isUseful={'kept' if args.keep_eval else 'DISABLED'})")

# Disable answer checks (isUseful) unless explicitly kept — they cause Start to FAILS with real LLM
if not args.keep_eval and not args.mock:
    import tabulairity.core as _core
    _core.isUseful = lambda q, r: True
    print("[Demo] isUseful stubbed to True (disable self_eval checks) — use --keep-eval to re-enable")

# --- chat fx ---
chatFx = {
    "isYes": lambda x, y: tb.ynToBool(x),
    "isNo": lambda x, y: not tb.ynToBool(x),
    "getColor": lambda x, y: tb.getColor(x),
    "getYN": lambda x, y: tb.getYN(x),
    "dogOrCat": lambda x, y: "dogs" * ("dog" in x.lower()) + "cats" * ("cat" in x.lower() and "dog" not in x.lower()),
    "null": lambda x, y: True,
}

# fake LLM when --mock
if args.mock:
    import tabulairity.core as core
    orig_ask = core.askChatQuestion
    orig_useful = core.isUseful

    def _fake_ask(prompt, persona, model="dummy", **kw):
        if args.delay:
            time.sleep(args.delay)
        low = prompt.lower()
        if "favorite color" in low:
            return "My favorite color is blue."
        if "dogs or cats" in low:
            return "I love dogs more than cats, dogs are great!"
        if "blue dogs" in low or "purple dogs" in low:
            return "Yes."
        if "would you like to chat" in low:
            return "Yes! I would love to chat."
        return "Yes."

    def _fake_useful(q, r):
        return True

    core.askChatQuestion = _fake_ask
    core.isUseful = _fake_useful
    # also ensure the fx that call getYN/getColor don't hit network
    # override those to trivial
    chatFx["getYN"] = lambda x, y: "yes" if "yes" in x.lower() else "no"
    chatFx["getColor"] = lambda x, y: "blue"
    chatFx["isYes"] = lambda x, y: "yes" in x.lower()
    chatFx["isNo"] = lambda x, y: "no" in x.lower()
    print("[Demo] --mock enabled (fake LLM, isUseful stubbed)")

scriptToRole = lambda role: pd.read_csv("examples/networks/BlueDogs.csv").replace("*role", role)

# --- viz on (default off until this call) ---
url = tb.vizOn(open_browser=True)
print(f"[Demo] Viz URL: {url}")
time.sleep(0.6)  # let browser connect

personas = ["Jim Smith from east Oklahoma"]
if args.two:
    personas = ["Jim Smith from east Oklahoma", "my niece Sally"]

for persona in personas:
    print(f"\n=== Running chatNet for: {persona} ===")
    script = scriptToRole(persona)
    # For viz debugging, disable self_eval unless --keep-eval (prevents Start FAILS on real LLM)
    if not args.keep_eval:
        script["self_eval"] = False
    G = tb.buildChatNet(script)
    print(f"  graph id={G.graph.get('viz_id')} nodes={len(G.nodes)} edges={len(G.edges)}")
    result = tb.walkChatNet(G, chatFx, verbosity=1)
    print(f"  -> success={result.get('success')} errors={result.get('errors')}")
    # also demo a standalone askChatQuestion (appears in side panel)
    try:
        r = tb.core.askChatQuestion("What is 2+2?", "a helpful math tutor", tokens=20)
        print(f"  standalone askChatQuestion: {r[:80]}")
    except Exception as e:
        print(f"  standalone askChatQuestion failed (expected with --mock off and no LLM): {e}")

print("\n[Demo] All chatNets finished. Graph colors will reset after ~4s.")
print("[Demo] Side panel is capped at 100 entries (FIFO).")
print("[Demo] Instant (<2s) chatNets skip animations; add --delay 0.8 to force animations.")
print("[Demo] Press Ctrl+C to exit (vizOff will be called).")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    tb.vizOff()
    print("[Demo] vizOff done.")
