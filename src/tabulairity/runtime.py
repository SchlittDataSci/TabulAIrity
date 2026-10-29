"""Small, shared execution context for walkers, nested walks, and the workbench."""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, replace
from threading import Event
from contextlib import contextmanager
from time import sleep


class WalkControl:
    """Cooperative pause/stop. In-flight blocking requests cannot be interrupted."""

    def __init__(self, paused=None, stopped=None):
        self.paused = paused if paused is not None else Event()
        self.stopped = stopped if stopped is not None else Event()

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()

    def stop(self):
        self.stopped.set()
        self.paused.clear()

    async def checkpoint(self):
        while self.paused.is_set() and not self.stopped.is_set():
            await asyncio.sleep(0.05)
        if self.stopped.is_set():
            raise asyncio.CancelledError("Execution stopped")


@dataclass
class WalkContext:
    slots: object
    nets: dict
    fx: dict
    workers: int = 1
    observer: object = None
    control: object = None
    path: tuple = ()
    node: object = None
    model: object = None
    graph: object = None
    loop: object = None
    in_slot: bool = False
    executor: object = None
    # Product of ancestor iterable/structural fan-outs. Automatic expansions
    # partition the root worker budget instead of multiplying it at each depth.
    expansion_fanout: int = 1
    execution_id: str = ""

    def child(self, source, index):
        return replace(self, path=self.path + (f"{source}[{index}]",))

    def emit(self, event, **data):
        if self.observer is not None:
            try:
                self.observer(
                    {
                        "event": event,
                        "path": list(self.path),
                        "node": self.node,
                        "network": self.graph.graph.get("name", "")
                        if self.graph is not None
                        else "",
                        **data,
                    }
                )
            except Exception:
                # Observability must never determine an extraction's result.
                pass

    @contextmanager
    def inference_slot(self):
        """Share slots with synchronous model calls inside a custom chatFX."""
        if self.in_slot:
            yield
            return
        if self.control is not None:
            while self.control.paused.is_set() and not self.control.stopped.is_set():
                sleep(0.05)
            if self.control.stopped.is_set():
                raise asyncio.CancelledError("Execution stopped")
        gate = getattr(self.slots, "gate", None)
        if gate is not None:
            while not gate.acquire(False):
                if self.control is not None and self.control.stopped.is_set():
                    raise asyncio.CancelledError("Execution stopped")
                sleep(0.025)
            try:
                yield
            finally:
                gate.release()
        elif self.loop is not None and self.loop.is_running():
            try:
                current = asyncio.get_running_loop()
            except RuntimeError:
                current = None
            if current is self.loop:
                raise RuntimeError(
                    "An async chatFX should call inference with await asyncio.to_thread(tb.askChatQuestion, ...)"
                )
            future = asyncio.run_coroutine_threadsafe(self.slots.acquire(), self.loop)
            future.result()
            try:
                yield
            finally:
                self.loop.call_soon_threadsafe(self.slots.release)
        else:
            yield

    async def checkpoint(self):
        if self.control is not None:
            await self.control.checkpoint()


walk_context = ContextVar("tabulairity_walk_context", default=None)
