"""What a run cost and where the time went: Claude tokens -> dollars, plus a timer per stage.

One RunStats per process (`STATS`), since each clipper run is its own process. The
summary is printed at the end and appended to clips.json under "runs", so the web UI
and you can see cost per run and per clip.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

# $ per million tokens: (input, output, cache read). Cache writes (5-minute TTL) cost
# 1.25x input; the Batch API halves everything. Update when prices change.
PRICES = {
    "claude-fable-5-1": (10.00, 50.00, 0.25),
    "claude-fable-5": (10.00, 50.00, 1.00),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-sonnet-4-6": (3.00, 15.00, 0.30),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}
CACHE_WRITE = 1.25
BATCH_DISCOUNT = 0.5


def price_for(model: str) -> tuple[float, float, float] | None:
    # Exact id first, then the longest known prefix (covers dated/suffixed ids).
    if model in PRICES:
        return PRICES[model]
    matches = [k for k in PRICES if model.startswith(k)]
    return PRICES[max(matches, key=len)] if matches else None


def _fmt_secs(s: float) -> str:
    s = round(s)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    return f"{m}m {s:02d}s" if m < 60 else f"{m // 60}h {m % 60:02d}m"


class RunStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.usage: dict[tuple[str, bool], dict[str, int]] = {}
        self.stages: dict[str, float] = {}
        self.requests = 0

    def add(self, msg, batch: bool = False) -> None:
        """Record one API response's token usage (safe to call from worker threads)."""
        u = getattr(msg, "usage", None)
        if u is None:
            return
        key = (getattr(msg, "model", "") or "unknown", batch)
        with self._lock:
            self.requests += 1
            row = self.usage.setdefault(key, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
            row["input"] += getattr(u, "input_tokens", 0) or 0
            row["output"] += getattr(u, "output_tokens", 0) or 0
            row["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            row["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0

    @contextmanager
    def stage(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.monotonic() - t0

    def cost(self) -> float | None:
        """Estimated dollars, or None if a model's price is unknown."""
        total = 0.0
        for (model, batch), r in self.usage.items():
            p = price_for(model)
            if p is None:
                return None
            inp, out, read = p
            dollars = (r["input"] * inp + r["output"] * out + r["cache_read"] * read
                       + r["cache_write"] * inp * CACHE_WRITE) / 1e6
            total += dollars * (BATCH_DISCOUNT if batch else 1.0)
        return total

    def tokens(self) -> dict[str, int]:
        out = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        for r in self.usage.values():
            for k in out:
                out[k] += r[k]
        return out

    def summary(self, n_clips: int | None = None) -> str:
        lines = []
        if self.stages:
            total = sum(self.stages.values())
            parts = ", ".join(f"{k} {_fmt_secs(v)}" for k, v in self.stages.items())
            lines.append(f"Time:  {_fmt_secs(total)} ({parts})")
        if self.requests:
            t = self.tokens()
            cost = self.cost()
            line = (f"Claude: {self.requests} request(s), {t['input'] + t['cache_read'] + t['cache_write']:,} "
                    f"input + {t['output']:,} output tokens")
            if cost is not None:
                line += f" = ${cost:.2f}"
                if n_clips:
                    line += f" (${cost / n_clips:.3f} per clip)"
            else:
                line += " (unknown model price; add it to stats.PRICES)"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self, kind: str, n_clips: int) -> dict:
        cost = self.cost()
        return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind,
                "clips": n_clips, "seconds": {k: round(v, 1) for k, v in self.stages.items()},
                "requests": self.requests, "tokens": self.tokens(),
                "cost_usd": round(cost, 4) if cost is not None else None,
                "models": sorted({m for m, _ in self.usage})}


STATS = RunStats()
