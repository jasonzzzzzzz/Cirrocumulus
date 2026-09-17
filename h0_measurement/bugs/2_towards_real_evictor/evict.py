"""
evict.py -- PRACTICAL (deployable) eviction scores for the H0 eviction corner.

WHY THIS EXISTS. alloc.quant_metrics builds its eviction corner by ranking tokens
on the true sensitivity a_i*||v_i - o||, where a_i is the CURRENT step's attention.
That ranking needs the very attention weights eviction exists to avoid computing,
so it is an upper bound on every real evictor, not a baseline any system can field.
Measuring the interior against it deflates every gain, and it deflates hardest on
sharp heads -- exactly the heads the verdict table was calling STOP.

An Evictor here maintains a selection score from PAST attention only. Nothing in
this module ever sees the current step's logits. The oracle corner is NOT an
Evictor: it stays hard-wired in alloc.py (err_evict / gain_e) so it can never be
configured away, and `oracle_evict_advantage_*` reports what each practical
evictor loses against it.

ALIGNMENT (this is the bug the module was written to fix). The score is indexed by
KV-cache position, but the cache changes shape under us between decode steps. The
old inline code required len(prev_a) >= len(current), which is false on essentially
every step, so the practical score silently degraded to None and never populated a
single column. The base class handles alignment once, for all evictors:

  * cache grew by one (full attention): append one slot for the new token.
  * cache length unchanged (sliding window, DynamicCache trims the front): roll
    left by one -- oldest position drops, new slot at the end.
  * anything else (static/wrapping cache, a batched or skipped step): position
    identity cannot be trusted, so the evictor RESETS and reports no history
    rather than scoring a mis-aligned vector. Loud, not silent.

A position that has never been observed (the token generated this step) has no
history at all. Every real evictor keeps the newest token by recency, so score()
ranks such positions strictly first rather than letting them be evicted at birth.
This is applied at scoring time only -- it never contaminates the accumulator.

ADDING AN EVICTOR: subclass Evictor, set n_bufs, implement _accum/_raw, decorate
with @register("name"). Nothing else in the pipeline changes -- run_h0 discovers
it through the registry and alloc.py emits its columns automatically.

KNOWN CONSERVATISM: accumulation starts at the first decode step, because the
probe only captures decode queries. Deployed H2O/SnapKV also accumulate over the
prefill, so our H2O is WEAKER than the real thing at small n_decode. That biases
gains UP, so treat H2O numbers at n_decode<=8 as a floor on the practical corner's
strength, and raise n_decode when the H2O column is the headline.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

_REG: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REG[name] = cls
        return cls
    return deco


def available() -> list[str]:
    return sorted(_REG)


class Evictor:
    """Selection score over KV positions, built from PAST attention only.

    Call contract, once per decode step per (layer, head):

        s = ev.score(fin)     # None until enough history; aligns the state
        ...                   # use s, which is ordered like logits[fin]
        ev.observe(a, fin)    # a = THIS step's attention over the fin positions

    score() must precede observe() in a step -- that ordering is what makes the
    score lagged, and it is what a deployed evictor is restricted to.
    """
    name = "base"
    n_bufs = 1

    def __init__(self, **kw):
        if kw:
            raise TypeError(f"{type(self).__name__} got unknown options "
                            f"{sorted(kw)}; valid: none")
        self.reset()

    # ------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Forget everything. Selection history does not cross prompts."""
        self._bufs: list[torch.Tensor] = []
        self._fresh = torch.zeros(0, dtype=torch.bool)
        self._Lc = 0
        self._steps = 0

    def ready(self) -> bool:
        return self._steps >= 1

    # -------------------------------------------------------- subclass hooks
    def _accum(self, a: torch.Tensor, fin: torch.Tensor) -> None:
        """Fold this step's attention into the buffers. Position space."""
        raise NotImplementedError

    def _raw(self) -> torch.Tensor:
        """Current score over all cache positions, float32, position space."""
        raise NotImplementedError

    # ------------------------------------------------------------ alignment
    @staticmethod
    def _prep(fin: torch.Tensor) -> torch.Tensor:
        if fin.dtype != torch.bool:
            raise TypeError(f"fin must be a bool mask, got {fin.dtype}")
        return fin.detach().to("cpu")

    def _alloc(self, Lc: int) -> None:
        self._bufs = [torch.zeros(Lc, dtype=torch.float32)
                      for _ in range(self.n_bufs)]
        self._fresh = torch.ones(Lc, dtype=torch.bool)
        self._Lc = Lc

    def _align(self, fin: torch.Tensor) -> None:
        Lc = fin.numel()
        if self._Lc == 0:
            self._alloc(Lc)
            return
        if Lc == self._Lc + 1:
            # full attention: one new token at the end.
            z = torch.zeros(1, dtype=torch.float32)
            self._bufs = [torch.cat([b, z]) for b in self._bufs]
            self._fresh = torch.cat([self._fresh,
                                     torch.ones(1, dtype=torch.bool)])
        elif Lc == self._Lc:
            # sliding window: front dropped, new token at the end.
            for b in self._bufs:
                b.copy_(torch.roll(b, -1))
                b[-1] = 0.0
            self._fresh = torch.roll(self._fresh, -1)
            self._fresh[-1] = True
        else:
            # Position identity is not recoverable from a length change we do
            # not model (static/wrapping cache, >1 token per step, a new prompt
            # without reset). Scoring on it would silently mis-attribute history
            # to the wrong tokens, which is the exact class of bug this module
            # replaces. Drop the history instead.
            self.reset()
            self._alloc(Lc)
            return
        self._Lc = Lc

    # ---------------------------------------------------------------- public
    def score(self, fin: torch.Tensor) -> torch.Tensor | None:
        """Score over the fin positions, ordered like logits[fin]. None = no
        usable history yet (first decode step of a prompt)."""
        fin = self._prep(fin)
        self._align(fin)
        if not self.ready():
            return None
        s = self._raw()[fin].double().clone()
        fr = self._fresh[fin]
        if bool(fr.any()):
            seen = s[~fr]
            mx = float(seen.max().item()) if seen.numel() else 0.0
            s[fr] = mx + 1.0        # never evict a token at birth
        return s

    def observe(self, a: torch.Tensor, fin: torch.Tensor) -> None:
        """Record this step's true attention over the fin positions."""
        fin = self._prep(fin)
        if self._Lc == 0:
            self._alloc(fin.numel())     # first use: nothing to mis-align yet
        if fin.numel() != self._Lc:
            raise RuntimeError(
                f"{self.name}.observe got cache length {fin.numel()} but the "
                f"state is aligned to {self._Lc}; call score(fin) first, once "
                f"per step, before observe()")
        n = int(fin.sum().item())
        if a.numel() != n:
            raise ValueError(f"{self.name}.observe: attention has {a.numel()} "
                             f"entries but fin selects {n} positions")
        self._accum(a.detach().to("cpu", torch.float32).reshape(-1), fin)
        self._fresh[fin] = False
        self._steps += 1


@register("tova")
class Tova(Evictor):
    """Last step's attention (TOVA). The cheapest deployable score, and the one
    the inline code was already trying to compute."""
    n_bufs = 1

    def _accum(self, a, fin):
        self._bufs[0].zero_()
        self._bufs[0][fin] = a

    def _raw(self):
        return self._bufs[0]


@register("h2o")
class H2O(Evictor):
    """Accumulated attention over every step seen so far (H2O heavy-hitters).
    See the module docstring on prefill: our accumulator starts at decode."""
    n_bufs = 1

    def _accum(self, a, fin):
        self._bufs[0][fin] += a

    def _raw(self):
        return self._bufs[0]


@register("snapkv")
class SnapKV(Evictor):
    """Attention pooled over the last `window` steps, then max-pooled over
    neighbouring positions (SnapKV). The neighbour pooling is the part that
    actually distinguishes it from H2O: it keeps contiguous spans rather than
    isolated spikes, which is what makes it robust when the needle moves.

    Options:  window (steps kept, default 4)   pool (odd kernel, default 7; 1 off)

    Memory is `window` position-space vectors per (layer, head) -- window times
    what tova/h2o cost. Drop `window` first if a large model runs the host out
    of RAM.
    """

    def __init__(self, window: int = 4, pool: int = 7):
        window, pool = int(window), int(pool)
        if window < 1:
            raise ValueError(f"snapkv window must be >= 1, got {window}")
        if pool < 1 or pool % 2 == 0:
            raise ValueError(f"snapkv pool must be odd and >= 1, got {pool}")
        self.window, self.pool = window, pool
        self.n_bufs = window
        super().__init__()

    def _accum(self, a, fin):
        b = self._bufs[self._steps % self.window]   # ring; _steps not yet bumped
        b.zero_()
        b[fin] = a

    def _raw(self):
        tot = self._bufs[0].clone()
        for b in self._bufs[1:]:
            tot += b
        if self.pool > 1:
            tot = F.max_pool1d(tot[None, None], self.pool, stride=1,
                               padding=self.pool // 2)[0, 0]
        return tot


@register("slm")
class StreamingLLM(Evictor):
    """Attention sinks plus recency (StreamingLLM). Uses no attention statistics
    at all, so it scores from the very first step and forms the floor of the
    practical family: any evictor that cannot beat this is not earning its
    bookkeeping.

    Options:  sinks (leading positions always kept, default 4)
    """
    n_bufs = 0

    def __init__(self, sinks: int = 4):
        self.sinks = int(sinks)
        if self.sinks < 0:
            raise ValueError(f"slm sinks must be >= 0, got {self.sinks}")
        super().__init__()

    def ready(self) -> bool:
        return True

    def _accum(self, a, fin):
        pass

    def score(self, fin):
        fin = self._prep(fin)
        self._align(fin)
        L = int(fin.sum().item())
        s = torch.arange(L, dtype=torch.float64)          # recency
        n = min(self.sinks, L)
        if n:  # above every recency score, and the earliest sink ranks highest
            s[:n] = float(L) + torch.arange(n, 0, -1, dtype=torch.float64)
        return s


# --------------------------------------------------------------- construction
def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    return v


def make(spec: str) -> tuple[str, Evictor]:
    """Build one evictor from a spec string. Returns (label, evictor).

        "h2o"                        -> h2o with defaults
        "snapkv:window=8,pool=13"    -> snapkv, options overridden
        "snapkv:window=8@snap8"      -> same, labelled snap8 (needed only when
                                        the same evictor appears twice)

    The label becomes a column-name fragment (`gain_practical_<label>_b<budget>`),
    so it must be alphanumeric with underscores.
    """
    body, _, alias = str(spec).strip().partition("@")
    name, _, opts = body.strip().partition(":")
    name = name.strip().lower()
    if name not in _REG:
        raise KeyError(f"unknown evictor {name!r}; available: {available()}")
    kw = {}
    for part in (o for o in opts.split(",") if o.strip()):
        k, sep, v = part.partition("=")
        if not sep:
            raise ValueError(f"evictor option {part!r} in {spec!r} is not key=value")
        kw[k.strip()] = _coerce(v)
    label = (alias.strip() or name).lower()
    if not label or not label.replace("_", "").isalnum():
        raise ValueError(f"evictor label {label!r} must be alphanumeric "
                         f"(underscores allowed) -- it becomes a column name")
    return label, _REG[name](**kw)


def parse_specs(cfg) -> list[str]:
    """Normalise the `evictors` config value to a list of spec strings.

    Accepts a YAML list, or a string from --override. A string is split on ';'
    when it contains one, else on ',' when it carries no options -- so both
    `evictors=tova,h2o` and `evictors=snapkv:window=8,pool=7` mean what they look
    like. 'none' / '' / [] disables the practical corner (oracle only).
    """
    if cfg is None:
        return []
    if isinstance(cfg, (list, tuple)):
        specs = [str(x).strip() for x in cfg]
    else:
        s = str(cfg).strip()
        if s.lower() in ("", "none", "off", "false", "[]"):
            return []
        if ";" in s:
            specs = s.split(";")
        elif "=" in s:
            specs = [s]
        else:
            specs = s.split(",")
    return [x.strip() for x in specs if x.strip()]


def make_many(specs) -> dict[str, Evictor]:
    """Build the configured evictors, keyed by label. Fails on a duplicate
    label rather than silently dropping a column."""
    out: dict[str, Evictor] = {}
    for spec in parse_specs(specs):
        label, ev = make(spec)
        if label in out:
            raise ValueError(
                f"duplicate evictor label {label!r}; give one an alias, "
                f"e.g. '{spec}@{label}_alt'")
        out[label] = ev
    return out


def labels(specs) -> list[str]:
    """Labels the given config will produce, without building the evictors."""
    return list(make_many(specs))
