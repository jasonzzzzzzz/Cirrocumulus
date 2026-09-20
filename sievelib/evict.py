"""
evict.py -- WHO the eviction corner is, and HOW MUCH it is allowed to keep.

Two biases in the H0 eviction corner, both flattering it, both fixed here.

E2 -- WHO. `quant_metrics` ranked tokens by the true sensitivity a_i*||v_i - o||,
where a_i is the CURRENT step's attention. That ranking needs the very weights
eviction exists to avoid computing: it is an upper bound on every real evictor,
not a baseline any system can field. The oracle STAYS -- gain-vs-oracle is a valid
lower bound and `oracle_evict_advantage` (what H2O/SnapKV-style scoring loses to
it, per head, at scale) is publishable on its own. What changes is which corner
the VERDICT keys off: the best corner a DEPLOYABLE system could field, with the
oracle reported alongside as the bound.

E1 -- HOW MUCH. The corner keeps B*L/maxb tokens, linear in L, while head support
grows as L^0.63-0.92 (bugs/3_context_sweep_and_reports/findings.md). At B=3 the
corner holds 15.8x a head's support at 4k and 70.8x at 64k, so it may be winning
at 128k on slack rather than on merit. `corner_tokens` makes the budget a policy
axis: `frac` is the literature's definition, `abs` caps the corner at a multiple
of the head's own measured support and records what it did not spend.

P0 -- ALIGNMENT, the bug that made all of this moot. The old inline score required
len(prev_a) >= len(current logits); prev_a is always exactly one SHORTER, so the
guard was false on essentially every step and the practical corner has NEVER run,
in any campaign. Alignment is handled once here, for every evictor:

  * cache grew by one (full attention): append a slot for the new token.
  * cache length unchanged (sliding window; DynamicCache trims the front): roll
    left by one -- oldest position drops, new slot at the end.
  * anything else (static/wrapping cache, >1 token per step, a missed reset):
    position identity is not recoverable, so the evictor RESETS and reports no
    history rather than scoring a mis-aligned vector. Loud, not silent.

A position never observed (the token generated this step) has no history at all.
Every real evictor keeps the newest token by recency, so score() ranks such
positions strictly first rather than letting them be evicted at birth. Applied at
scoring time only -- it never contaminates the accumulator.

ADDING AN EVICTOR: subclass Evictor, set n_bufs, implement _accum/_raw, decorate
with @register("name"). run_h0 discovers it through the registry and alloc.py
emits its columns automatically; nothing else changes.

KNOWN CONSERVATISM: accumulation starts at the first decode step, because the
probe only captures decode queries. Deployed H2O/SnapKV also accumulate over the
prefill, so `accum` here is WEAKER than the real thing at small n_decode. That
biases gains UP, so treat `accum` at n_decode<=8 as a floor on the practical
corner's strength, and raise n_decode when it is the headline.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# The oracle is a corner, but not a stateful evictor: its score is the current
# step's a_i*||v_i-o||, which alloc.quant_metrics already has and run_h0 does
# not. It is therefore named here but implemented in alloc.py, and it is ALWAYS
# measured -- listing it in `evictors` only affects nothing, and omitting it
# cannot switch off the bound.
ORACLE = "oracle"

_REG: dict[str, type] = {}
_ALIAS = {"tova": "last_step", "h2o": "accum", "snapkv": "window",
          "streamingllm": "recency", "slm": "recency", "sink": "recency"}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REG[name] = cls
        return cls
    return deco


def available() -> list[str]:
    """Every corner name accepted in `evictors`, oracle included."""
    return [ORACLE] + sorted(_REG)


class Evictor:
    """Selection score over KV positions, built from PAST attention only.

    Call contract, once per decode step per (layer, head):

        s = ev.score(fin)     # None until enough history; aligns the state
        ...                   # use s, ordered like logits[fin]
        ev.observe(a, fin)    # a = THIS step's attention over the fin positions

    score() must precede observe() in a step -- that ordering is what makes the
    score lagged, and it is the only information a deployed evictor has.

    score() may be called MORE THAN ONCE in a step (run_h0 asks for the ranked
    tensor for the corner and the rank_bump=False tensor for the interior). Only
    the first call aligns; later calls before observe() re-read the same state.
    """
    name = "base"
    n_bufs = 1
    # Consecutive decode steps of history the score needs to be the evictor it
    # claims to be. None = unbounded (a running sum over EVERY step), which a
    # sparse measurement schedule cannot feed -- see decode_plan.
    history_steps: int | None = 1
    # Survives the block resets a sparse schedule does between measured steps.
    # False for every evictor whose score is a window over RECENT steps: a
    # window that spans a gap is not the evictor it claims to be. True only for
    # `first`, whose score is one old snapshot by construction (R5).
    persistent: bool = False

    def __init__(self, **kw):
        if kw:
            raise TypeError(f"{type(self).__name__} got unknown options "
                            f"{sorted(kw)}")
        self.reset()

    # ------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Forget everything. Selection history does not cross prompts."""
        self._bufs: list[torch.Tensor] = []
        self._fresh = torch.zeros(0, dtype=torch.bool)
        self._Lc = 0
        self._steps = 0
        self._pending = False       # aligned for a step that is not yet observed

    def ready(self) -> bool:
        return self._steps >= 1

    def bytes_per_slot(self) -> int:
        """Host RAM per cache position: float32 buffers + the fresh mask."""
        return 4 * self.n_bufs + 1

    def _unseen_pos(self) -> torch.Tensor:
        """Position space: True where the score carries NO information -- the
        history never observed the position. Subclasses whose score is a
        snapshot (Lag) widen this to every position the snapshot missed."""
        return self._fresh

    def unseen(self, fin: torch.Tensor) -> torch.Tensor:
        """Positions (ordered like logits[fin]) the current score knows nothing
        about. Call AFTER score() in the same step, so the state is aligned.

        A score of 0 there is ABSENCE of evidence, not evidence of zero
        attention. Treated as a magnitude it gets 0 bits from the allocator --
        it EVICTS the token appended this very step, which the current query
        attends to: the job214* R3 defect (R3-report.md section 2). The corner
        bumps these positions (score) and the interior floors them at the top
        tier (alloc.quant_metrics, interior_unseen)."""
        return self._unseen_pos()[self._prep(fin)].clone()

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

    @staticmethod
    def _grow(b: torch.Tensor, n: int = 1) -> torch.Tensor:
        """Cache grew by n: append zero slots for the new positions."""
        return torch.cat([b, torch.zeros(n, dtype=b.dtype)])

    @staticmethod
    def _shift(b: torch.Tensor) -> None:
        """Sliding window: oldest position drops, new zero slot at the end."""
        b.copy_(torch.roll(b, -1))
        b[-1] = 0.0

    def _align(self, fin: torch.Tensor) -> None:
        Lc = fin.numel()
        # [REGRESSION] A second score() in the same step used to land in the
        # "length unchanged" branch below and ROLL the state left by one -- the
        # sliding-window rule applied to a cache that had not moved. run_h0 calls
        # score() twice per step for every interior evictor (ranked for the
        # corner, rank_bump=False for the allocator), so the default `accum` lost
        # its oldest position (the sink) and shifted all history by one slot per
        # step: after 4 steps a 2.8-mass sink read 0.7 and a heavy hitter was
        # smeared over 4 positions. Same step, same length -> nothing to do.
        if self._pending:
            if Lc == self._Lc:
                return
            # length changed without an observe(): position identity is gone,
            # so start over exactly as the "anything else" branch does
            self.reset()
            self._alloc(Lc)
            self._pending = True
            return
        self._pending = True
        if self._Lc == 0:
            self._alloc(Lc)
            return
        if Lc == self._Lc + 1 or (Lc > self._Lc and self.persistent):
            # [REGRESSION] A sparse schedule leaves unprobed gaps, so the cache
            # can grow by MORE than one between two score() calls. The generic
            # branch below treats that as "position identity is gone" and
            # resets -- which silently re-calibrated `first` at every block and
            # made it identical to last_step (caught by the R5 smoke run: its
            # unseen fraction read 1/L at step 4 instead of 4/L). Growth is the
            # one shape where identity SURVIVES: the old positions are
            # untouched and the new ones are simply unseen. Only a persistent
            # evictor may cross a gap; a window over recent steps must not.
            n = Lc - self._Lc
            self._bufs = [self._grow(b, n) for b in self._bufs]
            self._fresh = torch.cat([self._fresh,
                                     torch.ones(n, dtype=torch.bool)])
        elif Lc == self._Lc:
            for b in self._bufs:
                self._shift(b)
            self._fresh = torch.roll(self._fresh, -1)
            self._fresh[-1] = True
        else:
            self.reset()
            self._alloc(Lc)
            self._pending = True
            return
        self._Lc = Lc

    # ---------------------------------------------------------------- public
    def score(self, fin: torch.Tensor, rank_bump: bool = True) -> torch.Tensor | None:
        """Score over the fin positions, ordered like logits[fin]. None = no
        usable history yet (first decode step of a prompt).

        `rank_bump` controls the "never evict a token at birth" rule, which
        rewrites freshly-appended positions to max+1 so they sort first.

        THAT RULE IS ORDINAL ONLY, AND IT MUST NOT REACH THE ALLOCATOR.
        A corner consumes this tensor as a RANKING, where max+1 just means
        "first". The R3 interior consumes it as a MAGNITUDE: it normalises the
        tensor into a distribution and forms w2p = (ap*||v-op||)^2. There the
        "+1" is in units of accumulated attention (the buffer sums to ~1 per
        observed step), so a handful of fresh tokens absorb 20-33% of the total
        mass -- measured, and WORSE on concentrated heads (33% sharp vs 20%
        diffuse) because `mx` is larger there. That is an ordinal convention
        masquerading as a sensitivity, and it biases hardest on exactly the
        high-gain heads the interior exists to serve.

        So: rank_bump=True for corners (the deployed policy -- do not evict a
        token you have never observed), rank_bump=False for the interior, which
        needs the honest accumulated magnitude. A deployed allocator would still
        floor new tokens at high precision, but that is a separate, explicit
        policy rather than an artifact of a tie-break.
        """
        fin = self._prep(fin)
        self._align(fin)
        if not self.ready():
            return None
        s = self._raw()[fin].double().clone()
        if not rank_bump:
            return s
        fr = self._unseen_pos()[fin]
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
        self._pending = False


@register("last_step")
class LastStep(Evictor):
    """Previous step's attention (TOVA). The cheapest deployable score, and the
    one the broken inline code was already trying to compute."""
    n_bufs = 1

    def _accum(self, a, fin):
        self._bufs[0].zero_()
        self._bufs[0][fin] = a

    def _raw(self):
        return self._bufs[0]


@register("accum")
class Accum(Evictor):
    """Running sum of attention received (H2O heavy-hitters). See the module
    docstring on prefill: our accumulator starts at decode."""
    n_bufs = 1
    history_steps = None            # every step since the prompt began

    def _accum(self, a, fin):
        self._bufs[0][fin] += a

    def _raw(self):
        return self._bufs[0]


@register("lag")
class Lag(Evictor):
    """Attention from EXACTLY `k` decode steps ago -- a pure staleness probe.

    Not a published evictor and not meant as one. It exists for R3's staleness
    sweep: the design question SIEVE actually faces is not "lagged vs
    clairvoyant" but "how fast does an allocation decay after the step it was
    computed at". `last_step` answers k=1 only; a real system that allocates
    once at prefill and decodes 2,000 tokens is operating at k=2000.

    Sweeping k therefore prices the two candidate architectures directly:
      k=1 flat vs k=large   ->  allocate once, never re-budget (cheapest)
      error rising in k     ->  re-budgeting during decode is mandatory, and
                                the slope says how often
    Note `ready()` needs k steps of history, so a run must decode far enough
    past the first quant step for the deepest k to score -- otherwise the
    completeness guard in alloc.py withholds the row, by design.

    Options:  k (steps of lag, default 1)
    Memory:   k float32 vectors + k bool COVERAGE vectors per (layer, head).

    [DEFECT, job214*] The k-1 tokens generated after the snapshot HAVE been
    observed since (so `_fresh` is False for them) but are ABSENT from the
    snapshot, so they scored 0: the corner evicted the newest tokens and the
    interior gave them 0 bits. Measured on Llama-3.2-1B: lag-2 corner 2.05-3.65x
    the oracle as implemented vs 1.37-1.42x with them protected. Each snapshot
    now carries its coverage; positions it did not cover are `unseen`.

    [REGRESSION] The ring used to be a private list of vectors that _align never
    saw, so on a growing cache the k-steps-ago vector was SHORTER than the state
    and copying it back raised a size mismatch at step k -- every lag >= 2
    crashed on the first full-attention step that needed it (the unit test used
    a fixed-length cache, which never grows). The ring now lives in `_bufs`, the
    one place alignment reaches, and the host-RAM estimate finally counts it.
    """
    n_bufs = 1

    def __init__(self, k: int = 1, **kw):
        k = int(k)
        if k < 1:
            raise ValueError(f"lag k must be >= 1, got {k}")
        self.k = k
        self.n_bufs = 2 * k            # k attention snapshots + k coverage masks
        self.history_steps = k
        super().__init__(**kw)

    def ready(self) -> bool:
        return self._steps >= self.k

    def bytes_per_slot(self) -> int:
        return 4 * self.k + self.k + 1    # float32 snapshots, bool coverage, fresh

    def _alloc(self, Lc: int) -> None:
        super()._alloc(Lc)
        # coverage lives in _bufs so _grow/_shift keep it position-aligned
        self._bufs[self.k:] = [torch.zeros(Lc, dtype=torch.bool)
                               for _ in range(self.k)]

    def _accum(self, a, fin):
        slot = self._steps % self.k               # ring; _steps not yet bumped
        b, cov = self._bufs[slot], self._bufs[self.k + slot]
        b.zero_()
        b[fin] = a
        cov.zero_()
        cov[fin] = True

    def _raw(self):
        # The next write goes to slot _steps % k, which holds the observation
        # from exactly k steps before it.
        return self._bufs[self._steps % self.k]

    def _unseen_pos(self):
        return ~self._bufs[self.k + self._steps % self.k]


@register("first")
class First(Evictor):
    """The FIRST probed step's attention, frozen -- never updated again.

    R5 / C4. The router is claimed to read a head's phase from ONE offline
    calibration pass, and `lag:k` cannot price that claim: reaching k = 4096
    would need 4096 buffers. `first` is the k -> infinity end of the same axis at
    the cost of one buffer, so `interior_lag_cost<B>_first` at step t IS the
    price of keeping the prefill-time allocation t steps later, and the gap to
    `last_step` (k = 1) is what re-budgeting every step buys.

    Positions appended after the snapshot are `unseen`: the snapshot says
    nothing about them (it is not evidence of zero attention), so the corner
    bumps them and the interior floors them at the top tier, exactly as for
    `lag:k` -- see Evictor.unseen and alloc.waterfill_floor.

    Options:  none
    Memory:   one float32 snapshot + one bool coverage mask per (layer, head).
    """
    n_bufs = 2                     # [0] the snapshot, [1] its coverage
    history_steps = 1              # one probed step, at the very beginning
    persistent = True              # a sparse schedule must NOT reset it

    def bytes_per_slot(self) -> int:
        return 4 + 1 + 1           # float32 snapshot, bool coverage, fresh mask

    def _alloc(self, Lc: int) -> None:
        super()._alloc(Lc)
        self._bufs[1] = torch.zeros(Lc, dtype=torch.bool)

    def _accum(self, a, fin):
        if self._steps:            # already frozen; later steps change nothing
            return
        self._bufs[0].zero_()
        self._bufs[0][fin] = a
        self._bufs[1].zero_()
        self._bufs[1][fin] = True

    def _raw(self):
        return self._bufs[0]

    def _unseen_pos(self):
        return ~self._bufs[1]


@register("window")
class Window(Evictor):
    """Attention over the last `window` steps, max-pooled over neighbouring
    positions (SnapKV). The neighbour pooling is what actually distinguishes it
    from `accum`: it keeps contiguous spans rather than isolated spikes, which
    is what makes it robust when the needle moves.

    Options:  window (steps kept, default 4)   pool (odd kernel, default 7; 1 off)

    Memory is `window` position-space vectors per (layer, head) -- `window` times
    what last_step/accum cost. Drop it first if a large model runs the host out
    of RAM.
    """

    def __init__(self, window: int = 4, pool: int = 7):
        window, pool = int(window), int(pool)
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if pool < 1 or pool % 2 == 0:
            raise ValueError(f"pool must be odd and >= 1, got {pool}")
        self.window, self.pool = window, pool
        self.n_bufs = window
        self.history_steps = window
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


@register("recency")
class Recency(Evictor):
    """Attention sinks plus recency (StreamingLLM). Uses no attention statistics
    at all, so it scores from the very first step and forms the FLOOR of the
    practical family: an evictor that cannot beat this is not earning its
    bookkeeping.

    Options:  sinks (leading positions always kept, default 4)
    """
    n_bufs = 0
    history_steps = 0

    def __init__(self, sinks: int = 4):
        self.sinks = int(sinks)
        if self.sinks < 0:
            raise ValueError(f"sinks must be >= 0, got {self.sinks}")
        super().__init__()

    def ready(self) -> bool:
        return True

    def _accum(self, a, fin):
        pass

    def score(self, fin, rank_bump: bool = True):
        # Positional, so there is no magnitude to be honest about: `recency` is
        # ordinal by construction and must never be used as an interior score.
        # The signature matches the base class so callers need no special case.
        fin = self._prep(fin)
        self._align(fin)
        L = int(fin.sum().item())
        s = torch.arange(L, dtype=torch.float64)          # recency
        n = min(self.sinks, L)
        if n:  # above every recency score, and the earliest sink ranks highest
            s[:n] = float(L) + torch.arange(n, 0, -1, dtype=torch.float64)
        return s


# ------------------------------------------------------------- budget policy
def corner_tokens(policy: str, B: float, L: int, maxb: int,
                  n95: float | None, kappa: float, floor: int) -> int:
    """How many tokens the eviction corner may keep at `maxb` bits.

    frac : B*L/maxb -- the literature's definition. Linear in L, so it grows
           faster than head support does and the corner gets structurally
           cheaper as context grows.
    abs  : min(frac, max(kappa*n95, floor)) -- capped at a multiple of the
           head's OWN measured support. Never keeps MORE TOKENS than `frac`;
           `corner_bits_used` records the bits it declined to spend.

    Fewer tokens does NOT mean a weaker corner. Eviction-corner error is not
    monotone in K: every kept token is quantized at maxb, so extending the
    keep-set down the tail adds low-weight tokens carrying quantization noise,
    and the renormalised softmax can come out WORSE. On sharp heads the `abs`
    corner is routinely both cheaper and more accurate than `frac` -- which is a
    stronger statement of E1 than "the corner has slack", and the reason the
    comparison is reported per cell rather than assumed to have a sign.
    """
    m_frac = max(1, int(round(B * L / maxb)))
    if policy == "frac":
        return m_frac
    if policy == "abs":
        if n95 is None or not math.isfinite(float(n95)):
            return m_frac            # no support estimate -> nothing to cap to
        return max(1, min(m_frac, int(max(kappa * float(n95), floor))))
    raise ValueError(f"unknown corner policy {policy!r}; use 'frac' or 'abs'")


def corner_tag(corner: "CornerSpec") -> str:
    """Short filesystem-safe fingerprint of a corner configuration.

    For putting the corner INTO the parquet filename, so a results directory
    holding several corner configurations is readable at `ls` rather than only
    after opening each file. Two-letter evictor prefixes, one-letter policies,
    and only the parameters that actually applied:

        oracle,last_step,accum,window,recency + frac,abs   -> or-la-ac-wi-re_fa
        oracle,accum + frac                                -> or-ac_f
        ... with corner_kappa=8                            -> or-ac_fa_k8

    Deliberately lossy -- it disambiguates, it does not reconstruct. The full
    effective config is written to the sidecar JSON beside the parquet.
    """
    ev = "-".join(lab[:2] for lab in labels(corner.evictors)) or "none"
    pol = "".join(p[0] for p in corner.policies)
    tag = f"{ev}_{pol}"
    if "abs" in corner.policies:
        d = CornerSpec()
        if float(corner.kappa) != float(d.kappa):
            tag += f"_k{corner.kappa:g}"
        if int(corner.floor) != int(d.floor):
            tag += f"_f{int(corner.floor)}"
    if not corner.kstar:
        tag += "_noks"
    return tag


def config_record(corner: "CornerSpec") -> dict:
    """The FULL effective corner configuration, for the sidecar JSON.

    Unlike `corner_tag` this is lossless: everything needed to reproduce the
    corner columns, including the parameters that only matter for one policy.
    """
    return {"evictors": list(corner.evictors),
            "evictor_labels": labels(corner.evictors),
            "oracle": corner.oracle_label,
            "practical": list(corner.practical),
            "policies": list(corner.policies),
            "kappa": corner.kappa if "abs" in corner.policies else None,
            "floor": corner.floor if "abs" in corner.policies else None,
            "kstar": corner.kstar,
            "kstar_points": corner.kstar_points if corner.kstar else None,
            "kstar_tol": corner.kstar_tol if corner.kstar else None,
            "state_bytes_per_layer_head_token": state_bytes_per_slot(corner),
            "tag": corner_tag(corner)}


def state_bytes_per_slot(corner: "CornerSpec") -> int:
    """Host RAM the lagged evictor state costs per (layer, head, cache position).

    Each stateful evictor holds `n_bufs` float32 position-space buffers plus one
    bool "never observed" mask, so the total is
        n_layers * n_heads * ctx * state_bytes_per_slot(corner)
    bytes, live for the duration of one prompt and freed at the boundary. At
    ctx 131072 that is ~19 GB for llama33-70b under the default corner set, which
    the SLURM scripts have to have actually requested -- hence a single helper
    both run_h0.py and the submit scripts size themselves from.
    """
    total = 0
    for spec in corner.evictors:
        _, ev = make(spec)
        if ev is not None:
            total += ev.bytes_per_slot()
    return total


@dataclass(frozen=True)
class StepRole:
    """What one decode step does in run_h0's loop."""
    probe: bool     # capture queries and drive the evictors (score + observe)
    row: bool       # emit metrics rows
    quant: bool     # run the bit sweep -- the expensive path
    fresh: bool     # first step of a probed block: evictor history starts here


def decode_plan(n_decode: int, quant_every: int = 1, measure_steps=None,
                specs=()) -> list[StepRole]:
    """Per-step roles for the decode loop.

    DENSE (measure_steps empty): every step is probed and emits rows, and every
    `quant_every`-th step is quantized -- exactly the pre-R5 behaviour.

    SPARSE (R5): only the listed steps emit rows, and each is quantized. The
    decode runs to max(measure_steps) with the probe OFF in between, which is
    what makes a several-thousand-token generation affordable: a probed step
    costs a per-head pass over all layers, an unprobed one is a plain forward.
    Each measured step is preceded by just enough probed WARM steps for every
    configured evictor to hold the history it claims (`history_steps`), and
    history restarts at each block, so a lagged score is never computed across a
    gap. An evictor with unbounded history (`accum`, a running sum over EVERY
    step) cannot be honestly fed by a sparse schedule and is refused -- with a
    gap it would silently become `last_step` under another name.
    """
    if not measure_steps:
        qe = max(1, int(quant_every))
        return [StepRole(True, True, t % qe == 0, t == 0)
                for t in range(int(n_decode))]
    ms = sorted({int(x) for x in measure_steps})
    if ms[0] < 0:
        raise ValueError(f"measure_steps must be >= 0, got {ms[0]}")
    need = 0
    for sp in parse_specs(specs):
        label, ev = make(sp)
        if ev is None:
            continue
        if ev.history_steps is None:
            raise ValueError(
                f"evictor {label!r} accumulates over EVERY decode step, so a "
                f"sparse measure_steps schedule cannot feed it. Use last_step / "
                f"lag:k=N / window for a sparse run, or drop measure_steps and "
                f"decode densely.")
        need = max(need, int(ev.history_steps))
    want = set(ms)
    probe = set()
    for m in ms:
        probe.update(range(max(0, m - need), m + 1))
    return [StepRole(t in probe, t in want, t in want,
                     t in probe and (t - 1) not in probe)
            for t in range(ms[-1] + 1)]


@dataclass(frozen=True)
class CornerSpec:
    """Everything about corner construction that a run can configure.

    These defaults must stay identical to `defaults:` in h0_measurement/models.yaml
    -- they are the same shipped configuration, and a drift between them would
    make every RAM/cost figure wrong depending on which one the caller hit.
    """
    # Kept identical to `defaults:` in h0_measurement/models.yaml -- see the note
    # there. Lean since R3: window/recency cost 18 of 28 B/slot for 6-22% and
    # 1-12% of head wins, and `abs` was refuted by E1.
    evictors: tuple = (ORACLE, "accum")
    policies: tuple = ("frac",)
    kappa: float = 4.0          # abs policy: keep kappa * n95 tokens ...
    floor: int = 256            # ... but never fewer than this
    kstar: bool = True          # the K* slack diagnostic
    kstar_points: int = 12      # geometric grid resolution
    kstar_tol: float = 0.10     # "within 10% of the full-budget corner"

    # R3 / E2b -- THE SYMMETRIC CELL.
    # Demoting the corner to lagged attention left the comparison asymmetric:
    # the interior still water-fills on the CURRENT step's sensitivity
    # w2 = (a*||v-o||)^2, which is exactly the oracle information we just took
    # away from the corner. `interior_scores` names the lagged score(s) to ALSO
    # run the allocator on, so both sides decide from the same information.
    # Empty tuple = skip (the pre-R3 behaviour, oracle interior only).
    #
    # Default is `accum` alone rather than every evictor: it wins 74-93% of
    # corners, so it is the score a deployable system would actually carry, and
    # each extra entry costs one waterfill + one exact_error per (head, budget).
    interior_scores: tuple = ("accum",)
    # Rank the corner by the practical SENSITIVITY w2p as well as by the raw
    # lagged attention. Raw pa is literature-faithful (H2O/SnapKV rank on
    # attention); w2p is apples-to-apples with a w2p interior, and the gap
    # between them isolates how much of the interior's edge is mixed-precision
    # SHAPE rather than score information the corner lacks.
    interior_rank_corner: bool = True

    @property
    def oracle_label(self) -> str | None:
        """Label the oracle corner is reported under, or None if this run did
        not ask for it. It is a CONFIGURED corner like any other -- listed by
        default, because gain_best<B>/err_evict<B> and the monotonicity check
        are defined against it, but a run may drop it."""
        for spec in self.evictors:
            label, ev = make(spec)
            if ev is None:
                return label
        return None

    @property
    def practical(self) -> tuple:
        return tuple(lab for lab, ev in (make(s) for s in self.evictors)
                     if ev is not None)

    @classmethod
    def from_cfg(cls, c: dict) -> "CornerSpec":
        specs = parse_specs(c.get("evictors", cls.evictors))
        pol = c.get("corner_policies", cls.policies)
        if isinstance(pol, str):
            pol = [p.strip() for p in pol.split(",") if p.strip()]
        pol = tuple(pol)
        for p in pol:
            if p not in ("frac", "abs"):
                raise ValueError(f"unknown corner policy {p!r}; use frac / abs")
        if not pol:
            raise ValueError("corner_policies is empty; 'frac' is the status quo")
        return cls(evictors=tuple(specs), policies=pol,
                   kappa=float(c.get("corner_kappa", cls.kappa)),
                   floor=int(c.get("corner_floor", cls.floor)),
                   kstar=bool(c.get("kstar", cls.kstar)),
                   kstar_points=int(c.get("kstar_points", cls.kstar_points)),
                   kstar_tol=float(c.get("kstar_tol", cls.kstar_tol)),
                   interior_scores=_parse_interior(c, cls, specs),
                   interior_rank_corner=bool(c.get("interior_rank_corner",
                                                   cls.interior_rank_corner)))


def _parse_interior(c: dict, cls, specs) -> tuple:
    """R3: which lagged score(s) the ALLOCATOR also runs on.

    Validated against the evictors this run actually configures, because an
    interior score with no matching evictor would silently produce nothing --
    exactly the class of silent-empty-column bug that hid the practical corner
    for three campaigns (bugs/2 P0).
    """
    explicit = "interior_scores" in c
    v = c.get("interior_scores", cls.interior_scores)
    if v is None or v is False:
        return ()
    if isinstance(v, str):
        v = [x.strip() for x in v.split(",") if x.strip()]
    want = tuple(v)
    have = tuple(lab for lab, ev in (make(sp) for sp in specs) if ev is not None)
    bad = [x for x in want if x not in have]
    # An EXPLICIT request for a score this run does not compute is an error --
    # that is the silent-empty-column class that hid the practical corner for
    # three campaigns. But the class DEFAULT naming `accum` must degrade
    # gracefully, or `SIEVE_EVICTORS=oracle` (documented as the way to reproduce
    # the pre-corner cost) would fail on a default it never asked for.
    if bad and not explicit:
        return tuple(x for x in want if x in have)
    if bad:
        raise ValueError(
            f"interior_scores {bad} are not practical evictors in this run "
            f"(configured: {list(have)}). The allocator can only use a score "
            f"the run actually computes.")
    # `recency` is a position index, not an attention magnitude. Normalising it
    # into a distribution would make the allocator spend bits by position, which
    # is not a lagged estimate of anything.
    ordinal = [x for x in want if x.startswith("recency")]
    if ordinal:
        raise ValueError(
            f"interior_scores {ordinal} are ORDINAL scores (position only). The "
            f"interior normalises its score into a distribution, so it needs an "
            f"attention magnitude -- use accum / last_step / window / lag.")
    return want


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


def make(spec: str) -> tuple[str, "Evictor | None"]:
    """Build one corner from a spec string. Returns (label, evictor|None);
    None means the oracle, which alloc.py computes rather than tracking.

        "accum"                      -> H2O, defaults
        "window:window=8,pool=13"    -> SnapKV, options overridden
        "window:window=8@snap8"      -> same, labelled snap8 (needed only when
                                        the same evictor appears twice)

    Paper names are accepted as aliases: tova/h2o/snapkv/streamingllm.
    """
    body, _, alias = str(spec).strip().partition("@")
    name, _, opts = body.strip().partition(":")
    name = _ALIAS.get(name.strip().lower(), name.strip().lower())
    label = (alias.strip() or name).lower()
    if not label or not label.replace("_", "").isalnum():
        raise ValueError(f"corner label {label!r} must be alphanumeric "
                         f"(underscores allowed) -- it becomes a column name")
    if name == ORACLE:
        if opts:
            raise ValueError("the oracle corner takes no options")
        return label, None
    if name not in _REG:
        raise KeyError(f"unknown evictor {name!r}; available: {available()}")
    kw = {}
    for part in (o for o in opts.split(",") if o.strip()):
        k, sep, v = part.partition("=")
        if not sep:
            raise ValueError(f"option {part!r} in {spec!r} is not key=value")
        kw[k.strip()] = _coerce(v)
    return label, _REG[name](**kw)


def parse_specs(cfg) -> list[str]:
    """Normalise the `evictors` config value to a list of spec strings.

    Accepts a YAML list, or a string from --override. A string is split on ';'
    when it contains one, else on ',' when it carries no options -- so both
    `evictors=oracle,accum` and `evictors=window:window=8,pool=7` mean what they
    look like. 'none' / '' / [] leaves only the oracle corner.
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
    specs = [x.strip() for x in specs if x.strip()]
    for sp in specs:                       # fail early, before the GPU is held
        make(sp)
    return specs


def make_many(specs) -> dict[str, Evictor]:
    """Build the STATEFUL evictors, keyed by label. The oracle is dropped -- it
    has no state to carry -- so this is exactly what run_h0 must drive per head.
    Fails on a duplicate label rather than silently losing a column."""
    out: dict[str, Evictor] = {}
    for spec in parse_specs(specs):
        label, ev = make(spec)
        if ev is None:
            continue
        if label in out:
            raise ValueError(f"duplicate corner label {label!r}; give one an "
                             f"alias, e.g. '{spec}@{label}_alt'")
        out[label] = ev
    return out


def labels(specs) -> list[str]:
    """Corner labels the config produces, oracle first, without building state."""
    seen, out = set(), []
    for spec in parse_specs(specs):
        label, ev = make(spec)
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out
