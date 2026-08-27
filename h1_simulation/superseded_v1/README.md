# superseded_v1 — the ORIGINAL H1 scripts

These are the scripts that produced the **first-pass** H1 figures, restored
byte-for-byte. They are archived for provenance. **Do not use them to generate
figures for the paper** — three of their results were overturned by the audit.

| file | produces |
|---|---|
| `h1_sim.py` | prints the four-scheme error tables; the base simulation |
| `h1_robust.py` | `fig1_curves`, `fig2_alloc`, `fig3_envelope` + cascade / correlated-value checks |
| `h1_tau.py` | `fig4_tau` + the τ sweep and eviction-fraction table |
| `fig*_ORIGINAL.png` | the exact figures they produced |

Run order: `h1_sim.py` → `h1_robust.py` → `h1_tau.py` (the latter two import from
`h1_sim`). They are self-contained apart from numpy/matplotlib and do **not**
depend on `sievelib`.

---

## What is wrong with them

Kept verbatim, so the flaws are still present. In order of severity:

**1. Gains are reported against uniform bits only.** `h1_tau.py` compares
water-filling to the best *uniform* baseline and finds the gain grows with τ
(1.0× → 70×). Compared against the better of *both* corners the curve is
non-monotonic: it peaks near τ≈1.25 and collapses to 1.0× by τ≈2, because at high
τ the optimum becomes nearly two-tier and converges to eviction. This is the
finding that forced proposal v5.

**2. Mixed cost units.** `h1_sim.py` uses `C_MSE` with `c_0 = 1.0` for eviction and
τ-relative constants for the quantized tiers — i.e. it implicitly assumes τ=1.
`h1_tau.py` does scale by τ² but then applies `min(0.999, τ²·c_b)`, a clamp that
prevents any tier from costing more than eviction. That is exactly the situation
that arises once τ>1, so the clamp suppresses the effect in (1).

**3. Error is computed through the linearized cost**, not by exact recomputation.
On synthetic heads the first-order expansion overstates the gain by 20–60%
(`lin_ratio` ≈ 1.2–1.6 in the corrected code).

Also: noise constants are taken from TurboQuant's published table rather than
measured on real keys, and the quantizer is never actually run — noise is injected
as Gaussian draws.

## What survived unchanged

- the one-pass cascade retaining 81–101% of the oracle gain (`h1_robust.py`)
- the QJL dead-zone result — 1- and 2-bit tiers off the lower convex envelope
  (`h1_sim.py:convex_envelope`, `fig3`). The corrected version adds that the
  envelope is τ-dependent and strips tiers from the bottom for *both* variants.
- robustness to correlated values, ρ up to 0.9

## Current equivalent

`../run_h1.py` — absolute cost units, no clamp, the real TurboQuant quantizer from
`sievelib/quant.py`, error by exact recomputation, gains against the best corner.
It regenerates `docs/fig1_curves.png` and `docs/fig4_tau.png`.

---

## Where `C_MSE` / `C_PROD` come from, and whether a fixed-variance noise model is ever legitimate

Two separate questions: where the numbers came from, and whether the *modeling approach*
(inject Gaussian noise with a fixed variance, instead of running a real quantizer) is ever
legitimate.

### Where the real numbers come from

`C_MSE = {0: 1.0, 1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}` in `h1_sim.py:32`. The module
docstring says these are "TurboQuant's tabulated distortion constants" (`h1_sim.py:18`) —
i.e. typed in from a published table in the TurboQuant paper, not measured on this
codebase's own data.

You can see exactly what kind of number that is by reading `../../sievelib/quant.py`, which
is the *corrected* pipeline's real quantizer:

1. Split each key vector into norm (`gamma`) and direction (`x = k/gamma`).
2. Apply a random rotation `y = Rx` (`quant.py:81`) — this is the trick that makes the
   published table possible at all: rotating a fixed vector by a random (Haar) rotation
   makes its coordinates behave like i.i.d. draws from a roughly Gaussian distribution,
   regardless of what the original key's coordinates looked like (a concentration-of-measure
   argument).
3. Scalar-quantize each rotated coordinate independently with a **Lloyd-Max quantizer** —
   the provably-optimal 1-D quantizer for a given source distribution — at `2^bits` levels
   (`quant.py:25`, `lloyd_max_levels`).

So "TurboQuant's distortion constant at b bits" is the mean-squared error of the *optimal*
scalar quantizer applied to a standard normal, at `2^b` levels — a number you can compute in
closed form / by numerical convergence (which is literally what `lloyd_max_levels()` does),
independent of any specific dataset. That's why it can be published as a fixed table: it's a
property of "optimally quantize a Gaussian at b bits," not a property of any particular
model's keys.

The corrected pipeline no longer trusts that published table, though — it **measures** the
same quantity on real keys instead: `sievelib/alloc.py`'s `noise_model()` runs the real
quantizer and computes `sig2[b] = Var(measured_logit − true_logit)` directly. That's the
difference between "assume keys look like the theoretical ideal" and "check."

### Is Gaussian-noise-injection-with-a-fixed-variance ever a legitimate stand-in?

Yes, in a specific, well-understood regime — but the boundaries of that regime are exactly
where this simulation's flaws live.

**Where it holds up:**

- **High-rate quantization theory** says that as bit-width grows, quantization error becomes
  well-approximated as uniform/Gaussian-like and only weakly correlated with the input — the
  classical result behind why audio/image/video codecs allocate bits using MSE tables in the
  first place. The random rotation step in TurboQuant's pipeline is specifically there to
  push you into this regime even at *low* bit counts, by decorrelating each coordinate's
  error from the signal's original structure.
- **Dithered quantization** (add random dither before quantizing, subtract after) has a
  proven result that the resulting error is *exactly* uniform and *statistically
  independent* of the input — in that case, injecting synthetic i.i.d. noise with the right
  variance isn't an approximation, it's mathematically equivalent to the real thing.
- **As a cheap first-pass feasibility screen**, it's standard and reasonable: before spending
  engineering effort building a real quantizer + exact recomputation pipeline, you want a
  fast way to ask "does this idea have any chance of working at all, roughly?" That's
  precisely the role `h1_sim.py` was meant to play — it's not that the *approach* was wrong,
  it's that its results were then mistaken for a load-bearing conclusion rather than a rough
  screen (which is exactly why this README says "do not use these to generate figures for
  the paper").
- **For ranking/allocation only**, a variance-based proxy is standard practice — this is
  literally how bit-allocation works in real codecs (greedy/Lagrangian allocation against
  per-block distortion tables), and it's what the corrected `sievelib/alloc.py` still does:
  `waterfill()` uses the linear `w² × sig2[b]` proxy to *choose* bit-widths, on the theory
  that even an imperfect ranking signal is good enough to pick roughly the right allocation,
  as long as the *final reported number* is checked against reality.

**Where it breaks — and this is exactly what flaws 2 and 3 above are about:**

- **Low bit-widths (1-2 bits)** are precisely where the Gaussian/high-rate approximation is
  weakest — the true error at 1-2 levels is a small number of discrete, signal-correlated
  outcomes, not smooth Gaussian noise. The variance constant can still be *correct on
  average*, but it can't capture the shape (heavy tails, discreteness) that matters once
  that error gets pushed through something nonlinear.
- **Softmax is exactly that nonlinearity.** A second-moment summary (variance) throws away
  tail behavior, but `exp()` amplifies tails — a rare large logit error matters far more to
  the softmax output than the variance number suggests. That's the core reason `exact_error()`
  exists in the corrected code: it doesn't trust that "right variance ⇒ right downstream
  error" holds once you're inside a softmax.
- **A table calibrated on "idealized rotated-Gaussian coordinates" isn't the same as "this
  real model's keys."** Real LLM KV caches are known to have outlier channels and structure
  that a fixed published constant can't see — which is exactly why the corrected pipeline
  switched from *importing* TurboQuant's table to *measuring* `sig2` by actually running the
  quantizer on real keys.

So: the modeling choice isn't illegitimate in general — it's a well-founded approximation in
the high-rate/dithered regime, and a reasonable choice for a first-pass screen or for ranking
candidates. The bug wasn't "using a noise model," it was using an *unmeasured, un-scaled* one
(flaw 2) and trusting its output as the final answer instead of verifying it against a real
quantizer (flaw 3) — precisely in the low-bit, high-τ regime where the approximation is least
trustworthy.
