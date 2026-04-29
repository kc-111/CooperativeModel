"""Global constrained optimization for L_final(F1..F4) over [0, 100]^4.

Problem (`objective.py`):
    maximise   L_final(F)
    subject to F ∈ [0, 100]^4         (box bounds)

Algorithm: batched-multistart with projected gradient ascent.

Why not scipy.optimize.shgo (the textbook "find all local minima" tool)?
We tried it — see `objective.py` for the SHGO-compatible `neg_with_grad`
interface.  SHGO is the correct *abstract* algorithm for this problem
(Sobol-sample → simplicial homology → L-BFGS-B from each minimiser
candidate), but it serialises the L-BFGS-B inner loop.  Each call to our
objective creates a fresh `Simulator(...)` with 9 samples and ~1 s setup
overhead per call; SHGO at `n=256, iters=1, maxiter=30` makes thousands of
these serial calls and runs for hours.  The custom batched ascent below
exploits the fact that 100 candidates × 9 perturbations = 900 samples in a
SINGLE Simulator call costs the same as 9 samples — so batching across
*candidates* (not just across FD perturbations) gives a ~100× speedup.

The algorithm is morally identical to SHGO with sobol sampling:
  1. Sample 200k Sobol + 16 corner points, batched in one Simulator call.
  2. Greedy ε-separated peak selection — discrete analogue of the
     "minimiser candidate" check (lower than every neighbour).
  3. Projected gradient ascent on all K candidates simultaneously: each
     outer iteration is one Simulator call evaluating 9·K perturbed
     points (the candidate plus 8 axis perturbations) followed by a
     batched backtracking line search.
  4. Re-dedupe + corner augmentation.
  5. Gradient + Hessian + discrete-max classification (verify_optima.py)
     to certify INTERIOR / BOUNDARY maxima.

If `Simulator` ever gains a persistent / low-overhead invocation, swap
this script back to `scipy.optimize.shgo` — the `Objective` class already
supports it via `obj.neg_with_grad`.
"""

import sys, os, time, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objective import Objective
from verify_optima import grad_hess, classify, KIND_COLOR


# ── Search settings ─────────────────────────────────────────────────────────
N_SCAN     = 200_000     # Sobol scan size
N_TOP      = 100         # candidates carried into refinement
EPSILON    = 15.0        # min L2 separation between distinct optima
N_ITER     = 30          # batched gradient-ascent outer iterations
H_REFINE   = 2.0         # FD step during refinement
H_VERIFY   = 2.0         # FD step for classifier


# ── Build the optimization problem ──────────────────────────────────────────
obj = Objective(t_final=72.0, N1=0.05, N2=0.05, device='cuda', fd_h=H_REFINE)


def batched_gradient_ascent(s_arr, n_iter=N_ITER, h=H_REFINE):
    """Project-gradient-ascent K candidates in lockstep with batched FD.

    Per outer iter: one Simulator call on 9·K points (each candidate plus
    its 8 axis perturbations), then a backtracking line search done in
    batched groups of K.  Total: ~3-5 Simulator calls per outer iter,
    each batched.
    """
    K = len(s_arr)
    s = np.asarray(s_arr, dtype=float).clip(0, 100).copy()
    L_now = obj.evaluate_batch(s)
    alpha = np.full(K, 30.0)

    for _ in range(n_iter):
        h_eff = np.minimum.reduce([np.full((K, 4), h), s, 100 - s])
        h_eff = np.where(h_eff <= 0, h, h_eff)
        pts = [s.copy()]
        for i in range(4):
            sp = s.copy(); sp[:, i] = np.minimum(s[:, i] + h_eff[:, i], 100); pts.append(sp)
            sm = s.copy(); sm[:, i] = np.maximum(s[:, i] - h_eff[:, i], 0);   pts.append(sm)
        big = np.vstack(pts)
        Lvals = obj.evaluate_batch(big).reshape(9, K)
        L_now = Lvals[0]
        g = np.zeros((K, 4))
        for i in range(4):
            g[:, i] = (Lvals[1+2*i] - Lvals[2+2*i]) / (2 * h_eff[:, i])
        gn = np.linalg.norm(g, axis=1, keepdims=True)
        d  = g / np.where(gn > 1e-9, gn, 1.0)

        active = (gn[:, 0] > 1e-3)
        for _ in range(6):
            if not active.any():
                break
            s_try = (s + alpha[:, None] * d).clip(0, 100)
            L_try = obj.evaluate_batch(s_try)
            improved = (L_try > L_now + 1e-4) & active
            s = np.where(improved[:, None], s_try, s)
            L_now = np.where(improved, L_try, L_now)
            alpha = np.where(improved, np.minimum(alpha * 1.5, 50.0), alpha * 0.5)
            active = active & ~improved
        if (alpha < 0.05).all():
            break

    return s, L_now


def greedy_eps_separated(points, scores, epsilon, k_max=None):
    order = np.argsort(scores)[::-1]
    kept = []
    for idx in order:
        p = points[idx]
        if all(np.linalg.norm(p - q) >= epsilon for q, _ in kept):
            kept.append((p.copy(), float(scores[idx])))
            if k_max is not None and len(kept) >= k_max:
                break
    return kept


# ── Phase 1: Sobol scan ─────────────────────────────────────────────────────
print(f'Phase 1 — Sobol scan, N={N_SCAN}')
from scipy.stats.qmc import Sobol
sob = Sobol(d=4, scramble=True, seed=42)
m = int(np.ceil(np.log2(N_SCAN)))
F_scan = 100.0 * sob.random_base2(m=m)[:N_SCAN]
corners = np.array(list(itertools.product([0., 100.], repeat=4)))
F_scan = np.vstack([corners, F_scan])
t0 = time.time()
L_scan = obj.evaluate_batch(F_scan)
print(f'  {len(F_scan)} pts in {time.time()-t0:.1f}s, '
      f'best={L_scan.max():.3f}')


# ── Phase 2: ε-separated peak selection ─────────────────────────────────────
peaks = greedy_eps_separated(F_scan, L_scan, EPSILON, k_max=N_TOP)
print(f'Phase 2 — top {len(peaks)} ε-separated peaks (eps={EPSILON})')


# ── Phase 3: batched projected gradient ascent ──────────────────────────────
print(f'Phase 3 — batched gradient ascent ({N_ITER} iters)')
t0 = time.time()
s_init = np.array([p for p, _ in peaks])
s_opt, L_opt = batched_gradient_ascent(s_init)
print(f'  {time.time()-t0:.1f}s, '
      f'best pre={max(L for _, L in peaks):.3f}, '
      f'best post={L_opt.max():.3f}')


# ── Phase 4: dedupe + corner augmentation ───────────────────────────────────
final = greedy_eps_separated(s_opt, L_opt, EPSILON)
corner_L = obj.evaluate_batch(corners)
for c, Lc in zip(corners, corner_L):
    if all(np.linalg.norm(c - so) >= 1e-6 for so, _ in final):
        final.append((c, float(Lc)))
final = sorted(final, key=lambda x: -x[1])
print(f'Phase 4 — {len(final)} candidates after dedupe')


# ── Phase 5: gradient + Hessian + discrete-max classification ───────────────
print(f'Phase 5 — classify (h={H_VERIFY})')
t0 = time.time()
classified = []
for s, L_val in final:
    L0, g, H, on_lo, on_hi, h_vec, L_plus, L_minus = grad_hess(s, h=H_VERIFY)
    kind, info = classify(s, L0, g, H, on_lo, on_hi,
                          L_plus=L_plus, L_minus=L_minus)
    classified.append((s, L_val, kind, info, L0, g))
print(f'  classification: {time.time()-t0:.1f}s')


# ── Report ──────────────────────────────────────────────────────────────────
print(f'\n{"="*108}')
print(f' {len(classified)} CANDIDATES, classified by gradient + Hessian')
print(f'{"="*108}')
print(f'{"#":<4} {"F1":>7} {"F2":>7} {"F3":>7} {"F4":>7} {"Total":>7} '
      f'{"L_final":>9} {"|g_free|":>9} {"max eig":>9}  status')
print('-' * 108)
for i, (s, L_val, kind, info, L0, g) in enumerate(classified, 1):
    free = info['free']
    g_free = np.linalg.norm(g[free]) if free.any() else 0.0
    max_eig = info['eigvals'].max() if info['eigvals'].size > 0 else float('nan')
    print(f'{i:<4} {s[0]:>7.2f} {s[1]:>7.2f} {s[2]:>7.2f} {s[3]:>7.2f} '
          f'{sum(s):>7.2f} {L_val:>9.4f} {g_free:>9.4f} {max_eig:>9.4f}  '
          f'{KIND_COLOR[kind]}')

genuine  = [c for c in classified if c[2] in ('interior_max', 'boundary_max')]
interior = [c for c in genuine if c[2] == 'interior_max']
boundary = [c for c in genuine if c[2] == 'boundary_max']
print(f'\nSummary: {len(genuine)} genuine local maxima '
      f'({len(interior)} interior, {len(boundary)} boundary), '
      f'{len(classified)-len(genuine)} discarded as saddle/not-max.')
print(f'Total simulator evaluations: {obj.n_evals}')

if len(genuine) > 1:
    print(f'\nTop genuine maxima (gap from #1):')
    g0 = genuine[0][1]
    for i, (s, L_val, kind, _, _, _) in enumerate(genuine[:20], 1):
        gap = g0 - L_val
        pct = 100 * gap / g0 if g0 > 0 else 0
        kind_short = 'INT ' if kind == 'interior_max' else 'BDY '
        print(f'  #{i:<2} {kind_short} ({s[0]:>5.1f},{s[1]:>5.1f},{s[2]:>5.1f},{s[3]:>5.1f}) '
              f'L={L_val:>7.3f}  -{gap:>5.2f} ({pct:>4.1f}%)')
