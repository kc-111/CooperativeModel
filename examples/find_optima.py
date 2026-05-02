"""Find local maxima of L_final over (F1, F2, F3, F4) in [0, 100]^4.

Closed-batch ODE model: F1..F4 are the initial sugar amounts loaded at t=0;
N1, N2 are fixed inocula; we maximise the spatially-averaged L (lactic acid)
at t = t_final.

Pipeline:
    1. Batched landscape scan over many random points (cheap with grid_size=1).
    2. Greedy epsilon-separated selection of the top peaks.
    3. scipy.optimize.minimize (L-BFGS-B with box constraints [0, 100]) from
       each selected peak.
    4. Dedup refined optima by Euclidean distance.

Usage:
    python examples/find_optima.py
"""

import sys, os, time, itertools
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator

# ── Config ──────────────────────────────────────────────────────────────
N1, N2, Sn, L0 = 0.05, 0.05, 0.0, 0.0
T_FINAL = 48.0
BOUNDS = [(0.0, 10.0)] * 4
EPSILON = 1.0          # min Euclidean distance between distinct optima
N_RANDOM = 30       # random points for landscape scan
N_TOP = 30              # number of seeded peaks to refine
DEVICE = 'cuda'         # 'cpu' if no GPU
SEED = 42
U_imp = 0.5

def eval_batch(sugars):
    """Vectorised L_final for sugars of shape [B, 4]."""
    sugars = np.asarray(sugars, dtype=float).reshape(-1, 4)
    B = len(sugars)
    samples = np.zeros((B, 8))
    samples[:, 0] = N1
    samples[:, 1] = N2
    samples[:, 2] = Sn
    samples[:, 3] = L0
    samples[:, 4:] = sugars
    r = Simulator(
        samples=samples.tolist(),
        mode='flow_through', t_final=T_FINAL, grid_size=32,
        U_imp=U_imp, diffusion_scale=0.1, flow_rate=0.05,
        device=DEVICE,
    ).run()
    return np.atleast_1d(np.asarray(r.L_final))


def neg_L(x):
    """Negative L_final for a single point — used by scipy.optimize."""
    return -float(eval_batch(np.asarray(x).reshape(1, 4))[0])


def greedy_select(points, scores, epsilon, k):
    """Pick up to k points greedily, scoring high to low, with min separation."""
    order = np.argsort(scores)[::-1]
    chosen = []
    for idx in order:
        p = points[idx]
        if all(np.linalg.norm(p - points[j]) >= epsilon for j, _ in chosen):
            chosen.append((idx, scores[idx]))
            if len(chosen) >= k:
                break
    return chosen


def main():
    rng = np.random.default_rng(SEED)

    # ── Step 1: landscape scan (random interior + 16 corners) ──────────
    print(f'Landscape scan: {N_RANDOM} random + 16 corner points...')
    t0 = time.time()
    random_pts = rng.uniform(0, 100, size=(N_RANDOM, 4))
    corners = np.array(list(itertools.product([0.0, 100.0], repeat=4)))
    all_pts = np.vstack([corners, random_pts])
    all_L = eval_batch(all_pts)
    print(f'  scan: {time.time() - t0:.1f}s  ({len(all_pts)} evaluations)')

    # ── Step 2: epsilon-separated seeds ────────────────────────────────
    seeds = greedy_select(all_pts, all_L, EPSILON, N_TOP)
    print(f'Selected {len(seeds)} epsilon-separated seeds (eps={EPSILON})')

    # ── Step 3: L-BFGS-B refinement from each seed ─────────────────────
    print(f'Refining with scipy.optimize.minimize (L-BFGS-B)...')
    t0 = time.time()
    refined = []
    for idx, _ in seeds:
        x0 = all_pts[idx]
        res = minimize(
            neg_L, x0=x0, method='L-BFGS-B',
            bounds=BOUNDS,
            options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6},
        )
        refined.append((res.x, -res.fun))
    print(f'  refinement: {time.time() - t0:.1f}s')

    # ── Step 4: dedup ──────────────────────────────────────────────────
    refined.sort(key=lambda r: -r[1])
    final = []
    for x, L_val in refined:
        if all(np.linalg.norm(x - x_other) >= EPSILON for x_other, _ in final):
            final.append((x, L_val))

    # ── Report ─────────────────────────────────────────────────────────
    print(f'\n{"=" * 64}')
    print(f' {len(final)} DISTINCT LOCAL OPTIMA  (epsilon = {EPSILON})')
    print(f'{"=" * 64}')
    print(f'{"#":<4} {"F1":>7} {"F2":>7} {"F3":>7} {"F4":>7} '
          f'{"sumF":>7} {"L_final":>10}')
    print('-' * 56)
    for i, (x, L_val) in enumerate(final, 1):
        print(f'{i:<4} {x[0]:>7.2f} {x[1]:>7.2f} {x[2]:>7.2f} {x[3]:>7.2f} '
              f'{x.sum():>7.2f} {L_val:>10.4f}')

    if len(final) > 1:
        best_x, best_L = final[0]
        print(f'\nGap from #1 (L = {best_L:.4f}):')
        for i, (x, L_val) in enumerate(final[1:], 2):
            gap = best_L - L_val
            pct = 100.0 * gap / best_L if best_L > 0 else float('nan')
            dist = np.linalg.norm(best_x - x)
            print(f'  #{i}: -{gap:.3f} ({pct:.1f}%), distance = {dist:.1f}')


if __name__ == '__main__':
    main()
