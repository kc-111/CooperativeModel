"""Find local maxima of L_final over (R1, R2, R3, R4) initial conditions.

Runs in the **well-mixed limit** (grid_shape=(1, 1, 1), flow_cache_path=None,
device='cpu'): the optimisation needs hundreds–thousands of objective
evaluations and the 32^3 PDE costs ~1 s/sample on GPU, while the 9-ODE
well-mixed system costs ~10 ms/sample on CPU.

Pipeline:
    1. Batched landscape scan (random interior + 16 corners).
    2. Greedy epsilon-separated selection of the top peaks.
    3. scipy.optimize.minimize (L-BFGS-B with box constraints).
    4. Dedup refined optima by Euclidean distance.

For the cyclic 4-cycle Liebig model with the symmetric defaults, the
four single-pair corners are expected to land out as the four local
optima: (R1,R2 hi; R3,R4 lo), (R2,R3 hi; ...), etc.

Usage:
    python examples/find_optima.py
"""

import sys, os, time, itertools
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator

# ── Config ──────────────────────────────────────────────────────────────
N0 = 0.01                 # per-species initial biomass
L0 = 0.0
T_FINAL = 24.0
R_MIN, R_MAX = 0.05, 2.0  # resource bounds; R_MIN well below R* ~ K*D/(mu-D)
BOUNDS = [(R_MIN, R_MAX)] * 4
# Eps-separation for the final dedup is set in (R1..R4) Euclidean space.
# 1.0 is roughly half the diagonal of the {LO, HI}^4 corner-set in our
# bounds, so near-corner L-BFGS-B refinements collapse into one optimum
# per corner while keeping the four corners distinguishable.
EPSILON = 1.0
N_RANDOM = 400
N_TOP = 30
DEVICE = 'cpu'
SEED = 42


def eval_batch(resources):
    """Vectorised L_final for resources of shape [B, 4]."""
    resources = np.asarray(resources, dtype=float).reshape(-1, 4)
    B = len(resources)
    # IC: [N1..N4, L, R1..R4, T1..T4, F1..F4]
    samples = np.zeros((B, 17))
    samples[:, 0] = N0
    samples[:, 1] = N0
    samples[:, 2] = N0
    samples[:, 3] = N0
    samples[:, 4] = L0
    samples[:, 5:9] = resources
    # T1..T4 (indices 9..12) and F1..F4 (indices 13..16) stay at zero —
    # no warfare and no accumulated byproducts at t=0.
    r = Simulator(
        samples=samples.tolist(),
        t_final=T_FINAL, grid_shape=(1, 1, 1),
        flow_cache_path=None,
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

    print(f'Landscape scan: {N_RANDOM} random + 16 corner points (well-mixed)...')
    t0 = time.time()
    random_pts = rng.uniform(R_MIN, R_MAX, size=(N_RANDOM, 4))
    corners = np.array(list(itertools.product([R_MIN, R_MAX], repeat=4)))
    all_pts = np.vstack([corners, random_pts])
    all_L = eval_batch(all_pts)
    print(f'  scan: {time.time() - t0:.1f}s  ({len(all_pts)} evaluations)')
    print(f'  L range: [{all_L.min():.3f}, {all_L.max():.3f}], '
          f'mean={all_L.mean():.3f}')

    seeds = greedy_select(all_pts, all_L, EPSILON, N_TOP)
    print(f'Selected {len(seeds)} epsilon-separated seeds (eps={EPSILON})')

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

    refined.sort(key=lambda r: -r[1])
    final = []
    for x, L_val in refined:
        if all(np.linalg.norm(x - x_other) >= EPSILON for x_other, _ in final):
            final.append((x, L_val))

    # Classify each optimum by which pair-corner it is closest to.
    # P_i = high on resources (i, i+1) modulo 4; LO elsewhere.
    pair_templates = {
        'P1 (N1: R1,R2)': np.array([R_MAX, R_MAX, R_MIN, R_MIN]),
        'P2 (N2: R2,R3)': np.array([R_MIN, R_MAX, R_MAX, R_MIN]),
        'P3 (N3: R3,R4)': np.array([R_MIN, R_MIN, R_MAX, R_MAX]),
        'P4 (N4: R4,R1)': np.array([R_MAX, R_MIN, R_MIN, R_MAX]),
    }

    def classify(x):
        best_name, best_d = None, np.inf
        for name, tpl in pair_templates.items():
            d = float(np.linalg.norm(x - tpl))
            if d < best_d:
                best_name, best_d = name, d
        return best_name, best_d

    print(f'\n{"=" * 72}')
    print(f' {len(final)} DISTINCT LOCAL OPTIMA  (epsilon = {EPSILON})')
    print(f'{"=" * 72}')
    print(f'{"#":<4} {"R1":>6} {"R2":>6} {"R3":>6} {"R4":>6} '
          f'{"L_final":>10}   {"closest pair":<16} {"d":>5}')
    print('-' * 72)
    for i, (x, L_val) in enumerate(final, 1):
        name, d = classify(x)
        print(f'{i:<4} {x[0]:>6.2f} {x[1]:>6.2f} {x[2]:>6.2f} {x[3]:>6.2f} '
              f'{L_val:>10.4f}   {name:<16} {d:>5.2f}')

    if len(final) > 1:
        best_x, best_L = final[0]
        print(f'\nGap from #1 (L = {best_L:.4f}):')
        for i, (x, L_val) in enumerate(final[1:], 2):
            gap = best_L - L_val
            pct = 100.0 * gap / best_L if best_L > 0 else float('nan')
            dist = np.linalg.norm(best_x - x)
            print(f'  #{i}: -{gap:.3f} ({pct:.1f}%), distance = {dist:.2f}')


if __name__ == '__main__':
    main()
