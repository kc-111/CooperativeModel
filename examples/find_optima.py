"""Find N distinct local optima for L_final over F1..F4 ∈ [0, 100].

Closed batch reactor: F1..F4 are initial sugar amounts loaded at t=0.

Strategy:
  1. Batch-evaluate 100k random points to map the landscape
  2. Greedy epsilon-separated peak selection (top-down by L_final)
  3. L-BFGS-B refinement from each peak
  4. Final dedup and report
"""

import sys, os, time, itertools
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator

# ── Config ──────────────────────────────────────────────────────────────
N1, N2, Sn, L = 0.05, 0.05, 0.0, 0.0
EPSILON = 15.0        # min Euclidean distance between distinct optima
N_RANDOM = 100_000    # random samples for landscape scan
N_TOP = 500           # top candidates to cluster from
DEVICE = 'cuda'

def eval_batch(sugars_array):
    """Evaluate L_final for a batch of [B, 4] initial sugar loadings."""
    B = len(sugars_array)
    samples = np.zeros((B, 8))
    samples[:, 0] = N1
    samples[:, 1] = N2
    samples[:, 4:] = sugars_array
    r = Simulator(
        samples=samples.tolist(),
        mode='batch', t_final=72.0, grid_size=1,
        omega=0.0, diffusion_scale=0.0, device=DEVICE
    ).run()
    return r.L_final if B > 1 else np.array([r.L_final])

def eval_single(sugars):
    """Evaluate L_final for a single [4] initial sugar loading."""
    return eval_batch(sugars.reshape(1, -1))[0]

def refine(sugars_init):
    """Local optimisation via L-BFGS-B. Returns (optimal_sugars, L_final)."""
    result = minimize(
        lambda s: -eval_single(np.array(s)),
        x0=sugars_init, method='L-BFGS-B',
        bounds=[(0, 100)] * 4,
        options={'maxiter': 100, 'ftol': 1e-10}
    )
    return result.x, -result.fun

def greedy_select(points, scores, epsilon):
    """Greedy epsilon-separated selection, highest score first."""
    order = np.argsort(scores)[::-1]
    selected = []
    for idx in order:
        p = points[idx]
        if all(np.linalg.norm(p - s) >= epsilon for s in [points[j] for j, _ in selected]):
            selected.append((idx, scores[idx]))
    return selected

# ── Step 1: Landscape scan ──────────────────────────────────────────────
print(f'Scanning landscape with {N_RANDOM} random points + 16 corners...')
t0 = time.time()

# Random interior points in [0, 100]^4
rng = np.random.default_rng(42)
random_sugars = rng.uniform(0, 100, size=(N_RANDOM, 4))

# All 16 corners (each F either 0 or 100)
corners = np.array(list(itertools.product([0, 100], repeat=4)), dtype=float)

# Combine
all_sugars = np.vstack([corners, random_sugars])
all_L = eval_batch(all_sugars)
print(f'Landscape scan: {time.time() - t0:.1f}s')

# ── Step 2: Epsilon-separated peak selection ────────────────────────────
top_idx = np.argsort(all_L)[-N_TOP:][::-1]
top_sugars = all_sugars[top_idx]
top_L = all_L[top_idx]

candidates = greedy_select(top_sugars, top_L, EPSILON)
print(f'\nFound {len(candidates)} epsilon-separated peaks (eps={EPSILON})')

# ── Step 3: Refine each candidate ──────────────────────────────────────
print(f'Refining {len(candidates)} candidates with L-BFGS-B...')
t0 = time.time()
optima = []
for i, (idx, _) in enumerate(candidates):
    s_init = all_sugars[idx]
    s_opt, L_opt = refine(s_init)
    optima.append((s_opt, L_opt))
print(f'Refinement: {time.time() - t0:.1f}s')

# ── Step 4: Final dedup (refinement may merge basins) ──────────────────
final = []
for s, L_val in sorted(optima, key=lambda x: -x[1]):
    if all(np.linalg.norm(s - so) >= EPSILON for so, _ in final):
        final.append((s, L_val))

# ── Report ──────────────────────────────────────────────────────────────
print(f'\n{"="*72}')
print(f' {len(final)} DISTINCT LOCAL OPTIMA  (epsilon = {EPSILON})')
print(f'{"="*72}')
print(f'{"#":<4} {"F1":>7} {"F2":>7} {"F3":>7} {"F4":>7} {"Total":>7} {"L_final":>10}')
print('-' * 55)
for i, (s, L_val) in enumerate(final, 1):
    print(f'{i:<4} {s[0]:>7.2f} {s[1]:>7.2f} {s[2]:>7.2f} {s[3]:>7.2f} '
          f'{sum(s):>7.2f} {L_val:>10.4f}')

# Show the gap between the global optimum and the rest
if len(final) > 1:
    print(f'\nGap from #1:')
    for i, (s, L_val) in enumerate(final[1:], 2):
        gap = final[0][1] - L_val
        pct = 100 * gap / final[0][1]
        dist = np.linalg.norm(np.array(final[0][0]) - s)
        print(f'  #{i}: -{gap:.2f} ({pct:.1f}%), distance={dist:.1f}')
