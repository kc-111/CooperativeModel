"""Find the initial sugar loading (F1..F4 in [0,100]) that maximises L_final.

Closed batch reactor: F1..F4 are the amounts loaded into the sealed vessel
at t=0, consumed over t_final hours.  No continuous feed.

Strategy:
  1. Coarse grid search (11 points per sugar → 14,641 samples, one batch)
  2. Refine around the best region with scipy L-BFGS-B
"""

import sys, os, time, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator

# Fixed biological ICs
N1, N2, Sn, L = 0.05, 0.05, 0.0, 0.0

# ── Step 1: Coarse grid search ──────────────────────────────────────────
grid_pts = np.linspace(0, 100, 11)  # 0, 10, 20, ..., 100
combos = list(itertools.product(grid_pts, repeat=4))
print(f'Grid search: {len(combos)} sugar combinations')

samples = [[N1, N2, Sn, L, f1, f2, f3, f4] for f1, f2, f3, f4 in combos]

t0 = time.time()
r = Simulator(
    samples=samples,
    mode='batch', t_final=72.0, grid_size=1,
    omega=0.0, diffusion_scale=0.0,
    device='cuda'
).run()
print(f'Grid search time: {time.time() - t0:.2f}s')

L_vals = r.L_final  # [B] array
best_idx = np.argmax(L_vals)
best_combo = combos[best_idx]

print(f'\n── Grid search results ──')
print(f'Best L_final:    {L_vals[best_idx]:.4f}')
print(f'Best sugar load: F1={best_combo[0]:.0f}, F2={best_combo[1]:.0f}, '
      f'F3={best_combo[2]:.0f}, F4={best_combo[3]:.0f}')
print(f'Total sugar:     {sum(best_combo):.0f}')

# Top 10
top_idx = np.argsort(L_vals)[-10:][::-1]
print(f'\nTop 10 sugar loadings:')
print(f'{"Rank":<5} {"F1":>5} {"F2":>5} {"F3":>5} {"F4":>5} {"Total":>6} {"L_final":>10}')
print('-' * 48)
for rank, idx in enumerate(top_idx, 1):
    c = combos[idx]
    print(f'{rank:<5} {c[0]:>5.0f} {c[1]:>5.0f} {c[2]:>5.0f} {c[3]:>5.0f} '
          f'{sum(c):>6.0f} {L_vals[idx]:>10.4f}')

# ── Step 2: Refine with scipy ───────────────────────────────────────────
from scipy.optimize import minimize

def objective(sugars):
    """Negative L_final (minimise → maximise L)."""
    f1, f2, f3, f4 = sugars
    r = Simulator(
        samples=[[N1, N2, Sn, L, f1, f2, f3, f4]],
        mode='batch', t_final=72.0, grid_size=1,
        omega=0.0, diffusion_scale=0.0,
        device='cuda'
    ).run()
    return -r.L_final

bounds = [(0, 100)] * 4
x0 = list(best_combo)

print(f'\nRefining with L-BFGS-B from grid-search optimum...')
t0 = time.time()
result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                  options={'maxiter': 200, 'ftol': 1e-8})
print(f'Refinement time: {time.time() - t0:.2f}s')

sugars_opt = result.x
print(f'\n── Optimised result ──')
print(f'L_final:     {-result.fun:.6f}')
print(f'Sugar load:  F1={sugars_opt[0]:.2f}, F2={sugars_opt[1]:.2f}, '
      f'F3={sugars_opt[2]:.2f}, F4={sugars_opt[3]:.2f}')
print(f'Total:       {sum(sugars_opt):.2f}')
print(f'Converged:   {result.success}  ({result.message})')

# ── Comparison ──────────────────────────────────────────────────────────
print(f'\n── Comparison with reference configurations ──')
ref_loadings = [
    ([100, 0, 0, 0],        'F1 only'),
    ([0, 100, 0, 0],        'F2 only'),
    ([0, 0, 100, 0],        'F3 only'),
    ([0, 0, 0, 100],        'F4 only'),
    ([25, 25, 25, 25],      'equal mix'),
    ([100, 100, 100, 100],  'all max'),
    (list(sugars_opt),      'optimised'),
]
ref_data = [[N1, N2, Sn, L] + sugars for sugars, _ in ref_loadings]
r_ref = Simulator(
    samples=ref_data,
    mode='batch', t_final=72.0, grid_size=1,
    omega=0.0, diffusion_scale=0.0,
    device='cuda'
).run()
print(f'{"Config":<14} {"F1":>6} {"F2":>6} {"F3":>6} {"F4":>6} {"Total":>6} {"L_final":>10}')
print('-' * 62)
for i, (sugars, label) in enumerate(ref_loadings):
    t = sum(sugars)
    print(f'{label:<14} {sugars[0]:>6.1f} {sugars[1]:>6.1f} {sugars[2]:>6.1f} {sugars[3]:>6.1f} '
          f'{t:>6.1f} {r_ref.L_final[i]:>10.4f}')
