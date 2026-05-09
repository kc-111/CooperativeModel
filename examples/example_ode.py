"""Example: well-mixed (0D-equivalent) bioreactor — lactic acid comparison.

Closed batch reactor with ``grid_shape=(1, 1, 1)``, no diffusion, no
advection (``flow_cache_path=None``).  The system reduces to 8 coupled
ODEs (N1, N2, Sn, L, F1..F4).

Usage:
    python examples/example_ode.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator


CANDIDATES = [
    # Top 5 local optima from scripts/find_optima_wellmixed.py with the
    # split-death (microbial + chemical) kinetics.  Each near-edge config
    # drops one sugar to escape per-sugar toxicity (K_tox_3 = 55 is the
    # tightest, so F3 ≈ 0 wins).
    ('#1  starve-F3',    59.02, 84.94,  0.47, 85.34),   # L ~ 61.21
    ('#2  starve-F1',     5.72, 80.10, 92.78, 77.21),   # L ~ 60.65
    ('#3  low-F1 INT',    9.42, 97.56, 76.11, 78.61),   # L ~ 59.90
    ('#4  starve-F2',    43.89,  2.16, 82.63, 89.62),   # L ~ 59.45
    ('#5  low-F4 INT',   96.32, 78.20, 86.68, 11.41),   # L ~ 57.85
]
samples = [[0.05, 0.05, 0.0, 0.0, F1, F2, F3, F4]
           for (_, F1, F2, F3, F4) in CANDIDATES]

start_time = time.time()
r = Simulator(
    samples=samples,
    t_final=72.0, grid_shape=(1, 1, 1),
    flow_cache_path=None,
    device='cuda',
).run()
end_time = time.time()

print(f'\nWall time: {end_time - start_time:.2f}s')
print(f'Samples:  {r.n_samples}')

print(f'\n{"Sample":<18} {"L_final":>10} {"Sn_final":>10}')
print('-' * 40)
for i, (label, *_) in enumerate(CANDIDATES):
    print(f'{label:<18} {r.L_final[i]:>10.4f} {r.Sn_final[i]:>10.4f}')

print(f'\nAll final values:')
fv = r.final_values()
for name, vals in fv.items():
    print(f'  {name}: {vals}')
