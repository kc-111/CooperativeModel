"""Example: Well-mixed (ODE) bioreactor — lactic acid comparison.

Closed batch reactor with grid_size=1, no diffusion, no advection.
The system reduces to 8 coupled ODEs (N1, N2, Sn, L, F1..F4) with
F1..F4 as the initial sugar loading at t=0, consumed over t_final
hours by the co-culture (CoA + CoB).

Usage:
    python examples/example_ode.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator

# 5 representative local optima from find_optima.py (matches the set used
# in pde_compare.py and gif_optima.py). F1..F4 are initial sugar amounts.
CANDIDATES = [
    ('#1  top INT',     58.59, 91.77, 17.22, 68.32),
    ('#2  far INT',     92.41, 28.85,  7.77, 95.90),
    ('#11 low-F4 INT',  87.41, 94.17, 28.40, 16.79),
    ('#18 top BDY',     66.34, 44.16, 32.92, 99.86),
    ('#20 F1=100 BDY', 100.00, 36.09,  6.25, 75.97),
]
samples = [[0.05, 0.05, 0.0, 0.0, F1, F2, F3, F4]
           for (_, F1, F2, F3, F4) in CANDIDATES]

start_time = time.time()
r = Simulator(
    samples=samples,
    mode='batch', t_final=72.0, grid_size=1,
    U_imp=0.0, diffusion_scale=0.0,
    device='cuda'
).run()
end_time = time.time()

print(f'\nWall time: {end_time - start_time:.2f}s')
print(f'Samples:  {r.n_samples}')

# Final lactic acid per sample
print(f'\n{"Sample":<18} {"L_final":>10} {"Sn_final":>10}')
print('-' * 40)
for i, (label, *_) in enumerate(CANDIDATES):
    print(f'{label:<18} {r.L_final[i]:>10.4f} {r.Sn_final[i]:>10.4f}')

print(f'\nAll final values:')
fv = r.final_values()
for name, vals in fv.items():
    print(f'  {name}: {vals}')
