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

# Each row is [N1, N2, Sn, L, F1, F2, F3, F4] — initial conditions at t=0.
# F1..F4 are the starting sugar amounts in the sealed reactor.
samples = [
    [0.05, 0.05, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0],    # F1 only
    [0.05, 0.05, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0],    # F2 only
    [0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0],    # F3 only
    [0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],    # F4 only
    [0.05, 0.05, 0.0, 0.0, 25.0, 25.0, 25.0, 25.0],  # equal mix
]

start_time = time.time()
r = Simulator(
    samples=samples,
    mode='batch', t_final=72.0, grid_size=1,
    omega=0.0, diffusion_scale=0.0,
    device='cuda'
).run()
end_time = time.time()

print(f'\nWall time: {end_time - start_time:.2f}s')
print(f'Samples:  {r.n_samples}')

# Final lactic acid per sample
sugar_labels = ['F1 only', 'F2 only', 'F3 only', 'F4 only', 'equal mix']
print(f'\n{"Sample":<12} {"L_final":>10} {"Sn_final":>10}')
print('-' * 34)
for i, label in enumerate(sugar_labels):
    print(f'{label:<12} {r.L_final[i]:>10.4f} {r.Sn_final[i]:>10.4f}')

print(f'\nAll final values:')
fv = r.final_values()
for name, vals in fv.items():
    print(f'  {name}: {vals}')
