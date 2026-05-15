"""Example: well-mixed (0D-equivalent) bioreactor — lactic acid comparison.

Closed batch reactor with ``grid_shape=(1, 1, 1)``, no diffusion, no
advection (``flow_cache_path=None``).  The system reduces to 9 coupled
ODEs (N1..N4, L, R1..R4) in the 4-species Liebig consumer-resource model.

Each of the four "pair corners" in the (R1, R2, R3, R4) initial space
gives a local maximum of L(t_final) — only the species whose pair P_i is
supplied with both resources can grow, so single-pair supply
configurations are intrinsically separated from all-max.

Usage:
    python examples/example_ode.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator


# Four single-pair corners + the all-max reference.  Resource budget T = 4
# (matches the symmetric symmetric defaults in ModelParameters).  Each
# corner concentrates the budget on one P_i pair and starves the antipodal
# resources, isolating one species' growth.
S_HI = 2.0
S_LO = 0.05
CANDIDATES = [
    # label,                R1,    R2,    R3,    R4
    ('A  pair P_1 (N1)',  S_HI,  S_HI,  S_LO,  S_LO),
    ('B  pair P_2 (N2)',  S_LO,  S_HI,  S_HI,  S_LO),
    ('C  pair P_3 (N3)',  S_LO,  S_LO,  S_HI,  S_HI),
    ('D  pair P_4 (N4)',  S_HI,  S_LO,  S_LO,  S_HI),
    ('all-max (ref)',     1.0,   1.0,   1.0,   1.0),
]

# IC layout: [N1..N4, L, R1..R4, T1..T4]
N0 = 0.01
samples = [[N0, N0, N0, N0, 0.0, R1, R2, R3, R4,
            0.0, 0.0, 0.0, 0.0]
           for (_, R1, R2, R3, R4) in CANDIDATES]

start_time = time.time()
r = Simulator(
    samples=samples,
    t_final=24.0, grid_shape=(1, 1, 1),
    flow_cache_path=None,
    device='cpu',
).run()
end_time = time.time()

print(f'\nWall time: {end_time - start_time:.2f}s')
print(f'Samples:  {r.n_samples}')

print(f'\n{"Sample":<22} {"L_final":>10}')
print('-' * 36)
for i, (label, *_) in enumerate(CANDIDATES):
    print(f'{label:<22} {r.L_final[i]:>10.4f}')

print(f'\nAll final values:')
fv = r.final_values()
for name, vals in fv.items():
    print(f'  {name}: {vals}')
