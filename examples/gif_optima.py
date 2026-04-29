"""Render flow-through GIFs for the 5 representative optima from find_optima.py.

Mirrors examples/example.py (grid_size=100, omega=-0.25, D=0.1, flow_rate=5)
but uses the F-points from find_optima.py instead of the corner cases.
GIFs are written to the current working directory as
``flow_through_sample{0..4}.gif`` — run from the repo root to overwrite the
existing files there.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator


# Same 5 candidates as pde_compare.py
# (label, F1, F2, F3, F4)
CANDIDATES = [
    ('#1  top INT',     58.59, 91.77, 17.22, 68.32),
    ('#2  far INT',     92.41, 28.85,  7.77, 95.90),
    ('#11 low-F4 INT',  87.41, 94.17, 28.40, 16.79),
    ('#18 top BDY',     66.34, 44.16, 32.92, 99.86),
    ('#20 F1=100 BDY', 100.00, 36.09,  6.25, 75.97),
]

samples = [[0.05, 0.05, 0.0, 0.0, F1, F2, F3, F4]
           for (_, F1, F2, F3, F4) in CANDIDATES]

t0 = time.time()
r = Simulator(
    samples=samples,
    mode='flow_through', t_final=72.0, grid_size=100,
    omega=-0.25, diffusion_scale=0.1, flow_rate=5.0,
    device='cuda',
).run()
print(f'Simulation: {time.time() - t0:.1f}s')
print(f'L_final per sample:  {r.L_final}')
print(f'Sn_final per sample: {r.Sn_final}')

for i, (lbl, *_) in enumerate(CANDIDATES):
    out = f'flow_through_sample{i}.gif'
    print(f'  rendering {out}  ({lbl})')
    r.gif(out, sample=i)
