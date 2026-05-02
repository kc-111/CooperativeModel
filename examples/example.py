"""Example: 2D PDE simulation of the cooperative bioreactor.

Runs the 5 representative local optima from find_optima.py through both
PDE modes:

  - mode='batch'        : closed 50x50 reactor with stirring + diffusion.
                          Uniform IC, no inlet/outlet. The ranking of
                          L_final closely matches the well-mixed ODE
                          (Spearman ρ ≈ 0.9).

  - mode='flow_through' : open 50x50 reactor with stirring, diffusion,
                          and inlet (top-left) / outlet (bottom-right)
                          patches. Bacteria are seeded as random colonies
                          at low local sugar; the inlet feeds at the IC
                          F-values; the outlet drains. Steady-state L is
                          much lower than batch (washout regime over 72h)
                          and the ranking does NOT match ODE — this mode
                          demonstrates 2D advection-reaction-diffusion
                          dynamics, not equilibrium yield.

Usage:
    python examples/example.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import torch
from CooperativeModel import Simulator

CANDIDATES = [
    ('#1  top INT',     58.59, 91.77, 17.22, 68.32),
    ('#2  far INT',     92.41, 28.85,  7.77, 95.90),
    ('#11 low-F4 INT',  87.41, 94.17, 28.40, 16.79),
    ('#18 top BDY',     66.34, 44.16, 32.92, 99.86),
    ('#20 F1=100 BDY', 100.00, 36.09,  6.25, 75.97),
]
samples = [[0.05, 0.05, 0.0, 0.0, F1 * 0.1, F2 * 0.1, F3 * 0.1, F4 * 0.1]
           for (_, F1, F2, F3, F4) in CANDIDATES]

U_imp = 0.5

# --- PDE flow-through (open reactor with inlet/outlet) — for visualisation ---
start = time.time()
r_flow = Simulator(
    samples=samples,
    mode='flow_through', t_final=48.0, grid_size=32,
    U_imp=U_imp, diffusion_scale=0.1, flow_rate=0.05,
    device='cuda',
).run()
print(f'\nPDE flow-through: {time.time() - start:.1f}s')
print(f'  L_final per sample:  {r_flow.L_final}')
print(f'  Sn_final per sample: {r_flow.Sn_final}')

# Visualise each sample (flow-through has the more interesting 2D dynamics)
for i in range(r_flow.n_samples):
    r_flow.gif(f'flow_through_sample{i}.gif', sample=i)
