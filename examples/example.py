"""Example: 2D cooperative bioreactor simulation.

Usage:
    python examples/example.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator

# --- Multi-sample flow-through using samples tensor ---
# Each row is [N1, N2, Sn, L, F1, F2, F3, F4]
samples = [
    [0.05, 0.05, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
    [0.05, 0.05, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0],
    [0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0],
    [0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
    [0.05, 0.05, 0.0, 0.0, 25.0, 25.0, 25.0, 25.0],
]

start_time = time.time()
r = Simulator(
    samples=samples,
    mode='flow_through', t_final=72.0, grid_size=100,
    omega=-0.25, diffusion_scale=0.1, flow_rate=5.0,
    device='cuda'
).run()
print(r.final_values()) # This contains the final values for each sample
end_time = time.time()
print(f'Flow-through time: {end_time - start_time:.2f} seconds')
print(f'Samples: {r.n_samples}')
print(f'L_final per sample:  {r.L_final}')
print(f'Sn_final per sample: {r.Sn_final}')

# Visualise each sample
for i in range(r.n_samples):
    r.gif(f'flow_through_sample{i}.gif', sample=i)
