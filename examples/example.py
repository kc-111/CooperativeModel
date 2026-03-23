"""Example: 2D cooperative bioreactor simulation.

Usage:
    python examples/example.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator

# --- Batch culture ---
# start_time = time.time()
# r = Simulator(N1=0.01, N2=0.1, F1=100.0, omega=-2.0).run()
# end_time = time.time()
# print(f'Batch time: {end_time - start_time:.2f} seconds')
# print(f'Batch:  L={r.L_final:.2f}, Sn={r.Sn_final:.2f}')
# r.gif('batch.gif')

# --- Flow-through reactor ---
start_time = time.time()
r = Simulator(
    N1=0.05, N2=0.05, Sn=0.0, L=0.0, F1=100.0, F2=100.0, F3=100.0, F4=100.0,
    mode='flow_through', t_final=72.0, grid_size=100, 
    omega=-0.25, diffusion_scale=0.1, flow_rate=5.0
).run()
end_time = time.time()
print(f'Flow-through time: {end_time - start_time:.2f} seconds')
print(f'Flow:   L={r.L_final:.2f}, Sn={r.Sn_final:.2f}')
r.gif('flow_through.gif')
