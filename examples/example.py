"""Example: 3D fermentation batch bioreactor on a cached steady flow.

Runs one representative local optimum (from ``find_optima.py``) through
the closed cylindrical reactor (no inlet, no outlet — a sealed batch
fermentation).  The flow field is loaded from ``flow_cache.h5``
produced once by ``scripts/solve_flow.py``.

Channels: [N1..N4, L, R1..R4, T1..T4].

Usage:
    python scripts/solve_flow.py --out flow_cache.h5     # once
    python examples/find_optima.py                       # well-mixed search
    python examples/example.py                           # 3D render
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator


# One of the four pair-corner optima from the well-mixed Liebig
# consumer-resource model.  Pair P_1 = {R1, R2}: load R1 and R2 high,
# R3 and R4 low; species N_1 grows and dominates lactate production.
S_HI = 2.0
S_LO = 0.05
N0 = 0.01

CANDIDATES = [
    ('A  pair P_1 (N1)',  S_HI,  S_HI,  S_LO,  S_LO),
    # ('B  pair P_2 (N2)',  S_LO,  S_HI,  S_HI,  S_LO),
    # ('C  pair P_3 (N3)',  S_LO,  S_LO,  S_HI,  S_HI),
    # ('D  pair P_4 (N4)',  S_HI,  S_LO,  S_LO,  S_HI),
]
samples = [[N0, N0, N0, N0, 0.0, R1, R2, R3, R4, 0.0, 0.0, 0.0, 0.0]
           for (_, R1, R2, R3, R4) in CANDIDATES]

FLOW_CACHE = 'flow_cache.h5'
if not os.path.isfile(FLOW_CACHE):
    raise SystemExit(
        f"flow cache '{FLOW_CACHE}' not found. "
        f"Run: python scripts/solve_flow.py --out {FLOW_CACHE}"
    )

start = time.time()
r = Simulator(
    samples=samples,
    t_final=24.0, grid_shape=(32, 32, 32),
    flow_cache_path=FLOW_CACHE,
    ic_mode='octant', ic_octant=(1, 1, 1),
    device='cuda',
).run()
print(f'\n3D PDE on cached flow: {time.time() - start:.1f}s')
print(f'  L_final per sample:  {r.L_final}')

for i in range(r.n_samples):
    r.gif(f'sample{i}_midz.gif',     sample=i, view='midz')
    r.gif(f'sample{i}_vertical.gif', sample=i, view='midy')
    r.gif(f'sample{i}_topdown.gif',  sample=i, view='topdown')
