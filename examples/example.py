"""Example: 3D fermentation batch bioreactor on a cached steady flow.

Runs one representative local optimum (from ``find_optima.py``) through
the closed cylindrical reactor (no inlet, no outlet — a sealed batch
fermentation).  The flow field is loaded from ``flow_cache.h5``
produced once by ``scripts/solve_flow.py``; this script never re-solves
the flow.

Conservation argument (well-mixed -> 3D)
----------------------------------------
``find_optima.py`` performs the BO/optimisation in the well-mixed limit
(grid_shape=(1, 1, 1)) where each evaluation is an 8-ODE solve.  The 3D
PDE evaluates *the same kinetics* at every fluid cell, so the well-mixed
optimum F* is a local optimum of the 3D objective up to two finite-size
corrections:

  1. Volumetric dilution from the octant IC.  ``ic_mode='octant'`` puts
     all cells + sugars into 1/8 of the cylinder, so the fluid-averaged
     L_final reported by ``r.L_final`` is approximately
     (1/8) x L_well_mixed in the perfect-mixing-after-startup limit.
  2. Advective dieback.  Cells advected out of the active octant before
     sugar arrives there see g1_total = 0 and microbial death dt1 = 0.39
     (Sn = 0 outside the active octant, no nisin protection yet), so they
     decay roughly as exp(-0.39 t) until sugar reaches them via the
     impeller's chaotic streamlines.  Net: viable biomass at late times
     is below the well-mixed value.

The local-optimum *ranking* is preserved (the 4D landscape shape doesn't
change), but absolute fluid-mean L is lower.  To recover well-mixed L
values in the 3D run, switch ``ic_mode='uniform'`` (cells + sugars
distributed everywhere from t=0) — that's the IC ``find_optima.py``
implicitly assumes.

Three GIFs are rendered with the wall boundary contoured:
    sample0_midz.gif     — mid-z (xy) slice through the impeller plane
    sample0_vertical.gif — mid-y vertical (xz) slice; full reactor height
    sample0_topdown.gif  — z-aggregated fluid mean (top-down view)

Usage:
    python scripts/solve_flow.py --out flow_cache.h5     # once
    python examples/find_optima.py                       # well-mixed search
    python examples/example.py                           # 3D render
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator


CANDIDATES = [
    # #1 well-mixed local optimum from scripts/find_optima_wellmixed.py.
    # F3 ≈ 0 escapes per-sugar toxicity (K_tox_3 = 55 is the tightest) while
    # F1, F2, F4 stay high enough to drive sustained growth + L production.
    ('#1  starve-F3',   59.02, 84.94, 0.47, 85.34),
    # ('#2  starve-F1',    5.72, 80.10, 92.78, 77.21),
    # ('#3  low-F1 INT',   9.42, 97.56, 76.11, 78.61),
    # ('#4  starve-F2',   43.89,  2.16, 82.63, 89.62),
    # ('#5  low-F4 INT', 96.32, 78.20, 86.68, 11.41),
]
samples = [[0.05, 0.05, 0.0, 0.0, F1, F2, F3, F4]
           for (_, F1, F2, F3, F4) in CANDIDATES]

FLOW_CACHE = 'flow_cache.h5'
if not os.path.isfile(FLOW_CACHE):
    raise SystemExit(
        f"flow cache '{FLOW_CACHE}' not found. "
        f"Run: python scripts/solve_flow.py --out {FLOW_CACHE}"
    )

start = time.time()
r = Simulator(
    samples=samples,
    t_final=48.0, grid_shape=(32, 32, 32),
    flow_cache_path=FLOW_CACHE,
    ic_mode='octant', ic_octant=(1, 1, 1),
    device='cuda',
).run()
print(f'\n3D PDE on cached flow: {time.time() - start:.1f}s')
print(f'  L_final per sample:  {r.L_final}')
print(f'  Sn_final per sample: {r.Sn_final}')

for i in range(r.n_samples):
    r.gif(f'sample{i}_midz.gif',     sample=i, view='midz')
    r.gif(f'sample{i}_vertical.gif', sample=i, view='midy')   # XZ slice
    r.gif(f'sample{i}_topdown.gif',  sample=i, view='topdown')
