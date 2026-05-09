"""CLI to solve the steady 3D bioreactor flow once and cache to HDF5.

Usage::

    python scripts/solve_flow.py --out flow_cache.h5            # 32^3, defaults
    python scripts/solve_flow.py --grid 16 16 16 --out small.h5 # smoke

The Stage-2 species transport / BO loop loads this cache via
``CooperativeModel.flow_3d.load_flow`` and never re-solves the flow.
"""

import argparse
import math
import os
import sys
import time

# Allow running as a plain script from the repo root.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
)

import torch

from CooperativeModel.config import GridConfig
from CooperativeModel.flow_3d import save_flow, solve_steady_flow
from CooperativeModel.velocity_fields import cylinder_mask


def _parse_dtype(s):
    return {'float32': torch.float32, 'float64': torch.float64}[s]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', type=int, nargs=3, default=[32, 32, 32],
                   metavar=('NX', 'NY', 'NZ'),
                   help='Grid points (default 32 32 32).')
    p.add_argument('--extent', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   metavar=('LX', 'LY', 'LZ'),
                   help='Domain size in cm (default 1.0 1.0 1.0).')
    p.add_argument('--F0', type=float, default=10.0,
                   help='Peak body-force magnitude.')
    p.add_argument('--r-imp', type=float, default=None,
                   help='Radial blade position (default Lx/4).')
    p.add_argument('--z-imp', type=float, default=None,
                   help='Axial blade position (default Lz/2).')
    p.add_argument('--sigma-r', type=float, default=None,
                   help='Radial Gaussian width (default Lx/16).')
    p.add_argument('--sigma-z', type=float, default=None,
                   help='Axial Gaussian width (default Lz/16).')
    p.add_argument('--theta-0', type=float, default=0.0,
                   help='Blade azimuth [rad] (default 0).')
    p.add_argument('--sigma-theta', type=float, default=math.pi / 6,
                   help='Angular blade width [rad] (default pi/6).')
    p.add_argument('--nu', type=float, default=1e-3,
                   help='Kinematic viscosity (default 1e-3).')
    p.add_argument('--dt', type=float, default=None,
                   help='Pseudo-time step (default CFL-derived).')
    p.add_argument('--tol', type=float, default=1e-4,
                   help='Steady-state convergence tolerance.')
    p.add_argument('--max-iters', type=int, default=20000,
                   help='Maximum outer iterations.')
    p.add_argument('--pressure-iters', type=int, default=200,
                   help='Inner red-black GS sweeps per outer step.')
    p.add_argument('--device', type=str, default='cpu',
                   help='Torch device (default cpu).')
    p.add_argument('--dtype', type=_parse_dtype, default='float64',
                   help='float32 or float64 (default float64).')
    p.add_argument('--out', type=str, default='flow_cache.h5',
                   help='Output HDF5 path.')
    p.add_argument('--no-progress', action='store_true',
                   help='Disable tqdm progress bars.')
    args = p.parse_args()

    grid = GridConfig(
        Nx=args.grid[0], Ny=args.grid[1], Nz=args.grid[2],
        Lx=args.extent[0], Ly=args.extent[1], Lz=args.extent[2],
    )
    mask = cylinder_mask(grid, device=args.device, dtype=args.dtype)

    print(f'Solving NS on {grid.Nx}x{grid.Ny}x{grid.Nz} '
          f'(extent {grid.Lx} x {grid.Ly} x {grid.Lz} cm)')
    t0 = time.time()
    u, v, w, metadata = solve_steady_flow(
        grid, mask=mask,
        F0=args.F0,
        r_imp=args.r_imp, z_imp=args.z_imp,
        sigma_r=args.sigma_r, sigma_z=args.sigma_z,
        theta_0=args.theta_0, sigma_theta=args.sigma_theta,
        nu=args.nu, dt=args.dt, tol=args.tol,
        max_iters=args.max_iters, pressure_iters=args.pressure_iters,
        device=args.device, dtype=args.dtype,
        progress=not args.no_progress,
    )
    elapsed = time.time() - t0
    metadata['elapsed_s'] = float(elapsed)

    save_flow(args.out, u, v, w, mask, metadata)
    print(f"\nDone in {elapsed:.1f}s.  "
          f"Iters: {metadata['n_iters']}.  "
          f"Final residual: {metadata['converged_residual']:.3e}.")
    print(f"Cache written to {args.out}")


if __name__ == '__main__':
    main()
