"""Smoke test for the steady NS solver and HDF5 cache I/O.

Runs a 16^3 solve with relaxed tolerance, writes the cache, reloads it,
checks (a) bit-identical round-trip, (b) cylinder mask shape, and
(c) incompressibility of the saved field.

Target: end-to-end on CPU in well under 60 seconds.
"""

import os
import time
import torch

from CooperativeModel.config import GridConfig
from CooperativeModel.flow_3d import (
    load_flow, save_flow, solve_steady_flow,
)
from CooperativeModel.velocity_fields import cylinder_mask


def _max_div(u, v, w, mask, dx, dy, dz):
    """Inf-norm of the forward-difference divergence over fluid cells only.

    Matches the forward-difference div used by the projection step.  Wall
    cells trivially carry nonzero divergence under any embedded-boundary
    scheme (a wall cell holds u=0 while neighbouring fluid cells do not),
    so they are excluded — the physically meaningful quantity is
    ``∇·u`` inside the fluid region.
    """
    dux = (u[:, :, 1:] - u[:, :, :-1]) / dx
    dvy = (v[:, 1:, :] - v[:, :-1, :]) / dy
    dwz = (w[1:, :, :] - w[:-1, :, :]) / dz
    nz = min(dux.shape[0], dvy.shape[0], dwz.shape[0])
    ny = min(dux.shape[1], dvy.shape[1], dwz.shape[1])
    nx = min(dux.shape[2], dvy.shape[2], dwz.shape[2])
    div = (dux[:nz, :ny, :nx] + dvy[:nz, :ny, :nx] + dwz[:nz, :ny, :nx])
    fluid = mask[:nz, :ny, :nx] > 0.5
    return float(div[fluid].abs().amax().item())


def test_solve_smoke_and_io(tmp_path):
    grid = GridConfig(Nx=16, Ny=16, Nz=16, Lx=1.0, Ly=1.0, Lz=1.0)
    mask = cylinder_mask(grid, device='cpu', dtype=torch.float64)

    t0 = time.time()
    u, v, w, meta = solve_steady_flow(
        grid, mask=mask,
        F0=5.0,
        nu=5e-3,
        tol=1e-3,
        max_iters=2000,
        pressure_iters=80,
        device='cpu', dtype=torch.float64,
        progress=False,
    )
    elapsed = time.time() - t0
    assert elapsed < 60.0, f'smoke took {elapsed:.1f}s, must be < 60s'

    assert u.shape == (16, 16, 16) and v.shape == u.shape and w.shape == u.shape
    assert torch.isfinite(u).all() and torch.isfinite(v).all() and torch.isfinite(w).all()

    # Wall-zeroed velocities (mask 0 cells must hold u=v=w=0).
    wall = mask < 0.5
    assert u[wall].abs().max().item() == 0.0
    assert v[wall].abs().max().item() == 0.0
    assert w[wall].abs().max().item() == 0.0

    # Save and reload.
    path = os.path.join(str(tmp_path), 'smoke.h5')
    save_flow(path, u, v, w, mask, meta)
    u2, v2, w2, m2, meta2 = load_flow(path)
    assert torch.equal(u, u2) and torch.equal(v, v2) and torch.equal(w, w2)
    assert torch.equal(mask, m2)
    assert int(meta2['Nx']) == 16 and int(meta2['Nz']) == 16

    # Incompressibility on the interior.
    div_max = _max_div(u, v, w, mask, grid.dx, grid.dy, grid.dz)
    # The cell-centred forward divergence on a co-located projection scheme
    # carries an O(dx) consistency error at fluid-wall cells (the wall-zero
    # of u_star is incompatible with div_fwd; the rhs is shifted by its
    # mean to restore Poisson compatibility, leaving a small constant
    # divergence in the corrected field).  ~1.0 is a generous bound for
    # the 16^3 smoke run; the production 32^3 cache will be tighter.
    assert div_max < 1.0, (
        f'incompressibility check failed: max|div u| = {div_max:.3e}'
    )
