"""Confirm the impeller body force is non-axisymmetric.

A single-blade angular Gaussian must give different force magnitudes at the
same (r, z) but opposite azimuths.  If a future change accidentally drops
the angular factor (or sets sigma_theta -> infinity), this test fails.
"""

import math
import torch

from CooperativeModel.config import GridConfig
from CooperativeModel.velocity_fields import impeller_body_force


def _force_magnitude_along_blade(F, theta_query, grid, theta_0):
    """Return |f| at (r=r_imp, theta=theta_query, z=z_imp)."""
    cx = grid.Lx * 0.5
    cy = grid.Ly * 0.5
    r_imp = grid.Lx / 4.0
    # Map (r, theta) -> Cartesian cell index.
    x = cx + r_imp * math.cos(theta_query)
    y = cy + r_imp * math.sin(theta_query)
    z_imp_idx = grid.Nz // 2
    ix = min(max(int(x / grid.dx), 0), grid.Nx - 1)
    iy = min(max(int(y / grid.dy), 0), grid.Ny - 1)
    fvec = F[:, z_imp_idx, iy, ix]
    return float(torch.linalg.vector_norm(fvec).item())


def test_blade_localised_in_theta():
    grid = GridConfig(Nx=32, Ny=32, Nz=32, Lx=1.0, Ly=1.0, Lz=1.0)
    F0 = 10.0
    theta_0 = 0.0
    sigma_theta = math.pi / 6
    f = impeller_body_force(grid, F0=F0, theta_0=theta_0,
                            sigma_theta=sigma_theta,
                            device='cpu', dtype=torch.float64)

    on_blade = _force_magnitude_along_blade(f, theta_0,             grid, theta_0)
    opposite = _force_magnitude_along_blade(f, theta_0 + math.pi,   grid, theta_0)

    # On-blade force is order F0; opposite-side force is exp(-(pi)^2/(2 sigma_t^2))
    # smaller — for sigma_theta = pi/6 this ratio is ~exp(-18).
    assert on_blade > 0.5 * F0, f'on-blade |f|={on_blade:.3e} too small'
    assert opposite < 0.05 * F0, (
        f'opposite-side |f|={opposite:.3e} should be ~0 (axisymmetry broken)'
    )
    assert on_blade > 100.0 * (opposite + 1e-30), (
        f'force is not localised in theta: on={on_blade:.3e}, opp={opposite:.3e}'
    )
