"""Velocity field generator for 2D bioreactor stirring.

Steady single-cell mean flow as the curl of a stream function — one
centered circulation cell with no-penetration on every wall.
"""

import math
import torch


def bioreactor_flow(grid_cfg, U_imp=0.5, device='cpu', dtype=torch.float64):
    """Steady single-cell mean flow as the curl of a stream function.

    psi(x, y) = A sin(pi x / Lx) sin(pi y / Ly)
        vx =  d psi/dy =  A (pi / Ly) sin(pi x / Lx) cos(pi y / Ly)
        vy = -d psi/dx = -A (pi / Lx) cos(pi x / Lx) sin(pi y / Ly)

    Topology: one centered circulation cell. psi = 0 on all four walls
    so the field is divergence-free with no-penetration BCs by
    construction. A is rescaled numerically so max |v| equals U_imp.

    Args:
        grid_cfg: GridConfig instance.
        U_imp:    Peak mean-flow speed [cm/h] — the only stirring knob.
                  Set to 0 for a quiescent (no-stirring) reactor.

    Returns:
        [1, 2, Ny, Nx] velocity field (channel 0 = vx, channel 1 = vy).
    """
    Nx, Ny = grid_cfg.Nx, grid_cfg.Ny
    Lx, Ly = grid_cfg.Lx, grid_cfg.Ly
    dx, dy = grid_cfg.dx, grid_cfg.dy

    x = torch.linspace(0.5 * dx, Lx - 0.5 * dx, Nx, device=device, dtype=dtype)
    y = torch.linspace(0.5 * dy, Ly - 0.5 * dy, Ny, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    kx = math.pi / Lx
    ky = math.pi / Ly

    sin_kx = torch.sin(kx * X)
    cos_kx = torch.cos(kx * X)
    sin_ky = torch.sin(ky * Y)
    cos_ky = torch.cos(ky * Y)

    vx = ky * sin_kx * cos_ky
    vy = -kx * cos_kx * sin_ky

    speed = torch.sqrt(vx * vx + vy * vy)
    peak = float(speed.max())
    A = (U_imp / peak) if peak > 0 else 0.0
    return torch.stack([A * vx, A * vy], dim=0).unsqueeze(0)
