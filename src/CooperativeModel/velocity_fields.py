"""Velocity field generator for 2D bioreactor stirring.

Rigid-body-like rotation derived from a stream function, so it is
automatically divergence-free and satisfies no-penetration BCs.
"""

import torch


def rigid_body_vortex(grid_cfg, omega=1.0, device='cpu', dtype=torch.float64):
    """Rigid-body-like rotation with no-penetration boundary conditions.

    Stream function:  psi = A * (a^2 - u^2) * (b^2 - v^2)

    where u = x - Lx/2, v = y - Ly/2, a = Lx/2, b = Ly/2.  This vanishes on
    all four walls (divergence-free, no-penetration).

    Near the centre the velocity field is:
        vx = -omega * (y - Ly/2)
        vy =  omega * (x - Lx/2)
    i.e. rigid-body rotation with angular velocity omega.  Speed increases
    linearly with distance from the centre, then smoothly drops to zero at
    the walls.

    Args:
        grid_cfg: GridConfig instance.
        omega: Angular velocity [rad/h].  Positive = counter-clockwise,
               negative = clockwise.

    Returns:
        [1, 2, Ny, Nx] velocity field.
    """
    Nx, Ny = grid_cfg.Nx, grid_cfg.Ny
    Lx, Ly = grid_cfg.Lx, grid_cfg.Ly
    dx, dy = grid_cfg.dx, grid_cfg.dy

    a, b = Lx / 2.0, Ly / 2.0
    A = omega / (2.0 * a * b)

    x = torch.linspace(0.5 * dx, Lx - 0.5 * dx, Nx, device=device, dtype=dtype)
    y = torch.linspace(0.5 * dy, Ly - 0.5 * dy, Ny, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    u = X - a
    v = Y - b

    vx = -2.0 * A * (a**2 - u**2) * v
    vy = 2.0 * A * u * (b**2 - v**2)

    return torch.stack([vx, vy], dim=0).unsqueeze(0)
