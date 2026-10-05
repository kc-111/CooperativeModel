"""Cylindrical fluid mask and prescribed Gaussian stirring force."""

import math
import torch


def _cell_centres(grid, device, dtype):
    """Return (X, Y, Z) cell-centre coordinates, each [Nz, Ny, Nx]."""
    xs = (torch.arange(grid.Nx, device=device, dtype=dtype) + 0.5) * grid.dx
    ys = (torch.arange(grid.Ny, device=device, dtype=dtype) + 0.5) * grid.dy
    zs = (torch.arange(grid.Nz, device=device, dtype=dtype) + 0.5) * grid.dz
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing='ij')
    return X, Y, Z


def cylinder_mask(grid, device='cpu', dtype=torch.float64):
    """Return a [Nz, Ny, Nx] mask with 1 for fluid and 0 for walls.

    The cylinder is centered at (Lx/2, Ly/2), with radius min(Lx, Ly)/2.
    End caps and box-edge cells are walls.
    """
    X, Y, _ = _cell_centres(grid, device=device, dtype=dtype)
    cx = grid.Lx * 0.5
    cy = grid.Ly * 0.5
    radius = 0.5 * min(grid.Lx, grid.Ly)
    r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    inside_cyl = r <= radius
    # End-caps as walls (top and bottom planes).
    end_cap = (
        (torch.arange(grid.Nz, device=device).reshape(-1, 1, 1) == 0)
        | (torch.arange(grid.Nz, device=device).reshape(-1, 1, 1) == grid.Nz - 1)
    )
    # Exclude box-edge cells to keep projection and transport stencils consistent.
    box_edge = (
        (torch.arange(grid.Ny, device=device).reshape(1, -1, 1) == 0)
        | (torch.arange(grid.Ny, device=device).reshape(1, -1, 1) == grid.Ny - 1)
        | (torch.arange(grid.Nx, device=device).reshape(1, 1, -1) == 0)
        | (torch.arange(grid.Nx, device=device).reshape(1, 1, -1) == grid.Nx - 1)
    )
    mask = (inside_cyl & ~end_cap & ~box_edge).to(dtype)
    return mask


def azimuthal_unit(grid, device='cpu', dtype=torch.float64):
    """Return tangential directions [-sin(theta), cos(theta), 0].

    The axis is at (Lx/2, Ly/2); output shape is [3, Nz, Ny, Nx].
    """
    X, Y, _ = _cell_centres(grid, device=device, dtype=dtype)
    cx = grid.Lx * 0.5
    cy = grid.Ly * 0.5
    dx = X - cx
    dy = Y - cy
    r = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-30)
    cos_t = dx / r
    sin_t = dy / r
    tx = -sin_t
    ty = cos_t
    tz = torch.zeros_like(tx)
    return torch.stack([tx, ty, tz], dim=0)


def impeller_body_force(
    grid,
    F0=10.0,
    r_imp=None,
    z_imp=None,
    sigma_r=None,
    sigma_z=None,
    theta_0=0.0,
    sigma_theta=math.pi / 6,
    device='cpu',
    dtype=torch.float64,
):
    """Return a stationary Gaussian tangential force, [3, Nz, Ny, Nx].

    F0 is the peak acceleration [cm/h^2]. The radial and vertical centers
    are r_imp and z_imp; sigma_r and sigma_z are their widths.
    The angular center and width are theta_0 and sigma_theta [rad].
    Defaults: r_imp=Lx/4, z_imp=Lz/2, sigma_r=Lx/16, sigma_z=Lz/16.
    """
    if r_imp is None:
        r_imp = 0.25 * grid.Lx
    if z_imp is None:
        z_imp = 0.5 * grid.Lz
    if sigma_r is None:
        sigma_r = grid.Lx / 16.0
    if sigma_z is None:
        sigma_z = grid.Lz / 16.0

    X, Y, Z = _cell_centres(grid, device=device, dtype=dtype)
    cx = grid.Lx * 0.5
    cy = grid.Ly * 0.5
    dx_c = X - cx
    dy_c = Y - cy
    r = torch.sqrt(dx_c * dx_c + dy_c * dy_c).clamp(min=1e-30)
    theta = torch.atan2(dy_c, dx_c)  # in (-pi, pi]

    chi_rz = torch.exp(
        -(((r - r_imp) ** 2) / (2.0 * sigma_r ** 2)
          + ((Z - z_imp) ** 2) / (2.0 * sigma_z ** 2))
    )

    # Periodic distance on the circle.
    delta = torch.remainder(theta - theta_0 + math.pi, 2.0 * math.pi) - math.pi
    chi_theta = torch.exp(-(delta ** 2) / (2.0 * sigma_theta ** 2))

    chi = chi_rz * chi_theta  # [Nz, Ny, Nx]

    that = azimuthal_unit(grid, device=device, dtype=dtype)  # [3, Nz, Ny, Nx]
    f = F0 * chi.unsqueeze(0) * that  # [3, Nz, Ny, Nx]
    return f
