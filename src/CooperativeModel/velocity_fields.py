"""3D bioreactor geometry and impeller body-force.

Stage-1 utilities consumed by ``flow_3d.solve_steady_flow``:

  - ``cylinder_mask(grid)``        — fluid/wall mask for a cylinder inscribed
                                     in the (x, y) cross-section of the cube,
                                     with z = 0 and z = Nz - 1 end-caps marked
                                     as walls.
  - ``impeller_body_force(...)``   — non-axisymmetric azimuthal body force
                                     after Pericleous & Patel (1987).  A single
                                     localised blade at ``theta_0`` breaks
                                     theta-symmetry to unlock chaotic
                                     Lagrangian streamlines.
  - ``azimuthal_unit(grid)``       — Cartesian theta-hat at every cell centre.

Shape conventions:
    Scalar fields :  [Nz, Ny, Nx]
    Vector fields :  [3, Nz, Ny, Nx]  with channels [vx, vy, vz]
"""

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
    """Cylinder fluid/wall mask, [Nz, Ny, Nx].

    Cylinder axis is z; the cylinder is inscribed in the (x, y) cross-section
    of the cube (radius = min(Lx, Ly) / 2, centred at (Lx/2, Ly/2)).  The
    z = 0 and z = Nz - 1 end-cap planes are also marked as wall.

    Args:
        grid: ``GridConfig`` instance.
        device, dtype: Torch placement.

    Returns:
        mask: ``[Nz, Ny, Nx]`` float tensor; 1.0 = fluid, 0.0 = wall.
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
    # Also force the four side faces of the bounding box (j=0, j=Ny-1,
    # i=0, i=Nx-1) to wall.  With Lx = Ly and radius = 0.5*Lx, a few
    # otherwise-fluid cells near (y, x) = (Ly/2, 0) and similar corners
    # touch the box edge.  The replicate padding in the projection chain
    # (``_div(_grad(p))``) silently collapses the +y / -y face contribution
    # at those cells, leaving the operator rank-deficient and breaking the
    # discrete identity ``div_open(grad_open(p)) = lap_open(p)``.  Excluding
    # them removes the issue at negligible volumetric cost (≤ 1% of fluid
    # cells at 32^3) and means the species advection's open-face MAC
    # divergence agrees with the flow projection at every fluid cell.
    box_edge = (
        (torch.arange(grid.Ny, device=device).reshape(1, -1, 1) == 0)
        | (torch.arange(grid.Ny, device=device).reshape(1, -1, 1) == grid.Ny - 1)
        | (torch.arange(grid.Nx, device=device).reshape(1, 1, -1) == 0)
        | (torch.arange(grid.Nx, device=device).reshape(1, 1, -1) == grid.Nx - 1)
    )
    mask = (inside_cyl & ~end_cap & ~box_edge).to(dtype)
    return mask


def azimuthal_unit(grid, device='cpu', dtype=torch.float64):
    """Cartesian theta-hat at every cell centre, [3, Nz, Ny, Nx].

    Channel order is [tx, ty, tz] = [-sin(theta), cos(theta), 0].
    The radial origin is the cylinder axis (Lx/2, Ly/2).
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
    """Non-axisymmetric impeller body force, [3, Nz, Ny, Nx].

    Force model (Pericleous & Patel 1987-style azimuthal momentum source,
    with theta-localisation to break axisymmetry):

        f(r, z, theta)  =  F0 * chi_rz(r, z) * chi_theta(theta) * theta_hat

    where::

        chi_rz(r, z)    = exp(-((r - r_imp)^2 / (2 sigma_r^2)
                              +  (z - z_imp)^2 / (2 sigma_z^2)))
        chi_theta(t)    = exp(-d_circ(t, theta_0)^2 / (2 sigma_theta^2))
        d_circ(a, b)    = min(|a - b|, 2*pi - |a - b|)   # periodic distance

    A single localised blade at ``theta_0`` (width ``sigma_theta``) breaks the
    rotational symmetry of an axisymmetric toroidal forcing.  In a steady NS
    solve this produces non-axisymmetric mean flow whose Lagrangian
    streamlines are chaotic — characteristic of real stirred tanks.

    Args:
        grid: ``GridConfig``.
        F0: Peak body-force magnitude [cm/h^2].
        r_imp: Radial position of the blade.  Default ``Lx / 4``.
        z_imp: Axial position of the blade.  Default ``Lz / 2``.
        sigma_r: Radial Gaussian width.  Default ``Lx / 16``.
        sigma_z: Axial Gaussian width.  Default ``Lz / 16``.
        theta_0: Azimuth of the blade [rad], default 0.
        sigma_theta: Angular Gaussian width [rad], default pi/6.

    Returns:
        f: ``[3, Nz, Ny, Nx]`` body-force vector field; channels [fx, fy, fz].
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
