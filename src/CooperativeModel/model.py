"""2D bioreactor model combining reaction kinetics, diffusion, and advection.

The PDE system for each field y_k is:
    dy_k/dt = R_k(y) + nabla.(D_k nabla y_k) - nabla.(v y_k)

State tensor shape convention:
    Internal:  [B, 8, H, W]  (batch, channels, height, width)
    Solver:    [B, 8*H*W]    (flat vector for Tsit5)

Channels: [N1, N2, Sn, L, F1, F2, F3, F4]
"""

import torch
from .kinetics import compute_reaction_rates
from .spatial_operators import Divergence, Advection
from .tsit5_solver import Tsit5SolverTorch


def compute_cfl_limit(dx, dy, D_tensor, vel_tensor=None, safety=0.4):
    """Compute the CFL time-step limit for diffusion + advection.

    For explicit time integration, stability requires:
        dt_diffusion < dx^2 / (2 * D_max * ndim)       (von Neumann)
        dt_advection < dx / |v_max|                      (Courant)

    Args:
        dx, dy: Grid spacings [cm].
        D_tensor: [1, 8, 1, 1] diffusion coefficients.
        vel_tensor: Optional [1, 2, H, W] velocity field.
        safety: Safety factor (default 0.4).

    Returns:
        h_max_cfl: Maximum stable time step [h].
    """
    D_max = D_tensor.abs().max().item()
    h_min_grid = min(dx, dy)

    limits = []

    # Diffusion CFL: dt < dx^2 / (2 * D * ndim)
    if D_max > 0:
        dt_diff = h_min_grid ** 2 / (2.0 * D_max * 2.0)  # 2D
        limits.append(dt_diff)

    # Advection CFL: dt < dx / |v_max|
    if vel_tensor is not None:
        v_max = vel_tensor.abs().max().item()
        if v_max > 0:
            dt_adv = h_min_grid / v_max
            limits.append(dt_adv)

    if not limits:
        return 10.0  # no spatial operators, no constraint

    return safety * min(limits)


class BioreactorRHS:
    """Right-hand side function for the 2D bioreactor PDE system.

    Callable with signature ``__call__(t, y_flat, args)`` matching the Tsit5
    solver interface. Reshapes between flat and spatial representations internally.

    Args:
        params: dict from ``ModelParameters.to_tensors()``.
        grid_cfg: ``GridConfig`` instance.
        diffusion_tensor: ``[1, 8, 1, 1]`` diffusion coefficients.
        velocity_field: ``[1, 2, H, W]`` velocity field.
        wall_mask: Optional ``[1, 1, H, W]`` wall mask.
    """

    def __init__(self, params, grid_cfg, diffusion_tensor, velocity_field,
                 wall_mask=None):
        self.params = params
        self.H = grid_cfg.Ny
        self.W = grid_cfg.Nx
        self.D = diffusion_tensor
        self.vel = velocity_field

        self.has_diffusion = diffusion_tensor.abs().max().item() > 0
        self.has_advection = velocity_field.abs().max().item() > 0

        dx, dy = grid_cfg.dx, grid_cfg.dy
        self.divergence = Divergence(dx, dy, wall_mask=wall_mask)
        self.advection = Advection(dx, dy, wall_mask=wall_mask)

    @torch.no_grad()
    def __call__(self, t, y_flat, args=None):
        """
        Args:
            t: Current time (scalar).
            y_flat: [B, 8*H*W] flat state vector.
            args: Unused (interface compatibility with Tsit5).

        Returns:
            [B, 8*H*W] flat time derivative.
        """
        B = y_flat.shape[0]
        y = y_flat.reshape(B, 8, self.H, self.W)

        # Clamp before kinetics to prevent issues from intermediate RK stages
        y = y.clamp(min=0.0)

        # Local reaction kinetics
        dydt = compute_reaction_rates(y, self.params)

        # Diffusion: nabla.(D nabla y)
        if self.has_diffusion:
            dydt += self.divergence(y, self.D)

        # Advection: -nabla.(v y)
        if self.has_advection:
            dydt += self.advection(y, self.vel)

        return dydt.reshape(B, -1)


def simulate(config, initial_state, velocity_field=None):
    """Run a 2D bioreactor simulation.

    Args:
        config: ``SimulationConfig`` instance.
        initial_state: ``[B, 8, Ny, Nx]`` initial condition tensor.
        velocity_field: Optional ``[1, 2, Ny, Nx]`` velocity field
                        (broadcasts over B).  Defaults to zero (pure diffusion).

    Returns:
        results: ``[B, n_output, 8, Ny, Nx]`` solution tensor.
        t_eval:  ``[n_output]`` time points.
    """
    device = config.device
    dtype = config.dtype
    grid = config.grid

    # Convert parameters to tensors
    params = config.model.to_tensors(device=device, dtype=dtype)

    # Diffusion coefficients [1, 8, 1, 1]
    D = config.diffusion.to_tensor(device=device, dtype=dtype)

    # Velocity field
    if velocity_field is None:
        vel = torch.zeros(1, 2, grid.Ny, grid.Nx, device=device, dtype=dtype)
    else:
        vel = velocity_field.to(device=device, dtype=dtype)

    # Initial state → flat
    y0_spatial = initial_state.to(device=device, dtype=dtype)
    B = y0_spatial.shape[0]
    y0_flat = y0_spatial.reshape(B, -1)  # [B, 8*H*W]

    # Build RHS
    rhs = BioreactorRHS(params, grid, D, vel)

    # CFL-limited time step
    scfg = config.solver
    h_cfl = compute_cfl_limit(grid.dx, grid.dy, D, vel)
    h_max = min(scfg.h_max, h_cfl)
    h0 = min(scfg.h0, h_max)

    # Build solver
    solver = Tsit5SolverTorch(
        atol=scfg.atol,
        rtol=scfg.rtol,
        h_max=h_max,
        maxiters=scfg.maxiters,
    )

    # Time evaluation points
    t_eval = torch.linspace(0, scfg.t_final, scfg.n_output,
                            device=device, dtype=dtype)
    t_span = (0.0, scfg.t_final)

    # Solve
    results_flat = solver.solve(rhs, y0_flat, t_span, t_eval,
                                args=None, h0=h0)
    # results_flat: [B, n_output, 8*H*W]

    results = results_flat.reshape(B, len(t_eval), 8, grid.Ny, grid.Nx)
    return results, t_eval
