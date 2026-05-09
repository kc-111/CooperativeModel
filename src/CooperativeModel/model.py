"""3D bioreactor model combining reaction kinetics and advection.

The PDE system for each species y_k is

    dy_k/dt = R_k(y) - nabla.(v y_k)

State tensor shape convention:

    Internal :  [B, 8, Nz, Ny, Nx]   (batch, channels, depth, height, width)
    Solver   :  [B, 8 * Nz * Ny * Nx] (flat vector for Tsit5)

Channels: ``[N1, N2, Sn, L, F1, F2, F3, F4]``.  The velocity field is treated
as fixed (cached from Stage 1); the well-mixed limit is recovered by passing
zero velocity through ``simulate``.

Mixing is driven entirely by chaotic advection from the non-axisymmetric
impeller — there is no explicit sub-grid diffusion term.  First-order upwind
on the advection step provides the modest numerical diffusion needed to keep
sharp fronts well-resolved on the 32^3 grid.
"""

import torch

from .kinetics import compute_reaction_rates
from .spatial_operators import Advection
from .tsit5_solver import Tsit5SolverTorch


def compute_cfl_limit(dx, dy, dz, vel_tensor=None, safety=0.4):
    """CFL-limited time step for explicit advection in 3D.

    Args:
        dx, dy, dz: cell sizes.
        vel_tensor: optional ``[1, 3, Nz, Ny, Nx]`` velocity field.
        safety: safety factor (default 0.4).

    Returns:
        h_max_cfl: maximum stable time step [h].
    """
    h_min = min(dx, dy, dz)
    if vel_tensor is None:
        return 10.0
    v_max = vel_tensor.abs().max().item()
    if v_max <= 0:
        return 10.0
    return safety * h_min / v_max


class BioreactorRHS:
    """RHS callable for the 3D bioreactor PDE.

    Signature ``__call__(t, y_flat, args)`` matches the Tsit5 solver
    interface; reshaping between flat and spatial layouts is internal.

    Args:
        params: dict from ``ModelParameters.to_tensors()``.
        grid_cfg: ``GridConfig`` instance.
        velocity_field:   ``[1, 3, Nz, Ny, Nx]`` velocity field.
        wall_mask:        optional ``[1, 1, Nz, Ny, Nx]`` wall mask
                          (1 = wall, 0 = fluid).
    """

    def __init__(self, params, grid_cfg, velocity_field, wall_mask=None):
        self.params = params
        self.Nz = grid_cfg.Nz
        self.Ny = grid_cfg.Ny
        self.Nx = grid_cfg.Nx
        self.vel = velocity_field

        self.has_advection = velocity_field.abs().max().item() > 0

        dx, dy, dz = grid_cfg.dx, grid_cfg.dy, grid_cfg.dz
        self.advection = Advection(dx, dy, dz, wall_mask=wall_mask)

    @torch.no_grad()
    def __call__(self, t, y_flat, args=None):
        """``y_flat`` is ``[B, 8*Nz*Ny*Nx]``; returns the same shape."""
        B = y_flat.shape[0]
        y = y_flat.reshape(B, 8, self.Nz, self.Ny, self.Nx)

        # Clamp non-negative before kinetics; intermediate RK stages can
        # otherwise produce small negative concentrations that feed back into
        # the rate laws.
        y = y.clamp(min=0.0)

        dydt = compute_reaction_rates(y, self.params)

        if self.has_advection:
            dydt = dydt + self.advection(y, self.vel)

        return dydt.reshape(B, -1)


def simulate(config, initial_state, velocity_field=None, wall_mask=None):
    """Run a 3D bioreactor simulation on a fixed velocity field.

    Args:
        config: ``SimulationConfig`` instance.
        initial_state: ``[B, 8, Nz, Ny, Nx]`` initial condition tensor.
        velocity_field: optional ``[1, 3, Nz, Ny, Nx]`` velocity field
                        (broadcasts over the batch).  ``None`` ⇒ zero
                        velocity (well-mixed / 0D limit).
        wall_mask:      optional ``[1, 1, Nz, Ny, Nx]`` wall mask
                        (1 = wall, 0 = fluid).

    Returns:
        results: ``[B, n_output, 8, Nz, Ny, Nx]``.
        t_eval:  ``[n_output]``.
    """
    device = config.device
    dtype = config.dtype
    grid = config.grid

    params = config.model.to_tensors(device=device, dtype=dtype)

    if velocity_field is None:
        vel = torch.zeros(1, 3, grid.Nz, grid.Ny, grid.Nx,
                          device=device, dtype=dtype)
    else:
        vel = velocity_field.to(device=device, dtype=dtype)

    y0_spatial = initial_state.to(device=device, dtype=dtype)
    B = y0_spatial.shape[0]
    y0_flat = y0_spatial.reshape(B, -1)

    rhs = BioreactorRHS(params, grid, vel, wall_mask=wall_mask)

    scfg = config.solver
    h_cfl = compute_cfl_limit(grid.dx, grid.dy, grid.dz, vel)
    h_max = min(scfg.h_max, h_cfl)
    h0 = min(scfg.h0, h_max)

    solver = Tsit5SolverTorch(
        atol=scfg.atol,
        rtol=scfg.rtol,
        h_max=h_max,
        maxiters=scfg.maxiters,
    )

    t_eval = torch.linspace(0, scfg.t_final, scfg.n_output,
                            device=device, dtype=dtype)
    t_span = (0.0, scfg.t_final)

    results_flat = solver.solve(rhs, y0_flat, t_span, t_eval,
                                args=None, h0=h0)
    results = results_flat.reshape(
        B, len(t_eval), 8, grid.Nz, grid.Ny, grid.Nx,
    )
    return results, t_eval
