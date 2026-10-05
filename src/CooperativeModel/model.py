"""Reaction and advection on a fixed velocity field.

State shape: [B, 13, Nz, Ny, Nx]; solver states are flattened per sample.
Channels: [N1..N4, L, R1..R4, T1..T4].
"""

import torch

from .kinetics import compute_reaction_rates
from .spatial_operators import Advection
from .tsit5_solver import Tsit5SolverTorch


N_CHANNELS = 13


def compute_cfl_limit(dx, dy, dz, vel_tensor=None, safety=0.4):
    """Return the advection time-step limit [h] from cell sizes and velocity.

    vel_tensor has shape [1, 3, Nz, Ny, Nx]; absent or zero velocity
    returns 10 hours.
    """
    h_min = min(dx, dy, dz)
    if vel_tensor is None:
        return 10.0
    v_max = vel_tensor.abs().max().item()
    if v_max <= 0:
        return 10.0
    return safety * h_min / v_max


class BioreactorRHS:
    """Reaction-advection RHS with internal state reshaping.

    params comes from ModelParameters.to_tensors(); grid_cfg is GridConfig.
    velocity_field has shape [1, 3, Nz, Ny, Nx]. The optional wall_mask
    has shape [1, 1, Nz, Ny, Nx], with 1=wall and 0=fluid.
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
        """``y_flat`` is ``[B, 13*Nz*Ny*Nx]``; returns the same shape."""
        B = y_flat.shape[0]
        y = y_flat.reshape(B, N_CHANNELS, self.Nz, self.Ny, self.Nx)

        # Clamp negative intermediate stages before evaluating kinetics.
        y = y.clamp(min=0.0)

        dydt = compute_reaction_rates(y, self.params)

        if self.has_advection:
            dydt = dydt + self.advection(y, self.vel)

        return dydt.reshape(B, -1)


def simulate(config, initial_state, velocity_field=None, wall_mask=None):
    """Integrate reactions and transport on a fixed velocity field.

    initial_state: [B, 13, Nz, Ny, Nx].
    velocity_field: optional [1, 3, Nz, Ny, Nx]; None uses zero velocity.
    wall_mask: optional [1, 1, Nz, Ny, Nx], with 1=wall and 0=fluid.
    Returns results [B, n_output, 13, Nz, Ny, Nx] and t_eval [n_output].
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
        B, len(t_eval), N_CHANNELS, grid.Nz, grid.Ny, grid.Nx,
    )
    return results, t_eval
