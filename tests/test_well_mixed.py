"""Regression tests that pin Stage-2 transport against well-known limits.

(a) Spec test     : 1x1x1 grid + zero velocity + zero diffusion ≡ 0D ODE.
(b) Locality test : non-uniform IC, no velocity, no diffusion → at every voxel
                    the 3D trajectory equals an independent 0D ODE on that
                    voxel's IC.  Catches kinetics-shape bugs in BioreactorRHS.
(c) Conservation  : uniform IC over fluid cells + cached flow + zero diffusion
                    → species mass change inside the fluid region is bounded
                    by O(rtol) over the integration window.
"""

import math
import os
import torch

from CooperativeModel.config import (
    GridConfig, ModelParameters, SolverConfig, SimulationConfig,
)
from CooperativeModel.kinetics import compute_reaction_rates
from CooperativeModel.model import simulate, BioreactorRHS, N_CHANNELS
from CooperativeModel.initial_conditions import uniform
from CooperativeModel.tsit5_solver import Tsit5SolverTorch


def _make_config(grid, t_final=2.0, n_output=21, rtol=1e-7, atol=1e-9):
    """Build a SimulationConfig with relaxed solver knobs for fast tests."""
    return SimulationConfig(
        model=ModelParameters(),
        grid=grid,
        solver=SolverConfig(t_final=t_final, n_output=n_output,
                            rtol=rtol, atol=atol, h0=0.01),
        device='cpu', dtype=torch.float64,
    )


def _solve_ode(y0, params, t_final, n_output, rtol=1e-9, atol=1e-12):
    """Reference 0D solve: just `compute_reaction_rates` integrated by Tsit5.

    ``y0`` is ``[B, 9]``; returned ``y`` is ``[B, n_output, 9]``.
    """
    B = y0.shape[0]

    def rhs(t, y_flat, args=None):
        y_5d = y_flat.reshape(B, N_CHANNELS, 1, 1, 1).clamp(min=0.0)
        return compute_reaction_rates(y_5d, params).reshape(B, -1)

    solver = Tsit5SolverTorch(atol=atol, rtol=rtol, h_max=1.0,
                              maxiters=1_000_000)
    t_eval = torch.linspace(0.0, t_final, n_output, dtype=torch.float64)
    out = solver.solve(rhs, y0, (0.0, t_final), t_eval, args=None,
                       h0=0.01, progress=False)
    return out  # [B, n_output, 9]


def test_spec_well_mixed_matches_0D():
    """1x1x1 grid, zero velocity, zero diffusion -- must match 0D ODE."""
    grid = GridConfig(Nx=1, Ny=1, Nz=1, Lx=1.0, Ly=1.0, Lz=1.0)
    cfg = _make_config(grid, t_final=2.0, n_output=21)

    ic = uniform(grid, N1=0.01, N2=0.01, N3=0.01, N4=0.01, L=0.0,
                 R1=2.0, R2=2.0, R3=2.0, R4=2.0,
                 device=cfg.device, dtype=cfg.dtype)  # [1, 9, 1, 1, 1]

    results, t_eval = simulate(cfg, ic, velocity_field=None, wall_mask=None)
    # results: [1, n_output, 9, 1, 1, 1]
    pde = results.reshape(1, -1, N_CHANNELS)

    params = cfg.model.to_tensors(device=cfg.device, dtype=cfg.dtype)
    y0 = ic.reshape(1, N_CHANNELS)
    ode = _solve_ode(y0, params, cfg.solver.t_final, cfg.solver.n_output)

    diff = (pde - ode).abs().max().item()
    scale = max(pde.abs().max().item(), ode.abs().max().item(), 1.0)
    rel = diff / scale
    assert rel < 5e-4, (
        f'1x1x1 PDE deviates from 0D ODE: max abs={diff:.3e}, rel={rel:.3e}'
    )


def test_locality_per_voxel_matches_0D():
    """Zero velocity + zero diffusion + non-uniform IC.

    Each voxel must trace its own 0D trajectory.  Catches any leakage
    between voxels in the 3D RHS reshape/kinetics path.
    """
    grid = GridConfig(Nx=4, Ny=4, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0)
    cfg = _make_config(grid, t_final=2.0, n_output=11)

    torch.manual_seed(0)
    ic = torch.zeros(1, N_CHANNELS, grid.Nz, grid.Ny, grid.Nx, dtype=torch.float64)
    # Per-channel non-uniform IC: small biomass on each species, L=0,
    # randomised resource levels.
    for ch in range(4):
        ic[0, ch] = 0.01 + 0.005 * torch.rand_like(ic[0, ch])
    # L starts at 0 (ch 4).
    for ch in (5, 6, 7, 8):
        ic[0, ch] = 0.5 + 1.5 * torch.rand_like(ic[0, ch])

    results, _ = simulate(cfg, ic, velocity_field=None, wall_mask=None)
    # results: [1, n_output, 9, 4, 4, 4]

    # Reference: solve a 0D ODE per voxel in one batched solve.
    n_vox = grid.Nz * grid.Ny * grid.Nx
    y0 = ic.permute(0, 2, 3, 4, 1).reshape(n_vox, N_CHANNELS)
    params = cfg.model.to_tensors(device=cfg.device, dtype=cfg.dtype)
    ode = _solve_ode(y0, params, cfg.solver.t_final, cfg.solver.n_output)
    # ode: [n_vox, n_output, 9] -> [1, n_output, 9, Nz, Ny, Nx]
    ode_grid = (ode.reshape(grid.Nz, grid.Ny, grid.Nx, cfg.solver.n_output, N_CHANNELS)
                   .permute(3, 4, 0, 1, 2)
                   .unsqueeze(0))

    diff = (results - ode_grid).abs().max().item()
    scale = max(results.abs().max().item(), ode_grid.abs().max().item(), 1.0)
    rel = diff / scale
    assert rel < 1e-3, (
        f'per-voxel PDE deviates from per-voxel ODE: max abs={diff:.3e}, '
        f'rel={rel:.3e}'
    )


def test_conservation_uniform_field_zero_diffusion(tmp_path):
    """Uniform IC over fluid cells + cached steady flow + zero diffusion.

    A passive scalar (no kinetics source) advected by the cached flow
    must conserve its total mass inside the fluid region.  The mask-aware
    upwind FV advection zeros face velocities at fluid-wall interfaces, so
    no mass leaks into walls — even though the local divergence error of
    the embedded-boundary projection redistributes the field within fluid.

    Pointwise uniformity is *not* preserved, because the discrete ``nabla
    . v`` carries an O(dx) consistency error at fluid cells adjacent to
    walls (a co-located projection artefact); the physically meaningful
    invariant on a closed vessel is global conservation.
    """
    from CooperativeModel.flow_3d import solve_steady_flow, save_flow, load_flow
    from CooperativeModel.velocity_fields import cylinder_mask

    grid = GridConfig(Nx=12, Ny=12, Nz=12, Lx=1.0, Ly=1.0, Lz=1.0)
    fluid = cylinder_mask(grid, device='cpu', dtype=torch.float64)
    u, v, w, meta = solve_steady_flow(
        grid, mask=fluid, F0=3.0, nu=8e-3,
        tol=1e-2, max_iters=600, pressure_iters=40,
        device='cpu', dtype=torch.float64, progress=False,
    )
    path = os.path.join(str(tmp_path), 'flow.h5')
    save_flow(path, u, v, w, fluid, meta)
    u, v, w, fluid, _ = load_flow(path)

    cfg = _make_config(grid, t_final=2.0, n_output=11, rtol=1e-7, atol=1e-9)

    R_const = 1.0
    # Set biomass to zero so resources have no kinetics source/sink: pure
    # passive scalar transport under advection only.
    ic = uniform(grid, N1=0.0, N2=0.0, N3=0.0, N4=0.0, L=0.0,
                 R1=R_const, R2=R_const, R3=R_const, R4=R_const,
                 mask=fluid, device=cfg.device, dtype=cfg.dtype)

    vel = torch.stack([u, v, w], dim=0).unsqueeze(0)         # [1,3,Nz,Ny,Nx]
    wall_mask = (1.0 - fluid).reshape(1, 1, *fluid.shape)    # 1=wall

    results, _ = simulate(cfg, ic, velocity_field=vel, wall_mask=wall_mask)
    # results: [1, n_output, 9, Nz, Ny, Nx]

    # R1-R4 have no kinetics source in the absence of biomass; their total
    # mass over fluid cells must therefore be conserved by advection alone.
    fluid_b = fluid > 0.5
    for ch in (5, 6, 7, 8):
        m0 = results[0, 0, ch][fluid_b].sum().item()
        for k in range(results.shape[1]):
            mk = results[0, k, ch][fluid_b].sum().item()
            rel_err = abs(mk - m0) / max(abs(m0), 1.0)
            assert rel_err < 5e-4, (
                f'R{ch-4} lost mass at step {k}: '
                f'm0={m0:.6e}, mk={mk:.6e}, rel={rel_err:.3e}'
            )

    # Walls must remain identically zero — the wall_mask zeroing in the RHS
    # plus mask-aware face fluxes guarantee no scalar leaks into walls.
    wall_b = ~fluid_b
    for ch in range(N_CHANNELS):
        max_wall = results[0, :, ch][:, wall_b].abs().max().item()
        assert max_wall < 1e-12, (
            f'channel {ch} leaked into walls: max|c| in wall = {max_wall:.3e}'
        )
