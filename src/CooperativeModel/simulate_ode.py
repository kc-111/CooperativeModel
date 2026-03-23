"""Simple simulator interface for collaborators.

Pass the 8 ODE initial conditions, get results. Everything else is handled.

    from CooperativeModel import Simulator

    # Batch culture
    r = Simulator(N1=0.01, N2=0.1, F1=100.0).run()

    # Flow-through reactor
    r = Simulator(N1=0.01, N2=0.1, F1=100.0,
                  mode='flow_through', t_final=72.0, omega=-2.0).run()

    print(r.L_final, r.elapsed)
    r.gif('output.gif')
"""

import time
import torch

from .config import SimulationConfig, GridConfig, SolverConfig, DiffusionConfig
from .initial_conditions import uniform, random_inoculation
from .velocity_fields import rigid_body_vortex
from .model import BioreactorRHS, simulate, compute_cfl_limit
from .tsit5_solver import Tsit5SolverTorch


CHANNEL_NAMES = ['N1', 'N2', 'Sn', 'L', 'F1', 'F2', 'F3', 'F4']


class _FlowThroughRHS:
    """Internal: wraps BioreactorRHS with inlet/outlet source-sink terms."""

    def __init__(self, base_rhs, inlet_mask, outlet_mask, c_feed, flow_rate, H, W):
        self.base_rhs = base_rhs
        self.inlet_mask = inlet_mask
        self.outlet_mask = outlet_mask
        self.c_feed = c_feed
        self.flow_rate = flow_rate
        self.H, self.W = H, W

    @torch.no_grad()
    def __call__(self, t, y_flat, args=None):
        B = y_flat.shape[0]
        dydt = self.base_rhs(t, y_flat, args).reshape(B, 8, self.H, self.W)
        y = y_flat.reshape(B, 8, self.H, self.W).clamp(min=0.0)
        dydt += self.flow_rate * (self.c_feed - y) * self.inlet_mask
        dydt -= self.flow_rate * y * self.outlet_mask
        return dydt.reshape(B, -1)


class SimResults:
    """Results from a simulation. Provides easy access to data and plotting."""

    def __init__(self, results, t_eval, elapsed, grid_cfg, zone_boxes=None):
        self.results = results    # [1, T, 8, H, W]
        self.t_eval = t_eval      # [T]
        self.elapsed = elapsed    # seconds
        self._grid_cfg = grid_cfg
        self._zone_boxes = zone_boxes

    @property
    def L_final(self):
        return self.results[0, -1, 3].mean().item()

    @property
    def Sn_final(self):
        return self.results[0, -1, 2].mean().item()

    def final_values(self):
        """Dict of spatially-averaged final values for all 8 channels."""
        return {name: self.results[0, -1, i].mean().item()
                for i, name in enumerate(CHANNEL_NAMES)}

    def spatial_average(self):
        """Spatially-averaged time series as numpy array [T, 8]."""
        return self.results[0].mean(dim=(-2, -1)).detach().cpu().numpy()

    def gif(self, path='simulation.gif', curve_channels=None):
        """Save all-channel GIF with aggregate time-series curves."""
        from .visualization import animate_all_fields_with_curves
        if curve_channels is None:
            curve_channels = [2, 3]
        animate_all_fields_with_curves(
            self.results, self.t_eval,
            curve_channels=curve_channels,
            zone_boxes=self._zone_boxes,
            save_path=path,
        )

    def snapshot(self, path='snapshot.png', time_idx=-1):
        """Save spatial heatmap of all channels at a given time."""
        from .visualization import plot_snapshot
        import matplotlib.pyplot as plt
        fig = plot_snapshot(self.results, self.t_eval, time_idx=time_idx,
                           grid_cfg=self._grid_cfg)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def timeseries(self, path='timeseries.png'):
        """Save spatially-averaged concentration time series."""
        from .visualization import plot_spatial_average
        import matplotlib.pyplot as plt
        fig = plot_spatial_average(self.results, self.t_eval)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)


class Simulator:
    """2D bioreactor simulator.

    Args:
        N1, N2, Sn, L, F1, F2, F3, F4: Initial concentrations (ODE variables).
            Default feed: F4=100 (maltose), others zero.
        mode: 'batch' or 'flow_through' (default: 'flow_through').
            - batch: closed system, no inflow/outflow.
            - flow_through: fresh medium (nutrients only) fed at upper-left
              corner, products drained at lower-right corner.
        t_final: Integration time [hours]. Default 72.
        n_output: Number of output time points. Default 145.
        grid_size: Spatial grid points per side. Default 100.
        omega: Vortex angular velocity [rad/h]. Default -0.25 (clockwise).
        diffusion_scale: Multiplier on all diffusion coefficients.
            Default 0.1 (10x slower than physical). 1.0 = physical.
        flow_rate: Turnover rate in inlet/outlet zones [h^-1].
            Only used in flow_through mode. Default 5.0.
        n_colonies: Number of random bacterial colonies for flow_through IC.
        device: 'cpu' or 'cuda'.

    Examples::

        r = Simulator(N1=0.05, N2=0.05, F4=100.0).run()
        r.gif('flow_through.gif')

        # Batch culture
        r = Simulator(N1=0.01, N2=0.1, F1=100.0, mode='batch',
                      t_final=24.0, grid_size=50, omega=-2.0).run()
    """

    def __init__(self, N1=0.05, N2=0.05, Sn=0.0, L=0.0,
                 F1=0.0, F2=0.0, F3=0.0, F4=100.0,
                 mode='flow_through', t_final=72.0, n_output=145, grid_size=100,
                 omega=-0.25, diffusion_scale=0.1,
                 flow_rate=5.0, n_colonies=8, device='cpu'):
        self.ic = dict(N1=N1, N2=N2, Sn=Sn, L=L, F1=F1, F2=F2, F3=F3, F4=F4)
        self.mode = mode
        self.t_final = t_final
        self.n_output = n_output
        self.grid_size = grid_size
        self.omega = omega
        self.diffusion_scale = diffusion_scale
        self.flow_rate = flow_rate
        self.n_colonies = n_colonies
        self.device = device

    def run(self):
        """Execute the simulation. Returns SimResults."""
        if self.mode == 'batch':
            return self._run_batch()
        elif self.mode == 'flow_through':
            return self._run_flow_through()
        else:
            raise ValueError(f"mode must be 'batch' or 'flow_through', got '{self.mode}'")

    def _make_config(self):
        diff = DiffusionConfig()
        if self.diffusion_scale != 1.0:
            s = self.diffusion_scale
            diff = DiffusionConfig(
                D_N1=diff.D_N1*s, D_N2=diff.D_N2*s, D_Sn=diff.D_Sn*s, D_L=diff.D_L*s,
                D_F1=diff.D_F1*s, D_F2=diff.D_F2*s, D_F3=diff.D_F3*s, D_F4=diff.D_F4*s,
            )
        grid = GridConfig(Nx=self.grid_size, Ny=self.grid_size)
        solver = SolverConfig(t_final=self.t_final, n_output=self.n_output)
        config = SimulationConfig(grid=grid, solver=solver, diffusion=diff,
                                  device=self.device, dtype=torch.float64)
        return config, grid

    def _run_batch(self):
        config, grid = self._make_config()
        dtype = torch.float64

        y0 = uniform(grid, device=self.device, dtype=dtype, **self.ic)
        vel = (rigid_body_vortex(grid, omega=self.omega, device=self.device, dtype=dtype)
               if self.omega != 0.0 else None)

        t0 = time.time()
        results, t_eval = simulate(config, y0, velocity_field=vel)
        elapsed = time.time() - t0

        print(f'Simulation ({self.grid_size}x{self.grid_size}, '
              f'{self.t_final}h, batch): {elapsed:.1f}s')
        return SimResults(results, t_eval, elapsed, grid)

    def _run_flow_through(self):
        config, grid = self._make_config()
        dtype = torch.float64
        H, W = grid.Ny, grid.Nx

        vel = rigid_body_vortex(grid, omega=self.omega if self.omega != 0.0 else -2.0,
                                device=self.device, dtype=dtype)

        # Inlet/outlet corner zones
        corner = max(grid.Nx // 10, 4)
        inlet_mask = torch.zeros(1, 1, H, W, dtype=dtype)
        inlet_mask[:, :, H - corner:, :corner] = 1.0
        outlet_mask = torch.zeros(1, 1, H, W, dtype=dtype)
        outlet_mask[:, :, :corner, W - corner:] = 1.0

        # Feed: nutrients only (from the IC nutrient values)
        c_feed = torch.zeros(1, 8, 1, 1, dtype=dtype)
        c_feed[0, 4] = self.ic['F1']
        c_feed[0, 5] = self.ic['F2']
        c_feed[0, 6] = self.ic['F3']
        c_feed[0, 7] = self.ic['F4']

        # IC: random bacterial colonies + trace nutrients
        y0 = random_inoculation(
            grid, n_colonies=self.n_colonies,
            N1_amount=self.ic['N1'], N2_amount=self.ic['N2'],
            F1=self.ic['F1'] * 0.05, F2=self.ic['F2'] * 0.05,
            F3=self.ic['F3'] * 0.05, F4=self.ic['F4'] * 0.05,
            colony_radius=grid.Lx * 0.06,
            device=self.device, dtype=dtype,
        )

        # Build RHS
        params = config.model.to_tensors(device=self.device, dtype=dtype)
        D = config.diffusion.to_tensor(device=self.device, dtype=dtype)
        base_rhs = BioreactorRHS(params, grid, D, vel)
        rhs = _FlowThroughRHS(base_rhs, inlet_mask, outlet_mask, c_feed,
                               self.flow_rate, H, W)

        # CFL-limited time step
        h_cfl = compute_cfl_limit(grid.dx, grid.dy, D, vel)
        h_max = min(config.solver.h_max, h_cfl)
        h0 = min(config.solver.h0, h_max)

        # Solve
        y0_flat = y0.reshape(1, -1)
        t_eval = torch.linspace(0, config.solver.t_final, config.solver.n_output,
                                dtype=dtype)
        tsit5 = Tsit5SolverTorch(atol=config.solver.atol, rtol=config.solver.rtol,
                                  h_max=h_max)

        t0 = time.time()
        results_flat = tsit5.solve(rhs, y0_flat, (0.0, config.solver.t_final),
                                   t_eval, h0=h0)
        elapsed = time.time() - t0

        results = results_flat.reshape(1, len(t_eval), 8, H, W)
        print(f'Simulation ({self.grid_size}x{self.grid_size}, '
              f'{self.t_final}h, flow_through): {elapsed:.1f}s')

        zone_boxes = [
            {'xy': (0, H - corner), 'width': corner, 'height': corner,
             'color': 'red', 'label': 'IN'},
            {'xy': (W - corner, 0), 'width': corner, 'height': corner,
             'color': 'red', 'label': 'OUT'},
        ]
        return SimResults(results, t_eval, elapsed, grid, zone_boxes=zone_boxes)
