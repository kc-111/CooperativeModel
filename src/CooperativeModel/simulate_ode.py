"""Simple simulator interface for collaborators.

Pass the 8 ODE initial conditions, get results. Everything else is handled.

    from CooperativeModel import Simulator

    # Single sample
    r = Simulator(N1=0.01, N2=0.1, F1=100.0).run()

    # Multiple samples — per-variable arrays
    r = Simulator(N1=[0.01, 0.05, 0.1], N2=[0.1, 0.05, 0.01],
                  F1=100.0).run()

    # Multiple samples — full [B, 8] tensor
    r = Simulator(samples=[[0.01, 0.1, 0, 0, 100, 0, 0, 0],
                           [0.05, 0.05, 0, 0, 100, 0, 0, 0]]).run()

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
    """Results from a simulation. Provides easy access to data and plotting.

    When the simulation has multiple samples (B > 1), scalar properties
    return numpy arrays of shape ``[B]`` and ``spatial_average()`` returns
    ``[B, T, 8]``.  Visualisation methods accept a ``sample`` index
    (default 0).
    """

    def __init__(self, results, t_eval, elapsed, grid_cfg, zone_boxes=None):
        self.results = results    # [B, T, 8, H, W]
        self.t_eval = t_eval      # [T]
        self.elapsed = elapsed    # seconds
        self._grid_cfg = grid_cfg
        self._zone_boxes = zone_boxes

    @property
    def n_samples(self):
        """Number of samples (batch dimension)."""
        return self.results.shape[0]

    @property
    def L_final(self):
        """Spatially-averaged final lactic acid.  Scalar if B=1, [B] array otherwise."""
        vals = self.results[:, -1, 3].mean(dim=(-2, -1))  # [B]
        if self.n_samples == 1:
            return vals.item()
        return vals.detach().cpu().numpy()

    @property
    def Sn_final(self):
        """Spatially-averaged final nisin.  Scalar if B=1, [B] array otherwise."""
        vals = self.results[:, -1, 2].mean(dim=(-2, -1))  # [B]
        if self.n_samples == 1:
            return vals.item()
        return vals.detach().cpu().numpy()

    def final_values(self):
        """Dict of spatially-averaged final values for all 8 channels.

        Values are scalars if B=1, numpy arrays of shape ``[B]`` otherwise.
        """
        vals = self.results[:, -1].mean(dim=(-2, -1))  # [B, 8]
        if self.n_samples == 1:
            return {name: vals[0, i].item()
                    for i, name in enumerate(CHANNEL_NAMES)}
        vals_np = vals.detach().cpu().numpy()
        return {name: vals_np[:, i]
                for i, name in enumerate(CHANNEL_NAMES)}

    def spatial_average(self):
        """Spatially-averaged time series.

        Returns ``[T, 8]`` numpy array if B=1, ``[B, T, 8]`` if B>1.
        """
        avg = self.results.mean(dim=(-2, -1))  # [B, T, 8]
        if self.n_samples == 1:
            return avg[0].detach().cpu().numpy()
        return avg.detach().cpu().numpy()

    def gif(self, path='simulation.gif', curve_channels=None, sample=0):
        """Save all-channel GIF with aggregate time-series curves.

        Args:
            path: Output file path.
            curve_channels: Channels to plot as time series (default: [2, 3]).
            sample: Sample index to visualise (default 0).
        """
        from .visualization import animate_all_fields_with_curves
        if curve_channels is None:
            curve_channels = [2, 3]
        animate_all_fields_with_curves(
            self.results[sample:sample+1], self.t_eval,
            curve_channels=curve_channels,
            zone_boxes=self._zone_boxes,
            save_path=path,
        )

    def snapshot(self, path='snapshot.png', time_idx=-1, sample=0):
        """Save spatial heatmap of all channels at a given time.

        Args:
            path: Output file path.
            time_idx: Index into t_eval.
            sample: Sample index to visualise (default 0).
        """
        from .visualization import plot_snapshot
        import matplotlib.pyplot as plt
        fig = plot_snapshot(self.results[sample:sample+1], self.t_eval,
                           time_idx=time_idx, grid_cfg=self._grid_cfg)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def timeseries(self, path='timeseries.png', sample=0):
        """Save spatially-averaged concentration time series.

        Args:
            path: Output file path.
            sample: Sample index to visualise (default 0).
        """
        from .visualization import plot_spatial_average
        import matplotlib.pyplot as plt
        fig = plot_spatial_average(self.results[sample:sample+1], self.t_eval)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)


class Simulator:
    """2D bioreactor simulator with multi-sample support.

    Args:
        N1, N2, Sn, L, F1, F2, F3, F4: Initial concentrations (ODE variables).
            Each can be a scalar (one sample) or a 1-D list/array/tensor of
            length B (multiple samples).  Scalars are broadcast to match.
            Default feed: F4=100 (maltose), others zero.
        samples: Optional ``[B, 8]`` tensor (or nested list) specifying all 8
            IC values for B samples at once.  Overrides the individual
            N1..F4 parameters when provided.  Column order is
            ``[N1, N2, Sn, L, F1, F2, F3, F4]``.
        mode: 'batch' or 'flow_through' (default: 'flow_through').
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

        # Single sample (original API, fully backward compatible)
        r = Simulator(N1=0.05, N2=0.05, F4=100.0).run()

        # Multiple samples with per-variable arrays
        r = Simulator(N1=[0.01, 0.05, 0.1], N2=[0.1, 0.05, 0.01],
                      F1=100.0, mode='batch', t_final=24.0).run()
        print(r.n_samples)   # 3
        print(r.L_final)     # numpy array of shape [3]

        # Multiple samples with full tensor
        r = Simulator(samples=[[0.01, 0.1, 0, 0, 100, 0, 0, 0],
                               [0.05, 0.05, 0, 0, 0, 0, 0, 100]]).run()
    """

    def __init__(self, N1=0.05, N2=0.05, Sn=0.0, L=0.0,
                 F1=0.0, F2=0.0, F3=0.0, F4=100.0,
                 *, samples=None,
                 mode='flow_through', t_final=72.0, n_output=145, grid_size=100,
                 omega=-0.25, diffusion_scale=0.1,
                 flow_rate=5.0, n_colonies=8, device='cpu'):
        self._ic = self._normalize_ic(N1, N2, Sn, L, F1, F2, F3, F4, samples)
        self.mode = mode
        self.t_final = t_final
        self.n_output = n_output
        self.grid_size = grid_size
        self.omega = omega
        self.diffusion_scale = diffusion_scale
        self.flow_rate = flow_rate
        self.n_colonies = n_colonies
        self.device = device

    @staticmethod
    def _normalize_ic(N1, N2, Sn, L, F1, F2, F3, F4, samples):
        """Convert IC specification to a ``[B, 8]`` CPU float64 tensor.

        Accepts scalars, lists, numpy arrays, and torch tensors (on any
        device).  The result is always stored on CPU; the ``run()`` method
        moves data to the target device.
        """
        if samples is not None:
            s = torch.as_tensor(samples).to(dtype=torch.float64, device='cpu')
            if s.ndim == 1:
                s = s.unsqueeze(0)
            if s.ndim != 2 or s.shape[1] != 8:
                raise ValueError(
                    f"samples must be [B, 8] or [8]; got shape {list(s.shape)}")
            return s

        vals = [N1, N2, Sn, L, F1, F2, F3, F4]
        # Determine B (length-1 inputs are treated as scalars)
        B = 1
        for v in vals:
            if not isinstance(v, (int, float)):
                n = len(v) if not isinstance(v, torch.Tensor) else v.numel()
                if n <= 1:
                    continue
                if B == 1:
                    B = n
                elif n != B:
                    raise ValueError(
                        f"Array IC parameters must all have the same length; "
                        f"got {B} and {n}")

        result = torch.zeros(B, 8, dtype=torch.float64)
        for i, v in enumerate(vals):
            t = torch.as_tensor(v).to(dtype=torch.float64, device='cpu').flatten()
            if t.numel() == 1:
                result[:, i] = t.item()
            else:
                result[:, i] = t
        return result

    @property
    def n_samples(self):
        """Number of samples (batch dimension)."""
        return self._ic.shape[0]

    @property
    def ic(self):
        """Dict of IC values.  Scalars when B=1, tensors when B>1."""
        names = ['N1', 'N2', 'Sn', 'L', 'F1', 'F2', 'F3', 'F4']
        if self.n_samples == 1:
            return {n: self._ic[0, i].item() for i, n in enumerate(names)}
        return {n: self._ic[:, i] for i, n in enumerate(names)}

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

        B = self.n_samples
        label = f'{B} sample{"s" if B > 1 else ""}'
        print(f'Simulation ({self.grid_size}x{self.grid_size}, '
              f'{self.t_final}h, batch, {label}): {elapsed:.1f}s')
        return SimResults(results, t_eval, elapsed, grid)

    def _run_flow_through(self):
        config, grid = self._make_config()
        dtype = torch.float64
        H, W = grid.Ny, grid.Nx
        B = self.n_samples
        ic_vals = self._ic  # [B, 8]

        vel = rigid_body_vortex(grid, omega=self.omega if self.omega != 0.0 else -2.0,
                                device=self.device, dtype=dtype)

        # Inlet/outlet corner zones
        corner = max(grid.Nx // 10, 4)
        inlet_mask = torch.zeros(1, 1, H, W, device=self.device, dtype=dtype)
        inlet_mask[:, :, H - corner:, :corner] = 1.0
        outlet_mask = torch.zeros(1, 1, H, W, device=self.device, dtype=dtype)
        outlet_mask[:, :, :corner, W - corner:] = 1.0

        # Feed: nutrients only (from the IC nutrient values, per sample)
        c_feed = torch.zeros(B, 8, 1, 1, device=self.device, dtype=dtype)
        c_feed[:, 4, 0, 0] = ic_vals[:, 4]  # F1
        c_feed[:, 5, 0, 0] = ic_vals[:, 5]  # F2
        c_feed[:, 6, 0, 0] = ic_vals[:, 6]  # F3
        c_feed[:, 7, 0, 0] = ic_vals[:, 7]  # F4

        # IC: random bacterial colonies + trace nutrients (independent per sample)
        y0 = random_inoculation(
            grid, n_colonies=self.n_colonies,
            N1_amount=ic_vals[:, 0], N2_amount=ic_vals[:, 1],
            F1=ic_vals[:, 4] * 0.05, F2=ic_vals[:, 5] * 0.05,
            F3=ic_vals[:, 6] * 0.05, F4=ic_vals[:, 7] * 0.05,
            colony_radius=grid.Lx * 0.06,
            n_samples=B,
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
        y0_flat = y0.reshape(B, -1)
        t_eval = torch.linspace(0, config.solver.t_final, config.solver.n_output,
                                dtype=dtype)
        tsit5 = Tsit5SolverTorch(atol=config.solver.atol, rtol=config.solver.rtol,
                                  h_max=h_max)

        t0 = time.time()
        results_flat = tsit5.solve(rhs, y0_flat, (0.0, config.solver.t_final),
                                   t_eval, h0=h0)
        elapsed = time.time() - t0

        results = results_flat.reshape(B, len(t_eval), 8, H, W)
        label = f'{B} sample{"s" if B > 1 else ""}'
        print(f'Simulation ({self.grid_size}x{self.grid_size}, '
              f'{self.t_final}h, flow_through, {label}): {elapsed:.1f}s')

        zone_boxes = [
            {'xy': (0, H - corner), 'width': corner, 'height': corner,
             'color': 'red', 'label': 'IN'},
            {'xy': (W - corner, 0), 'width': corner, 'height': corner,
             'color': 'red', 'label': 'OUT'},
        ]
        return SimResults(results, t_eval, elapsed, grid, zone_boxes=zone_boxes)
