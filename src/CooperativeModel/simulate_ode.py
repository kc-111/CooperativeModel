"""Stage-2 simulator: species transport on a cached steady flow.

The flow field is produced once by ``scripts/solve_flow.py`` and reloaded
from HDF5; ``Simulator(...).run()`` is the BO-loop entry point and never
re-solves the flow.  Pass ``flow_cache_path=None`` to recover the
well-mixed (0D-equivalent) limit by running with zero velocity and a
single-cell grid, which is what ``examples/example_ode.py`` does.

Channel ordering is ``[N1, N2, Sn, L, F1, F2, F3, F4]``::

    from CooperativeModel import Simulator

    # Multi-sample run on the cached 32^3 flow.
    r = Simulator(
        N1=[0.05]*5, N2=[0.05]*5,
        F1=[10, 25, 50, 75, 100], F2=25.0, F3=25.0, F4=25.0,
        grid_shape=(32, 32, 32),
        flow_cache_path='flow_cache.h5',
        t_final=48.0,
    ).run()
    print(r.L_final)
    r.gif('sample0.gif', sample=0)
"""

import time
import torch

from .config import (
    SimulationConfig, GridConfig, SolverConfig,
)
from .flow_3d import load_flow
from .initial_conditions import uniform, octant as octant_ic
from .model import simulate


CHANNEL_NAMES = ['N1', 'N2', 'Sn', 'L', 'F1', 'F2', 'F3', 'F4']


class SimResults:
    """Results from a 3D simulation.

    Internal state tensor: ``[B, T, 8, Nz, Ny, Nx]``.  Spatial averages and
    finals are taken **only over fluid cells** (the wall mask is honoured),
    so wall voxels do not dilute the reported means.

    For B > 1 scalar properties return numpy arrays of shape ``[B]``;
    ``spatial_average()`` returns ``[B, T, 8]``.  Visualisation methods
    accept a ``sample`` index (default 0) and render a mid-z slice with
    the wall boundary overlaid as a contour.
    """

    def __init__(self, results, t_eval, elapsed, grid_cfg, fluid_mask):
        self.results = results       # [B, T, 8, Nz, Ny, Nx]
        self.t_eval = t_eval         # [T]
        self.elapsed = elapsed       # seconds
        self._grid_cfg = grid_cfg
        # fluid_mask: [Nz, Ny, Nx] float (1=fluid, 0=wall) or None for 1x1x1.
        if fluid_mask is None:
            Nz, Ny, Nx = grid_cfg.Nz, grid_cfg.Ny, grid_cfg.Nx
            fluid_mask = torch.ones(Nz, Ny, Nx, dtype=results.dtype,
                                    device=results.device)
        self._fluid = fluid_mask
        self._fluid_count = float(fluid_mask.sum().item())

    @property
    def n_samples(self):
        """Number of samples (batch dimension)."""
        return self.results.shape[0]

    def _fluid_mean(self, field):
        """Mean of ``field`` (any leading shape, trailing ``[Nz, Ny, Nx]``)
        over fluid cells only."""
        m = self._fluid
        # Broadcast m over leading dims.
        return (field * m).sum(dim=(-3, -2, -1)) / max(self._fluid_count, 1.0)

    @property
    def L_final(self):
        """Fluid-averaged final lactic acid.  Scalar if B=1, ``[B]`` otherwise."""
        vals = self._fluid_mean(self.results[:, -1, 3])  # [B]
        if self.n_samples == 1:
            return vals.item()
        return vals.detach().cpu().numpy()

    @property
    def Sn_final(self):
        """Fluid-averaged final nisin.  Scalar if B=1, ``[B]`` otherwise."""
        vals = self._fluid_mean(self.results[:, -1, 2])
        if self.n_samples == 1:
            return vals.item()
        return vals.detach().cpu().numpy()

    def final_values(self):
        """Dict of fluid-averaged final values for all 8 channels."""
        vals = self._fluid_mean(self.results[:, -1])  # [B, 8]
        if self.n_samples == 1:
            return {name: vals[0, i].item()
                    for i, name in enumerate(CHANNEL_NAMES)}
        vals_np = vals.detach().cpu().numpy()
        return {name: vals_np[:, i]
                for i, name in enumerate(CHANNEL_NAMES)}

    def spatial_average(self):
        """Fluid-averaged time series.

        Returns ``[T, 8]`` numpy if B=1, ``[B, T, 8]`` if B>1.
        """
        avg = self._fluid_mean(self.results)  # [B, T, 8]
        if self.n_samples == 1:
            return avg[0].detach().cpu().numpy()
        return avg.detach().cpu().numpy()

    def _midz_view(self, sample):
        """Return a ``[1, T, 8, Ny, Nx]`` mid-z slice for visualisation."""
        Nz = self.results.shape[-3]
        zmid = Nz // 2
        slc = self.results[sample:sample + 1, :, :, zmid]  # [1, T, 8, Ny, Nx]
        mask2d = self._fluid[zmid].detach().cpu().numpy()
        return slc, mask2d

    def _midy_view(self, sample):
        """Return a ``[1, T, 8, Nz, Nx]`` mid-y vertical slice (XZ plane).

        This is a vertical cross-section through the cylinder axis at
        ``y = Ly/2``: rows are z (height), columns are x.
        """
        Ny = self.results.shape[-2]
        ymid = Ny // 2
        slc = self.results[sample:sample + 1, :, :, :, ymid]  # [1,T,8,Nz,Nx]
        mask2d = self._fluid[:, ymid].detach().cpu().numpy()
        return slc, mask2d

    def _midx_view(self, sample):
        """Return a ``[1, T, 8, Nz, Ny]`` mid-x vertical slice (YZ plane).

        This is a vertical cross-section through the cylinder axis at
        ``x = Lx/2``: rows are z (height), columns are y.
        """
        Nx = self.results.shape[-1]
        xmid = Nx // 2
        slc = self.results[sample:sample + 1, :, :, :, :, xmid]  # [1,T,8,Nz,Ny]
        mask2d = self._fluid[:, :, xmid].detach().cpu().numpy()
        return slc, mask2d

    def _topdown_view(self, sample):
        """Return a ``[1, T, 8, Ny, Nx]`` z-aggregated view (fluid-mean
        along the cylinder axis) plus the 2-D footprint mask."""
        m = self._fluid                                               # [Nz, Ny, Nx]
        col_count = m.sum(dim=0).clamp(min=1.0)                       # [Ny, Nx]
        # Weighted sum along z then divide by column count.
        weighted = self.results[sample:sample + 1] * m                # broadcasts over [B,T,8,Nz,Ny,Nx]
        proj = weighted.sum(dim=-3) / col_count                       # [1, T, 8, Ny, Nx]
        footprint = (m.sum(dim=0) > 0).to(dtype=m.dtype).detach().cpu().numpy()
        return proj, footprint

    def _ortho_views(self, sample):
        """Return mid-z, mid-y, mid-x slices and their 2-D wall masks.

        Each slice is shaped ``[1, T, 8, H, W]`` for visualisation; the
        masks are 2-D arrays.
        """
        Nz, Ny, Nx = self.results.shape[-3], self.results.shape[-2], self.results.shape[-1]
        zmid, ymid, xmid = Nz // 2, Ny // 2, Nx // 2
        # Mid-z: (Ny, Nx) plane  -> already (..., Nz=zmid, Ny, Nx)
        slz = self.results[sample:sample + 1, :, :, zmid]                  # [1,T,8,Ny,Nx]
        sly = self.results[sample:sample + 1, :, :, :, ymid]               # [1,T,8,Nz,Nx]
        slx = self.results[sample:sample + 1, :, :, :, :, xmid]            # [1,T,8,Nz,Ny]
        mz = self._fluid[zmid].detach().cpu().numpy()                      # [Ny, Nx]
        my = self._fluid[:, ymid].detach().cpu().numpy()                   # [Nz, Nx]
        mx = self._fluid[:, :, xmid].detach().cpu().numpy()                # [Nz, Ny]
        return (slz, sly, slx), (mz, my, mx)

    def gif(self, path='simulation.gif', curve_channels=None, sample=0,
            view='midz'):
        """Save an animated GIF with fluid-averaged time-series curves.

        Args:
            path: output file path.
            curve_channels: channels to plot as time series (default: all 8,
                automatically split into a biomass+products panel and a
                sugars panel so each is auto-scaled to its own data range).
            sample: sample index to visualise.
            view: ``'midz'`` (default) renders the mid-z (xy) slice;
                  ``'midy'`` renders the mid-y vertical (xz) slice through
                  the impeller — full reactor height visible;
                  ``'midx'`` renders the mid-x vertical (yz) slice;
                  ``'topdown'`` renders a z-aggregated fluid-mean view
                  (top-down looking down the cylinder axis);
                  ``'ortho'`` renders mid-z + mid-y + mid-x side-by-side
                  per channel.
        """
        if curve_channels is None:
            curve_channels = list(range(8))
        means_full = self._fluid_mean(self.results[sample:sample + 1])     # [1, T, 8]
        means_np = means_full[0].detach().cpu().numpy()

        if view == 'midz':
            from .visualization import animate_all_fields_with_curves
            slc, mask2d = self._midz_view(sample)
            animate_all_fields_with_curves(
                slc, self.t_eval,
                curve_channels=curve_channels,
                mask2d=mask2d,
                curve_means=means_np,
                title_prefix='mid-z (xy)',
                save_path=path,
            )
        elif view == 'midy':
            from .visualization import animate_all_fields_with_curves
            slc, mask2d = self._midy_view(sample)
            animate_all_fields_with_curves(
                slc, self.t_eval,
                curve_channels=curve_channels,
                mask2d=mask2d,
                curve_means=means_np,
                title_prefix='mid-y (xz, vertical)',
                save_path=path,
            )
        elif view == 'midx':
            from .visualization import animate_all_fields_with_curves
            slc, mask2d = self._midx_view(sample)
            animate_all_fields_with_curves(
                slc, self.t_eval,
                curve_channels=curve_channels,
                mask2d=mask2d,
                curve_means=means_np,
                title_prefix='mid-x (yz, vertical)',
                save_path=path,
            )
        elif view == 'topdown':
            from .visualization import animate_all_fields_with_curves
            slc, mask2d = self._topdown_view(sample)
            animate_all_fields_with_curves(
                slc, self.t_eval,
                curve_channels=curve_channels,
                mask2d=mask2d,
                curve_means=means_np,
                title_prefix='z-mean',
                save_path=path,
            )
        elif view == 'ortho':
            from .visualization import animate_orthoviews
            slices, masks = self._ortho_views(sample)
            animate_orthoviews(
                slices, masks, self.t_eval,
                curve_channels=curve_channels,
                curve_means=means_np,
                save_path=path,
            )
        else:
            raise ValueError(
                f"view must be 'midz', 'midy', 'midx', 'topdown', "
                f"or 'ortho'; got '{view}'"
            )

    def snapshot(self, path='snapshot.png', time_idx=-1, sample=0):
        """Save mid-z-slice heatmap of all channels at a given time."""
        from .visualization import plot_snapshot
        import matplotlib.pyplot as plt
        slc, mask2d = self._midz_view(sample)
        fig = plot_snapshot(slc, self.t_eval, time_idx=time_idx,
                            grid_cfg=self._grid_cfg, mask2d=mask2d)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def timeseries(self, path='timeseries.png', sample=0):
        """Save fluid-averaged concentration time series."""
        from .visualization import plot_spatial_average
        import matplotlib.pyplot as plt
        means_full = self._fluid_mean(self.results[sample:sample + 1])
        fig = plot_spatial_average(means_full[0].detach().cpu().numpy(),
                                   self.t_eval)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)


class Simulator:
    """3D bioreactor simulator with multi-sample support.

    Args:
        N1, N2, Sn, L, F1, F2, F3, F4: per-channel initial concentrations
            (uniform over fluid cells).  Each can be a scalar or a 1-D
            sequence/tensor of length B (multi-sample run).
        samples: optional ``[B, 8]`` tensor (or nested list) of full
            initial conditions; overrides the per-channel arguments.
        t_final: integration time [hours].  Default 72.
        n_output: number of output time points.  Default 145.
        grid_shape: ``(Nz, Ny, Nx)``.  Default ``(32, 32, 32)``.  Use
            ``(1, 1, 1)`` together with ``flow_cache_path=None`` for the
            well-mixed (0D-equivalent) limit.
        flow_cache_path: path to the HDF5 cache produced by
            ``scripts/solve_flow.py``.  When ``None`` the simulation runs
            with zero velocity and an all-fluid mask of the requested
            ``grid_shape`` (well-mixed regime).
        mixing_scale: multiplier on the cached velocity field.  Default
            1.0 (use the flow as solved).  Scaling is mass-conserving — a
            divergence-free field stays divergence-free under uniform
            scaling, so the open-face MAC divergence remains zero.  Use
            this knob to dial the mixing rate up/down without re-solving
            the NS problem (e.g. ``mixing_scale=2.0`` halves the eddy
            turnover time).  Ignored when ``flow_cache_path is None``.
        ic_mode: ``'uniform'`` (default) loads each species into every
            fluid cell at its specified value — a spatially uniform
            initial state, the right setting for the BO objective.
            ``'octant'`` loads each species into a single octant of the
            vessel only (zero elsewhere) so chaotic advection has visible
            work to do; useful for visualisation and mixing diagnostics.
        ic_octant: 3-tuple of +-1 selecting which octant to fill when
            ``ic_mode='octant'``.  Default ``(+1, +1, +1)``.  Ignored when
            ``ic_mode='uniform'``.
        device: 'cpu' or 'cuda'.

    Example::

        r = Simulator(samples=[[0.05, 0.05, 0, 0, 25, 25, 25, 25]],
                      grid_shape=(32, 32, 32),
                      flow_cache_path='flow_cache.h5',
                      t_final=48.0).run()
    """

    def __init__(self, N1=0.05, N2=0.05, Sn=0.0, L=0.0,
                 F1=25.0, F2=25.0, F3=25.0, F4=25.0,
                 *, samples=None,
                 t_final=72.0, n_output=145,
                 grid_shape=(32, 32, 32),
                 flow_cache_path='flow_cache.h5',
                 mixing_scale=1.0,
                 ic_mode='uniform',
                 ic_octant=(1, 1, 1),
                 device='cpu'):
        self._ic = self._normalize_ic(N1, N2, Sn, L, F1, F2, F3, F4, samples)
        self.t_final = t_final
        self.n_output = n_output
        self.grid_shape = tuple(grid_shape)
        if len(self.grid_shape) != 3:
            raise ValueError(
                f'grid_shape must be (Nz, Ny, Nx); got {self.grid_shape}'
            )
        self.flow_cache_path = flow_cache_path
        self.mixing_scale = float(mixing_scale)
        if ic_mode not in ('uniform', 'octant'):
            raise ValueError(
                f"ic_mode must be 'uniform' or 'octant'; got '{ic_mode}'"
            )
        self.ic_mode = ic_mode
        self.ic_octant = tuple(ic_octant)
        self.device = device

    @staticmethod
    def _normalize_ic(N1, N2, Sn, L, F1, F2, F3, F4, samples):
        """Convert IC specification to a ``[B, 8]`` CPU float64 tensor."""
        if samples is not None:
            s = torch.as_tensor(samples).to(dtype=torch.float64, device='cpu')
            if s.ndim == 1:
                s = s.unsqueeze(0)
            if s.ndim != 2 or s.shape[1] != 8:
                raise ValueError(
                    f'samples must be [B, 8] or [8]; got shape {list(s.shape)}'
                )
            return s

        vals = [N1, N2, Sn, L, F1, F2, F3, F4]
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
                        f'Array IC parameters must all have the same length; '
                        f'got {B} and {n}'
                    )

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
        if self.n_samples == 1:
            return {n: self._ic[0, i].item() for i, n in enumerate(CHANNEL_NAMES)}
        return {n: self._ic[:, i] for i, n in enumerate(CHANNEL_NAMES)}

    def _make_config(self, grid):
        solver = SolverConfig(t_final=self.t_final, n_output=self.n_output)
        return SimulationConfig(
            grid=grid, solver=solver,
            device=self.device, dtype=torch.float64,
        )

    def run(self):
        """Execute the simulation.  Returns ``SimResults``."""
        Nz, Ny, Nx = self.grid_shape
        dtype = torch.float64

        # Velocity / mask: from cache or zero-flow / all-fluid.
        if self.flow_cache_path is not None:
            u, v, w, fluid, _meta = load_flow(self.flow_cache_path)
            if fluid.shape != (Nz, Ny, Nx):
                raise ValueError(
                    f'flow cache mask shape {tuple(fluid.shape)} does not '
                    f'match grid_shape {self.grid_shape}'
                )
            fluid = fluid.to(device=self.device, dtype=dtype)
            vel = torch.stack(
                [u.to(device=self.device, dtype=dtype),
                 v.to(device=self.device, dtype=dtype),
                 w.to(device=self.device, dtype=dtype)],
                dim=0,
            ).unsqueeze(0)                                          # [1,3,Nz,Ny,Nx]
            if self.mixing_scale != 1.0:
                vel = vel * self.mixing_scale
            wall_mask = (1.0 - fluid).reshape(1, 1, Nz, Ny, Nx)
        else:
            fluid = torch.ones(Nz, Ny, Nx, device=self.device, dtype=dtype)
            vel = None
            wall_mask = None

        # Geometry: use unit cube extents (the cached HDF5 stores its own
        # Lx/Ly/Lz, but with H/D=1 cylinder fitting the cube the scaling is
        # uniform — all spatial operators only need dx/dy/dz).
        grid = GridConfig(Nx=Nx, Ny=Ny, Nz=Nz)
        config = self._make_config(grid)

        # Build IC ``[B, 8, Nz, Ny, Nx]``.
        ic_kwargs = {name: self._ic[:, i] for i, name in enumerate(CHANNEL_NAMES)}
        if self.ic_mode == 'octant':
            y0 = octant_ic(grid, mask=fluid, octant=self.ic_octant,
                           device=self.device, dtype=dtype, **ic_kwargs)
        else:
            y0 = uniform(grid, mask=fluid,
                         device=self.device, dtype=dtype, **ic_kwargs)

        t0 = time.time()
        results, t_eval = simulate(config, y0,
                                   velocity_field=vel, wall_mask=wall_mask)
        elapsed = time.time() - t0

        B = self.n_samples
        if self.flow_cache_path is None:
            regime = 'well-mixed'
        else:
            regime = f'cache={self.flow_cache_path}'
            if self.mixing_scale != 1.0:
                regime += f', mixing_scale={self.mixing_scale:g}'
        label = f'{B} sample{"s" if B > 1 else ""}'
        print(f'Simulation ({Nz}x{Ny}x{Nx}, {self.t_final}h, '
              f'{regime}, {label}): {elapsed:.1f}s')

        return SimResults(results, t_eval, elapsed, grid, fluid)
