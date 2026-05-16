"""Visualisation utilities for the 3D bioreactor simulator.

The renderers operate on **mid-z slices** produced by ``SimResults``;
inputs therefore have shape ``[1, T, 13, Ny, Nx]``.  The wall-fluid
boundary is overlaid as a white contour from a 2-D ``mask2d``
(1=fluid, 0=wall) so the cylinder geometry is visible in every frame.

Time-series curves are pre-computed over the full 3-D fluid region by
``SimResults._fluid_mean`` and passed in as a ``[T, 13]`` array; the
plotting code does not redo the spatial reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


CHANNEL_NAMES = [
    'N1', 'N2', 'N3', 'N4', 'L (lactic acid)',
    'R1', 'R2', 'R3', 'R4',
    'T1', 'T2', 'T3', 'T4',
]

# Curves are split into two auto-scaled panels because biomass/product/toxin
# (channels 0-4, 9-12) and resources (channels 5-8) routinely differ by
# orders of magnitude — a single shared y-axis collapses one group to a
# flat line.
_LO_GROUP = (0, 1, 2, 3, 4, 9, 10, 11, 12)   # N1..N4, L, T1..T4
_HI_GROUP = (5, 6, 7, 8)                       # R1..R4

N_CHANNELS = len(CHANNEL_NAMES)
L_CH = 4   # lactic acid channel index


def _add_mask_contour(ax, mask2d):
    """Overlay the fluid/wall boundary as a thin white contour."""
    if mask2d is None:
        return
    ax.contour(mask2d, levels=[0.5], colors='white', linewidths=0.6)


def _setup_curve_panels(fig, gs, row_idx, ncols_total, t, curve_means,
                        curve_channels):
    """Lay out one or two curve panels with independent y-axis auto-scaling.

    If ``curve_channels`` spans both the biomass/product group (0-3) and the
    sugar group (4-7) the bottom strip is split into two side-by-side panels,
    each auto-scaled to its own data.  Otherwise a single full-width panel is
    used.

    Lactic acid (channel 3) is drawn on a *twin* y-axis inside the
    biomass+products panel — under octant initial conditions the
    fluid-averaged L lags far behind N1, so a shared y-axis collapses it
    to a near-flat line near zero.  The twin axis lets L use its own
    auto-scaled range while still sharing the time axis.

    Returns:
        lines:  list of (Line2D, channel_idx) — one entry per plotted channel.
        vlines: list of axvline objects (one per panel) for the time cursor.
    """
    import matplotlib.pyplot as plt  # local: keeps the helper self-contained

    lo = [c for c in curve_channels if c in _LO_GROUP]
    hi = [c for c in curve_channels if c in _HI_GROUP]

    if lo and hi:
        half = max(1, ncols_total // 2)
        ax_lo = fig.add_subplot(gs[row_idx, :half])
        ax_hi = fig.add_subplot(gs[row_idx, half:])
        groups = [(ax_lo, lo, 'biomass + products'),
                  (ax_hi, hi, 'sugars')]
    else:
        ax = fig.add_subplot(gs[row_idx, :])
        groups = [(ax, list(curve_channels), None)]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lines = []
    vlines = []
    for (ax, channels, label) in groups:
        twin_ax = None
        primary = [c for c in channels if c != L_CH]
        secondary = [c for c in channels if c == L_CH]
        # Only use a twin axis when L sits alongside other curves; if L is the
        # only channel in this panel, give it the primary axis itself.
        if secondary and primary:
            twin_ax = ax.twinx()
        else:
            primary = list(channels)
            secondary = []

        for ch in primary:
            line, = ax.plot([], [], label=CHANNEL_NAMES[ch],
                            linewidth=2, color=colors[ch % len(colors)])
            lines.append((line, ch))
        for ch in secondary:
            line, = twin_ax.plot([], [], label=CHANNEL_NAMES[ch] + ' (right)',
                                 linewidth=2, color=colors[ch % len(colors)],
                                 linestyle='--')
            lines.append((line, ch))

        ax.set_xlim(t[0], t[-1])
        y_max = max(float(curve_means[:, ch].max()) for ch in primary)
        y_min = min(float(curve_means[:, ch].min()) for ch in primary)
        y_min = min(y_min, 0.0)
        pad = 0.1 * max(y_max - y_min, 1e-12)
        ax.set_ylim(y_min, y_max + pad if y_max > 0 else 0.1)
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Fluid average')
        if label is not None:
            ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        vline = ax.axvline(t[0], color='k', linestyle='--',
                           alpha=0.5, linewidth=1)
        vlines.append(vline)

        if twin_ax is not None:
            l_max = max(float(curve_means[:, ch].max()) for ch in secondary)
            l_min = min(float(curve_means[:, ch].min()) for ch in secondary)
            l_min = min(l_min, 0.0)
            l_pad = 0.1 * max(l_max - l_min, 1e-12)
            twin_ax.set_ylim(l_min, l_max + l_pad if l_max > 0 else 0.1)
            twin_ax.set_ylabel('L (right axis)')
            # Combine legends from both axes onto the primary axis.
            h1, lbl1 = ax.get_legend_handles_labels()
            h2, lbl2 = twin_ax.get_legend_handles_labels()
            ax.legend(h1 + h2, lbl1 + lbl2, fontsize=8, loc='best')
        else:
            ax.legend(fontsize=8, loc='best')
    return lines, vlines


def plot_snapshot(slice_results, t_eval, time_idx, grid_cfg=None,
                  mask2d=None, channels=None, figsize=None,
                  vmin=None, vmax=None):
    """Plot mid-z-slice heatmaps of selected channels at a given time.

    Args:
        slice_results: ``[1, T, 13, Ny, Nx]`` mid-z slice of the result.
        t_eval: ``[T]`` time points.
        time_idx: index into t_eval.
        grid_cfg: optional ``GridConfig`` for axis labels.
        mask2d: optional 2-D fluid mask to draw as a contour.
        channels: list of channel indices to plot (default: all 13).
    """
    data = slice_results[0, time_idx].detach().cpu().numpy()  # [13, Ny, Nx]
    t = t_eval[time_idx].item()

    if channels is None:
        channels = list(range(N_CHANNELS))
    nc = len(channels)
    ncols = min(4, nc)
    nrows = (nc + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for idx, ch in enumerate(channels):
        ax = axes[idx // ncols, idx % ncols]
        im = ax.imshow(data[ch], origin='lower', aspect='equal',
                       vmin=vmin, vmax=vmax, cmap='viridis')
        _add_mask_contour(ax, mask2d)
        ax.set_title(f'{CHANNEL_NAMES[ch]}\nt = {t:.1f} h', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if grid_cfg:
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')

    for idx in range(nc, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    return fig


def plot_spatial_average(means, t_eval, channels=None, figsize=(10, 6)):
    """Plot fluid-averaged concentrations over time.

    Args:
        means: ``[T, 13]`` fluid-averaged time series (numpy or tensor).
        t_eval: ``[T]`` time points.
        channels: list of channel indices (default: all 13).
    """
    if hasattr(means, 'detach'):
        means = means.detach().cpu().numpy()
    means = np.asarray(means)
    t = t_eval.detach().cpu().numpy() if hasattr(t_eval, 'detach') else np.asarray(t_eval)

    if channels is None:
        channels = list(range(N_CHANNELS))

    fig, ax = plt.subplots(figsize=figsize)
    for ch in channels:
        ax.plot(t, means[:, ch], label=CHANNEL_NAMES[ch])

    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Concentration')
    ax.set_title('Fluid-averaged concentrations')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def animate_all_fields_with_curves(slice_results, t_eval, channels=None,
                                   curve_channels=None, mask2d=None,
                                   curve_means=None, interval=150,
                                   figsize=None, title_prefix=None,
                                   save_path=None):
    """Animate mid-z-slice heatmaps with fluid-averaged curves below.

    Args:
        slice_results: ``[1, T, 13, Ny, Nx]`` mid-z slice tensor.
        t_eval: ``[T]`` time points.
        channels: heatmap channels (default: all 13).
        curve_channels: channels to plot as time series (default: all 13).
        mask2d: 2-D fluid mask drawn as a contour overlay on every panel.
        curve_means: ``[T, 13]`` fluid-averaged time series to plot.  If
            ``None``, the mean is computed over the 2-D slice (suitable
            for the well-mixed limit where slice == volume).
        interval: ms between frames.
        save_path: ``.gif`` or ``.mp4`` to save (optional).
    """
    if channels is None:
        channels = list(range(N_CHANNELS))
    if curve_channels is None:
        curve_channels = list(range(N_CHANNELS))  # all variables

    nc = len(channels)
    ncols = min(4, nc)
    nrows_maps = (nc + ncols - 1) // ncols

    data = slice_results[0].detach().cpu().numpy()  # [T, 13, Ny, Nx]
    t = t_eval.detach().cpu().numpy() if hasattr(t_eval, 'detach') else np.asarray(t_eval)

    if curve_means is None:
        curve_means = data.mean(axis=(-2, -1))  # [T, 13]
    else:
        curve_means = np.asarray(curve_means)

    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows_maps + 3)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows_maps + 1, ncols,
                          height_ratios=[1] * nrows_maps + [1.0],
                          hspace=0.45, wspace=0.35)

    # Per-frame normalisation: each frame's vmax = that frame's slice max.
    # Concentrations sweep orders of magnitude as the reaction-advection-
    # diffusion system evolves; a global vmax washes out spatial structure.
    # Re-scaling per frame preserves the *pattern* of mixing in each frame.
    ims = []
    hm_axes = []
    for idx, ch in enumerate(channels):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        hm_axes.append(ax)
        vmax0 = max(float(np.nanmax(data[0, ch])), 1e-12)
        im = ax.imshow(data[0, ch], origin='lower', aspect='equal',
                       vmin=0, vmax=vmax0, cmap='viridis')
        _add_mask_contour(ax, mask2d)
        ax.set_title(CHANNEL_NAMES[ch], fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append((im, ch))

    for idx in range(nc, nrows_maps * ncols):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.set_visible(False)

    lines, vlines = _setup_curve_panels(
        fig, gs, nrows_maps, ncols, t, curve_means, curve_channels,
    )

    title_text = (lambda fr: f't = {t[fr]:.1f} h' if not title_prefix
                  else f'{title_prefix}   t = {t[fr]:.1f} h')
    suptitle = fig.suptitle(title_text(0), fontsize=13, y=0.98)

    def update(frame):
        for im, ch in ims:
            frame_data = data[frame, ch]
            im.set_data(frame_data)
            vmax_f = max(float(np.nanmax(frame_data)), 1e-12)
            im.set_clim(vmin=0, vmax=vmax_f)
        for line, ch in lines:
            line.set_data(t[:frame + 1], curve_means[:frame + 1, ch])
        for vline in vlines:
            vline.set_xdata([t[frame], t[frame]])
        suptitle.set_text(title_text(frame))
        return ([im for im, _ in ims]
                + [line for line, _ in lines]
                + list(vlines) + [suptitle])

    anim = animation.FuncAnimation(fig, update, frames=len(t),
                                   interval=interval, blit=False)
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', dpi=100)
        else:
            anim.save(save_path, writer='ffmpeg', dpi=100)

    return anim


def animate_orthoviews(slices, masks, t_eval, channels=None,
                       curve_channels=None, curve_means=None,
                       interval=150, figsize=None, save_path=None):
    """Animate three orthogonal slices (mid-z, mid-y, mid-x) per channel.

    Args:
        slices: tuple ``(slz, sly, slx)`` of three ``[1, T, 13, H, W]``
            mid-plane tensors (z-, y-, x-perpendicular planes).
        masks: tuple ``(mz, my, mx)`` of 2-D fluid masks for each plane,
            used as white-contour overlays.
        t_eval: ``[T]`` time points.
        channels: heatmap channels (default: all 13).
        curve_channels: channels to plot as time series (default: all 13).
        curve_means: ``[T, 13]`` fluid-averaged time series.
        interval: ms between frames.
        save_path: ``.gif`` or ``.mp4`` (optional).

    Layout: 13 rows (one per channel) by 3 columns (z, y, x slices), with
    a fluid-averaged time-series strip below.
    """
    if channels is None:
        channels = list(range(N_CHANNELS))
    if curve_channels is None:
        curve_channels = list(range(N_CHANNELS))

    slz, sly, slx = slices
    mz, my, mx = masks
    plane_data = [
        (slz[0].detach().cpu().numpy(), mz, 'mid-z (xy)'),  # [T, 13, Ny, Nx]
        (sly[0].detach().cpu().numpy(), my, 'mid-y (xz)'),  # [T, 13, Nz, Nx]
        (slx[0].detach().cpu().numpy(), mx, 'mid-x (yz)'),  # [T, 13, Nz, Ny]
    ]
    t = t_eval.detach().cpu().numpy() if hasattr(t_eval, 'detach') else np.asarray(t_eval)

    if curve_means is None:
        curve_means = plane_data[0][0].mean(axis=(-2, -1))
    curve_means = np.asarray(curve_means)

    nrows = len(channels)
    if figsize is None:
        figsize = (12, 1.6 * nrows + 3)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows + 1, 3,
                          height_ratios=[1] * nrows + [0.9 * max(nrows / 4, 1)],
                          hspace=0.5, wspace=0.3)

    ims = []  # list of (im, channel_idx, plane_idx)
    for r, ch in enumerate(channels):
        for c, (data_p, mask_p, plane_label) in enumerate(plane_data):
            ax = fig.add_subplot(gs[r, c])
            vmax0 = max(float(np.nanmax(data_p[0, ch])), 1e-12)
            im = ax.imshow(data_p[0, ch], origin='lower', aspect='equal',
                           vmin=0, vmax=vmax0, cmap='viridis')
            _add_mask_contour(ax, mask_p)
            if r == 0:
                ax.set_title(plane_label, fontsize=10)
            if c == 0:
                ax.set_ylabel(CHANNEL_NAMES[ch], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ims.append((im, ch, c))

    lines, vlines = _setup_curve_panels(
        fig, gs, nrows, 3, t, curve_means, curve_channels,
    )

    suptitle = fig.suptitle(f'orthogonal slices   t = {t[0]:.1f} h',
                            fontsize=13, y=0.995)

    def update(frame):
        for im, ch, c in ims:
            data_p, _, _ = plane_data[c]
            frame_data = data_p[frame, ch]
            im.set_data(frame_data)
            vmax_f = max(float(np.nanmax(frame_data)), 1e-12)
            im.set_clim(vmin=0, vmax=vmax_f)
        for line, ch in lines:
            line.set_data(t[:frame + 1], curve_means[:frame + 1, ch])
        for vline in vlines:
            vline.set_xdata([t[frame], t[frame]])
        suptitle.set_text(f'orthogonal slices   t = {t[frame]:.1f} h')
        return ([im for im, _, _ in ims]
                + [line for line, _ in lines]
                + list(vlines) + [suptitle])

    anim = animation.FuncAnimation(fig, update, frames=len(t),
                                   interval=interval, blit=False)
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', dpi=100)
        else:
            anim.save(save_path, writer='ffmpeg', dpi=100)
    return anim
