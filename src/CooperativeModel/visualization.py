"""Visualization utilities for 2D bioreactor simulations.

Provides spatial heatmaps, time-series of spatial averages, comparison plots,
and animations of field evolution.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CHANNEL_NAMES = [
    'N1 (CoA)', 'N2 (CoB)', 'Sn (nisin)', 'L (lactic acid)',
    'F1 (glucose)', 'F2 (fructose)', 'F3 (sucrose)', 'F4 (maltose)',
]


def plot_snapshot(results, t_eval, time_idx, grid_cfg=None, channels=None,
                  figsize=None, vmin=None, vmax=None):
    """Plot spatial heatmaps of selected channels at a given time index.

    Args:
        results: [1, T, 8, H, W] solution tensor.
        t_eval: [T] time points.
        time_idx: Index into t_eval.
        grid_cfg: Optional GridConfig for axis labels.
        channels: List of channel indices to plot (default: all 8).
        figsize: Figure size tuple.

    Returns:
        matplotlib Figure.
    """
    data = results[0, time_idx].detach().cpu().numpy()  # [8, H, W]
    t = t_eval[time_idx].item()

    if channels is None:
        channels = list(range(8))
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
        ax.set_title(f'{CHANNEL_NAMES[ch]}\nt = {t:.1f} h', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if grid_cfg:
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')

    for idx in range(nc, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    return fig


def plot_spatial_average(results, t_eval, channels=None, figsize=(10, 6)):
    """Plot spatially-averaged concentrations over time.

    Args:
        results: [1, T, 8, H, W] solution tensor.
        t_eval: [T] time points.
        channels: List of channel indices (default: all 8).

    Returns:
        matplotlib Figure.
    """
    data = results[0].detach().cpu().numpy()  # [T, 8, H, W]
    t = t_eval.detach().cpu().numpy()
    means = data.mean(axis=(-2, -1))  # [T, 8]

    if channels is None:
        channels = list(range(8))

    fig, ax = plt.subplots(figsize=figsize)
    for ch in channels:
        ax.plot(t, means[:, ch], label=CHANNEL_NAMES[ch])

    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Concentration')
    ax.set_title('Spatially-averaged concentrations')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_comparison(results_2d, t_eval_2d, results_ode=None, t_eval_ode=None,
                    channels=None, figsize=(14, 7)):
    """Compare 2D spatial average with well-mixed ODE solution.

    Args:
        results_2d: [1, T, 8, H, W] tensor from 2D simulation.
        t_eval_2d: [T] time points for 2D.
        results_ode: Optional [1, T_ode, 8] tensor from ODE.
        t_eval_ode: Optional [T_ode] time points for ODE.
        channels: List of channel indices (default: all 8).

    Returns:
        matplotlib Figure.
    """
    data_2d = results_2d[0].detach().cpu().numpy()
    t_2d = t_eval_2d.detach().cpu().numpy()
    means_2d = data_2d.mean(axis=(-2, -1))  # [T, 8]

    if channels is None:
        channels = list(range(8))

    ncols = min(4, len(channels))
    nrows = (len(channels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, ch in enumerate(channels):
        ax = axes[idx // ncols, idx % ncols]
        ax.plot(t_2d, means_2d[:, ch], 'b-', linewidth=2, label='2D (spatial avg)')
        if results_ode is not None and t_eval_ode is not None:
            ode_data = results_ode[0].detach().cpu().numpy()
            t_ode = t_eval_ode.detach().cpu().numpy()
            ax.plot(t_ode, ode_data[:, ch], 'r--', linewidth=2, label='ODE (well-mixed)')
        ax.set_title(CHANNEL_NAMES[ch], fontsize=10)
        ax.set_xlabel('Time [h]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(len(channels), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.suptitle('2D vs Well-Mixed Comparison', fontsize=13)
    plt.tight_layout()
    return fig


def animate_field(results, t_eval, channel=0, interval=100, figsize=(6, 5),
                  save_path=None):
    """Create an animation of a single channel over time.

    Args:
        results: [1, T, 8, H, W] solution tensor.
        t_eval: [T] time points.
        channel: Channel index to animate.
        interval: Milliseconds between frames.
        save_path: Optional path to save (.gif or .mp4).

    Returns:
        matplotlib FuncAnimation object.
    """
    data = results[0, :, channel].detach().cpu().numpy()  # [T, H, W]
    t = t_eval.detach().cpu().numpy()
    vmin, vmax = data.min(), data.max()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data[0], origin='lower', aspect='equal',
                   vmin=vmin, vmax=vmax, cmap='viridis')
    title = ax.set_title(f'{CHANNEL_NAMES[channel]}  t = {t[0]:.1f} h')
    plt.colorbar(im, ax=ax)

    def update(frame):
        im.set_data(data[frame])
        title.set_text(f'{CHANNEL_NAMES[channel]}  t = {t[frame]:.1f} h')
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=len(t),
                                   interval=interval, blit=False)
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow')
        else:
            anim.save(save_path, writer='ffmpeg')

    return anim


def animate_all_fields(results, t_eval, channels=None, interval=150,
                       figsize=None, save_path=None):
    """Create a multi-panel animation of several channels over time.

    Args:
        results: [1, T, 8, H, W] solution tensor.
        t_eval: [T] time points.
        channels: List of channel indices (default: all 8).
        interval: Milliseconds between frames.
        figsize: Figure size tuple.
        save_path: Optional path to save (.gif or .mp4).

    Returns:
        matplotlib FuncAnimation object.
    """
    if channels is None:
        channels = list(range(8))
    nc = len(channels)
    ncols = min(4, nc)
    nrows = (nc + ncols - 1) // ncols

    data = results[0].detach().cpu().numpy()  # [T, 8, H, W]
    t = t_eval.detach().cpu().numpy()

    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    ims = []
    for idx, ch in enumerate(channels):
        ax = axes[idx // ncols, idx % ncols]
        vmin, vmax = data[:, ch].min(), data[:, ch].max()
        im = ax.imshow(data[0, ch], origin='lower', aspect='equal',
                       vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(CHANNEL_NAMES[ch], fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append((im, ch))

    for idx in range(nc, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    suptitle = fig.suptitle(f't = {t[0]:.1f} h', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def update(frame):
        for im, ch in ims:
            im.set_data(data[frame, ch])
        suptitle.set_text(f't = {t[frame]:.1f} h')
        return [im for im, _ in ims] + [suptitle]

    anim = animation.FuncAnimation(fig, update, frames=len(t),
                                   interval=interval, blit=False)
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', dpi=100)
        else:
            anim.save(save_path, writer='ffmpeg', dpi=100)

    return anim


def animate_all_fields_with_curves(results, t_eval, channels=None,
                                   curve_channels=None, zone_boxes=None,
                                   interval=150, figsize=None, save_path=None):
    """Animate spatial heatmaps with spatially-averaged time series below.

    Top rows show the spatial fields.  The bottom row shows a progressively
    revealed time-series of selected channels so you can watch the system
    approach (or fail to reach) steady state.

    Args:
        results: [1, T, 8, H, W] solution tensor.
        t_eval: [T] time points.
        channels: Channels to show as heatmaps (default: all 8).
        curve_channels: Channels to plot as time series (default: [2, 3] = Sn, L).
        zone_boxes: Optional list of dicts to draw labeled rectangles on
            every heatmap.  Each dict has keys:
                'xy': (col, row) lower-left corner in array coords,
                'width': box width in cells,
                'height': box height in cells,
                'color': edge color (default 'red'),
                'label': text label (default '').
        interval: Milliseconds between frames.
        figsize: Figure size tuple.
        save_path: Optional path to save (.gif or .mp4).

    Returns:
        matplotlib FuncAnimation object.
    """
    if channels is None:
        channels = list(range(8))
    if curve_channels is None:
        curve_channels = [2, 3]  # Sn and L

    nc = len(channels)
    ncols = min(4, nc)
    nrows_maps = (nc + ncols - 1) // ncols

    data = results[0].detach().cpu().numpy()  # [T, 8, H, W]
    t = t_eval.detach().cpu().numpy()
    means = data.mean(axis=(-2, -1))  # [T, 8]

    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows_maps + 3)

    # Layout: heatmap rows + one time-series row
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows_maps + 1, ncols,
                          height_ratios=[1] * nrows_maps + [0.8],
                          hspace=0.4)

    # --- Heatmap axes ---
    ims = []
    hm_axes = []
    for idx, ch in enumerate(channels):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        hm_axes.append(ax)
        vmin, vmax = data[:, ch].min(), data[:, ch].max()
        im = ax.imshow(data[0, ch], origin='lower', aspect='equal',
                       vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(CHANNEL_NAMES[ch], fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append((im, ch))

    # --- Draw zone boxes on every heatmap ---
    if zone_boxes:
        from matplotlib.patches import Rectangle
        for ax in hm_axes:
            for box in zone_boxes:
                xy = box['xy']
                w, h = box['width'], box['height']
                color = box.get('color', 'red')
                rect = Rectangle(
                    (xy[0] - 0.5, xy[1] - 0.5), w, h,
                    linewidth=2, edgecolor=color, facecolor='none',
                )
                ax.add_patch(rect)
                label = box.get('label', '')
                if label:
                    ax.text(
                        xy[0] + w / 2, xy[1] + h / 2, label,
                        color=color, fontsize=7, fontweight='bold',
                        ha='center', va='center',
                    )

    for idx in range(nc, nrows_maps * ncols):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.set_visible(False)

    # --- Time-series axis (full width) ---
    ax_ts = fig.add_subplot(gs[nrows_maps, :])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lines = []
    for i, ch in enumerate(curve_channels):
        line, = ax_ts.plot([], [], label=CHANNEL_NAMES[ch],
                           linewidth=2, color=colors[i % len(colors)])
        lines.append((line, ch))
    ax_ts.set_xlim(t[0], t[-1])
    y_max = max(means[:, ch].max() for ch in curve_channels) * 1.1
    y_min = 0
    ax_ts.set_ylim(y_min, max(y_max, 0.1))
    ax_ts.set_xlabel('Time [h]')
    ax_ts.set_ylabel('Spatial average')
    ax_ts.legend(fontsize=9, loc='upper left')
    ax_ts.grid(True, alpha=0.3)
    vline = ax_ts.axvline(t[0], color='k', linestyle='--', alpha=0.5, linewidth=1)

    suptitle = fig.suptitle(f't = {t[0]:.1f} h', fontsize=13, y=0.98)

    def update(frame):
        for im, ch in ims:
            im.set_data(data[frame, ch])
        for line, ch in lines:
            line.set_data(t[:frame + 1], means[:frame + 1, ch])
        vline.set_xdata([t[frame], t[frame]])
        suptitle.set_text(f't = {t[frame]:.1f} h')
        return [im for im, _ in ims] + [line for line, _ in lines] + [vline, suptitle]

    anim = animation.FuncAnimation(fig, update, frames=len(t),
                                   interval=interval, blit=False)
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', dpi=100)
        else:
            anim.save(save_path, writer='ffmpeg', dpi=100)

    return anim
