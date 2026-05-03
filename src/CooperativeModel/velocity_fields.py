"""Velocity field generator for 2D bioreactor stirring.

Rushton-turbine vertical-slice mean flow: two stacked counter-rotating
vortices with a radial jet at mid-height, derived from a separable
stream function. Optional time-periodic blade-pass proxy via
``period`` / ``t``.
"""

import numpy as np
import torch

# Blade-pass proxy magnitudes — only active when ``period`` is set.
_INTENSITY_PULSE = 0.10   # +/- envelope on amplitude A(t)
_JET_WOBBLE      = 0.02   # +/- fraction of Ly for jet vertical wobble


def rushton_flow(y_grid, x_grid, dx, U_imp, period=None, t=0.0,
                 device="cpu", dtype=torch.float32):
    """2D vertical-slice Rushton-turbine velocity field [2, Y, X].

    Streamfunction psi(x,y,t) = -A(t) * sin(pi x/Lx) * sin(2 pi (y-d(t))/Ly)
    gives two stacked counter-rotating vortices with a radial jet at mid-
    height. ``A(t)`` and ``d(t)`` are blade-pass proxies; both vanish when
    ``period`` is None (steady field). The field is normalised so the peak
    speed equals ``U_imp`` (independent of grid).
    """
    Lx = x_grid * dx
    Ly = y_grid * dx
    ys = (torch.arange(y_grid, device=device, dtype=dtype) + 0.5) * dx
    xs = (torch.arange(x_grid, device=device, dtype=dtype) + 0.5) * dx
    Y, X = torch.meshgrid(ys, xs, indexing="ij")

    if period is None or period <= 0:
        A = 1.0
        delta = 0.0
    else:
        omega_t = 2.0 * np.pi * t / period
        A = 1.0 + _INTENSITY_PULSE * float(np.cos(omega_t))
        delta = _JET_WOBBLE * Ly * float(np.sin(omega_t))

    sx = torch.sin(np.pi * X / Lx)
    cx = torch.cos(np.pi * X / Lx)
    sy = torch.sin(2.0 * np.pi * (Y - delta) / Ly)
    cy = torch.cos(2.0 * np.pi * (Y - delta) / Ly)

    vx = -A * sx * (2.0 * np.pi / Ly) * cy
    vy = +A * (np.pi / Lx) * cx * sy

    peak = torch.sqrt(vx * vx + vy * vy).max()
    if peak > 1e-12:
        scale = U_imp / peak
        vx = vx * scale
        vy = vy * scale

    return torch.stack([vx, vy], dim=0)
