"""Initial-condition generators for the 3D bioreactor model.

Channel ordering: ``[N1, N2, Sn, L, F1, F2, F3, F4]``.

Two generators:

  * ``uniform``  — fills every fluid cell with the per-channel value.  The
    well-mixed limit (1 x 1 x 1) and the BO objective both use this path.
  * ``octant``   — fills only one octant of the vessel with the per-channel
    value, leaving the other 7/8 at zero.  The chaotic flow then has to
    redistribute species across the cylinder, producing visually obvious
    mixing in the GIFs while leaving ``uniform`` as the default for BO.

The 2D-specific generators (Gaussian blobs, stratified halves, edge
concentration, random/stratified colony inoculations) lived only inside
the deprecated ``flow_through`` mode and were removed during the 3D
migration.
"""

import torch


def _infer_batch_size(values):
    """Infer batch size from a list of scalar-or-array values, raising on
    inconsistent lengths."""
    B = 1
    for v in values:
        if isinstance(v, (int, float)):
            continue
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
    return B


def uniform(grid_cfg, N1=0.05, N2=0.05, Sn=0.0, L=0.0,
            F1=25.0, F2=25.0, F3=25.0, F4=25.0,
            mask=None, device='cpu', dtype=torch.float64):
    """Spatially-uniform initial condition, ``[B, 8, Nz, Ny, Nx]``.

    Each value can be a scalar (single sample) or a 1-D sequence/tensor of
    length B.  The well-mixed limit (1 x 1 x 1) and the full 3D vessel use
    the same path.

    Args:
        grid_cfg: ``GridConfig`` instance.
        N1, N2, Sn, L, F1..F4: per-channel concentrations.
        mask: optional fluid mask, ``[Nz, Ny, Nx]`` or
            ``[1, 1, Nz, Ny, Nx]`` (1 = fluid, 0 = wall).  When given, wall
            cells are zeroed in the returned IC.
        device, dtype: torch placement.
    """
    Nz, Ny, Nx = grid_cfg.Nz, grid_cfg.Ny, grid_cfg.Nx
    values = [N1, N2, Sn, L, F1, F2, F3, F4]
    B = _infer_batch_size(values)

    state = torch.zeros(B, 8, Nz, Ny, Nx, device=device, dtype=dtype)
    for i, v in enumerate(values):
        v_t = torch.as_tensor(v).to(device=device, dtype=dtype).flatten()
        if v_t.numel() == 1:
            state[:, i] = v_t.item()
        else:
            state[:, i] = v_t.reshape(B, 1, 1, 1)

    if mask is not None:
        m = mask.to(device=device, dtype=dtype)
        if m.dim() == 3:
            m = m.reshape(1, 1, Nz, Ny, Nx)
        state = state * m
    return state


def octant(grid_cfg, N1=0.05, N2=0.05, Sn=0.0, L=0.0,
           F1=25.0, F2=25.0, F3=25.0, F4=25.0,
           octant=(1, 1, 1),
           mask=None, device='cpu', dtype=torch.float64):
    """Initial condition concentrated in a single octant of the vessel.

    Each species takes its given value inside the chosen octant (cells where
    ``(sx*(x-Lx/2), sy*(y-Ly/2), sz*(z-Lz/2)) >= 0`` for ``octant=(sx,sy,sz)``)
    and zero elsewhere.  Wall cells are zeroed by ``mask`` as in ``uniform``.

    With chaotic advection from the non-axisymmetric impeller, this IC
    redistributes itself across the full cylinder over many turnovers, so the
    rendered GIFs show actual mixing rather than a constant field.

    Args:
        grid_cfg: ``GridConfig`` instance.
        N1, N2, Sn, L, F1..F4: per-channel concentrations inside the octant.
        octant: 3-tuple of +-1 selecting (x_sign, y_sign, z_sign) relative to
            the vessel centre ``(Lx/2, Ly/2, Lz/2)``.  Default ``(+1, +1, +1)``
            (front-right-top).
        mask: optional fluid mask, ``[Nz, Ny, Nx]`` or ``[1, 1, Nz, Ny, Nx]``.
        device, dtype: torch placement.
    """
    Nz, Ny, Nx = grid_cfg.Nz, grid_cfg.Ny, grid_cfg.Nx
    values = [N1, N2, Sn, L, F1, F2, F3, F4]
    B = _infer_batch_size(values)

    state = torch.zeros(B, 8, Nz, Ny, Nx, device=device, dtype=dtype)
    for i, v in enumerate(values):
        v_t = torch.as_tensor(v).to(device=device, dtype=dtype).flatten()
        if v_t.numel() == 1:
            state[:, i] = v_t.item()
        else:
            state[:, i] = v_t.reshape(B, 1, 1, 1)

    sx, sy, sz = (float(s) for s in octant)
    cx = grid_cfg.Lx * 0.5
    cy = grid_cfg.Ly * 0.5
    cz = grid_cfg.Lz * 0.5
    xs = (torch.arange(Nx, device=device, dtype=dtype) + 0.5) * grid_cfg.dx
    ys = (torch.arange(Ny, device=device, dtype=dtype) + 0.5) * grid_cfg.dy
    zs = (torch.arange(Nz, device=device, dtype=dtype) + 0.5) * grid_cfg.dz
    Z, Y, X = torch.meshgrid(zs, ys, xs, indexing='ij')
    in_oct = ((sx * (X - cx) >= 0)
              & (sy * (Y - cy) >= 0)
              & (sz * (Z - cz) >= 0)).to(dtype)
    state = state * in_oct.reshape(1, 1, Nz, Ny, Nx)

    if mask is not None:
        m = mask.to(device=device, dtype=dtype)
        if m.dim() == 3:
            m = m.reshape(1, 1, Nz, Ny, Nx)
        state = state * m
    return state
