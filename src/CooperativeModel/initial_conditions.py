"""Uniform and octant initial conditions for the bioreactor."""

import torch


N_CHANNELS = 13


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


def uniform(grid_cfg, N1=0.01, N2=0.01, N3=0.01, N4=0.01, L=0.0,
            R1=2.0, R2=2.0, R3=2.0, R4=2.0,
            T1=0.0, T2=0.0, T3=0.0, T4=0.0,
            mask=None, device='cpu', dtype=torch.float64):
    """Fill fluid cells with per-channel values; return [B, 13, Nz, Ny, Nx].

    Channel values are scalars or sequences of length B, ordered as
    [N1..N4, L, R1..R4, T1..T4]. The optional fluid mask (1=fluid)
    zeros wall cells and has shape [Nz, Ny, Nx] or [1, 1, Nz, Ny, Nx].
    """
    Nz, Ny, Nx = grid_cfg.Nz, grid_cfg.Ny, grid_cfg.Nx
    values = [N1, N2, N3, N4, L, R1, R2, R3, R4, T1, T2, T3, T4]
    B = _infer_batch_size(values)

    state = torch.zeros(B, N_CHANNELS, Nz, Ny, Nx, device=device, dtype=dtype)
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


def octant(grid_cfg, N1=0.01, N2=0.01, N3=0.01, N4=0.01, L=0.0,
           R1=2.0, R2=2.0, R3=2.0, R4=2.0,
           T1=0.0, T2=0.0, T3=0.0, T4=0.0,
           octant=(1, 1, 1),
           mask=None, device='cpu', dtype=torch.float64):
    """Fill one octant with per-channel values and zero all other cells.

    Return shape is [B, 13, Nz, Ny, Nx], with the same channels as uniform.
    The octant signs (+1 or -1) select each side of the vessel center.
    Values are concentrations within the octant, without volume rescaling.
    An optional fluid mask zeros wall cells.
    """
    Nz, Ny, Nx = grid_cfg.Nz, grid_cfg.Ny, grid_cfg.Nx
    values = [N1, N2, N3, N4, L, R1, R2, R3, R4, T1, T2, T3, T4]
    B = _infer_batch_size(values)

    state = torch.zeros(B, N_CHANNELS, Nz, Ny, Nx, device=device, dtype=dtype)
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
