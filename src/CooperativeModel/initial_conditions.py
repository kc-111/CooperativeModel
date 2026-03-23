"""Initial condition generators for the 2D bioreactor model.

All functions return a [1, 8, Ny, Nx] tensor with channel ordering:
    [N1, N2, Sn, L, F1, F2, F3, F4]
"""

import torch


def uniform(grid_cfg, N1=0.05, N2=0.05, Sn=0.0, L=0.0,
            F1=25.0, F2=25.0, F3=25.0, F4=25.0,
            device='cpu', dtype=torch.float64):
    """Spatially uniform initial conditions (well-mixed).

    With no advection, the 2D model should recover the ODE solution.
    """
    Ny, Nx = grid_cfg.Ny, grid_cfg.Nx
    state = torch.zeros(1, 8, Ny, Nx, device=device, dtype=dtype)
    for i, v in enumerate([N1, N2, Sn, L, F1, F2, F3, F4]):
        state[:, i] = v
    return state


def gaussian_blob(grid_cfg, centers, sigmas, backgrounds,
                  device='cpu', dtype=torch.float64):
    """Gaussian blob initial conditions.

    Each specified channel gets a Gaussian peak added on top of its
    background value.

    Args:
        grid_cfg: GridConfig instance.
        centers: dict mapping channel index to (cy, cx) centre in physical coords [cm].
        sigmas: dict mapping channel index to sigma (spread) [cm].
        backgrounds: list of 8 background values for each channel.
    """
    Ny, Nx = grid_cfg.Ny, grid_cfg.Nx
    dx, dy = grid_cfg.dx, grid_cfg.dy

    x = torch.linspace(0.5 * dx, grid_cfg.Lx - 0.5 * dx, Nx, device=device, dtype=dtype)
    y = torch.linspace(0.5 * dy, grid_cfg.Ly - 0.5 * dy, Ny, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    state = torch.zeros(1, 8, Ny, Nx, device=device, dtype=dtype)
    for i in range(8):
        state[:, i] = backgrounds[i]

    for ch, (cy, cx) in centers.items():
        s = sigmas.get(ch, 0.1)
        peak = max(backgrounds[ch], 1e-3) * 5.0  # 5x background at centre
        blob = peak * torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * s**2))
        state[:, ch] += blob

    return state


def stratified(grid_cfg, N1=0.05, N2=0.05, Sn=0.0, L=0.0, FT=100.0,
               device='cpu', dtype=torch.float64):
    """Stratified: bacteria on the left half, nutrients on the right half.

    Nutrients are split equally among F1-F4 and concentrated in the right half.
    Bacteria are concentrated in the left half. This creates a situation where
    diffusion (and advection) must transport nutrients to bacteria.
    """
    Ny, Nx = grid_cfg.Ny, grid_cfg.Nx
    mid = Nx // 2

    state = torch.zeros(1, 8, Ny, Nx, device=device, dtype=dtype)

    # Bacteria on left half (doubled to preserve spatial average)
    state[:, 0, :, :mid] = N1 * 2
    state[:, 1, :, :mid] = N2 * 2

    # Nisin and lactic acid uniform
    state[:, 2] = Sn
    state[:, 3] = L

    # Nutrients on right half (doubled to preserve spatial average)
    F_per = FT / 4.0
    for i in range(4, 8):
        state[:, i, :, mid:] = F_per * 2

    return state


def random_perturbation(grid_cfg, means=None, std_frac=0.05,
                        device='cpu', dtype=torch.float64):
    """Small random perturbations around a mean value.

    Useful for studying spontaneous pattern formation or symmetry breaking.

    Args:
        means: list of 8 mean values.
               Default: [0.05, 0.05, 0.0, 0.0, 25.0, 25.0, 25.0, 25.0].
        std_frac: standard deviation as a fraction of the mean.
    """
    if means is None:
        means = [0.05, 0.05, 0.0, 0.0, 25.0, 25.0, 25.0, 25.0]

    Ny, Nx = grid_cfg.Ny, grid_cfg.Nx
    state = torch.zeros(1, 8, Ny, Nx, device=device, dtype=dtype)

    for i, m in enumerate(means):
        if m > 0:
            noise = torch.randn(Ny, Nx, device=device, dtype=dtype) * (m * std_frac)
            state[:, i] = (m + noise).clamp(min=0.0)
        else:
            state[:, i] = 0.0

    return state


def edge_concentrated(grid_cfg, edge='left', decay_length=0.15, noise_frac=0.05,
                      N1=0.05, N2=0.05, Sn=0.0, L=0.0,
                      F1=25.0, F2=25.0, F3=25.0, F4=25.0,
                      device='cpu', dtype=torch.float64):
    """Concentrate mass near a domain edge with random perturbation.

    The spatial average of each channel equals the given ODE value, so total
    mass is conserved compared to the uniform case.

    Args:
        grid_cfg: GridConfig instance.
        edge: Which edge to concentrate near ('left', 'right', 'top', 'bottom').
        decay_length: Exponential decay length scale [cm].
        noise_frac: Relative amplitude of random noise (0.05 = 5%).
        N1..F4: ODE-equivalent initial values (become the spatial average).
    """
    Ny, Nx = grid_cfg.Ny, grid_cfg.Nx
    dx, dy = grid_cfg.dx, grid_cfg.dy
    Lx, Ly = grid_cfg.Lx, grid_cfg.Ly

    x = torch.linspace(0.5 * dx, Lx - 0.5 * dx, Nx, device=device, dtype=dtype)
    y = torch.linspace(0.5 * dy, Ly - 0.5 * dy, Ny, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Distance from chosen edge
    if edge == 'left':
        dist = X
    elif edge == 'right':
        dist = Lx - X
    elif edge == 'bottom':
        dist = Y
    elif edge == 'top':
        dist = Ly - Y
    else:
        raise ValueError(f"edge must be 'left', 'right', 'top', or 'bottom', got '{edge}'")

    # Exponential decay profile + noise
    profile = torch.exp(-dist / decay_length)
    noise = 1.0 + noise_frac * torch.randn(Ny, Nx, device=device, dtype=dtype)
    profile = (profile * noise).clamp(min=0.0)

    # Normalise so spatial average = 1 → value * profile has correct total mass
    profile = profile / profile.mean()

    state = torch.zeros(1, 8, Ny, Nx, device=device, dtype=dtype)
    values = [N1, N2, Sn, L, F1, F2, F3, F4]
    for i, v in enumerate(values):
        if v > 0:
            state[:, i] = v * profile
    return state


def random_inoculation(grid_cfg, n_colonies=6, colony_radius=0.08,
                       N1_amount=0.1, N2_amount=0.1,
                       Sn=0.0, L=0.0,
                       F1=0.0, F2=0.0, F3=0.0, F4=0.0,
                       device='cpu', dtype=torch.float64):
    """Scatter small bacterial colonies at random positions.

    Each colony is a Gaussian blob of N1 and/or N2 placed at a random
    location.  Colonies get independent random amounts of each strain,
    so some colonies may be N1-dominant, others N2-dominant.  Nutrients
    are set uniformly (e.g. trace background or zero).

    Args:
        grid_cfg: GridConfig instance.
        n_colonies: Number of colonies to place.
        colony_radius: Gaussian sigma of each colony [cm].
        N1_amount: Peak concentration scale for CoA per colony.
        N2_amount: Peak concentration scale for CoB per colony.
        Sn, L, F1..F4: Uniform background values for non-bacterial fields.
    """
    Ny, Nx = grid_cfg.Ny, grid_cfg.Nx
    dx, dy = grid_cfg.dx, grid_cfg.dy
    Lx, Ly = grid_cfg.Lx, grid_cfg.Ly

    x = torch.linspace(0.5 * dx, Lx - 0.5 * dx, Nx, device=device, dtype=dtype)
    y = torch.linspace(0.5 * dy, Ly - 0.5 * dy, Ny, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    state = torch.zeros(1, 8, Ny, Nx, device=device, dtype=dtype)

    # Uniform backgrounds for non-bacterial channels
    state[:, 2] = Sn
    state[:, 3] = L
    state[:, 4] = F1
    state[:, 5] = F2
    state[:, 6] = F3
    state[:, 7] = F4

    # Random colonies
    for _ in range(n_colonies):
        cx = torch.rand(1, device=device, dtype=dtype).item() * Lx
        cy = torch.rand(1, device=device, dtype=dtype).item() * Ly
        blob = torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * colony_radius**2))

        # Random fraction of N1 vs N2 per colony
        frac = torch.rand(1, device=device, dtype=dtype).item()
        peak_N1 = N1_amount * (0.2 + 1.6 * frac)       # varies 0.2x – 1.8x
        peak_N2 = N2_amount * (0.2 + 1.6 * (1 - frac))
        state[:, 0] += peak_N1 * blob
        state[:, 1] += peak_N2 * blob

    return state
