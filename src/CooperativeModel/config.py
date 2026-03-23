"""Configuration and parameters for the 2D bioreactor model.

All kinetic parameters from Table 1 of the cooperative model description PDF.
Grid, diffusion, and solver settings are also defined here.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class ModelParameters:
    """Kinetic parameters from Table 1."""

    # Max growth rates [h^-1]
    mu1: list = field(default_factory=lambda: [0.53, 0.5, 0.6, 0.55])   # CoA on F1-F4
    mu2: list = field(default_factory=lambda: [0.68, 0.64, 0.61, 0.7])  # CoB on F1-F4

    # Max death rates [h^-1]
    dt1: float = 0.39   # CoA
    dt2: float = 0.34   # CoB

    # CoB growth scaling factor
    sigma: float = 1.5

    # Nisin parameters
    alpha: float = 0.33     # production constant
    kp: float = 8.0         # production saturation constant
    rb: float = 0.060       # basal production rate
    kn: float = 0.065       # degradation rate [h^-1]
    ks: float = 1.2e3       # death inhibition constant
    km: float = 0.014       # cooperative saturation constant

    # Monod half-saturation constants
    K1: list = field(default_factory=lambda: [0.19, 0.2, 0.18, 0.17])   # CoA
    K2: list = field(default_factory=lambda: [0.72, 0.75, 0.65, 0.6])   # CoB

    # Yield constants
    gamma1: list = field(default_factory=lambda: [0.6, 0.7, 0.72, 0.78])     # CoA
    gamma2: list = field(default_factory=lambda: [0.575, 0.625, 0.6, 0.5])   # CoB

    # Lactic acid production yield
    YL: float = 1.0

    # Diauxic shift sharpness (applied to CoA only)
    n: float = 2.0

    def to_tensors(self, device='cpu', dtype=torch.float64):
        """Convert parameters to tensors shaped for broadcasting over [B, 4, H, W]."""
        def _t(vals):
            return torch.tensor(vals, device=device, dtype=dtype).reshape(1, 4, 1, 1)

        return {
            'mu1': _t(self.mu1), 'mu2': _t(self.mu2),
            'dt1': self.dt1, 'dt2': self.dt2,
            'sigma': self.sigma,
            'alpha': self.alpha, 'kp': self.kp, 'rb': self.rb,
            'kn': self.kn, 'ks': self.ks, 'km': self.km,
            'K1': _t(self.K1), 'K2': _t(self.K2),
            'gamma1': _t(self.gamma1), 'gamma2': _t(self.gamma2),
            'YL': self.YL, 'n': self.n,
        }


@dataclass
class GridConfig:
    """Spatial grid configuration."""

    Nx: int = 50       # grid points in x
    Ny: int = 50       # grid points in y
    Lx: float = 1.0    # domain size in x [cm]
    Ly: float = 1.0    # domain size in y [cm]

    @property
    def dx(self):
        return self.Lx / self.Nx

    @property
    def dy(self):
        return self.Ly / self.Ny


@dataclass
class DiffusionConfig:
    """Diffusion coefficients for each species [cm^2/h].

    Bacteria diffuse slowly; small molecules diffuse faster.
    """

    D_N1: float = 1e-6    # CoA (bacteria)
    D_N2: float = 1e-6    # CoB (bacteria)
    D_Sn: float = 5e-4    # nisin (small peptide)
    D_L:  float = 5e-4    # lactic acid (small molecule)
    D_F1: float = 1e-4    # glucose
    D_F2: float = 1e-4    # fructose
    D_F3: float = 1e-4    # sucrose
    D_F4: float = 1e-4    # maltose

    def to_tensor(self, device='cpu', dtype=torch.float64):
        """Return [1, 8, 1, 1] tensor of diffusion coefficients."""
        return torch.tensor(
            [self.D_N1, self.D_N2, self.D_Sn, self.D_L,
             self.D_F1, self.D_F2, self.D_F3, self.D_F4],
            device=device, dtype=dtype,
        ).reshape(1, 8, 1, 1)


@dataclass
class SolverConfig:
    """ODE solver settings."""

    t_final: float = 24.0
    n_output: int = 49       # linspace(0, t_final, n_output) → every 0.5 h
    atol: float = 1e-6
    rtol: float = 1e-6
    h0: float = 0.01
    h_max: float = 10.0
    maxiters: int = 1000000


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""

    model: ModelParameters = field(default_factory=ModelParameters)
    grid: GridConfig = field(default_factory=GridConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    device: str = 'cpu'
    dtype: torch.dtype = torch.float64
