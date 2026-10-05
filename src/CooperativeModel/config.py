"""Kinetic parameters, grid dimensions, and solver settings."""

import torch
from dataclasses import dataclass, field


@dataclass
class ModelParameters:
    """Growth, resource consumption, product formation, and inhibition parameters."""

    # Per-species maximum growth rates [time^-1]
    mu: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    # Half-saturation for uptake and half-inhibition for unused resources.
    K: float = 0.5

    # Resource inhibition Hill exponent.
    h_R: float = 4.0

    # Consumption of each required resource per unit biomass growth.
    c: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    # Per-species product yields.
    Y: list = field(default_factory=lambda: [0.995, 1.001, 1.005, 0.999])

    # Toxin production per unit biomass growth.
    beta: float = 1.0

    # Toxin decay rate [time^-1].
    gamma: float = 0.1

    # Half-inhibition constant for other species' toxins.
    K_T: float = 0.5

    # Toxin inhibition Hill exponent.
    h_T: float = 4.0

    def to_tensors(self, device='cpu', dtype=torch.float64):
        """Convert parameters to tensors shaped for broadcasting over
        ``[B, 4, Nz, Ny, Nx]``."""
        def _t(vals):
            return torch.tensor(vals, device=device, dtype=dtype).reshape(1, 4, 1, 1, 1)

        return {
            'mu': _t(self.mu),
            'K': self.K,
            'h_R': self.h_R,
            'c': _t(self.c),
            'Y': _t(self.Y),
            'beta': self.beta,
            'gamma': self.gamma,
            'K_T': self.K_T,
            'h_T': self.h_T,
        }


@dataclass
class GridConfig:
    """Cartesian grid dimensions and cell sizes for the cylindrical reactor."""

    Nx: int = 32       # grid points in x
    Ny: int = 32       # grid points in y
    Nz: int = 32       # grid points in z (cylinder axis)
    Lx: float = 1.0    # domain size in x [cm]
    Ly: float = 1.0    # domain size in y [cm]
    Lz: float = 1.0    # domain size in z [cm]

    @property
    def dx(self):
        return self.Lx / self.Nx

    @property
    def dy(self):
        return self.Ly / self.Ny

    @property
    def dz(self):
        return self.Lz / self.Nz


@dataclass
class SolverConfig:
    """ODE solver settings."""

    t_final: float = 24.0
    n_output: int = 49       # linspace(0, t_final, n_output)
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
    solver: SolverConfig = field(default_factory=SolverConfig)
    device: str = 'cpu'
    dtype: torch.dtype = torch.float64
