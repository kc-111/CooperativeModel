"""Configuration and parameters for the 3D bioreactor model.

Model: 4-species Liebig consumer-resource on a cyclic 4-cycle (batch mode
with cross-feeding); see ``kinetics.py`` for the equations.

Grid and solver settings are also defined here.  Explicit species
diffusion is not modelled: mixing in the 3D run is driven entirely by
chaotic advection from the non-axisymmetric impeller body force, plus the
modest numerical diffusion contributed by first-order upwind advection.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class ModelParameters:
    """Kinetic parameters for the 4-species Liebig consumer-resource model.

    The symmetric defaults below give four equal-height local optima of
    ``L(t_final)`` at the four "two-resources-high, two-resources-low"
    corners of the initial-condition box.  Symmetry can be broken by
    setting different per-species ``mu_i`` or ``Y_i``.
    """

    # Per-species maximum growth rates [time^-1]
    mu: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    # Monod half-saturation constant on each resource.  Scalar (not
    # per-species) for the symmetric base case: ``g_i = mu_i * min_{j in P_i}
    # R_j / (K + R_j)``.
    K: float = 0.5

    # Per-species stoichiometric coefficient: each species i consumes
    # ``c_i * g_i * N_i`` from each of its two paired resources.
    c: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    # Per-species lactate yield: dL/dt = sum_i Y_i * g_i * N_i.
    Y: list = field(default_factory=lambda: [0.995, 1.001, 1.005, 0.999])

    # Cross-feeding strength.  Each species secretes its antipodal resource
    # (the one not in its growth pair) at rate ``sigma * g_i * N_i``.  Small
    # sigma keeps the four single-pair corners well-separated; large sigma
    # bridges missing resources and erodes the multimodality.
    sigma: float = 0.1

    # Toxin production rate per unit nutrient-uptake flux: dT_i/dt has a
    # source term ``beta * g_i * N_i``.  Each species secretes its own
    # species-specific toxin and is immune to it.
    beta: float = 1.0

    # First-order toxin decay rate [time^-1].  Slow clearance lets toxin
    # pools build up during co-existence so the linear death term has time
    # to bite at any coexistence point (all-max **and** interior).
    gamma: float = 0.1

    # Linear toxin kill coefficient.  Each species i loses biomass at
    # ``delta * T_other * N_i``.  Sized so that multi-species partial
    # growth on intermediate R is killed harder than single-species
    # growth at a pair corner — with the now-tight Y the four pair
    # corners are nearly degenerate in L and any interior coexistence
    # point would otherwise beat them by stacking four partial-growth
    # contributions.  A strong linear kill restores the corner-as-optimum
    # topology.
    delta: float = 10.0

    def to_tensors(self, device='cpu', dtype=torch.float64):
        """Convert parameters to tensors shaped for broadcasting over
        ``[B, 4, Nz, Ny, Nx]``."""
        def _t(vals):
            return torch.tensor(vals, device=device, dtype=dtype).reshape(1, 4, 1, 1, 1)

        return {
            'mu': _t(self.mu),
            'K': self.K,
            'c': _t(self.c),
            'Y': _t(self.Y),
            'sigma': self.sigma,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
        }


@dataclass
class GridConfig:
    """3D Cartesian grid configuration.

    The cylinder axis is z; the cylinder is inscribed in the (x, y) cross-section
    of the cube. Wall mask is built by ``velocity_fields.cylinder_mask(grid)``.
    """

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
