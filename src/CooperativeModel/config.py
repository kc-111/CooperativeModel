"""Configuration and parameters for the 3D bioreactor model.

Model: 4-species Liebig consumer-resource on a cyclic 4-cycle (batch mode);
see ``kinetics.py`` for the equations.

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

    # Monod half-saturation on each resource.  Scalar (not per-species)
    # for the symmetric base case: ``Liebig_i = min_{j in P_i} R_j / (K + R_j)``.
    # K also serves as the Hill K for the R-poison inhibition term, so
    # the natural scale that separates LO from HI is shared between
    # uptake and inhibition.
    K: float = 0.5

    # Hill exponent for the resource-poison inhibition term.  Each
    # species i is repressed by ``R_{(i+2) mod 4}`` via
    # ``inh_R = K^h_R / (K^h_R + R_poison^h_R)``.  h_R = 4 gives a sharp
    # switch at the K scale.
    h_R: float = 4.0

    # Per-species stoichiometric coefficient: each species i consumes
    # ``c_i * g_i * N_i`` from each of its two paired resources.
    c: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    # Per-species lactate yield: dL/dt = sum_i Y_i * g_i * N_i.  Slight
    # per-species asymmetry breaks degeneracy between the four pair
    # corners so the optimiser does not see four exactly-equal optima.
    Y: list = field(default_factory=lambda: [0.995, 1.001, 1.005, 0.999])

    # Toxin production per unit growth flux: ``dT_i/dt`` has a source
    # term ``beta * g_i * N_i``.  Tying production to growth (not just
    # biomass) means a poisoned / Liebig-starved species produces no
    # toxin, so the toxin pool reflects which species are *actually*
    # active, not just present.
    beta: float = 1.0

    # First-order toxin decay rate [time^-1].  Slow clearance lets toxin
    # pools persist after the producing species' paired resources are
    # depleted, which is what closes the "depletion cascade" channel.
    gamma: float = 0.1

    # Hill K for toxin inhibition.  ``inh_T = K_T^h_T / (K_T^h_T +
    # T_other^h_T)`` where ``T_other = T_tot - T_i``.  With beta = 1.0,
    # gamma = 0.1 and mature biomass ~ 1, the steady toxin pool is on
    # the order 1-10, so K_T = 0.5 puts the inhibition threshold well
    # below the active-species toxin level (suppression is strong) but
    # safely above the initial T = 0 (no spurious self-inhibition at
    # t = 0).
    K_T: float = 0.5

    # Hill exponent for toxin inhibition.  h_T = 4 matches h_R for a
    # sharp on/off transition at the K_T scale.
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
