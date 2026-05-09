"""Configuration and parameters for the 3D bioreactor model.

All kinetic parameters from Table 1 of the cooperative model description PDF.
Grid and solver settings are also defined here.

Note: explicit species diffusion has been removed.  Mixing in the 3D model is
driven entirely by chaotic advection from the non-axisymmetric impeller body
force; the first-order upwind advection contributes its own numerical
diffusion which is sufficient to smooth steep fronts without an explicit
sub-grid eddy-diffusivity term.
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

    # ks reduced from 1200 -> 400 to soften the bistable nisin-protection cliff
    # so the survival/death boundary moves into the interior of the F-box.
    # Combined with K_coop > 0 (co-limitation requires all sugars) and CCR-on-
    # nisin (penalises high F_total), this places multiple local L_final
    # maxima at strictly-interior F* in (0, 100)^4.

    # CoB growth scaling factor
    sigma: float = 1.5

    # Nisin parameters
    alpha: float = 0.33     # production constant
    kp: float = 8.0         # production saturation constant
    rb: float = 0.060       # basal production rate
    kn: float = 0.065       # degradation rate [h^-1]
    ks: float = 400.0       # death inhibition constant (was 1.2e3; see note above)
    km: float = 0.014       # cooperative saturation constant

    # Monod half-saturation constants
    K1: list = field(default_factory=lambda: [0.19, 0.2, 0.18, 0.17])   # CoA
    K2: list = field(default_factory=lambda: [0.72, 0.75, 0.65, 0.6])   # CoB

    # Haldane substrate-inhibition constants for CoA per sugar.
    # Peak of g1_i sits at F_i* = sqrt(K1_i * Ki_inh_i):
    #   F1* ~ 75, F2* ~ 50, F3* ~ 30, F4* ~ 60
    # Setting Ki_inh -> infinity recovers pure-Monod kinetics.
    Ki_inh: list = field(default_factory=lambda: [3.0e4, 1.25e4, 5.0e3, 2.1e4])

    # Yield constants
    gamma1: list = field(default_factory=lambda: [0.6, 0.7, 0.72, 0.78])     # CoA
    gamma2: list = field(default_factory=lambda: [0.575, 0.625, 0.6, 0.5])   # CoB

    # Lactic acid production yield
    YL: float = 1.0

    # Diauxic shift sharpness (applied to CoA only)
    n: float = 2.0

    # Lactic-acid product inhibition of CoA growth.
    # Effective growth rate: g1_total <- g1_total / (1 + (L/Kp_L)**hL).
    # Setting Kp_L -> infinity recovers no product inhibition.
    Kp_L: float = 35.0
    hL:   float = 2.0

    # Osmotic-stress death.  High total sugar increases death rate of both
    # strains via:  dt_eff = dt_j * (1 + c_osm * (F_total/K_osm)**h_osm).
    # Setting c_osm -> 0 recovers no osmotic stress.
    c_osm: float = 0.0
    K_osm: float = 220.0
    h_osm: float = 2.0

    # Per-sugar specific toxicity (saturating Hill form):
    #   dt_eff *= 1 + c_tox * sum_i Hill_h(F_i / K_tox_i)
    #   Hill_h(x) = x**h_tox / (1 + x**h_tox)         # capped at 1 per sugar
    # Each term saturates so the four sugars together contribute at most
    # 4*c_tox to the death-rate multiplier (vs. an unbounded sum in the
    # linear form).  h_tox > 1 makes the threshold sigmoidal — near-zero
    # below K_tox_i, sharp rise above — which keeps moderate-F regions
    # viable while still creating a clear "death zone" at very high F.
    # With c_tox=0.35, h_tox=4, max tox_factor ≈ 2.4 → max It ≈ 0.94 (no
    # nisin), within reach of g1_total ≈ 0.5 once Sn climbs above ~0.005.
    # Setting c_tox -> 0 recovers no per-sugar specific toxicity.
    c_tox: float = 0.4
    K_tox: list = field(default_factory=lambda: [85.0, 75.0, 55.0, 80.0])
    h_tox: float = 4.0

    # Acid (lactate)-induced death of CoA — separate from product inhibition
    # of growth.  Models pH-mediated cell death once L is high.
    #   dt1_eff = dt1 * (1 + c_pH * (L/K_pH)**h_pH)
    # Setting c_pH -> 0 recovers no acid death.
    c_pH:  float = 0.0
    K_pH:  float = 50.0
    h_pH:  float = 4.0

    # Multiplicative co-limitation factor for nisin biosynthesis.
    # The four sugars are treated as complementary, non-substitutable inputs
    # to a single secondary-metabolite flux:  Pm *= prod_i F_i / (K_coop + F_i).
    K_coop: float = 2.0

    # Carbon catabolite repression (CCR) of nisin biosynthesis.
    #   Pm *= 1 / (1 + (F_total / K_ccr)**h_ccr)
    K_ccr: float = 300.0
    h_ccr: float = 2.0

    def to_tensors(self, device='cpu', dtype=torch.float64):
        """Convert parameters to tensors shaped for broadcasting over [B, 4, Nz, Ny, Nx]."""
        def _t(vals):
            return torch.tensor(vals, device=device, dtype=dtype).reshape(1, 4, 1, 1, 1)

        return {
            'mu1': _t(self.mu1), 'mu2': _t(self.mu2),
            'dt1': self.dt1, 'dt2': self.dt2,
            'sigma': self.sigma,
            'alpha': self.alpha, 'kp': self.kp, 'rb': self.rb,
            'kn': self.kn, 'ks': self.ks, 'km': self.km,
            'K1': _t(self.K1), 'K2': _t(self.K2), 'Ki_inh': _t(self.Ki_inh),
            'gamma1': _t(self.gamma1), 'gamma2': _t(self.gamma2),
            'YL': self.YL, 'n': self.n,
            'Kp_L': self.Kp_L, 'hL': self.hL,
            'c_osm': self.c_osm, 'K_osm': self.K_osm, 'h_osm': self.h_osm,
            'c_pH':  self.c_pH,  'K_pH':  self.K_pH,  'h_pH':  self.h_pH,
            'K_coop': self.K_coop,
            'K_ccr': self.K_ccr, 'h_ccr': self.h_ccr,
            'c_tox': self.c_tox, 'K_tox': _t(self.K_tox), 'h_tox': self.h_tox,
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
