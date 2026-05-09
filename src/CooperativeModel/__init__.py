"""3D Reaction-Advection Bioreactor Model for cooperative microbial consortia.

Two-stage architecture:
  Stage 1 — ``solve_steady_flow`` (slow, cached to HDF5).  Run once via
            ``scripts/solve_flow.py``.
  Stage 2 — ``Simulator(...).run()``: species transport on the cached
            flow (fast, reused for every BO evaluation).

Mixing in Stage 2 is driven entirely by chaotic advection from the
non-axisymmetric impeller body force — there is no explicit sub-grid
diffusion term.
"""

from .config import (
    SimulationConfig, ModelParameters, GridConfig, SolverConfig,
)
from .kinetics import compute_reaction_rates
from .velocity_fields import cylinder_mask, impeller_body_force, azimuthal_unit
from .flow_3d import solve_steady_flow, save_flow, load_flow
from .spatial_operators import Advection
from .model import simulate, BioreactorRHS, compute_cfl_limit
from .initial_conditions import uniform
from .simulate_ode import Simulator, SimResults

__all__ = [
    'SimulationConfig', 'ModelParameters', 'GridConfig', 'SolverConfig',
    'compute_reaction_rates',
    'cylinder_mask', 'impeller_body_force', 'azimuthal_unit',
    'solve_steady_flow', 'save_flow', 'load_flow',
    'Advection',
    'simulate', 'BioreactorRHS', 'compute_cfl_limit',
    'uniform',
    'Simulator', 'SimResults',
]
