"""Bioreactor simulations with reaction kinetics and cached fluid flow."""

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
