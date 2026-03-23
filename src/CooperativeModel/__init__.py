"""2D Reaction-Diffusion Bioreactor Model for cooperative microbial consortia."""

from .simulate_ode import Simulator, SimResults
from .config import SimulationConfig, ModelParameters, GridConfig, DiffusionConfig, SolverConfig
from .model import simulate, BioreactorRHS
from .kinetics import compute_reaction_rates
