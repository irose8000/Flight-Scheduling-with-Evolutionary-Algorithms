"""
Flight Departure Scheduling using Evolutionary Algorithms
"""

__version__ = "0.1.0"

from .problem import SingleMachineScheduling, Schedule
from .operators import (
    cx_edge_recombination,
    cx_mixed,
    mut_insertion,
    calculate_ranked_probabilities
)
from .diversity import (
    kendall_tau_distance,
    compute_sample_diversity,
    sharing_distance,
    sharing_function,
    calculate_shared_fitness
)
from .ea import departure_schedule_ea

__all__ = [
    'SingleMachineScheduling',
    'Schedule',
    'cx_edge_recombination',
    'cx_mixed',
    'mut_insertion',
    'calculate_ranked_probabilities',
    'kendall_tau_distance',
    'compute_sample_diversity',
    'sharing_distance',
    'sharing_function',
    'calculate_shared_fitness',
    'departure_schedule_ea'
]