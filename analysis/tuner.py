"""
Parameter tuning framework for the EA.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from .analyzer import EAAnalyzer


class ParameterTuner:
    """
    Framework for systematic parameter tuning using the GENERATE and TEST principle.
    """
    
    def __init__(self, scheduler, toolbox):
        """
        Initialize the parameter tuner.
        
        Args:
            scheduler: SingleMachineScheduling instance
            toolbox: DEAP toolbox with registered operators
        """
        self.scheduler = scheduler
        self.toolbox = toolbox
        self.results = []
        
        # Define default parameter ranges for tuning
        self.parameter_ranges = {
            'crossover_prob': [0.5, 0.7, 0.9],
            'mutation_prob': [0.1, 0.3, 0.5],
            'tournament_size': [2, 3, 5],
            'generation_switch_threshold': [0.5, 0.7, 0.9],
            'mixed_crossover_prob': [0.3, 0.5, 0.7],
            'population_size': [15, 30, 45],
            'sigma_share': [1.5, 2.0, 2.5],
            'linear_ranking_s': [1.5, 2.0, 2.5]
        }
        
    def generate_parameter_vectors(self):
        """
        Generate all combinations using full factorial design.
        
        Returns:
            List of parameter dictionaries
        """
        param_names = list(self.parameter_ranges.keys())
        param_values = list(self.parameter_ranges.values())
        
        parameter_vectors = []
        
        # Create all combinations
        for combo in itertools.product(*param_values):
            param_vector = dict(zip(param_names, combo))
            parameter_vectors.append(param_vector)
        
        print(f"Generated {len(parameter_vectors)} parameter vectors to test")
        return parameter_vectors
    
    def generate_reduced_parameter_vectors(self, n_samples=50):
        """
        Generate a reduced set using random sampling for computational efficiency.
        
        Args:
            n_samples: Number of parameter combinations to sample
            
        Returns:
            List of parameter dictionaries
        """
        param_names = list(self.parameter_ranges.keys())
        parameter_vectors = []
        
        for _ in range(n_samples):
            param_vector = {}
            for param_name in param_names:
                param_vector[param_name] = np.random.choice(self.parameter_ranges[param_name])
            parameter_vectors.append(param_vector)
        
        print(f"Generated {len(parameter_vectors)} reduced parameter vectors to test")
        return parameter_vectors
    
    def test_parameter_vector(self, param_vector, n_runs=3, generations=50):
        """
        Test a single parameter combination using the EAAnalyzer.
        
        Args:
            param_vector: Dictionary of parameter values
            n_runs: Number of runs per parameter combination
            generations: Number of generations to run
            
        Returns:
            Dictionary of results for this parameter combination
        """
        print(f"Testing parameters: {param_vector}")
        
        # Use existing EAAnalyzer to run tests
        analyzer = EAAnalyzer(self.scheduler, self.toolbox)
        ea_results = analyzer.run_tests(
            n_runs=n_runs, 
            verbose=False,
            generations=generations,
            **param_vector
        )
        
        # Extract results
        fitnesses = [result['best_fitness'] for result in ea_results['all_results']]
        
        result = {
            'parameters': param_vector.copy(),
            'avg_fitness': np.mean(fitnesses),
            'std_fitness':