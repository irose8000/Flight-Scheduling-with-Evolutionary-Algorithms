"""
Diversity preservation mechanisms for the EA.
Includes fitness sharing and diversity metrics.
"""

import random
import numpy as np
from itertools import combinations


def kendall_tau_distance(perm1, perm2):
    """
    Calculate the Kendall tau distance between two permutations.
    
    The Kendall tau distance counts the number of pairwise disagreements between two rankings.
    
    Args:
        perm1: First permutation
        perm2: Second permutation
        
    Returns:
        Number of inversions (pairwise disagreements)
    """
    n = len(perm1)
    pos2 = {val: i for i, val in enumerate(perm2)}
    transformed = [pos2[val] for val in perm1]
    
    # Count inversions in the transformed sequence
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if transformed[i] > transformed[j]:
                inversions += 1
    
    return inversions


def compute_sample_diversity(population, max_sample_size=100):
    """
    Compute diversity statistics using sampling method.
    
    This allows regular monitoring without excessive computational strain.
    
    Args:
        population: List of individuals
        max_sample_size: Maximum number of pairs to sample
        
    Returns:
        Mean Kendall tau distance of sampled pairs
    """
    n_pop = len(population)

    # Sample random pairs
    sample_size = min(max_sample_size, n_pop * (n_pop - 1) // 2)  # Maximum possible inversions
    pairs = random.sample(list(combinations(range(n_pop), 2)), sample_size)

    # Calculate sample pair distances
    distances = [kendall_tau_distance(population[i], population[j]) for i, j in pairs]

    return np.mean(distances) if distances else 0


def sharing_distance(ind1, ind2):
    """
    Calculate sharing distance between two individuals using normalized Kendall tau.
    
    Args:
        ind1: First individual
        ind2: Second individual
        
    Returns:
        Normalized distance between 0 and 1
    """
    max_distance = len(ind1) * (len(ind1) - 1) // 2  # Maximum possible inversions
    kt_distance = kendall_tau_distance(ind1, ind2)
    return kt_distance / max_distance if max_distance > 0 else 0


def sharing_function(distance, sigma_share, alpha):
    """
    Calculate sharing function value.
    
    Args:
        distance: Distance between two individuals
        sigma_share: Sharing radius
        alpha: Sharing function shape parameter
        
    Returns:
        Sharing value
    """
    if distance < sigma_share:
        return 1 - (distance / sigma_share) ** alpha
    return 0


def calculate_shared_fitness(population, sigma_share, alpha_share):
    """
    Calculate shared fitness for all individuals in population.
    
    Args:
        population: List of individuals
        sigma_share: Sharing radius
        alpha_share: Sharing function shape parameter
        
    Returns:
        List of shared fitness values
    """
    shared_fitnesses = []
    
    for i, ind in enumerate(population):
        raw_fitness = ind.fitness.values[0]
        niche_count = 0
        
        # Calculate niche count (sum of sharing function values)
        for j, other_ind in enumerate(population):
            distance = sharing_distance(ind, other_ind)
            niche_count += sharing_function(distance, sigma_share, alpha_share)
        
        # Shared fitness = raw fitness / niche count
        shared_fitness = raw_fitness / niche_count if niche_count > 0 else raw_fitness
        shared_fitnesses.append(shared_fitness)
    
    return shared_fitnesses