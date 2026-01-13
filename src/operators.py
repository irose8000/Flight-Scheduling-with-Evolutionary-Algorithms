"""
Genetic operators for the departure scheduling EA.
Includes custom crossover, mutation, and selection operators.
"""

import random
from collections import defaultdict


def cx_edge_recombination(ind1, ind2, creator):
    """
    Edge recombination crossover for permutation-based representation.
    
    Args:
        ind1: First parent individual
        ind2: Second parent individual
        creator: DEAP creator object for creating new individuals
        
    Returns:
        Tuple of two offspring individuals
    """
    def build_edge_table(parent1, parent2):
        edge_table = defaultdict(set)

        # Add edges from parent1
        for i in range(len(parent1)):
            current = parent1[i]
            prev_node = parent1[i-1]
            next_node = parent1[(i+1) % len(parent1)]
            edge_table[current].update([prev_node, next_node])

        # Add edges from parent2
        for i in range(len(parent2)):
            current = parent2[i]
            prev_node = parent2[i-1]
            next_node = parent2[(i+1) % len(parent2)]
            edge_table[current].update([prev_node, next_node])

        return edge_table

    def create_offspring(edge_table, start_node):
        offspring = [start_node]
        current = start_node
        remaining = set(range(len(ind1))) - {start_node}
        
        while remaining:
            # Remove current node from all edge lists
            for node in edge_table:
                edge_table[node].discard(current)
            
            # Find next node
            neighbors = edge_table[current] & remaining
            
            if neighbors:
                # Choose neighbor with fewest connections, break ties randomly
                next_node = min(neighbors, key=lambda x: len(edge_table[x] & remaining))
            else:
                # Random selection if no neighbors available
                next_node = random.choice(list(remaining))
            
            offspring.append(next_node)
            remaining.remove(next_node)
            current = next_node
        return offspring

    # Going back to original edge recombination function given parents
    edge_table = build_edge_table(ind1, ind2)
    offspring1 = create_offspring(edge_table.copy(), random.choice(ind1))
    offspring2 = create_offspring(edge_table.copy(), random.choice(ind2))

    return creator.Individual(offspring1), creator.Individual(offspring2)


def cx_mixed(ind1, ind2, ordered_not_edge_prob, creator, tools):
    """
    Mixed crossover: uses either order-one crossover or edge recombination.
    
    Args:
        ind1: First parent individual
        ind2: Second parent individual
        ordered_not_edge_prob: Probability of using order-one crossover vs edge recombination
        creator: DEAP creator object
        tools: DEAP tools module for cxOrdered
        
    Returns:
        Tuple of two offspring individuals
    """
    if random.random() < ordered_not_edge_prob:
        return tools.cxOrdered(ind1, ind2)
    else:
        # No need to update fitness values, as they are not created as part of edge recombination function
        return cx_edge_recombination(ind1, ind2, creator)


def mut_insertion(individual):
    """
    Insertion mutation operator.
    
    Selects two positions and moves the second element to after the first position.
    
    Args:
        individual: Individual to mutate
        
    Returns:
        Tuple containing the mutated individual
    """
    # Select first allele index (any but the last)
    first_allele_index = random.randrange(len(individual)-1)
    
    # Select second allele index (any after the first one)
    second_allele_index = random.randrange(first_allele_index+1, len(individual))
    
    # Remove the second allele and insert after first
    second_allele = individual.pop(second_allele_index)
    individual.insert(first_allele_index+1, second_allele)
    
    return individual,


def calculate_ranked_probabilities(population, linear_ranking_s):
    """
    Calculate selection probabilities using linear ranking.
    
    Args:
        population: List of individuals
        linear_ranking_s: Selection pressure parameter (1.0 to 2.0)
        
    Returns:
        List of selection probabilities for each individual
    """
    N = len(population)
    # Sort population and assign probabilities
    sorted_population = sorted(enumerate(population), key=lambda x: x[1].fitness.values[0])
    
    # Initialize a list of zeroes with length N
    selection_probabilities = [0.0] * N
    
    for rank, (orig_idx, ind) in enumerate(sorted_population):
        selection_prob = (2 - linear_ranking_s) / N + 2 * rank * (linear_ranking_s - 1) / (N * (N - 1))
        selection_probabilities[orig_idx] = selection_prob

    return selection_probabilities