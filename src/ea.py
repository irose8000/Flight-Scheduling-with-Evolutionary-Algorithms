"""
Main evolutionary algorithm for departure scheduling.
"""

import random
from deap import tools


def departure_schedule_ea(
    population, 
    toolbox, 
    ngen, 
    cxpb=0.7, 
    mutpb=0.3,
    tournament_size=3,
    generation_switch_threshold=0.7,
    sigma_share=0.5,
    alpha_share=1.0,
    linear_ranking_s=2.0,
    verbose=False
):
    """
    Execute the evolutionary algorithm for departure scheduling.
    
    Args:
        population: Initial population
        toolbox: DEAP toolbox with registered operators
        ngen: Number of generations
        cxpb: Crossover probability
        mutpb: Mutation probability
        tournament_size: Size of tournament for tournament selection
        generation_switch_threshold: When to switch from SUS to tournament selection
        sigma_share: Sharing radius for fitness sharing
        alpha_share: Sharing function shape parameter
        linear_ranking_s: Selection pressure for linear ranking
        verbose: Whether to print progress
        
    Returns:
        Tuple of (final_population, logbook)
    """
    from .diversity import compute_sample_diversity, calculate_shared_fitness
    from .operators import calculate_ranked_probabilities
    
    # Setup for evaluation
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x))
    stats.register("std", lambda x: (sum((xi - sum(x)/len(x))**2 for xi in x) / len(x))**0.5)
    stats.register("min", min)
    stats.register("max", max)
    logbook = tools.Logbook()
    
    # Initialization
    initial_fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, initial_fitnesses):
        ind.fitness.values = fit

    # EA Loop
    for generation in range(ngen):
        
        # Evaluation
        fitnesses = [ind.fitness.values[0] for ind in population]
        diversity = compute_sample_diversity(population)
        current_best = min(population, key=lambda x: x.fitness.values[0])
        record = stats.compile(population)
        logbook.record(gen=generation, diversity=diversity, **record)

        if verbose:
            print(f"Gen {generation}: diversity of {diversity:.2f}")
            print(record)
        
        # Parent Selection
        # Based on stage of optimization
        generation_ratio = generation / ngen

        # Diversity preservation: fitness sharing
        shared_fitnesses = calculate_shared_fitness(population, sigma_share, alpha_share)
        # Store original fitnesses to restore these values to population after parent selection has been performed
        original_fitnesses = [ind.fitness.values for ind in population]
        # Assign new shared fitness values for during parent selection
        for ind, shared_fit in zip(population, shared_fitnesses):
            ind.fitness.values = (shared_fit,)

        # Early stage: ranking & stochastic universal sampling for exploration
        if generation_ratio < generation_switch_threshold:
            # Reassign "fitnesses" (not actually fitnesses, but instead selection probabilities) 
            # based on shared fitnesses using linear ranking
            for ind, selection_prob in zip(population, calculate_ranked_probabilities(population, linear_ranking_s)):
                ind.fitness.values = (selection_prob,)

            # Select parents using stochastic universal sampling
            parents = tools.selStochasticUniversalSampling(population, len(population))

        # Later stage: Tournament selection for slightly less exploration
        else:
            parents = tools.selTournament(population, len(population), tournsize=tournament_size)

        # Restore original (not shared) fitness values
        for ind, orig_fit in zip(population, original_fitnesses):
            ind.fitness.values = orig_fit

        # Variation Operators
        offspring = []
        for i in range(0, len(parents), 2):
            parent1_copy = toolbox.clone(parents[i])

            if i + 1 < len(parents):
                parent2_copy = toolbox.clone(parents[i+1])
                # Crossover 
                if random.random() < cxpb:
                    child1, child2 = toolbox.mate(parent1_copy, parent2_copy)
                else:
                    child1, child2 = parent1_copy, parent2_copy
                
                offspring.extend([child1, child2])
            
            else:
                child1 = parent1_copy
                offspring.append(child1)
        
        # Mutation
        for o in offspring:
            if random.random() < mutpb:
                toolbox.mutate(o)

        # Update fitness evaluations
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Survivor Selection
        # Generational model
        population = offspring[:]
        # Elitism: Replace worst individual with best from previous generation
        worst_idx = max(range(len(population)), key=lambda i: population[i].fitness.values[0])
        population[worst_idx] = current_best
        
    return population, logbook