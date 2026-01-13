# example_run.py
from deap import base, creator, tools
from src.problem import SingleMachineScheduling
from src.ea import departure_schedule_ea
from src.operators import cx_mixed, mut_insertion
import random

# Setup problem
N_DEPARTURES = 10
scheduler = SingleMachineScheduling(n_jobs=N_DEPARTURES)

# Setup DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Register creation functions
def create_individual():
    return creator.Individual(random.sample(range(N_DEPARTURES), N_DEPARTURES))

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register evaluation
def evaluate_schedule(individual):
    wt_categories = [scheduler.jobs[i]["wake turbulence category"] for i in individual]
    return (scheduler.get_cumulative_wait_time(wt_categories),)

toolbox.register("evaluate", evaluate_schedule)

# Register operators
toolbox.register("mate", lambda ind1, ind2: cx_mixed(ind1, ind2, 0.5, creator, tools))
toolbox.register("mutate", mut_insertion)
toolbox.register("clone", lambda ind: creator.Individual(ind[:]))

# Run EA
population = toolbox.population(n=20)
final_pop, logbook = departure_schedule_ea(
    population, toolbox, ngen=100,
    cxpb=0.9, mutpb=0.2, verbose=True
)

# Get best solution
best = min(final_pop, key=lambda x: x.fitness.values[0])
print(f"\nBest fitness: {best.fitness.values[0]}")