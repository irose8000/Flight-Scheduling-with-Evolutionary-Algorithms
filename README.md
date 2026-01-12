# Flight Departure Scheduling with Evolutionary Algorithms

An evolutionary algorithm solution for optimizing aircraft departure schedules at single-runway airports. Minimizes cumulative wait times while respecting wake turbulence separation requirements.

## Overview

Air traffic congestion at bottleneck runways creates costly delays. This project implements an EA that finds departure schedules significantly better than the standard First-In-First-Out (FIFO) approach.

### Key Features

- Order-based permutation representation for departure sequences
- Hybrid parent selection (SUS + Tournament Selection)
- Mixed crossover operators (Order-one + Edge Recombination)
- Fitness sharing for diversity preservation
- Parameter tuning framework with visualization

## Installation

### Prerequisites
- Python 3.8+

### Setup
```bash
git clone https://github.com/yourusername/Flight-Scheduling-with-Evolutionary-Algorithms.git
cd Flight-Scheduling-with-Evolutionary-Algorithms
pip install -r requirements.txt
```

### Dependencies
- numpy
- matplotlib
- deap

## Quick Start

### Basic Usage
```python
from deap import base, creator, tools
from src.problem import SingleMachineScheduling
from src.ea import departure_schedule_ea
import random

# Setup problem
scheduler = SingleMachineScheduling(n_jobs=10)

# Setup DEAP (see notebooks/exploration.ipynb for full example)
# ... register operators ...

# Run EA
population = toolbox.population(n=20)
final_pop, logbook = departure_schedule_ea(
    population, toolbox, ngen=100,
    cxpb=0.9, mutpb=0.2
)

# Get best solution
best = min(final_pop, key=lambda x: x.fitness.values[0])
print(f"Best fitness: {best.fitness.values[0]}")
```

### Analysis
```python
from analysis.analyzer import EAAnalyzer

analyzer = EAAnalyzer(scheduler, toolbox)
results = analyzer.run_tests(n_runs=5, verbose=True)
analyzer.generate_report(run=1, fifo_baseline=5287)
```

### Parameter Tuning
```python
from analysis.tuner import ParameterTuner

tuner = ParameterTuner(scheduler, toolbox)
analysis = tuner.run_parameter_tuning(n_samples=15, n_runs=2)
```

## Algorithm Overview

- **Representation**: Order-based permutations (departure sequences)
- **Selection**: Hybrid SUS (early) → Tournament (late)
- **Crossover**: Order-one + Edge Recombination (90% rate)
- **Mutation**: Insertion (20% rate)
- **Diversity**: Fitness sharing with Kendall tau distance
- **Survival**: Generational model with elitism

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| population_size | 20 | Number of individuals |
| crossover_prob | 0.9 | Crossover rate |
| mutation_prob | 0.2 | Mutation rate |
| generations | 100 | Number of generations |
| tournament_size | 3 | Tournament selection size |


## References

- Bianco et al. (2006). Scheduling Models for Air Traffic Control in Terminal Areas
- Farrahi et al. (2017). Applying Graph Theory to Problems in Air Traffic Management
- Full references in [docs/references.md](docs/references.md)
