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

## For a full walkthrough, check out notebooks/flight_departure_scheduling_EA_final_draft.ipynb

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
