"""
EA performance analyzer for testing and visualization.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from src.problem import Schedule


class EAAnalyzer:
    """
    Framework for analyzing EA performance across multiple runs.
    """
    
    def __init__(self, scheduler, toolbox):
        """
        Initialize the analyzer.
        
        Args:
            scheduler: SingleMachineScheduling instance
            toolbox: DEAP toolbox with registered operators
        """
        self.scheduler = scheduler
        self.toolbox = toolbox
        self.results = {}

    def run_tests(self, n_runs=1, verbose=True, **ea_params):
        """
        Run the EA multiple times and collect results.
        
        Args:
            n_runs: Number of independent runs
            verbose: Whether to print progress
            **ea_params: Parameters to pass to the EA
            
        Returns:
            Dictionary of aggregated results
        """
        from src.ea import departure_schedule_ea
        
        print("Starting EA analysis...")
        all_results = []

        for run in range(n_runs):
            print(f"\n=== Run {run + 1}/{n_runs} ===")

            # Create fresh population for each run
            population = self.toolbox.population(ea_params.get('population_size', 20))

            start_time = time.time()
            try:
                final_pop, logbook = departure_schedule_ea(
                    population, 
                    self.toolbox, 
                    ea_params.get('generations', 100),
                    cxpb=ea_params.get('crossover_prob', 0.9),
                    mutpb=ea_params.get('mutation_prob', 0.2),
                    tournament_size=ea_params.get('tournament_size', 3),
                    generation_switch_threshold=ea_params.get('generation_switch_threshold', 0.7),
                    sigma_share=ea_params.get('sigma_share', 0.5),
                    alpha_share=ea_params.get('alpha_share', 1.0),
                    linear_ranking_s=ea_params.get('linear_ranking_s', 2.0),
                    verbose=verbose
                )
            except Exception as e:
                print(f"Error in run {run + 1}: {e}")
                final_pop = population

            best_individual = min(final_pop, key=lambda x: x.fitness.values[0])
            best_fitness = best_individual.fitness.values[0]
            end_time = time.time()

            runtime = end_time - start_time

            # Store results
            run_result = {
                'run': run,
                'runtime': runtime,
                'final_population': final_pop,
                'logbook': logbook,
                'best_individual': best_individual,
                'best_fitness': best_fitness
            }
            all_results.append(run_result)

            print(f"Run {run + 1} completed in {runtime:.2f}s")
            print(f"Best fitness: {best_fitness:.2f}")

        runtimes = [result["runtime"] for result in all_results]
        fitnesses = [result['best_fitness'] for result in all_results]

        self.results = {
            'all_results': all_results,
            'avg_runtime': np.mean(runtimes),
            'std_runtime': np.std(runtimes),
            'best_overall_fitness': min(fitnesses),
            'best_ind_last_gen': all_results[-1]['best_individual']
        }

        print(f"\n=== Overall Results ===")
        print(f"Average runtime: {self.results['avg_runtime']:.2f} ± {self.results['std_runtime']:.2f}s")
        print(f"Best overall fitness: {self.results['best_overall_fitness']:.2f}")

        optimal_sched = Schedule.from_individual(self.results['best_ind_last_gen'], self.scheduler.jobs)
        print(f"Optimized schedule:")
        print(optimal_sched)

        return self.results
        
    def generate_report(self, run, fifo_baseline):
        """
        Generate visualization of EA performance.
        
        Args:
            run: Which run to visualize (1-indexed)
            fifo_baseline: FIFO baseline fitness for comparison
        """
        logbook = self.results["all_results"][run-1]["logbook"]
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        fifo_fitness = np.full(len(gen), fifo_baseline)
        diversities = logbook.select("diversity")

        fig, ax1 = plt.subplots()

        # Primary axis plots
        line1 = ax1.plot(gen, fifo_fitness, "r-", label="FIFO Fitness")
        line2 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
        
        ax1.set_ylim(0, 8000)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.tick_params(axis='y', labelcolor='black')

        # Create second axis
        ax2 = ax1.twinx()
        line3 = ax2.plot(gen, diversities, "g--", label="Diversity")
        ax2.set_ylabel("Diversity", color="g")
        ax2.set_ylim(15, 25)
        ax2.tick_params(axis='y', labelcolor='g')

        # Legend from all lines
        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.title("EA Report")
        plt.tight_layout()
        plt.show()