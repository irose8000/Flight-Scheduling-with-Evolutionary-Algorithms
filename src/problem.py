"""
Problem representation for single-runway departure scheduling.
Contains the scheduling problem definition and schedule representation.
"""

import numpy as np


class SingleMachineScheduling:
    """
    Represents the single-runway departure scheduling problem with wake turbulence constraints.
    """
    
    def __init__(self, n_jobs=5):
        """
        Initialize the scheduling problem.
        
        Args:
            n_jobs: Number of aircraft departures to schedule
        """
        self.n_jobs = n_jobs
        self.__initData()
    
    def __initData(self):
        """Initialize aircraft data and separation matrix."""
        # From San Diego International (Single Runway), organized by first arrived
        # Source: https://www.flightaware.com/
        all_jobs = [
            {"registration": "EJA11CA", "type": "C560", "wake turbulence category": 2},
            {"registration": "EJM83", "type": "E550", "wake turbulence category": 1},
            {"registration": "JAL65", "type": "B789", "wake turbulence category": 3},
            {"registration": "SKW3429", "type": "E170", "wake turbulence category": 2},
            {"registration": "DLH467", "type": "A359", "wake turbulence category": 4},
            {"registration": "EJA606", "type": "C68A", "wake turbulence category": 2},
            {"registration": "EJA874", "type": "C700", "wake turbulence category": 1},
            {"registration": "SKW3352", "type": "E75L", "wake turbulence category": 2},
            {"registration": "SWA2397", "type": "B737", "wake turbulence category": 2},
            {"registration": "EJA624", "type": "C68A", "wake turbulence category": 2}    
        ]

        # Only the first n_jobs of all jobs in the list are included
        self.jobs = all_jobs[:self.n_jobs]

        # Wake turbulence (WT) categories from https://mondortiz.com/wake-turbulence-categories-heavy-medium-and-light/
        # Aircraft class separation time (in seconds) from https://www.sciencedirect.com/science/article/pii/S0969699704000304
        self.sep_matrix = np.array([
            #1-Light 2-Medium 3-Large 4-Heavy
            [   154,    108,     90,    68],   # Light leading
            [   206,    108,     90,    68],   # Medium leading  
            [   257,    144,    120,    90],   # Large leading
            [   309,    180,    150,    90]    # Heavy leading
        ])
    
    def get_separation_time(self, wt_category, leading_wt_category):
        """
        Returns required separation time in seconds based on wake turbulence (wt) category 
        of the aircraft and the one that departed immediately before (leading).
        
        Args:
            wt_category: Wake turbulence category of following aircraft
            leading_wt_category: Wake turbulence category of leading aircraft
            
        Returns:
            Required separation time in seconds
        """
        return self.sep_matrix[leading_wt_category - 1][wt_category - 1]

    def get_cumulative_wait_time(self, wt_categories_schedule):
        """
        Calculate the total cumulative wait time for a given schedule.
        
        Args:
            wt_categories_schedule: List of wake turbulence categories in departure order
            
        Returns:
            Total cumulative wait time in seconds
        """
        cumulative_wait_time = 0
        wait_so_far = 0
        
        for i in range(len(wt_categories_schedule)):
            if i == 0:
                # First job has no wait time
                wait = 0
            else:
                # Get separation time required
                separation_time = self.get_separation_time(
                    wt_category=wt_categories_schedule[i], 
                    leading_wt_category=wt_categories_schedule[i-1]
                )
                               
                # Current job's wait time is the wait time so far plus separation
                wait = wait_so_far + separation_time

                # Augment wait so far for the next job
                wait_so_far += separation_time
            
            # For each job, the wait time is added
            cumulative_wait_time += wait
            
        return cumulative_wait_time


class Schedule:
    """
    Represents a departure schedule with methods for analysis and display.
    
    DEAP individuals are generally used in the EA instead of instances of the Schedule class,
    but converting individuals into Schedules for deeper analysis can be useful.
    """
    
    def __init__(self, jobs, job_indices=None):
        """
        Initialize a schedule.
        
        Args:
            jobs: List of job/aircraft dictionaries
            job_indices: Order of jobs (defaults to FIFO if not provided)
        """
        self.jobs = jobs
        
        # If the job indices aren't provided, they will automatically be arranged to match the FIFO method
        self.job_indices = job_indices or list(range(len(jobs)))
    
    @classmethod
    def from_individual(cls, individual, jobs):
        """
        Create a Schedule from a DEAP individual.
        
        Args:
            individual: DEAP individual (list of job indices)
            jobs: List of job/aircraft dictionaries
            
        Returns:
            Schedule instance
        """
        return cls(jobs, list(individual))

    def get_wts(self):
        """
        Create the wake turbulence categories list for get_cumulative_wait_time.
        
        Returns:
            List of wake turbulence categories in scheduled order
        """
        return [self.jobs[i]["wake turbulence category"] for i in self.job_indices]
    
    def get_registrations(self):
        """
        Return list of aircraft registrations in scheduled order.
        
        Returns:
            List of registration strings
        """
        return [self.jobs[i]["registration"] for i in self.job_indices]
    
    def get_aircraft_types(self):
        """
        Return list of aircraft types in scheduled order.
        
        Returns:
            List of aircraft type strings
        """
        return [self.jobs[i]["type"] for i in self.job_indices]
    
    def __len__(self):
        """Return the number of jobs in the schedule."""
        return len(self.job_indices)
    
    def __str__(self):
        """Return a formatted string representation of the schedule."""
        lines = ["Schedule Order:"]
        for i, job_idx in enumerate(self.job_indices):
            job = self.jobs[job_idx]
            lines.append(f"  {i+1}. {job['registration']} ({job['type']}) - WT: {job['wake turbulence category']}")
        return "\n".join(lines)