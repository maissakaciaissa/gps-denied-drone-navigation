"""
PART 5: Simulation Engine
Main simulation loop implementation that coordinates everything.
"""

import time
from typing import Dict, List, Tuple
from backend.core.environment import Environment
from backend.core.drone import Drone
from backend.core.sensor import DroneSensor
from backend.game_theory.minimax import Minimax
from backend.game_theory.nash import NashEquilibriumSolver
from backend.game_theory.bayesian import BayesianGameSolver
from backend.game_theory.payoff import PayoffFunction
from backend.game_theory.strategies import DroneAction
from backend.simulation.logger import SimulationLogger
from backend.simulation.metrics import PerformanceMetrics


class SimulationEngine:
    """
    The brain that coordinates everything.
    Runs the simulation step-by-step using the selected algorithm.
    """
    
    def __init__(self, 
        environment: Environment, drone: Drone, payoff_function: PayoffFunction, algorithm_mode: str = "minimax"
    ):
        """
        Initialize the simulation engine.
        
        Args:
            environment: Environment instance
            drone: Drone instance
            payoff_function: PayoffFunction instance
            algorithm_mode: Options are "minimax", "nash" or "bayesian"
        """
        self.environment = environment
        self.drone = drone
        self.payoff_function = payoff_function
        
        # Sensor for environment perception
        self.sensor = DroneSensor(detection_range=5, initial_visibility=1.0)
        
        # Algorithm solvers
        self.minimax_solver = Minimax(payoff_function)
        self.nash_solver = NashEquilibriumSolver(payoff_function)
        self.bayesian_solver = BayesianGameSolver(payoff_function)
        
        # Simulation state
        self.step_counter = 0
        self.logger = SimulationLogger()
        self.metrics = PerformanceMetrics(environment, drone)
        
        # Current algorithm mode
        self.algorithm_mode = algorithm_mode
        
    def step(self, verbose: bool = False) -> Tuple[bool, str]:
        """
        Execute one time step of the simulation.
        
        1. Select which algorithm to use
        2. Get the recommended action from that algorithm
        3. Execute the action (move the drone)
        4. Update the environment state
        5. Log what happened
        6. Check if done (reached goal, crashed, or out of battery)
        
        Args:
            verbose: If True, print step details
            
        Returns:
            Tuple of (done, status_message)
            - done: True if simulation should stop
            - status_message: Reason for stopping or status update
        """
        self.step_counter += 1
        
        # Get state parameters for decision-making
        state_params = self._get_state_params()
        
        # Get available actions
        available_actions = self.drone.get_valid_actions()
        
        if not available_actions:
            status = "No valid actions available"
            self.logger.log_event('no_actions', status)
            return True, status
        
        # Select action using the current algorithm
        action = self.select_action(available_actions, state_params, verbose)
        
        # Execute the action
        old_position = self.drone.position
        success = self.drone.move(action)
        
        if not success:
            # Move failed (collision or invalid)
            status = f"Move failed at step {self.step_counter}"
            self.logger.log_event('collision', status, {'position': old_position})
            return True, status
        
        # Use sensor to detect current environment condition
        env_condition = self.sensor.sense_environment_condition(self.environment, self.drone.position)
        
        # Update sensor visibility based on detected condition
        self.sensor.update_visibility(env_condition)
        
        # Calculate payoff for this step
        distance_to_nearest_obstacle = self.environment.get_nearest_obstacle_distance(self.drone.position)
        if distance_to_nearest_obstacle is None:
            distance_to_nearest_obstacle = float('inf')  # No obstacles nearby
        
        drone_payoff, env_payoff = self.payoff_function.compute_payoff(
            action, env_condition,
            self.drone.position,
            self.environment.goal_pos,
            self.environment.distance_to_goal(self.environment.start_pos),
            self.drone.battery_capacity - self.drone.current_battery,
            self.drone.battery_capacity,
            distance_to_nearest_obstacle,
            len(self.drone.explored_cells),
            self.environment.width * self.environment.height,
            False,
            self.environment
        )
        
        # Update Bayesian beliefs if using Bayesian algorithm
        if self.algorithm_mode == "bayesian":
            self.bayesian_solver.update_beliefs(action, env_condition, state_params)
        
        # Log this step
        self.logger.log_step(
            step_number=self.step_counter,
            action=action.value,
            position=self.drone.position,
            battery_level=self.drone.current_battery,
            payoff=drone_payoff,
            distance_to_goal=self.environment.distance_to_goal(self.drone.position),
            additional_data={
                'algorithm': self.algorithm_mode,
                'env_condition': env_condition.value
            }
        )
        
        if verbose:
            print(
                f"Step {self.step_counter}: {action.value} → {self.drone.position}, "
                f"Battery: {self.drone.current_battery:.1f}, Payoff: {drone_payoff:.2f}"
            )
        
        # Check termination conditions
        if self.environment.is_goal_reached(self.drone.position):
            status = f"Goal reached at step {self.step_counter}!"
            self.logger.log_event('goal_reached', status, {'position': self.drone.position})
            return True, status
        
        if self.drone.current_battery <= 0:
            status = f"Battery depleted at step {self.step_counter}"
            self.logger.log_event('battery_depleted', status)
            return True, status
        
        # Continue simulation
        return False, "Running"
    
    def select_action(self, 
        available_actions: List[DroneAction], state_params: Dict, verbose: bool = False
    ) -> DroneAction:
        """
        Route to the appropriate algorithm based on the current mode.
        
        Args:
            available_actions: List of valid actions
            state_params: Current state parameters
            verbose: If True, print algorithm analysis
            
        Returns:
            Recommended drone action
        """
        if self.algorithm_mode == "minimax":
            return self.minimax_solver.solve(available_actions, state_params, verbose)
        
        elif self.algorithm_mode == "nash":
            return self.nash_solver.solve(state_params, available_actions)
        
        elif self.algorithm_mode == "bayesian":
            return self.bayesian_solver.solve(available_actions, state_params, verbose)
        
        else:
            raise ValueError(f"Unknown algorithm mode: {self.algorithm_mode}")
    
    def run_simulation(self, 
        max_steps: int = 1000, algorithm: str = "minimax", verbose: bool = False
    ) -> Dict:
        """
        Run the complete simulation from start to finish.
        
        Args:
            max_steps: Maximum number of steps before timeout
            algorithm: Which algorithm to use ("minimax", "nash", or "bayesian")
            verbose: If True, print step details
            
        Returns:
            Dictionary containing final metrics
        """
        # Set algorithm mode
        self.algorithm_mode = algorithm
        
        # Reset if needed
        if self.step_counter > 0:
            self.reset()
        
        # Record start time
        start_time = time.time()
        
        # Set metadata
        self.logger.set_metadata('algorithm', algorithm)
        self.logger.set_metadata('max_steps', max_steps)
        
        if verbose:
            print("="*70)
            print(f"STARTING SIMULATION - Algorithm: {algorithm.upper()}")
            print("="*70)
            print(f"Start: {self.environment.start_pos}, Goal: {self.environment.goal_pos}")
            print(f"Battery: {self.drone.current_battery}")
            print()
        
        # Main simulation loop
        done = False
        status_message = "Running"
        
        while not done and self.step_counter < max_steps:
            done, status_message = self.step(verbose)
        
        # Check timeout
        if self.step_counter >= max_steps and not done:
            status_message = f"Timeout: Reached max steps ({max_steps})"
            self.logger.log_event('timeout', status_message)
        
        # Calculate computation time
        computation_time = time.time() - start_time
        
        if verbose:
            print()
            print("="*70)
            print(f"SIMULATION COMPLETE: {status_message}")
            print("="*70)
        
        # Collect final metrics
        metrics = self.collect_metrics(computation_time, algorithm)
        
        return metrics
    
    def collect_metrics(self, computation_time: float, algorithm_name: str) -> Dict:
        """
        Gather all results at the end.
        
        Args:
            computation_time: Time taken to complete simulation
            algorithm_name: Name of algorithm used
            
        Returns:
            Dictionary containing all performance metrics
        """
        # Get collision count from logger
        collision_count = sum(
            1 for entry in self.logger.get_history() 
            if entry.get('event_type') == 'collision'
        )
        
        # Check if goal was reached
        success = self.environment.is_goal_reached(self.drone.position)
        
        # Generate comprehensive report
        report = self.metrics.generate_report(
            self.drone,
            success=success,
            collisions=collision_count,
            computation_time=computation_time,
            algorithm_name=algorithm_name
        )
        
        return report
    
    def reset(self):
        """
        Reset the simulation to restart.
        """
        # Reset simulation state
        self.step_counter = 0
        
        # Reset drone
        self.drone.position = self.environment.start_pos
        self.drone.current_battery = self.drone.battery_capacity
        self.drone.path = [self.environment.start_pos]
        self.drone.explored_cells = {self.environment.start_pos}
        
        # Reset sensor
        self.sensor.reset_visibility()
        
        # Reset Bayesian solver beliefs
        self.bayesian_solver.reset()
        
        # Create new logger
        self.logger = SimulationLogger()
        
        # Reset metrics
        self.metrics = PerformanceMetrics(self.environment, self.drone)
    
    def _get_state_params(self) -> Dict:
        """
        Get current state parameters for algorithm decision-making.
        
        Returns:
            Dictionary of state parameters
        """
        distance_to_nearest_obstacle = self.environment.get_nearest_obstacle_distance(self.drone.position)
        if distance_to_nearest_obstacle is None:
            distance_to_nearest_obstacle = float('inf')  # No obstacles
        
        return {
            'current_pos': self.drone.position,
            'goal_pos': self.environment.goal_pos,
            'initial_distance': self.environment.distance_to_goal(self.environment.start_pos),
            'battery_used': self.drone.battery_capacity - self.drone.current_battery,
            'total_battery': self.drone.battery_capacity,
            'distance_to_nearest_obstacle': distance_to_nearest_obstacle,
            'explored_cells': len(self.drone.explored_cells),
            'total_cells': self.environment.width * self.environment.height,
            'collision': False,
            'environment': self.environment
        }
    
    def get_logger(self) -> SimulationLogger:
        """Get the simulation logger."""
        return self.logger
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get the performance metrics tracker."""
        return self.metrics
    
    def set_algorithm(self, algorithm: str):
        """
        Set the algorithm mode.
        
        Args:
            algorithm: Algorithm name ("minimax", "nash", or "bayesian")
        """
        valid_algorithms = ["minimax", "nash", "bayesian"]
        if algorithm not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}, got: {algorithm}")
        self.algorithm_mode = algorithm
    
    def __repr__(self) -> str:
        """String representation of the engine."""
        return (
            f"SimulationEngine(algorithm={self.algorithm_mode}, "
            f"step={self.step_counter}, "
            f"drone_pos={self.drone.position})"
        )
