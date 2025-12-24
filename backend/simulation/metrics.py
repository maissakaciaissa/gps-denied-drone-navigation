"""
PART 6: Performance Metrics
This file measures and evaluates how well the drone performed.
It provides various metrics for analyzing simulation results.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from backend.core.environment import Environment
from backend.core.drone import Drone
from backend.game_theory.payoff import PayoffFunction
from backend.game_theory.strategies import DroneAction, EnvironmentCondition, EnvironmentStrategies



class PerformanceMetrics:
    """
    Measures and evaluates drone navigation performance.
    
    Provides various metrics including success rate, path efficiency,
    Pareto optimality, security level, and Nash efficiency.
    """
    
    def __init__(self, environment: Environment, payoff_function: Optional[PayoffFunction] = None):
        """
        Initialize the performance metrics calculator.
        
        Args:
            environment: Environment instance for distance calculations
            payoff_function: PayoffFunction for calculating security and efficiency metrics
        """
        self.environment = environment
        self.payoff_function = payoff_function or PayoffFunction()
        
    def calculate_success_rate(self, final_position: Tuple[int, int]) -> bool:
        """
        Check if the drone reached the goal.
        
        Args:
            final_position: The drone's final position
            
        Returns:
            True if goal reached, False otherwise
        """
        return self.environment.is_goal_reached(final_position)
    
    def calculate_path_efficiency(self, 
        actual_path: List[Tuple[int, int]],
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        Calculate path efficiency by comparing actual path to optimal path.
        
        Path efficiency only makes sense if the goal was reached.
        Returns 0.0 if goal not reached.
        
        Efficiency = optimal_distance / actual_distance
        - 1.0 = perfect (took optimal path)
        - 0.5 = took twice as long as needed
        - 0.0 = goal not reached or extremely inefficient
        
        Args:
            actual_path: List of positions the drone actually visited
            start_pos: Starting position (defaults to environment start)
            goal_pos: Goal position (defaults to environment goal)
            
        Returns:
            Path efficiency ratio (0.0 to 1.0)
        """
        if not actual_path or len(actual_path) < 2:
            return 0.0
        
        start = start_pos or self.environment.start_pos
        goal = goal_pos or self.environment.goal_pos
        
        # Check if goal was reached (final position must be at goal)
        final_position = actual_path[-1]
        if not self.environment.is_goal_reached(final_position):
            return 0.0
        
        # Optimal distance (Manhattan distance is a reasonable lower bound for grid navigation)
        optimal_distance = self.environment.manhattan_distance_to_goal(start)
        
        # Actual path length
        actual_distance = len(actual_path) - 1  # Number of moves
        
        if actual_distance == 0:
            return 0.0
        
        # Calculate efficiency
        efficiency = min(optimal_distance / actual_distance, 1.0)
        
        return efficiency
    
    def calculate_pareto_optimality(self, 
        solution: Dict, alternative_solutions: List[Dict]
    ) -> Tuple[bool, List[Dict]]:
        """
        Check if the solution is Pareto optimal.
        
        A solution is Pareto optimal if no other solution improves one objective
        without worsening another.
        
        The objectives compared are:
        - path_length (lower is better)
        - battery_used (lower is better)
        - collisions (lower is better)
        - success (higher is better)
        
        Args:
            solution: The solution to check (dict with metrics)
            alternative_solutions: List of alternative solutions to compare against
            
        Returns:
            Tuple of (is_pareto_optimal, list_of_dominating_solutions)
        """
        if not alternative_solutions:
            return True, []
        
        dominating_solutions = []
        
        # Extract objectives from solution
        sol_path = solution.get('path_length', float('inf'))
        sol_battery = solution.get('battery_used', float('inf'))
        sol_collisions = solution.get('collisions', float('inf'))
        sol_success = solution.get('success', False)
        
        for alt in alternative_solutions:
            alt_path = alt.get('path_length', float('inf'))
            alt_battery = alt.get('battery_used', float('inf'))
            alt_collisions = alt.get('collisions', float('inf'))
            alt_success = alt.get('success', False)
            
            # Check if alternative dominates solution
            # Alternative dominates if it's better in at least one dimension
            # and not worse in any other dimension
            
            better_in_some = False
            worse_in_any = False
            
            # Check path_length (lower is better)
            if alt_path < sol_path:
                better_in_some = True
            elif alt_path > sol_path:
                worse_in_any = True
            
            # Check battery_used (lower is better)
            if alt_battery < sol_battery:
                better_in_some = True
            elif alt_battery > sol_battery:
                worse_in_any = True
            
            # Check collisions (lower is better)
            if alt_collisions < sol_collisions:
                better_in_some = True
            elif alt_collisions > sol_collisions:
                worse_in_any = True
            
            # Check success (higher is better)
            if alt_success and not sol_success:
                better_in_some = True
            elif not alt_success and sol_success:
                worse_in_any = True
            
            # If better in some dimension and not worse in any, it dominates
            if better_in_some and not worse_in_any:
                dominating_solutions.append(alt)
        
        is_pareto = len(dominating_solutions) == 0
        return is_pareto, dominating_solutions
    
    def calculate_security_level(self,
        drone_action: DroneAction, state_params: Dict,
        env_conditions: Optional[List[EnvironmentCondition]] = None
    ) -> float:
        """
        Calculate the security level (worst-case guaranteed payoff) for an action.
        
        This is the minimum payoff the drone can expect no matter what the
        environment does. It's the minimax value.
        
        security_level = min_{env_condition} payoff(drone_action, env_condition)
        
        Args:
            drone_action: The drone action to evaluate
            state_params: Current state parameters
            env_conditions: List of possible environment conditions (defaults to all)
            
        Returns:
            Worst-case guaranteed payoff
        """
        if env_conditions is None:
            env_conditions = EnvironmentStrategies.get_all_pure_strategies()
        
        worst_payoff = float('inf')
        
        for env_condition in env_conditions:
            drone_payoff, _ = self.payoff_function.compute_payoff(
                drone_action,
                env_condition,
                state_params['current_pos'],
                state_params['goal_pos'],
                state_params['initial_distance'],
                state_params['battery_used'],
                state_params['total_battery'],
                state_params['distance_to_nearest_obstacle'],
                state_params['explored_cells'],
                state_params['total_cells'],
                state_params.get('collision', False),
                state_params.get('environment', None)
            )
            
            worst_payoff = min(worst_payoff, drone_payoff)
        
        return worst_payoff
    
    def calculate_nash_efficiency(self,
        nash_payoff: float, optimal_payoff: float
    ) -> float:
        """ 
        Calculate Nash efficiency (Price of Anarchy).
        
        Measures how good the Nash equilibrium is compared to the optimal solution.
        
        efficiency = nash_payoff / optimal_payoff
        - 1.0 = Nash is optimal
        - 0.5 = Nash gives half the optimal payoff
        
        Args:
            nash_payoff: Payoff at Nash equilibrium
            optimal_payoff: Theoretically optimal payoff
            
        Returns:
            Efficiency ratio (0.0 to 1.0, or > 1.0 if Nash is better than "optimal")
        """
        if optimal_payoff == 0:
            return 1.0 if nash_payoff == 0 else float('inf')
        
        return nash_payoff / optimal_payoff
    
    def generate_report(self,
        drone: Drone, success: bool, collisions: int = 0,
        computation_time: float = 0.0, algorithm_name: str = "Unknown",
        additional_metrics: Optional[Dict] = None,
        alternative_solutions: Optional[List[Dict]] = None,
        state_params: Optional[Dict] = None,
        last_action: Optional[DroneAction] = None,
        nash_payoff: Optional[float] = None,
        optimal_payoff: Optional[float] = None
    ) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Args:
            drone: The drone instance after simulation
            success: Whether the goal was reached
            collisions: Number of collisions that occurred
            computation_time: Time taken for computation (seconds)
            algorithm_name: Name of the algorithm used
            additional_metrics: Optional dictionary of additional metrics
            alternative_solutions: Optional list of alternative solutions for Pareto check
            state_params: Optional state parameters for security level calculation
            last_action: Optional last action taken for security level calculation
            nash_payoff: Optional Nash equilibrium payoff for efficiency calculation
            optimal_payoff: Optional optimal payoff for Nash efficiency calculation
            
        Returns:
            Dictionary containing all performance metrics
        """
        path_length = len(drone.path) - 1  # Number of moves

        battery_used_amount = drone.battery_capacity - drone.current_battery
        battery_used_percent = (battery_used_amount / drone.battery_capacity) * 100

        cells_explored = len(drone.explored_cells)
        total_cells = self.environment.width * self.environment.height
        exploration_rate = (cells_explored / total_cells) * 100
        
        # Calculate path efficiency (function now handles goal check internally)
        path_efficiency = self.calculate_path_efficiency(drone.path)
        
        # Optimal path length for reporting
        optimal_path_length = self.environment.manhattan_distance_to_goal(self.environment.start_pos)
        
        final_distance_to_goal = self.environment.distance_to_goal(drone.position)
        initial_distance = self.environment.distance_to_goal(self.environment.start_pos)
        distance_progress = ((initial_distance - final_distance_to_goal) / initial_distance) * 100
        
        report = {
            'algorithm': algorithm_name,
            'success': success,
            'path_length': path_length,
            'optimal_path_length': optimal_path_length,
            'path_efficiency': path_efficiency,
            'battery_used': battery_used_amount,
            'battery_used_percent': battery_used_percent,
            'battery_remaining': drone.current_battery,
            'collisions': collisions,
            'cells_explored': cells_explored,
            'exploration_rate': exploration_rate,
            'final_position': drone.position,
            'final_distance_to_goal': final_distance_to_goal,
            'distance_progress': distance_progress,
            'computation_time': computation_time,
        }
        
        # Calculate Pareto optimality if alternative solutions provided
        if alternative_solutions is not None:
            # Build a solution dict for comparison
            current_solution = {
                'path_length': path_length,
                'battery_used': battery_used_amount,
                'collisions': collisions,
                'success': success
            }
            is_pareto, _ = self.calculate_pareto_optimality(current_solution, alternative_solutions)
            report['pareto_optimal'] = is_pareto
        else:
            report['pareto_optimal'] = None
        
        # Calculate security level if state_params and last_action provided
        if state_params is not None and last_action is not None:
            security_level = self.calculate_security_level(last_action, state_params)
            report['security_level'] = security_level
        else:
            report['security_level'] = None
        
        # Calculate Nash efficiency if both payoffs provided
        if nash_payoff is not None and optimal_payoff is not None:
            nash_efficiency = self.calculate_nash_efficiency(nash_payoff, optimal_payoff)
            report['nash_efficiency'] = nash_efficiency
        else:
            report['nash_efficiency'] = None
        
        # Add additional metrics if provided (can override calculated values)
        if additional_metrics:
            report.update(additional_metrics)
        
        return report
    
    def generate_summary_string(self, report: Dict) -> str:
        """
        Generate a concise one-line summary like in the example:
        "Success: Yes, Path length: 52 (optimal was 45, efficiency: 86.5%), 
         Battery used: 71%, Collisions: 0, Pareto optimal: True, 
         Security level: 0.42, Nash efficiency: 0.89"
        
        Args:
            report: Performance report dictionary
            
        Returns:
            One-line summary string
        """
        parts = []
        
        # Success
        success = "Yes" if report.get('success', False) else "No"
        parts.append(f"Success: {success}")
        
        # Path length with optimal and efficiency
        path_length = report.get('path_length', 0)
        optimal_path = report.get('optimal_path_length', 0)
        efficiency = report.get('path_efficiency', 0) * 100
        parts.append(f"Path length: {path_length} (optimal was {optimal_path}, efficiency: {efficiency:.1f}%)")
        
        # Battery used
        battery_pct = report.get('battery_used_percent', 0)
        parts.append(f"Battery used: {battery_pct:.0f}%")
        
        # Collisions
        collisions = report.get('collisions', 0)
        parts.append(f"Collisions: {collisions}")
        
        # Pareto optimal
        pareto = report.get('pareto_optimal')
        if pareto is not None:
            parts.append(f"Pareto optimal: {pareto}")
        
        # Security level
        security = report.get('security_level')
        if security is not None:
            parts.append(f"Security level: {security:.2f}")
        
        # Nash efficiency
        nash_eff = report.get('nash_efficiency')
        if nash_eff is not None:
            parts.append(f"Nash efficiency: {nash_eff:.2f}")
        
        return ", ".join(parts)
    
    def calculate_overall_score(self, report: Dict, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate an overall performance score based on multiple metrics.
        
        Args:
            report: Performance report dictionary
            weights: Optional custom weights for each metric
            
        Returns:
            Overall score (0.0 to 100.0)
        """
        # Default weights
        default_weights = {
            'success': 30.0,
            'path_efficiency': 25.0,
            'battery_efficiency': 20.0,
            'collision_free': 15.0,
            'exploration': 10.0
        }
        
        w = weights or default_weights
        
        # Normalize weights to sum to 100
        total_weight = sum(w.values())
        w = {k: v / total_weight * 100 for k, v in w.items()}
        
        score = 0.0
        
        # Success component
        score += w['success'] if report.get('success', False) else 0
        
        # Path efficiency component
        score += w['path_efficiency'] * report.get('path_efficiency', 0)
        
        # Battery efficiency component (invert so lower usage is better)
        battery_percent = report.get('battery_used_percent', 100)
        battery_efficiency = max(0, 1 - (battery_percent / 100))
        score += w['battery_efficiency'] * battery_efficiency
        
        # Collision-free component
        collisions = report.get('collisions', 0)
        collision_free = 1.0 if collisions == 0 else max(0, 1 - (collisions * 0.1))
        score += w['collision_free'] * collision_free
        
        # Exploration component
        exploration_rate = report.get('exploration_rate', 0) / 100
        score += w['exploration'] * exploration_rate
        
        return min(score, 100.0)  # Cap at 100
    
    def format_report(self, report: Dict, verbose: bool = True) -> str:
        """
        Format a report as a readable string.
        
        Args:
            report: Performance report dictionary
            verbose: If True, include all details
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"PERFORMANCE REPORT: {report.get('algorithm', 'Unknown')}")
        lines.append("=" * 70)
        
        # Success status
        success = report.get('success', False)
        status = "✓ SUCCESS" if success else "✗ FAILED"
        lines.append(f"\nStatus: {status}")
        
        # Path metrics
        lines.append(f"\n--- Path Metrics ---")
        lines.append(f"Path Length: {report.get('path_length', 0)} moves")
        lines.append(f"Path Efficiency: {report.get('path_efficiency', 0):.2%}")
        lines.append(f"Final Distance to Goal: {report.get('final_distance_to_goal', 0):.2f}")
        
        # Battery metrics
        lines.append(f"\n--- Energy Metrics ---")
        lines.append(f"Battery Used: {report.get('battery_used', 0):.1f} ({report.get('battery_used_percent', 0):.1f}%)")
        lines.append(f"Battery Remaining: {report.get('battery_remaining', 0):.1f}")
        
        # Safety metrics
        lines.append(f"\n--- Safety Metrics ---")
        lines.append(f"Collisions: {report.get('collisions', 0)}")
        
        # Exploration metrics
        lines.append(f"\n--- Exploration Metrics ---")
        lines.append(f"Cells Explored: {report.get('cells_explored', 0)}")
        lines.append(f"Exploration Rate: {report.get('exploration_rate', 0):.1f}%")
        
        # Performance metrics
        if verbose:
            lines.append(f"\n--- Performance Metrics ---")
            lines.append(f"Computation Time: {report.get('computation_time', 0):.3f}s")
            
            pareto = report.get('pareto_optimal')
            if pareto is not None:
                pareto_str = "Yes" if pareto else "No"
                lines.append(f"Pareto Optimal: {pareto_str}")
            
            security = report.get('security_level')
            if security is not None:
                lines.append(f"Security Level: {security:.3f}")
            
            nash_eff = report.get('nash_efficiency')
            if nash_eff is not None:
                lines.append(f"Nash Efficiency: {nash_eff:.3f}")
            
            overall = report.get('overall_score')
            if overall is not None:
                lines.append(f"Overall Score: {overall:.1f}/100")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation of the metrics calculator."""
        return f"PerformanceMetrics(environment={self.environment})"
