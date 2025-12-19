"""
PART 1: Simulation Engine
This file runs comprehensive simulations of all strategy combinations
under different environmental scenarios (battery levels, obstacles, etc.)
"""

import numpy as np
from typing import Dict, List, Tuple
from backend.core.environment import Environment
from backend.game_theory.strategies import DroneStrategies, EnvironmentStrategies
from backend.game_theory.payoff import PayoffFunction
from backend.simulation.logger import SimulationLogger
from backend.game_theory.strategies import DroneAction, EnvironmentCondition


class StrategySimulation:
    """
    Simulates all strategy combinations under various scenarios.
    Tests how different drone strategies perform against different environment conditions
    with varying battery levels, obstacle proximities, and exploration states.
    """
    
    def __init__(self, environment: Environment, payoff_function: PayoffFunction):
        """
        Initialize the simulation engine.
        
        Args:
            environment: The environment to simulate in
            payoff_function: The payoff function to evaluate strategies
        """
        self.environment = environment
        self.payoff_function = payoff_function
        self.logger = SimulationLogger(log_name="strategy_simulation")
        
    def get_all_drone_strategy_scenarios(self) -> List[Tuple[str, object]]:
        """Get all predefined drone strategy scenarios."""
        return [
            ('Uniform', DroneStrategies.create_uniform_mixed_strategy()),
            ('Cautious', DroneStrategies.create_cautious_strategy()),
            ('Aggressive', DroneStrategies.create_aggressive_strategy()),
            ('Balanced', DroneStrategies.create_balanced_strategy()),
        ]
    
    def get_all_env_strategy_scenarios(self) -> List[Tuple[str, object]]:
        """Get all predefined environment strategy scenarios."""
        return [
            ('Uniform', EnvironmentStrategies.create_uniform_mixed_strategy()),
            ('Typical', EnvironmentStrategies.create_typical_conditions()),
            ('Adversarial', EnvironmentStrategies.create_adversarial_conditions()),
            ('Favorable', EnvironmentStrategies.create_favorable_conditions()),
        ]
    
    def generate_scenario_variations(self) -> List[Dict]:
        """
        Generate different scenario variations with varying parameters.
        
        Returns:
            List of scenario parameter dictionaries
        """
        scenarios = []
        
        # Scenario 1: High battery, safe position
        scenarios.append({
            'name': 'High Battery - Safe',
            'current_pos': (10, 10),
            'battery_used': 20,
            'total_battery': 100,
            'explored_cells': 80,
            'collision': False
        })
        
        # Scenario 2: Low battery, safe position
        scenarios.append({
            'name': 'Low Battery - Safe',
            'current_pos': (10, 10),
            'battery_used': 75,
            'total_battery': 100,
            'explored_cells': 250,
            'collision': False
        })
        
        # Scenario 3: Medium battery, near obstacle
        scenarios.append({
            'name': 'Medium Battery - Near Obstacle',
            'current_pos': (10, 9),  # Near obstacle at (11, 9) - safe position
            'battery_used': 50,
            'total_battery': 100,
            'explored_cells': 150,
            'collision': False
        })
        
        # Scenario 4: High battery, far from goal
        scenarios.append({
            'name': 'High Battery - Far From Goal',
            'current_pos': (3, 3),  # Far from goal, safe position (not on obstacle)
            'battery_used': 30,
            'total_battery': 100,
            'explored_cells': 50,
            'collision': False
        })
        
        # Scenario 5: Low battery, close to goal
        scenarios.append({
            'name': 'Low Battery - Close To Goal',
            'current_pos': (16, 16),
            'battery_used': 80,
            'total_battery': 100,
            'explored_cells': 300,
            'collision': False
        })
        
        return scenarios
    
    def calculate_expected_payoff(self, drone_strategy, env_strategy, state_params: Dict) -> float:
        """
        Calculate expected payoff for mixed strategy combination.
        
        Args:
            drone_strategy: Drone mixed strategy
            env_strategy: Environment mixed strategy
            state_params: Current state parameters
            
        Returns:
            Expected payoff value
        """
        expected_payoff = 0.0
        
        for drone_action, drone_prob in zip(drone_strategy.strategies, drone_strategy.probabilities):
            for env_cond, env_prob in zip(env_strategy.strategies, env_strategy.probabilities):
                payoff = self.payoff_function.compute_payoff(
                    drone_action, env_cond,
                    state_params['current_pos'],
                    state_params['goal_pos'],
                    state_params['initial_distance'],
                    state_params['battery_used'],
                    state_params['total_battery'],
                    state_params['distance_to_nearest_obstacle'],
                    state_params['explored_cells'],
                    state_params['total_cells'],
                    state_params['collision'],
                    state_params['environment']
                )
                expected_payoff += drone_prob * env_prob * payoff
        
        return expected_payoff
    
    def run_pure_strategy_simulation(self, scenarios: List[Dict]) -> List[Dict]:
        """
        Run simulation with pure strategies across all scenarios.
        
        Args:
            scenarios: List of scenario dictionaries
            
        Returns:
            List of results for each scenario
        """
        all_drone_actions = DroneStrategies.get_all_pure_strategies()
        all_env_conditions = EnvironmentStrategies.get_all_pure_strategies()
        all_results = []
        
        for scenario in scenarios:
            print("\n" + "=" * 70)
            print(f"SCENARIO: {scenario['name']}")
            print("=" * 70)
            
            # Build state params for this scenario
            state_params = {
                'current_pos': scenario['current_pos'],
                'goal_pos': self.environment.goal_pos,
                'initial_distance': self.environment.distance_to_goal(self.environment.start_pos),
                'battery_used': scenario['battery_used'],
                'total_battery': scenario['total_battery'],
                'distance_to_nearest_obstacle': self.environment.get_nearest_obstacle_distance(scenario['current_pos']),
                'explored_cells': scenario['explored_cells'],
                'total_cells': 400,
                'collision': scenario['collision'],
                'environment': self.environment
            }
            
            # Calculate distance to goal from current position
            current_distance = self.environment.distance_to_goal(scenario['current_pos'])
            
            print(f"Position: {scenario['current_pos']}, Battery: {scenario['battery_used']}/{scenario['total_battery']}")
            print(f"Distance to goal: {current_distance:.2f}")
            
            # Generate payoff matrices (returns tuple: drone_matrix, env_matrix)
            drone_matrix, env_matrix = self.payoff_function.generate_payoff_matrix(
                all_drone_actions, all_env_conditions, state_params
            )
            
            # Display matrix with both payoffs
            print(f"\n{'Action':<15}", end='')
            for cond in all_env_conditions:
                print(f"{cond.value:<22}", end='')
            print()
            print("-" * (15 + 22 * len(all_env_conditions)))
            
            for i, action in enumerate(all_drone_actions):
                print(f"{action.value:<15}", end='')
                for j in range(len(all_env_conditions)):
                    # Display as (drone_payoff, env_payoff)
                    print(f"({drone_matrix[i, j]:>6.1f},{env_matrix[i, j]:>6.1f})  ", end='')
                print()
            
            # Find best and worst combinations for drone
            best_idx = np.unravel_index(np.argmax(drone_matrix), drone_matrix.shape)
            worst_idx = np.unravel_index(np.argmin(drone_matrix), drone_matrix.shape)
            
            best_action = all_drone_actions[best_idx[0]]
            best_condition = all_env_conditions[best_idx[1]]
            worst_action = all_drone_actions[worst_idx[0]]
            worst_condition = all_env_conditions[worst_idx[1]]
            
            print(f"\nBest for Drone: {best_action.value} vs {best_condition.value} = ({drone_matrix[best_idx]:.1f}, {env_matrix[best_idx]:.1f})")
            print(f"Worst for Drone: {worst_action.value} vs {worst_condition.value} = ({drone_matrix[worst_idx]:.1f}, {env_matrix[worst_idx]:.1f})")
            
            # Log results
            self.logger.log_event('pure_strategies', f'Scenario: {scenario["name"]}', {
                'best_drone_payoff': float(np.max(drone_matrix)),
                'best_combination': f"{best_action.value} vs {best_condition.value}",
                'worst_drone_payoff': float(np.min(drone_matrix)),
                'worst_combination': f"{worst_action.value} vs {worst_condition.value}",
                'average_drone_payoff': float(np.mean(drone_matrix)),
                'average_env_payoff': float(np.mean(env_matrix))
            })
            
            all_results.append({
                'scenario': scenario['name'],
                'drone_matrix': drone_matrix,
                'env_matrix': env_matrix,
                'best_drone_payoff': float(np.max(drone_matrix)),
                'worst_drone_payoff': float(np.min(drone_matrix))
            })
        
        return all_results
    
    def run_mixed_strategy_simulation(self, drone_strategies_to_test=None, env_strategies_to_test=None) -> List[Dict]:
        """
        Run mixed strategy simulation with predefined AND custom strategies.
        Shows 6×5 payoff matrices (like pure strategies) for each combo in each scenario.
        
        Args:
            drone_strategies_to_test: Optional list of (name, strategy) tuples for drone strategies
            env_strategies_to_test: Optional list of (name, strategy) tuples for environment strategies
        
        Returns:
            List of results for each scenario
        """
        
        # Use provided strategies or define defaults
        if drone_strategies_to_test is None:
            drone_strategies_to_test = [
                ('Uniform', DroneStrategies.create_uniform_mixed_strategy()),
                ('Cautious', DroneStrategies.create_cautious_strategy()),
                ('Aggressive', DroneStrategies.create_aggressive_strategy()),
                ('Balanced', DroneStrategies.create_balanced_strategy()),
                # Custom: Exploration mode (favor movement in all directions)
                ('Custom_Exploration', DroneStrategies.create_custom_strategy({
                    DroneAction.MOVE_UP: 0.3,
                    DroneAction.MOVE_DOWN: 0.2,
                    DroneAction.MOVE_LEFT: 0.25,
                    DroneAction.MOVE_RIGHT: 0.25
                }))
            ]
        
        # Use provided strategies or define defaults
        if env_strategies_to_test is None:
            env_strategies_to_test = [
                ('Uniform', EnvironmentStrategies.create_uniform_mixed_strategy()),
                ('Typical', EnvironmentStrategies.create_typical_conditions()),
                ('Adversarial', EnvironmentStrategies.create_adversarial_conditions()),
                ('Favorable', EnvironmentStrategies.create_favorable_conditions()),
                # Custom: Danger zone (high obstacle probability)
                ('Custom_DangerZone', EnvironmentStrategies.create_custom_strategy({
                    EnvironmentCondition.CLEAR_PATH: 0.15,
                    EnvironmentCondition.OBSTACLE_AHEAD: 0.5,
                    EnvironmentCondition.LOW_VISIBILITY: 0.2,
                    EnvironmentCondition.SENSOR_NOISE: 0.1,
                    EnvironmentCondition.LIGHTING_CHANGE: 0.05
                }))
            ]
        
        scenario_variations = self.generate_scenario_variations()
        
        all_drone_actions = DroneStrategies.get_all_pure_strategies()
        all_env_conditions = EnvironmentStrategies.get_all_pure_strategies()
        
        all_results = []
        
        # Test each combination across all scenarios
        for drone_name, drone_strategy in drone_strategies_to_test:
            for env_name, env_strategy in env_strategies_to_test:
                print("\n" + "=" * 70)
                print(f"MIXED STRATEGY: {drone_name} vs {env_name}")
                print("=" * 70)
                
                # Show probability distributions
                print(f"\nDrone '{drone_name}' probabilities:")
                for action, prob in zip(drone_strategy.strategies, drone_strategy.probabilities):
                    print(f"  {action.value:<15} {prob:.2%}")
                
                print(f"\nEnvironment '{env_name}' probabilities:")
                for condition, prob in zip(env_strategy.strategies, env_strategy.probabilities):
                    print(f"  {condition.value:<20} {prob:.2%}")
                
                # Test this combo across all 5 scenarios - each scenario gets its own matrix
                combo_results = []
                for scenario in scenario_variations:
                    print("\n" + "-" * 70)
                    print(f"SCENARIO: {scenario['name']}")
                    print("-" * 70)
                    
                    # Build state parameters
                    state_params = {
                        'current_pos': scenario['current_pos'],
                        'goal_pos': self.environment.goal_pos,
                        'initial_distance': self.environment.distance_to_goal(self.environment.start_pos),
                        'battery_used': scenario['battery_used'],
                        'total_battery': scenario['total_battery'],
                        'distance_to_nearest_obstacle': self.environment.get_nearest_obstacle_distance(scenario['current_pos']),
                        'explored_cells': scenario['explored_cells'],
                        'total_cells': 400,
                        'collision': scenario['collision'],
                        'environment': self.environment
                    }
                    
                    current_distance = self.environment.distance_to_goal(scenario['current_pos'])
                    print(f"Position: {scenario['current_pos']}, Battery: {scenario['battery_used']}/{scenario['total_battery']}")
                    print(f"Distance to goal: {current_distance:.2f}")
                    
                    # Generate full payoff matrix for this scenario
                    drone_matrix, env_matrix = self.payoff_function.generate_payoff_matrix(
                        all_drone_actions, all_env_conditions, state_params
                    )
                    
                    # Display the full payoff matrix (like pure strategies)
                    print(f"\nPayoff Matrix (6 actions × 5 conditions):")
                    print()
                    
                    # Print header with condition names
                    print(f"{'Action':<15}", end='')
                    for condition in all_env_conditions:
                        print(f"{condition.value:<22}", end='')
                    print()
                    print("-" * (15 + 22 * len(all_env_conditions)))
                    
                    # Print each row with payoff tuples
                    for i in range(len(all_drone_actions)):
                        action = all_drone_actions[i]
                        print(f"{action.value:<15}", end='')
                        for j in range(len(all_env_conditions)):
                            # Display as (drone_payoff, env_payoff)
                            print(f"({drone_matrix[i, j]:>6.1f},{env_matrix[i, j]:>6.1f})  ", end='')
                        print()
                    
                    # Calculate expected payoff for mixed strategy
                    # E[u] = sum over all (action, condition) pairs: P(action) * P(condition) * payoff(action, condition)
                    total_drone_payoff = 0.0
                    total_env_payoff = 0.0
                    
                    for i, action in enumerate(all_drone_actions):
                        for j, condition in enumerate(all_env_conditions):
                            # Get payoffs from matrix
                            drone_payoff = drone_matrix[i, j]
                            env_payoff = env_matrix[i, j]
                            
                            # Weight by mixed strategy probabilities (with safety check)
                            # Check if action exists in drone strategy
                            if action in drone_strategy.strategies:
                                drone_prob = drone_strategy.probabilities[drone_strategy.strategies.index(action)]
                            else:
                                drone_prob = 0.0  # Action not in this strategy, zero probability
                            
                            # Check if condition exists in environment strategy
                            if condition in env_strategy.strategies:
                                env_prob = env_strategy.probabilities[env_strategy.strategies.index(condition)]
                            else:
                                env_prob = 0.0  # Condition not in this strategy, zero probability
                            
                            # Add to expected value
                            total_drone_payoff += drone_prob * env_prob * drone_payoff
                            total_env_payoff += drone_prob * env_prob * env_payoff
                    
                    # Display expected payoff (weighted sum)
                    print(f"\nExpected Payoff for {scenario['name']} (weighted by probabilities):")
                    print(f"  Drone:       {total_drone_payoff:>8.2f}")
                    print(f"  Environment: {total_env_payoff:>8.2f}")
                    
                    result = {
                        'scenario': scenario['name'],
                        'drone_strategy': drone_name,
                        'env_strategy': env_name,
                        'expected_drone_payoff': total_drone_payoff,
                        'expected_env_payoff': total_env_payoff,
                        'battery_used': scenario['battery_used'],
                        'position': scenario['current_pos']
                    }
                    
                    combo_results.append(result)
                    all_results.append(result)
                    
                    # Log this result
                    self.logger.log_event('mixed_strategy', 
                                        f"{drone_name} vs {env_name} - {scenario['name']}", {
                        'expected_drone_payoff': float(total_drone_payoff),
                        'expected_env_payoff': float(total_env_payoff),
                        'position': scenario['current_pos'],
                        'battery_used': scenario['battery_used']
                    })
                
                # Summary for this specific combination
                best_scenario = max(combo_results, key=lambda x: x['expected_drone_payoff'])
                worst_scenario = min(combo_results, key=lambda x: x['expected_drone_payoff'])
                avg_payoff = sum(r['expected_drone_payoff'] for r in combo_results) / len(combo_results)
                
                print("\n" + "-" * 70)
                print(f"SUMMARY for {drone_name} vs {env_name}:")
                print(f"  Best scenario:  {best_scenario['scenario']:<30} Drone Payoff = {best_scenario['expected_drone_payoff']:.2f}")
                print(f"  Worst scenario: {worst_scenario['scenario']:<30} Drone Payoff = {worst_scenario['expected_drone_payoff']:.2f}")
                print(f"  Average drone payoff: {avg_payoff:.2f}")
        
        # Overall summary across ALL combinations
        best_result = max(all_results, key=lambda x: x['expected_drone_payoff'])
        worst_result = min(all_results, key=lambda x: x['expected_drone_payoff'])
        avg_all = sum(r['expected_drone_payoff'] for r in all_results) / len(all_results)
        
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY - ALL MIXED STRATEGY COMBINATIONS")
        print("=" * 70)
        print(f"Best:  {best_result['drone_strategy']} vs {best_result['env_strategy']}")
        print(f"       Scenario: {best_result['scenario']}, Drone Payoff = {best_result['expected_drone_payoff']:.2f}")
        print(f"Worst: {worst_result['drone_strategy']} vs {worst_result['env_strategy']}")
        print(f"       Scenario: {worst_result['scenario']}, Drone Payoff = {worst_result['expected_drone_payoff']:.2f}")
        print(f"Average drone payoff across all combinations & scenarios: {avg_all:.2f}")
        
        return all_results
    
    def run_complete_simulation(self,
                               drone_strategies_to_test: List[Tuple[str, object]] = None,
                               env_strategies_to_test: List[Tuple[str, object]] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Run complete simulation with both pure and mixed strategies across all scenarios.
        
        Args:
            drone_strategies_to_test: List of (name, strategy) tuples for drone strategies.
            env_strategies_to_test: List of (name, strategy) tuples for environment strategies.
        
        Returns:
            Tuple of (pure_results, mixed_results)
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE STRATEGY SIMULATION")
        print("=" * 70)
        
        # Set metadata
        self.logger.set_metadata('environment', f"{self.environment.width}x{self.environment.height}")
        self.logger.set_metadata('obstacles', len(self.environment.obstacles))
        self.logger.set_metadata('payoff_weights', self.payoff_function.get_weights())
        
        # Generate all scenarios once
        scenarios = self.generate_scenario_variations()
        
        # PART 1: Run pure strategy simulation for ALL scenarios
        print("\n" + "=" * 70)
        print("PART 1: PURE STRATEGIES - ALL SCENARIOS")
        print("=" * 70)
        pure_results = self.run_pure_strategy_simulation(scenarios)
        
        # PART 2: Run mixed strategy simulation
        print("\n" + "=" * 70)
        print("PART 2: MIXED STRATEGIES - ALL COMBINATIONS × ALL SCENARIOS")
        print("=" * 70)
        num_drone = len(drone_strategies_to_test) if drone_strategies_to_test else 4
        num_env = len(env_strategies_to_test) if env_strategies_to_test else 4
        num_combos = num_drone * num_env
        print(f"Testing {num_combos} strategy combinations ({num_drone} drone × {num_env} env) across 5 scenarios = {num_combos * 5} evaluations")
        mixed_results = self.run_mixed_strategy_simulation(drone_strategies_to_test, env_strategies_to_test)
        
        # Save results
        json_path = self.logger.save_to_json()
        
        print(f"\n{'='*70}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*70}")
        
        # Calculate totals
        pure_combinations = 6 * 5 * len(scenarios)  # 6 actions × 5 conditions × 5 scenarios
        num_drone = len(drone_strategies_to_test) if drone_strategies_to_test else 4
        num_env = len(env_strategies_to_test) if env_strategies_to_test else 4
        mixed_combinations = num_drone * num_env * len(scenarios)
        
        print(f"Results saved to: {json_path}")
        print(f"Pure strategy tests: {pure_combinations} (6 actions × 5 conditions × {len(scenarios)} scenarios)")
        print(f"Mixed strategy tests: {mixed_combinations} ({num_drone} drone × {num_env} env × {len(scenarios)} scenarios)")
        print(f"Total combinations tested: {pure_combinations + mixed_combinations}")
        
        return pure_results, mixed_results
