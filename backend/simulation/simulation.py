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
            'current_pos': (11, 10),  # Near obstacle at (11, 9)
            'battery_used': 50,
            'total_battery': 100,
            'explored_cells': 150,
            'collision': False
        })
        
        # Scenario 4: High battery, far from goal
        scenarios.append({
            'name': 'High Battery - Far From Goal',
            'current_pos': (5, 5),
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
            
            # Generate payoff matrix
            payoff_matrix = self.payoff_function.generate_payoff_matrix(
                all_drone_actions, all_env_conditions, state_params
            )
            
            # Display matrix
            print(f"\n{'Action':<15}", end='')
            for cond in all_env_conditions:
                print(f"{cond.value:<18}", end='')
            print()
            print("-" * (15 + 18 * len(all_env_conditions)))
            
            for i, action in enumerate(all_drone_actions):
                print(f"{action.value:<15}", end='')
                for j in range(len(all_env_conditions)):
                    print(f"{payoff_matrix[i, j]:>17.4f} ", end='')
                print()
            
            # Find best and worst combinations
            best_idx = np.unravel_index(np.argmax(payoff_matrix), payoff_matrix.shape)
            worst_idx = np.unravel_index(np.argmin(payoff_matrix), payoff_matrix.shape)
            
            best_action = all_drone_actions[best_idx[0]]
            best_condition = all_env_conditions[best_idx[1]]
            worst_action = all_drone_actions[worst_idx[0]]
            worst_condition = all_env_conditions[worst_idx[1]]
            
            print(f"\nBest: {best_action.value} vs {best_condition.value} = {payoff_matrix[best_idx]:.4f}")
            print(f"Worst: {worst_action.value} vs {worst_condition.value} = {payoff_matrix[worst_idx]:.4f}")
            
            # Log results
            self.logger.log_event('pure_strategies', f'Scenario: {scenario["name"]}', {
                'best_payoff': float(np.max(payoff_matrix)),
                'best_combination': f"{best_action.value} vs {best_condition.value}",
                'worst_payoff': float(np.min(payoff_matrix)),
                'worst_combination': f"{worst_action.value} vs {worst_condition.value}",
                'average_payoff': float(np.mean(payoff_matrix))
            })
            
            all_results.append({
                'scenario': scenario['name'],
                'payoff_matrix': payoff_matrix,
                'best_payoff': float(np.max(payoff_matrix)),
                'worst_payoff': float(np.min(payoff_matrix))
            })
        
        return all_results
    
    def run_mixed_strategy_simulation(self) -> List[Dict]:
        """
        Run mixed strategy simulation with predefined AND custom strategies.
        Shows 6×5 payoff matrices (like pure strategies) for each combo in each scenario.
        
        Returns:
            List of results for each scenario
        """
        
        
        # Define ALL drone strategies to test (predefined + custom)
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
        
        # Define ALL environment strategies to test (predefined + custom)
        env_strategies_to_test = [
            ('Uniform', EnvironmentStrategies.create_uniform_mixed_strategy()),
            ('Typical', EnvironmentStrategies.create_typical_conditions()),
            ('Adversarial', EnvironmentStrategies.create_adversarial_conditions()),
            ('Favorable', EnvironmentStrategies.create_favorable_conditions()),
            # Custom: Danger zone (high obstacle probability)
            ('Custom_DangerZone', EnvironmentStrategies.create_custom_strategy({
                EnvironmentCondition.CLEAR_PATH: 0.2,
                EnvironmentCondition.OBSTACLE_AHEAD: 0.5,
                EnvironmentCondition.LOW_VISIBILITY: 0.2,
                EnvironmentCondition.SENSOR_NOISE: 0.1
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
                    
                    # Generate 6×5 matrix with expected payoffs based on mixed strategy probabilities
                    # Each cell = weighted sum of payoffs using drone and env probabilities
                    payoff_matrix = np.zeros((len(all_drone_actions), len(all_env_conditions)))
                    
                    for i, action in enumerate(all_drone_actions):
                        for j, condition in enumerate(all_env_conditions):
                            # Get base payoff for this action-condition pair
                            base_payoff = self.payoff_function.compute_payoff(
                                action, condition,
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
                            
                            # Weight by mixed strategy probabilities
                            drone_prob = drone_strategy.probabilities[drone_strategy.strategies.index(action)]
                            env_prob = env_strategy.probabilities[env_strategy.strategies.index(condition)]
                            
                            payoff_matrix[i, j] = base_payoff * drone_prob * env_prob
                    
                    # Display matrix (same format as pure strategies)
                    print(f"\n{'Action':<15}", end='')
                    for cond in all_env_conditions:
                        print(f"{cond.value:<18}", end='')
                    print()
                    print("-" * (15 + 18 * len(all_env_conditions)))
                    
                    for i, action in enumerate(all_drone_actions):
                        print(f"{action.value:<15}", end='')
                        for j in range(len(all_env_conditions)):
                            print(f"{payoff_matrix[i, j]:>17.4f} ", end='')
                        print()
                    
                    # Calculate overall expected payoff (sum of all weighted cells)
                    total_expected_payoff = np.sum(payoff_matrix)
                    print(f"\nExpected Payoff for this scenario: {total_expected_payoff:.4f}")
                    
                    result = {
                        'scenario': scenario['name'],
                        'drone_strategy': drone_name,
                        'env_strategy': env_name,
                        'expected_payoff': total_expected_payoff,
                        'battery_used': scenario['battery_used'],
                        'position': scenario['current_pos']
                    }
                    
                    combo_results.append(result)
                    all_results.append(result)
                    
                    # Log this result
                    self.logger.log_event('mixed_strategy_matrix', 
                                        f"{drone_name} vs {env_name} - {scenario['name']}", {
                        'expected_payoff': float(total_expected_payoff),
                        'position': scenario['current_pos'],
                        'battery_used': scenario['battery_used']
                    })
                
                # Summary for this specific combination
                best_scenario = max(combo_results, key=lambda x: x['expected_payoff'])
                worst_scenario = min(combo_results, key=lambda x: x['expected_payoff'])
                avg_payoff = sum(r['expected_payoff'] for r in combo_results) / len(combo_results)
                
                print("\n" + "-" * 70)
                print(f"SUMMARY for {drone_name} vs {env_name}:")
                print(f"  Best scenario:  {best_scenario['scenario']:<30} Payoff = {best_scenario['expected_payoff']:.4f}")
                print(f"  Worst scenario: {worst_scenario['scenario']:<30} Payoff = {worst_scenario['expected_payoff']:.4f}")
                print(f"  Average payoff: {avg_payoff:.4f}")
        
        # Overall summary across ALL combinations
        best_result = max(all_results, key=lambda x: x['expected_payoff'])
        worst_result = min(all_results, key=lambda x: x['expected_payoff'])
        avg_all = sum(r['expected_payoff'] for r in all_results) / len(all_results)
        
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY - ALL MIXED STRATEGY COMBINATIONS")
        print("=" * 70)
        print(f"Best:  {best_result['drone_strategy']} vs {best_result['env_strategy']}")
        print(f"       Scenario: {best_result['scenario']}, Payoff = {best_result['expected_payoff']:.4f}")
        print(f"Worst: {worst_result['drone_strategy']} vs {worst_result['env_strategy']}")
        print(f"       Scenario: {worst_result['scenario']}, Payoff = {worst_result['expected_payoff']:.4f}")
        print(f"Average payoff across all combinations & scenarios: {avg_all:.4f}")
        
        return all_results
    
    def run_complete_simulation(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Run complete simulation with both pure and mixed strategies across all scenarios.
        
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
        print(f"Testing 25 strategy combinations (5 drone × 5 env) across 5 scenarios = 125 evaluations")
        print(f"Predefined strategies: Uniform, Cautious, Aggressive, Balanced")
        print(f"Custom strategies: Exploration (drone), Danger Zone (environment)")
        mixed_results = self.run_mixed_strategy_simulation()
        
        # Save results
        json_path = self.logger.save_to_json()
        
        print(f"\n{'='*70}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*70}")
        
        # Calculate totals
        pure_combinations = 6 * 5 * len(scenarios)  # 6 actions × 5 conditions × 5 scenarios
        mixed_combinations = 5 * 5 * len(scenarios)  # 5 drone strategies × 5 env strategies × 5 scenarios
        
        print(f"Results saved to: {json_path}")
        print(f"Pure strategy tests: {pure_combinations} (6 actions × 5 conditions × 5 scenarios)")
        print(f"Mixed strategy tests: {mixed_combinations} (5 drone × 5 env × 5 scenarios)")
        print(f"Total combinations tested: {pure_combinations + mixed_combinations}")
        
        return pure_results, mixed_results
