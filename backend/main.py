"""
Main entry point for the Drone Visual Navigation Project
This demonstrates PART 1: Environment, Strategies, Payoff, and Logger
"""

import numpy as np
from backend.core.environment import Environment
from backend.game_theory.strategies import (
    DroneStrategies, EnvironmentStrategies, DroneAction, EnvironmentCondition
)
from backend.game_theory.payoff import PayoffFunction
from backend.simulation.logger import SimulationLogger

"""Test PART 1: Environment creation and functionality"""

def test_environment():
   
    print("=" * 70)
    print("TEST 1: ENVIRONMENT - Grid, Obstacles, and Position Validation")
    print("=" * 70)
    
    # Create a 20x20 grid environment
    env = Environment(
        width=20, 
        height=20, 
        start_pos=(1, 1), 
        goal_pos=(18, 18)
    )
    
    # Add obstacles to create a challenging environment
    obstacles = [
        # Vertical wall
        (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8),
        # Horizontal wall
        (6, 8), (7, 8), (8, 8), (9, 8), (10, 8), (11, 8),
        # Another vertical wall
        (11, 9), (11, 10), (11, 11), (11, 12), (11, 13),
        # Random obstacles
        (3, 10), (4, 10), (15, 5), (15, 6), (15, 7)
    ]
    env.add_obstacles(obstacles)
    
    # Test environment functions
    print(f"\n{env}")
    print(f"\nEnvironment State:")
    state = env.get_state()
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    # Test position validation
    #Position (25, 25) is outside the 20×20 grid , so it is impossible for it to be an obstacle , (position is outside the grid) → Show "N/A". 
    test_positions = [(1, 1), (5, 5), (18, 18), (25, 25), (0, 0)]
    print(f"\nPosition Validation Tests:")
    for pos in test_positions:
        valid = env.is_valid_position(pos)
        within_bounds = env.is_within_bounds(pos)
        is_obs = env.is_obstacle(pos) if within_bounds else "N/A"
        print(f"  Position {pos}: Valid={valid}, InBounds={within_bounds}, Obstacle={is_obs}")
    
    # Test distance calculations
    print(f"\nDistance to Goal from (1,1): {env.distance_to_goal((1, 1)):.2f}")
    print(f"Manhattan Distance to Goal from (1,1): {env.manhattan_distance_to_goal((1, 1))}")
    print(f"Nearest Obstacle Distance from (1,1): {env.get_nearest_obstacle_distance((1, 1)):.2f}")
    
    return env


def test_strategies():
    
    print("\n" + "=" * 70)
    print("TEST 2: STRATEGIES - Pure and Mixed Strategies")
    print("=" * 70)
    
    # Test Drone Strategies
    print("\n--- Drone Strategies ---")
    print(f"All pure drone strategies: {[action.value for action in DroneStrategies.get_all_pure_strategies()]}")
    
    # Create ALL different mixed strategies for drone
    # uniform = DroneStrategies.create_uniform_mixed_strategy()
    # cautious = DroneStrategies.create_cautious_strategy()
    aggressive = DroneStrategies.create_aggressive_strategy()
    # balanced = DroneStrategies.create_balanced_strategy()
    
    # print(f"\n1. Uniform Strategy (equal probabilities): {uniform}")
    # print(f"2. Cautious Strategy (prefer staying): {cautious}")
    print(f"3. Aggressive Strategy (prefer moving): {aggressive}")
    # print(f"4. Balanced Strategy (mixed approach): {balanced}")
    
    # Sample from each strategy
    print(f"\nSampling from Uniform Strategy (3 samples):")
    # for i in range(3):
    #     action = uniform.sample()
    #     print(f"  Sample {i+1}: {action.value}")
    
    # print(f"\nSampling from Cautious Strategy (3 samples):")
    # for i in range(3):
    #     action = cautious.sample()
    #     print(f"  Sample {i+1}: {action.value}")
    
    print(f"\nSampling from Aggressive Strategy (3 samples):")
    for i in range(3):
        action = aggressive.sample()
        print(f"  Sample {i+1}: {action.value}")
    
    # print(f"\nSampling from Balanced Strategy (3 samples):")
    # for i in range(3):
    #     action = balanced.sample()
    #     print(f"  Sample {i+1}: {action.value}")
    
    # Test Environment Strategies
    print("\n--- Environment Strategies ---")
    print(f"All pure environment conditions: {[cond.value for cond in EnvironmentStrategies.get_all_pure_strategies()]}")
    
    # Create ALL environment strategies
    # env_uniform = EnvironmentStrategies.create_uniform_mixed_strategy()
    #typical = EnvironmentStrategies.create_typical_conditions()
    adversarial = EnvironmentStrategies.create_adversarial_conditions()
    # favorable = EnvironmentStrategies.create_favorable_conditions()
    
    # print(f"\n1. Uniform Conditions (equal probabilities): {env_uniform}")
    #print(f"2. Typical Conditions (realistic): {typical}")
    print(f"3. Adversarial Conditions (hostile): {adversarial}")
    # print(f"4. Favorable Conditions (benign): {favorable}")
    
    # print(f"\nSampling from Typical Conditions (3 samples):")
    # for i in range(3):
    #     condition = typical.sample()
    #     print(f"  Sample {i+1}: {condition.value}")
    
    print(f"\nSampling from Adversarial Conditions (3 samples):")
    for i in range(3):
        condition = adversarial.sample()
        print(f"  Sample {i+1}: {condition.value}")
    
    # Test Custom Strategies
    print("\n--- Custom Strategies ---")
    
    # Custom drone strategy: Exploration mode (favor movement in all directions)
    custom_drone = DroneStrategies.create_custom_strategy({
        DroneAction.MOVE_UP: 0.3,
        DroneAction.MOVE_DOWN: 0.2,
        DroneAction.MOVE_LEFT: 0.25,
        DroneAction.MOVE_RIGHT: 0.25
    })
    print(f"\n5. Custom Drone (Exploration mode): {custom_drone}")
    
    # Custom environment strategy: Known danger zone (high obstacle probability)
    custom_env = EnvironmentStrategies.create_custom_strategy({
        EnvironmentCondition.CLEAR_PATH: 0.2,
        EnvironmentCondition.OBSTACLE_AHEAD: 0.5,
        EnvironmentCondition.LOW_VISIBILITY: 0.2,
        EnvironmentCondition.SENSOR_NOISE: 0.1
    })
    print(f"5. Custom Environment (Danger Zone): {custom_env}")
    
    print(f"\nSampling from Custom Drone Strategy (3 samples):")
    for i in range(3):
        action = custom_drone.sample()
        print(f"  Sample {i+1}: {action.value}")
    
    print(f"\nSampling from Custom Environment Strategy (3 samples):")
    for i in range(3):
        condition = custom_env.sample()
        print(f"  Sample {i+1}: {condition.value}")
    
    return aggressive, adversarial


def test_payoff_function():
    print("\n" + "=" * 70)
    print("TEST 3: PAYOFF FUNCTION")
    print("=" * 70)
    
    # Create environment for distance calculations
    env = Environment(width=20, height=20, start_pos=(1, 1), goal_pos=(18, 18))
    
    # Create payoff function with default weights 
    """The weights are design choices that define your mission priorities"""
    payoff_func = PayoffFunction(w1=0.4, w2=0.2, w3=0.3, w4=0.1)
    print(f"\n{payoff_func}")
    print(f"Weights: {payoff_func.get_weights()}")
    
    # Test individual components
    print("\n--- Component Tests ---")
    
    # Mission success
    mission1 = payoff_func.calculate_mission_success((18, 18), (18, 18), 24.0, False, env)
    mission2 = payoff_func.calculate_mission_success((10, 10), (18, 18), 24.0, False, env)
    mission3 = payoff_func.calculate_mission_success((5, 5), (18, 18), 24.0, True, env)
    print(f"Mission Success (at goal): {mission1:.3f}")
    print(f"Mission Success (halfway): {mission2:.3f}")
    print(f"Mission Success (collision): {mission3:.3f}")
    
    # Energy consumed
    energy1 = payoff_func.calculate_energy_consumed(30, 100)
    energy2 = payoff_func.calculate_energy_consumed(80, 100)
    print(f"\nEnergy Consumed (30/100): {energy1:.3f}")
    print(f"Energy Consumed (80/100): {energy2:.3f}")
    
    # Collision risk
    risk1 = payoff_func.calculate_collision_risk(5.0)
    risk2 = payoff_func.calculate_collision_risk(1.0)
    risk3 = payoff_func.calculate_collision_risk(0.5)
    print(f"\nCollision Risk (distance=5.0): {risk1:.3f}")
    print(f"Collision Risk (distance=1.0): {risk2:.3f}")
    print(f"Collision Risk (distance=0.5): {risk3:.3f}")
    
    # Map quality
    map1 = payoff_func.calculate_map_quality(50, 400)
    map2 = payoff_func.calculate_map_quality(200, 400)
    print(f"\nMap Quality (50/400 explored): {map1:.3f}")
    print(f"Map Quality (200/400 explored): {map2:.3f}")
    
    # Compute overall payoff for different scenarios
    print("\n--- Overall Payoff Examples ---")
    
    # Scenario 1: Good progress, safe
    drone_payoff1, env_payoff1 = payoff_func.compute_payoff(
        DroneAction.MOVE_UP, EnvironmentCondition.CLEAR_PATH,
        current_pos=(10, 10), goal_pos=(18, 18), initial_distance=24.0,
        battery_used=30, total_battery=100, distance_to_nearest_obstacle=5.0,
        explored_cells=150, total_cells=400, collision=False, environment=env
    )
    print(f"Scenario 1 (good progress, safe): Drone={drone_payoff1:.1f}, Env={env_payoff1:.1f}")
    
    # Scenario 2: Near obstacle, risky
    drone_payoff2, env_payoff2 = payoff_func.compute_payoff(
        DroneAction.MOVE_RIGHT, EnvironmentCondition.OBSTACLE_AHEAD,
        current_pos=(10, 10), goal_pos=(18, 18), initial_distance=24.0,
        battery_used=30, total_battery=100, distance_to_nearest_obstacle=0.8,
        explored_cells=150, total_cells=400, collision=False, environment=env
    )
    print(f"Scenario 2 (near obstacle, risky): Drone={drone_payoff2:.1f}, Env={env_payoff2:.1f}")
    
    # Scenario 3: Reached goal
    drone_payoff3, env_payoff3 = payoff_func.compute_payoff(
        DroneAction.STAY, EnvironmentCondition.CLEAR_PATH,
        current_pos=(18, 18), goal_pos=(18, 18), initial_distance=24.0,
        battery_used=70, total_battery=100, distance_to_nearest_obstacle=3.0,
        explored_cells=250, total_cells=400, collision=False, environment=env
    )
    print(f"Scenario 3 (reached goal): Drone={drone_payoff3:.1f}, Env={env_payoff3:.1f}")
    
    # Generate a simple payoff matrix
    print("\n--- Payoff Matrix Example ---")
    drone_actions = [DroneAction.MOVE_UP, DroneAction.STAY]
    env_conditions = [EnvironmentCondition.CLEAR_PATH, EnvironmentCondition.OBSTACLE_AHEAD]
    
    state_params = {
        'current_pos': (10, 10),
        'goal_pos': (18, 18),
        'initial_distance': 24.0,
        'battery_used': 30,
        'total_battery': 100,
        'distance_to_nearest_obstacle': 3.0,
        'explored_cells': 150,
        'total_cells': 400,
        'collision': False,
        'environment': env
    }
    
    drone_matrix, env_matrix = payoff_func.generate_payoff_matrix(drone_actions, env_conditions, state_params)
    print(f"\nPayoff Matrices (rows=actions, cols=conditions):")
    print(f"Actions: {[a.value for a in drone_actions]}")
    print(f"Conditions: {[c.value for c in env_conditions]}")
    print("\nDrone Payoff Matrix:")
    print(drone_matrix)
    print("\nEnvironment Payoff Matrix:")
    print(env_matrix)
    
    return payoff_func


def test_strategy_simulation(env, payoff_func, drone_strategies=None, env_strategies=None):
    """Test PART 1: Run comprehensive strategy simulation (both pure and mixed strategies)
    
    Args:
        env: Environment instance
        payoff_func: PayoffFunction instance
        drone_strategies: List of (name, strategy) tuples for drone strategies to test.
                         If None, uses default predefined strategies.
        env_strategies: List of (name, strategy) tuples for environment strategies to test.
                       If None, uses default predefined strategies.
    """
    from backend.simulation.simulation import StrategySimulation
    
    # Create simulation engine
    simulation = StrategySimulation(env, payoff_func)
    
    # Run complete simulation with both pure and mixed strategies
    pure_results, mixed_results = simulation.run_complete_simulation(drone_strategies, env_strategies)
    
    return pure_results, mixed_results


def test_logger():
    """Test PART 1: Simulation Logger"""
    print("\n" + "=" * 70)
    print("TEST 5: LOGGER - Recording Simulation Data")
    print("=" * 70)
    
    # Create logger
    logger = SimulationLogger(log_name="test_simulation")
    print(f"\n{logger}")
    
    # Simulate logging some steps
    print("\nLogging simulation steps...")
    positions = [(1, 1), (2, 1), (3, 1), (3, 2), (4, 2), (5, 2)]
    actions = ["move_right", "move_right", "move_up", "move_right", "move_right"]
    
    for i, (pos, action) in enumerate(zip(positions[:-1], actions)):
        battery = 100 - (i * 5)
        payoff = np.random.uniform(0.2, 0.8)
        distance = np.sqrt((18 - pos[0])**2 + (18 - pos[1])**2)
        
        logger.log_step(
            step_number=i+1,
            action=action,
            position=pos,
            battery_level=battery,
            payoff=payoff,
            distance_to_goal=distance,
            additional_data={'collision_risk': 0.1 * i}
        )
        print(f"  Step {i+1}: {action} at {pos}, battery={battery}%, payoff={payoff:.3f}")
    
    # Log an event
    logger.log_event('milestone', 'Completed 5 steps successfully', {'total_distance': 25.0})
    
    # Set metadata
    logger.set_metadata('algorithm', 'minimax')
    logger.set_metadata('environment_size', '20x20')
    
    # Get summary
    print("\n--- Logger Summary ---")
    summary = logger.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save logs
    print("\n--- Saving Logs ---")
    json_path = logger.save_to_json()
    csv_path = logger.save_to_csv()
    print(f"  JSON saved to: {json_path}")
    print(f"  CSV saved to: {csv_path}")
    
    return logger


def main():
    print("\n")
    print("=" * 70)
    print(" DRONE VISUAL NAVIGATION PROJECT - PART 1 DEMONSTRATION")
    print("=" * 70)
    
    # Test all components
    env = test_environment()
    #drone_strategy, env_strategy = test_strategies()
    #payoff_func = test_payoff_function()
    payoff_func = PayoffFunction(w1=0.4, w2=0.2, w3=0.3, w4=0.1)
    
    # Define strategies to test in mixed strategy simulation
    # Comment out strategies you don't want to test
    
    drone_strategies_to_test =  [
        ('Uniform', DroneStrategies.create_uniform_mixed_strategy()),
        ('Cautious', DroneStrategies.create_cautious_strategy()),
        ('Aggressive', DroneStrategies.create_aggressive_strategy()),
        ('Balanced', DroneStrategies.create_balanced_strategy()),
        # Custom: Exploration mode (favor movement in all directions, only 4 actions)
        ('Custom_Exploration', DroneStrategies.create_custom_strategy({
           DroneAction.MOVE_UP: 0.3,
           DroneAction.MOVE_DOWN: 0.2,
           DroneAction.MOVE_LEFT: 0.25,
           DroneAction.MOVE_RIGHT: 0.25
           # Note: STAY and ROTATE omitted - tests probability lookup fix
         }))
    ]
    
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
    
    # Run simulation with custom strategies
    pure_results, mixed_results = test_strategy_simulation(env, payoff_func,drone_strategies_to_test,env_strategies_to_test)
    
    # Test logger
    test_logger()


if __name__ == "__main__":
    main()