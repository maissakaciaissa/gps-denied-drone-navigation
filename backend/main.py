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


def test_minimax():
    """Test PART 2: Minimax Algorithm"""
    print("\n" + "=" * 70)
    print("TEST: MINIMAX ALGORITHM - Adversarial Decision Making")
    print("=" * 70)
    
    from backend.game_theory.minimax import Minimax
    from backend.core.drone import Drone
    
    # 1. Créer l'environnement
    print("\n--- Setup ---")
    env = Environment(width=20, height=20, start_pos=(5, 5), goal_pos=(15, 15))
    
    # Ajouter des obstacles pour rendre la décision intéressante
    obstacles = [
        (7, 6), (8, 6), (9, 6),  # Mur horizontal devant
        (7, 7), (7, 8),          # Mur vertical à gauche
    ]
    env.add_obstacles(obstacles)
    print(f"Environment: {env}")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    print(f"Obstacles: {len(obstacles)} obstacles placés")
    
    # 2. Créer le drone
    drone = Drone(environment=env, battery_capacity=100)
    print(f"\nDrone créé à position: {drone.get_position()}")
    print(f"Batterie: {drone.get_battery_percentage():.1f}%")
    
    # 3. Créer la fonction de payoff
    payoff_func = PayoffFunction(w1=0.4, w2=0.2, w3=0.3, w4=0.1)
    print(f"\nPayoff Function: {payoff_func}")
    
    # 4. Créer le solver Minimax
    minimax_solver = Minimax(payoff=payoff_func)
    print(f"\nMinimax Solver créé")
    
    # 5. Construire les state_params
    state_params = {
        'current_pos': drone.get_position(),
        'goal_pos': env.goal_pos,
        'initial_distance': env.distance_to_goal(env.start_pos),
        'battery_used': drone.battery_capacity - drone.get_battery_level(),
        'total_battery': drone.battery_capacity,
        'distance_to_nearest_obstacle': env.get_nearest_obstacle_distance(drone.get_position()),
        'explored_cells': len(drone.explored_cells),
        'total_cells': env.width * env.height,
        'collision': False,
        'environment': env
    }
    
    print("\n--- State Parameters ---")
    print(f"Position: {state_params['current_pos']}")
    print(f"Goal: {state_params['goal_pos']}")
    print(f"Distance to goal: {env.distance_to_goal(state_params['current_pos']):.2f}")
    print(f"Distance to nearest obstacle: {state_params['distance_to_nearest_obstacle']:.2f}")
    print(f"Battery used: {state_params['battery_used']}/{state_params['total_battery']}")
    
    # 6. Test 1: evaluate_action() pour une seule action
    print("\n" + "-" * 70)
    print("TEST 1: evaluate_action() - Évaluer une seule action")
    print("-" * 70)
    
    test_action = DroneAction.MOVE_UP
    worst_payoff, worst_condition = minimax_solver.evaluate_action(test_action, state_params)
    
    print(f"\nAction testée: {test_action.value}")
    print(f"Pire cas payoff: {worst_payoff:.2f}")
    print(f"Pire condition: {worst_condition.value}")
    
    # 7. Test 2: get_worst_case_payoff()
    print("\n" + "-" * 70)
    print("TEST 2: get_worst_case_payoff() - Version simplifiée")
    print("-" * 70)
    
    worst_only = minimax_solver.get_worst_case_payoff(test_action, state_params)
    print(f"\nAction: {test_action.value}")
    print(f"Pire cas payoff: {worst_only:.2f}")
    
    # 8. Test 3: minimax_decision() avec toutes les actions
    print("\n" + "-" * 70)
    print("TEST 3: minimax_decision() - Choisir la meilleure action")
    print("-" * 70)
    
    # Obtenir les actions valides du drone
    valid_actions = drone.get_valid_actions()
    print(f"\nActions valides: {[a.value for a in valid_actions]}")
    
    # Appeler minimax_decision avec verbose=True
    best_action, best_worst_case, analysis = minimax_solver.minimax_decision(
        valid_actions, state_params, verbose=True
    )
    
    print(f"\n✅ Décision finale: {best_action.value}")
    print(f"✅ Payoff garanti: {best_worst_case:.2f}")
    
    # 9. Test 4: solve() - Interface simple
    print("\n" + "-" * 70)
    print("TEST 4: solve() - Interface simplifiée")
    print("-" * 70)
    
    chosen_action = minimax_solver.solve(valid_actions, state_params, verbose=False)
    print(f"\nAction choisie par solve(): {chosen_action.value}")
    
    # 10. Test 5: Navigation multi-étapes avec Minimax
    print("\n" + "-" * 70)
    print("TEST 5: Navigation Multi-Étapes avec Minimax")
    print("-" * 70)
    
    # Réinitialiser le drone
    drone2 = Drone(environment=env, battery_capacity=100)
    max_steps = 5
    
    print(f"\nNavigation de {max_steps} étapes avec Minimax")
    print(f"Position initiale: {drone2.get_position()}")
    print(f"Objectif: {env.goal_pos}\n")
    
    for step in range(max_steps):
        # Construire state_params
        state_params = {
            'current_pos': drone2.get_position(),
            'goal_pos': env.goal_pos,
            'initial_distance': env.distance_to_goal(env.start_pos),
            'battery_used': drone2.battery_capacity - drone2.get_battery_level(),
            'total_battery': drone2.battery_capacity,
            'distance_to_nearest_obstacle': env.get_nearest_obstacle_distance(drone2.get_position()),
            'explored_cells': len(drone2.explored_cells),
            'total_cells': env.width * env.height,
            'collision': False,
            'environment': env
        }
        
        # Obtenir actions valides
        valid_actions = drone2.get_valid_actions()
        
        if not valid_actions:
            print("❌ Aucune action valide disponible!")
            break
        
        # Minimax choisit l'action
        action = minimax_solver.solve(valid_actions, state_params, verbose=False)
        
        # Exécuter l'action
        success = drone2.move(action)
        
        # Afficher
        distance_to_goal = env.distance_to_goal(drone2.get_position())
        print(f"Étape {step+1}: {action.value:15s} → {drone2.get_position()} | "
              f"Batterie: {drone2.get_battery_percentage():5.1f}% | "
              f"Distance: {distance_to_goal:5.2f}")
        
        # Vérifier si objectif atteint
        if env.is_goal_reached(drone2.get_position()):
            print(f"\n🎯 Objectif atteint en {step+1} étapes!")
            break
    
    print(f"\nPosition finale: {drone2.get_position()}")
    print(f"Distance finale à l'objectif: {env.distance_to_goal(drone2.get_position()):.2f}")
    print(f"Batterie restante: {drone2.get_battery_percentage():.1f}%")
    print(f"Cellules explorées: {len(drone2.explored_cells)}")
    
    print("\n" + "=" * 70)
    print("✅ TESTS MINIMAX TERMINÉS")
    print("=" * 70)


def test_nash():
    """Test PART 3: Nash Equilibrium Algorithm"""
    print("\n" + "=" * 70)
    print("TEST: NASH EQUILIBRIUM ALGORITHM - Strategic Decision Making")
    print("=" * 70)
    
    from backend.game_theory.nash import NashEquilibriumSolver
    from backend.core.drone import Drone
    
    # 1. Créer l'environnement
    print("\n--- Setup ---")
    env = Environment(width=20, height=20, start_pos=(5, 5), goal_pos=(15, 15))
    
    # Ajouter des obstacles pour rendre la décision intéressante
    obstacles = [
        (7, 6), (8, 6), (9, 6),  # Mur horizontal devant
        (7, 7), (7, 8),          # Mur vertical à gauche
    ]
    env.add_obstacles(obstacles)
    print(f"Environment: {env}")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    print(f"Obstacles: {len(obstacles)} obstacles placés")
    
    # 2. Créer le drone
    drone = Drone(environment=env, battery_capacity=100)
    print(f"\nDrone créé à position: {drone.get_position()}")
    print(f"Batterie: {drone.get_battery_percentage():.1f}%")
    
    # 3. Créer la fonction de payoff
    payoff_func = PayoffFunction(w1=0.4, w2=0.2, w3=0.3, w4=0.1)
    print(f"\nPayoff Function: {payoff_func}")
    
    # 4. Créer le solver Nash
    nash_solver = NashEquilibriumSolver(payoff_func)
    print(f"\nNash Equilibrium Solver créé")
    
    # 5. Construire les state_params
    state_params = {
        'current_pos': drone.get_position(),
        'goal_pos': env.goal_pos,
        'initial_distance': env.distance_to_goal(env.start_pos),
        'battery_used': drone.battery_capacity - drone.get_battery_level(),
        'total_battery': drone.battery_capacity,
        'distance_to_nearest_obstacle': env.get_nearest_obstacle_distance(drone.get_position()),
        'explored_cells': len(drone.explored_cells),
        'total_cells': env.width * env.height,
        'collision': False,
        'environment': env
    }
    
    print("\n--- State Parameters ---")
    print(f"Position: {state_params['current_pos']}")
    print(f"Goal: {state_params['goal_pos']}")
    print(f"Distance to goal: {env.distance_to_goal(state_params['current_pos']):.2f}")
    print(f"Distance to nearest obstacle: {state_params['distance_to_nearest_obstacle']:.2f}")
    print(f"Battery used: {state_params['battery_used']}/{state_params['total_battery']}")
    
    # 6. Test 1: find_nash_equilibrium() pour un sous-ensemble d'actions
    print("\n" + "-" * 70)
    print("TEST 1: find_nash_equilibrium() - Trouver l'équilibre Nash")
    print("-" * 70)
    
    # Utiliser un petit sous-ensemble pour la démonstration
    test_drone_actions = [DroneAction.MOVE_UP, DroneAction.MOVE_RIGHT, DroneAction.STAY]
    test_env_conditions = [EnvironmentCondition.CLEAR_PATH, EnvironmentCondition.OBSTACLE_AHEAD]
    
    print(f"\nActions testées: {[a.value for a in test_drone_actions]}")
    print(f"Conditions testées: {[c.value for c in test_env_conditions]}")
    
    # Générer les matrices de payoff
    drone_matrix, env_matrix = payoff_func.generate_payoff_matrix(
        test_drone_actions, test_env_conditions, state_params
    )
    
    print(f"\nMatrice de payoff Drone:")
    print(drone_matrix)
    print(f"\nMatrice de payoff Environnement:")
    print(env_matrix)
    
    # Trouver l'équilibre Nash
    nash_drone, nash_env = nash_solver.find_nash_equilibrium(
        drone_matrix, env_matrix, test_drone_actions, test_env_conditions
    )
    
    print(f"\n✅ Équilibre de Nash trouvé:")
    print(f"\nStratégie Drone (Nash):")
    for action, prob in zip(nash_drone.strategies, nash_drone.probabilities):
        print(f"  {action.value:20s}: {prob:6.2%}")
    
    print(f"\nStratégie Environnement (Nash):")
    for condition, prob in zip(nash_env.strategies, nash_env.probabilities):
        print(f"  {condition.value:20s}: {prob:6.2%}")
    
    # 7. Test 2: calculate_expected_payoff()
    print("\n" + "-" * 70)
    print("TEST 2: calculate_expected_payoff() - Calculer le payoff espéré")
    print("-" * 70)
    
    expected_payoff = nash_solver.calculate_expected_payoff(nash_drone, nash_env, drone_matrix)
    print(f"\nPayoff espéré à l'équilibre Nash: {expected_payoff:.2f}")
    
    # 8. Test 3: _verify_nash() - Vérification
    print("\n" + "-" * 70)
    print("TEST 3: _verify_nash() - Vérifier l'équilibre")
    print("-" * 70)
    
    is_nash = nash_solver._verify_nash(
        np.array(nash_drone.probabilities), np.array(nash_env.probabilities),
        drone_matrix, env_matrix
    )
    print(f"\n✅ Vérification: Est-ce un équilibre de Nash? {is_nash}")
    
    if is_nash:
        print("  - Condition d'indifférence satisfaite")
        print("  - Aucune déviation profitable pour aucun joueur")
    
    # 9. Test 4: solve() - Interface simple
    print("\n" + "-" * 70)
    print("TEST 4: solve() - Interface simplifiée")
    print("-" * 70)
    
    # Obtenir les actions valides du drone
    valid_actions = drone.get_valid_actions()
    print(f"\nActions valides: {[a.value for a in valid_actions]}")
    
    chosen_action = nash_solver.solve(state_params)
    print(f"\nAction choisie par solve(): {chosen_action.value}")
    
    # Afficher les stratégies Nash complètes
    full_nash_drone, full_nash_env = nash_solver.get_nash_strategies()
    if full_nash_drone and full_nash_env:
        print(f"\n✅ Stratégies Nash complètes:")
        print(f"\nDrone (top 3 actions):")
        probs = list(zip(full_nash_drone.strategies, full_nash_drone.probabilities))
        probs.sort(key=lambda x: x[1], reverse=True)
        for action, prob in probs[:3]:
            print(f"  {action.value:20s}: {prob:6.2%}")
        
        print(f"\nEnvironnement (top 3 conditions):")
        env_probs = list(zip(full_nash_env.strategies, full_nash_env.probabilities))
        env_probs.sort(key=lambda x: x[1], reverse=True)
        for condition, prob in env_probs[:3]:
            print(f"  {condition.value:20s}: {prob:6.2%}")
    
    # 10. Test 5: Navigation multi-étapes avec Nash Equilibrium
    print("\n" + "-" * 70)
    print("TEST 5: Navigation Multi-Étapes avec Nash Equilibrium")
    print("-" * 70)
    
    # Réinitialiser le drone
    drone2 = Drone(environment=env, battery_capacity=100)
    max_steps = 5
    
    print(f"\nNavigation de {max_steps} étapes avec Nash Equilibrium")
    print(f"Position initiale: {drone2.get_position()}")
    print(f"Objectif: {env.goal_pos}\n")
    
    for step in range(max_steps):
        # Construire state_params
        state_params = {
            'current_pos': drone2.get_position(),
            'goal_pos': env.goal_pos,
            'initial_distance': env.distance_to_goal(env.start_pos),
            'battery_used': drone2.battery_capacity - drone2.get_battery_level(),
            'total_battery': drone2.battery_capacity,
            'distance_to_nearest_obstacle': env.get_nearest_obstacle_distance(drone2.get_position()),
            'explored_cells': len(drone2.explored_cells),
            'total_cells': env.width * env.height,
            'collision': False,
            'environment': env
        }
        
        # Nash choisit l'action
        action = nash_solver.solve(state_params)
        
        # Exécuter l'action
        success = drone2.move(action)
        
        # Afficher
        distance_to_goal = env.distance_to_goal(drone2.get_position())
        print(f"Étape {step+1}: {action.value:15s} → {drone2.get_position()} | "
              f"Batterie: {drone2.get_battery_percentage():5.1f}% | "
              f"Distance: {distance_to_goal:5.2f}")
        
        # Vérifier si objectif atteint
        if env.is_goal_reached(drone2.get_position()):
            print(f"\n🎯 Objectif atteint en {step+1} étapes!")
            break
    
    print(f"\nPosition finale: {drone2.get_position()}")
    print(f"Distance finale à l'objectif: {env.distance_to_goal(drone2.get_position()):.2f}")
    print(f"Batterie restante: {drone2.get_battery_percentage():.1f}%")
    print(f"Cellules explorées: {len(drone2.explored_cells)}")
    
    print("\n" + "=" * 70)
    print("✅ TESTS NASH EQUILIBRIUM TERMINÉS")
    print("=" * 70)


def test_bayesian():
    """Test PART 4: Bayesian Game Solver"""
    print("\n" + "=" * 70)
    print("TEST: BAYESIAN GAME SOLVER - Decision Making Under Uncertainty")
    print("=" * 70)
    
    from backend.game_theory.bayesian import BayesianGameSolver
    from backend.core.drone import Drone
    
    # 1. Créer l'environnement
    print("\n--- Setup ---")
    env = Environment(width=20, height=20, start_pos=(5, 5), goal_pos=(15, 15))
    
    # Ajouter des obstacles pour un environnement incertain
    obstacles = [
        (7, 6), (8, 6), (9, 6),  # Mur horizontal devant
        (7, 7), (7, 8),          # Mur vertical à gauche
        (12, 10), (12, 11),      # Obstacles dispersés
    ]
    env.add_obstacles(obstacles)
    print(f"Environment: {env}")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    print(f"Obstacles: {len(obstacles)} obstacles placés")
    
    # 2. Créer le drone
    drone = Drone(environment=env, battery_capacity=100)
    print(f"\nDrone créé à position: {drone.get_position()}")
    print(f"Batterie: {drone.get_battery_percentage():.1f}%")
    
    # 3. Créer la fonction de payoff
    payoff_func = PayoffFunction(w1=0.4, w2=0.2, w3=0.3, w4=0.1)
    print(f"\nPayoff Function: {payoff_func}")
    
    # 4. Créer le solver Bayesian
    bayesian_solver = BayesianGameSolver(payoff_func)
    print(f"\nBayesian Solver créé")
    
    # 5. Initialiser les croyances (priors)
    print("\n--- Belief Initialization ---")
    beliefs = bayesian_solver.initialize_beliefs(
        prior_adversarial=0.4,
        prior_neutral=0.4,
        prior_favorable=0.2
    )
    print("Croyances initiales:")
    for env_type, prob in beliefs.items():
        print(f"  {env_type:15s}: {prob:6.2%}")
    
    # 6. Construire les state_params
    state_params = {
        'current_pos': drone.get_position(),
        'goal_pos': env.goal_pos,
        'initial_distance': env.distance_to_goal(env.start_pos),
        'battery_used': drone.battery_capacity - drone.get_battery_level(),
        'total_battery': drone.battery_capacity,
        'distance_to_nearest_obstacle': env.get_nearest_obstacle_distance(drone.get_position()),
        'explored_cells': len(drone.explored_cells),
        'total_cells': env.width * env.height,
        'collision': False,
        'environment': env
    }
    
    # 7. Test 1: Expected Utility Calculation
    print("\n" + "-" * 70)
    print("TEST 1: Expected Utility Calculation")
    print("-" * 70)
    
    print("\nUtilités espérées pour chaque action:")
    for action in [DroneAction.MOVE_UP, DroneAction.MOVE_RIGHT, DroneAction.STAY]:
        expected_util = bayesian_solver.expected_utility(action, state_params)
        print(f"  {action.value:15s}: {expected_util:7.2f}")
    
    # 8. Test 2: Bayesian Decision (verbose)
    print("\n" + "-" * 70)
    print("TEST 2: Bayesian Decision Making")
    print("-" * 70)
    
    available_actions = [DroneAction.MOVE_UP, DroneAction.MOVE_RIGHT, DroneAction.STAY]
    best_action, best_utility, analysis = bayesian_solver.bayesian_decision(
        available_actions, state_params, verbose=True
    )
    
    # 9. Test 3: Belief Update
    print("\n" + "-" * 70)
    print("TEST 3: Belief Update (Bayesian Learning)")
    print("-" * 70)
    
    # Simuler l'observation d'un obstacle
    print("\nSimulation: Le drone observe un OBSTACLE_AHEAD")
    updated_beliefs = bayesian_solver.update_beliefs(
        best_action,
        EnvironmentCondition.OBSTACLE_AHEAD,
        25.0
    )
    
    print("\nCroyances mises à jour:")
    for env_type, prob in updated_beliefs.items():
        print(f"  {env_type:15s}: {prob:6.2%}")
    print(f"\nNombre d'observations: {bayesian_solver.get_observation_count()}")
    
    # 10. Test 4: Solve Method
    print("\n" + "-" * 70)
    print("TEST 4: solve() Method")
    print("-" * 70)
    
    action = bayesian_solver.solve(list(DroneAction), state_params, verbose=False)
    print(f"\nAction recommandée: {action.value}")
    print(f"Croyances actuelles: {bayesian_solver.get_beliefs()}")
    
    # 11. Test 5: Learning Over Multiple Observations
    print("\n" + "-" * 70)
    print("TEST 5: Learning Over Multiple Observations")
    print("-" * 70)
    
    # Réinitialiser avec croyances uniformes
    bayesian_solver.reset()
    bayesian_solver.initialize_beliefs(0.33, 0.34, 0.33)
    
    print("\nCroyances initiales (uniformes):")
    for env_type, prob in bayesian_solver.get_beliefs().items():
        print(f"  {env_type:15s}: {prob:6.2%}")
    
    # Simuler 5 observations d'obstacles (preuve d'adversarial)
    print("\nSimulation: 5 observations d'obstacles...")
    for i in range(5):
        bayesian_solver.update_beliefs(
            DroneAction.MOVE_UP,
            EnvironmentCondition.OBSTACLE_AHEAD,
            30.0
        )
    
    print("\nCroyances après 5 obstacles:")
    for env_type, prob in bayesian_solver.get_beliefs().items():
        print(f"  {env_type:15s}: {prob:6.2%}")
    
    # Maintenant 10 observations de CLEAR_PATH (preuve de favorable)
    print("\nSimulation: 10 observations de chemins clairs...")
    for i in range(10):
        bayesian_solver.update_beliefs(
            DroneAction.MOVE_UP,
            EnvironmentCondition.CLEAR_PATH,
            85.0
        )
    
    print("\nCroyances après 10 chemins clairs:")
    for env_type, prob in bayesian_solver.get_beliefs().items():
        print(f"  {env_type:15s}: {prob:6.2%}")
    print(f"\nTotal observations: {bayesian_solver.get_observation_count()}")
    
    # 12. Test 6: Navigation Multi-Étapes avec Bayesian
    print("\n" + "-" * 70)
    print("TEST 6: Navigation Multi-Étapes avec Bayesian Learning")
    print("-" * 70)
    
    # Réinitialiser pour navigation
    bayesian_solver.reset()
    bayesian_solver.initialize_beliefs(0.3, 0.5, 0.2)
    
    drone2 = Drone(environment=env, battery_capacity=100)
    max_steps = 5
    
    print(f"\nNavigation de {max_steps} étapes avec Bayesian")
    print(f"Position initiale: {drone2.get_position()}")
    print(f"Objectif: {env.goal_pos}\n")
    
    for step in range(max_steps):
        # Construire state_params
        state_params = {
            'current_pos': drone2.get_position(),
            'goal_pos': env.goal_pos,
            'initial_distance': env.distance_to_goal(env.start_pos),
            'battery_used': drone2.battery_capacity - drone2.get_battery_level(),
            'total_battery': drone2.battery_capacity,
            'distance_to_nearest_obstacle': env.get_nearest_obstacle_distance(drone2.get_position()),
            'explored_cells': len(drone2.explored_cells),
            'total_cells': env.width * env.height,
            'collision': False,
            'environment': env
        }
        
        # Bayesian choisit l'action
        action = bayesian_solver.solve(list(DroneAction), state_params)
        
        # Simuler une observation (ici on sample une condition)
        observed_condition = EnvironmentCondition.CLEAR_PATH  # Simplified
        
        # Mettre à jour les croyances
        bayesian_solver.update_beliefs(action, observed_condition, 70.0)
        
        # Exécuter l'action
        success = drone2.move(action)
        
        # Afficher
        distance_to_goal = env.distance_to_goal(drone2.get_position())
        beliefs = bayesian_solver.get_beliefs()
        print(f"Étape {step+1}: {action.value:15s} → {drone2.get_position()} | "
              f"Batterie: {drone2.get_battery_percentage():5.1f}% | "
              f"Distance: {distance_to_goal:5.2f} | "
              f"Beliefs: Adv={beliefs['adversarial']:.1%}")
        
        # Vérifier si objectif atteint
        if env.is_goal_reached(drone2.get_position()):
            print(f"\n🎯 Objectif atteint en {step+1} étapes!")
            break
    
    print(f"\nPosition finale: {drone2.get_position()}")
    print(f"Distance finale à l'objectif: {env.distance_to_goal(drone2.get_position()):.2f}")
    print(f"Batterie restante: {drone2.get_battery_percentage():.1f}%")
    print(f"Cellules explorées: {len(drone2.explored_cells)}")
    print(f"Croyances finales:")
    for env_type, prob in bayesian_solver.get_beliefs().items():
        print(f"  {env_type:15s}: {prob:6.2%}")
    
    print("\n" + "=" * 70)
    print("✅ TESTS BAYESIAN SOLVER TERMINÉS")
    print("=" * 70)


def test_sensor():
    """Test PART 4: Drone Sensor - Environment Perception"""
    print("\n" + "=" * 70)
    print("TEST: DRONE SENSOR - Vision and Environment Perception")
    print("=" * 70)
    
    from backend.core.sensor import DroneSensor
    from backend.core.drone import Drone
    
    # 1. Create environment with obstacles
    print("\n--- Setup ---")
    env = Environment(width=20, height=20, start_pos=(10, 10), goal_pos=(15, 15))
    
    # Add obstacles around drone position
    obstacles = [
        (11, 10), (12, 10), (13, 10),  # Right side
        (10, 11), (10, 12),             # Below
        (9, 9), (8, 9)                  # Upper left
    ]
    env.add_obstacles(obstacles)
    print(f"Environment: {env}")
    print(f"Obstacles placed: {len(obstacles)}")
    
    # 2. Create drone and sensor
    drone = Drone(environment=env, battery_capacity=100)
    sensor = DroneSensor(detection_range=5)
    
    print(f"\nDrone position: {drone.get_position()}")
    print(f"Sensor detection range: {sensor.detection_range}")
    
    # 3. Test 1: detect_obstacles()
    print("\n" + "-" * 70)
    print("TEST 1: detect_obstacles() - Directional Obstacle Detection")
    print("-" * 70)
    
    detected = sensor.detect_obstacles(env, drone.get_position())
    print(f"\nObstacle detection from {drone.get_position()}:")
    for direction, distance in detected.items():
        if distance is not None:
            print(f"  {direction:6s}: Obstacle at {distance} cells away")
        else:
            print(f"  {direction:6s}: Clear (no obstacles within range)")
    
    # 4. Test 2: scan_environment()
    print("\n" + "-" * 70)
    print("TEST 2: scan_environment() - Full Environment Scan")
    print("-" * 70)
    
    visible_obstacles = sensor.scan_environment(env, drone.get_position())
    print(f"\nVisible obstacles from {drone.get_position()}:")
    print(f"  Total visible: {len(visible_obstacles)} obstacles")
    for obs_pos in visible_obstacles[:5]:  # Show first 5
        distance = np.sqrt((obs_pos[0] - drone.get_position()[0])**2 + 
                          (obs_pos[1] - drone.get_position()[1])**2)
        print(f"    Position: {obs_pos}, Distance: {distance:.2f}")
    if len(visible_obstacles) > 5:
        print(f"    ... and {len(visible_obstacles) - 5} more")
    
    # 5. Test 3: sense_environment_condition()
    print("\n" + "-" * 70)
    print("TEST 3: sense_environment_condition() - High-level Perception")
    print("-" * 70)
    
    condition = sensor.sense_environment_condition(env, drone.get_position())
    print(f"\nEnvironment condition detected: {condition.value}")
    
    # 6. Test 4: Sensor Info and Visibility
    print("\n" + "-" * 70)
    print("TEST 4: Sensor Information and Status")
    print("-" * 70)
    
    sensor_info = sensor.get_sensor_info()
    print(f"\nSensor Information:")
    for key, value in sensor_info.items():
        print(f"  {key}: {value}")
    
    print(f"\nSensor representation: {sensor}")
    
    # 7. Test 5: Multiple positions
    print("\n" + "-" * 70)
    print("TEST 5: Sensor Performance at Multiple Positions")
    print("-" * 70)
    
    test_positions = [(10, 10), (5, 5), (15, 15), (1, 1)]
    print("\nTesting sensor at different positions:")
    for pos in test_positions:
        if env.is_valid_position(pos):
            visible = sensor.scan_environment(env, pos)
            condition = sensor.sense_environment_condition(env, pos)
            detected_dirs = sensor.detect_obstacles(env, pos)
            blocked_dirs = sum(1 for d in detected_dirs.values() if d is not None)
            print(f"  Position {pos}: {len(visible)} visible obstacles, "
                  f"{blocked_dirs} blocked directions, Condition: {condition.value}")
    
    print("\n" + "=" * 70)
    print("✅ TESTS SENSOR TERMINÉS")
    print("=" * 70)


def test_engine():
    """Test PART 4: Simulation Engine - Comprehensive Testing"""
    print("\n" + "=" * 70)
    print("TEST: SIMULATION ENGINE - Drone Navigation Simulation")
    print("=" * 70)
    
    from backend.simulation.engine import SimulationEngine
    from backend.core.drone import Drone
    
    # Setup
    print("\n--- Setup ---")
    env = Environment(width=15, height=15, start_pos=(2, 2), goal_pos=(12, 12))
    obstacles = [(7, 7), (8, 7), (7, 8), (9, 9), (10, 5)]
    env.add_obstacles(obstacles)
    print(f"Environment: {env}")
    print(f"Obstacles: {len(obstacles)}")
    
    drone = Drone(environment=env, battery_capacity=150)
    payoff_func = PayoffFunction()
    
    engine = SimulationEngine(env, drone, payoff_func)
    print(f"\nEngine initialized: {engine}")
    
    # TEST 1: Run simulation with different algorithms
    print("\n" + "-" * 70)
    print("TEST 1: Algorithm Comparison")
    print("-" * 70)
    
    algorithms = ['minimax', 'nash', 'bayesian']
    results_summary = []
    
    for algo in algorithms:
        engine.reset()
        result = engine.run_simulation(max_steps=50, algorithm=algo, verbose=False)
        results_summary.append((algo, result))
        print(f"\n{algo.upper()} Results:")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['path_length']}")
        print(f"  Battery: {result['battery_used_percent']:.1f}%")
        print(f"  Collisions: {result['collisions']}")
    
    # TEST 2: Detailed results analysis
    print("\n" + "-" * 70)
    print("TEST 2: Detailed Results Analysis")
    print("-" * 70)
    
    # Use Nash (best performer) for detailed analysis
    engine.reset()
    result = engine.run_simulation(max_steps=50, algorithm='nash', verbose=False)
    
    print(f"\nDetailed Nash Results:")
    print(f"  Success: {result.get('success', False)}")
    print(f"  Path Length: {result.get('path_length', 0)} steps")
    print(f"  Battery Used: {result.get('battery_used_percent', 0):.1f}%")
    print(f"  Collisions: {result.get('collisions', 0)}")
    print(f"  Cells Explored: {result.get('cells_explored', 0)}")
    print(f"  Exploration Rate: {result.get('exploration_rate', 0):.1%}")
    print(f"  Path Efficiency: {result.get('path_efficiency', 0):.2f}")
    print(f"  Computation Time: {result.get('computation_time', 0):.3f}s")
    print(f"  Final Distance to Goal: {result.get('final_distance_to_goal', 0):.2f}")
    
    # TEST 3: Engine state and configuration
    print("\n" + "-" * 70)
    print("TEST 3: Engine State & Configuration")
    print("-" * 70)
    
    print(f"\nEngine Configuration:")
    print(f"  Algorithm: {engine.algorithm_mode}")
    print(f"  Step Counter: {engine.step_counter}")
    print(f"  Drone Position: {engine.drone.position}")
    print(f"  Drone Battery: {engine.drone.get_battery_percentage():.1f}%")
    print(f"  Environment Size: {engine.environment.width}x{engine.environment.height}")
    print(f"  Goal Position: {engine.environment.goal_pos}")
    
    # TEST 4: Multiple runs with reset
    print("\n" + "-" * 70)
    print("TEST 4: Multiple Runs with Reset")
    print("-" * 70)
    
    print(f"\nRunning 3 consecutive simulations with Nash:")
    for i in range(3):
        engine.reset()
        result = engine.run_simulation(max_steps=50, algorithm='nash', verbose=False)
        print(f"  Run {i+1}: Steps={result['path_length']:3d}, "
              f"Battery={result['battery_used_percent']:5.1f}%, "
              f"Success={result['success']}")
    
    # TEST 5: Path tracking
    print("\n" + "-" * 70)
    print("TEST 5: Path Tracking")
    print("-" * 70)
    
    engine.reset()
    result = engine.run_simulation(max_steps=30, algorithm='nash', verbose=False)
    path = engine.drone.path  # Get path from drone object
    
    print(f"\nPath taken ({len(path)} positions):")
    print(f"  Start: {path[0] if path else 'N/A'}")
    if len(path) > 5:
        print(f"  Intermediate: {path[1]} -> {path[len(path)//2]} -> {path[-2]}")
    print(f"  End: {path[-1] if path else 'N/A'}")
    print(f"  Goal: {env.goal_pos}")
    print(f"  Goal Reached: {result['success']}")
    print(f"  Total positions visited: {len(path)}")
    
    print("\n" + "=" * 70)
    print("✅ SIMULATION ENGINE TESTS COMPLETE")
    print("=" * 70)


def test_compare():
    """Test PART 4: Algorithm Comparator - Multi-algorithm Performance Analysis"""
    print("\n" + "=" * 70)
    print("TEST: ALGORITHM COMPARATOR - Performance Comparison")
    print("=" * 70)
    
    from backend.evaluation.compare import AlgorithmComparator
    from backend.core.drone import Drone
    
    # Setup
    print("\n--- Setup ---")
    env = Environment(width=15, height=15, start_pos=(2, 2), goal_pos=(12, 12))
    obstacles = [(7, 7), (8, 7), (7, 8), (10, 10), (5, 9)]
    env.add_obstacles(obstacles)
    print(f"Environment: {env}")
    print(f"Obstacles: {len(obstacles)}")
    
    drone = Drone(environment=env, battery_capacity=150)
    payoff_func = PayoffFunction()
    
    comparator = AlgorithmComparator(env, drone, payoff_func)
    print(f"\nAlgorithm Comparator created: {comparator}")
    
    # TEST 1: Multi-algorithm comparison
    print("\n" + "-" * 70)
    print("TEST 1: Multi-Algorithm Comparison")
    print("-" * 70)
    
    print("\nTesting: Minimax, Nash Equilibrium, Bayesian")
    print("Trials per algorithm: 3")
    print("Max steps per trial: 100\n")
    
    all_results = comparator.run_all_algorithms(
        trials=3,
        max_steps=100,
        verbose=True
    )
    
    # TEST 2: Results summary by algorithm
    print("\n" + "-" * 70)
    print("TEST 2: Results Summary by Algorithm")
    print("-" * 70)
    
    for algo, results in all_results.items():
        print(f"\n{algo.upper()} Results:")
        print(f"  Trials completed: {len(results)}")
        successes = sum(1 for r in results if r.get('success', False))
        print(f"  Successful: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
        
        if results:
            avg_steps = np.mean([r.get('path_length', 0) for r in results])
            avg_battery = np.mean([r.get('battery_used_percent', 0) for r in results])
            avg_time = np.mean([r.get('computation_time', 0) for r in results])
            print(f"  Average steps: {avg_steps:.1f}")
            print(f"  Average battery used: {avg_battery:.1f}%")
            print(f"  Average computation time: {avg_time:.3f}s")
    
    # TEST 3: Statistical comparison
    print("\n" + "-" * 70)
    print("TEST 3: Statistical Comparison")
    print("-" * 70)
    
    comparison = comparator.compare_metrics()
    
    print("\nComparison by Algorithm:")
    for algo, metrics in comparison.items():
        if isinstance(metrics, dict) and 'success_rate' in metrics:
            print(f"\n{algo.upper()}:")
            print(f"  Success Rate: {metrics.get('success_rate', 0):.1%}")
            
            if 'path_length' in metrics:
                pl = metrics['path_length']
                print(f"  Path Length: {pl.get('mean', 0):.1f} ± {pl.get('std', 0):.1f}")
            
            if 'battery_used_percent' in metrics:
                bat = metrics['battery_used_percent']
                print(f"  Battery Used: {bat.get('mean', 0):.1f}% ± {bat.get('std', 0):.1f}%")
            
            if 'computation_time' in metrics:
                ct = metrics['computation_time']
                print(f"  Computation Time: {ct.get('mean', 0):.3f}s ± {ct.get('std', 0):.3f}s")
    
    # TEST 4: Detailed comparison report
    print("\n" + "-" * 70)
    print("TEST 4: Detailed Comparison Report")
    print("-" * 70)
    
    report = comparator.generate_comparison_report()
    print(report)
    
    # TEST 5: Winner identification
    print("\n" + "-" * 70)
    print("TEST 5: Winner Identification")
    print("-" * 70)
    
    # Identify winners manually from comparison
    print("\nBest Algorithm by Metric:")
    
    # Success rate winner
    success_rates = {}
    for algo, metrics in comparison.items():
        if isinstance(metrics, dict) and 'success_rate' in metrics:
            success_rates[algo] = metrics['success_rate']
    if success_rates:
        best_success = max(success_rates, key=success_rates.get)
        print(f"  Highest Success Rate: {best_success} ({success_rates[best_success]:.1%})")
    
    # Shortest path winner
    path_lengths = {}
    for algo, metrics in comparison.items():
        if isinstance(metrics, dict) and 'path_length' in metrics:
            path_lengths[algo] = metrics['path_length'].get('mean', float('inf'))
    if path_lengths:
        best_path = min(path_lengths, key=path_lengths.get)
        print(f"  Shortest Path: {best_path} ({path_lengths[best_path]:.1f} steps)")
    
    # Battery efficiency winner
    battery_usage = {}
    for algo, metrics in comparison.items():
        if isinstance(metrics, dict) and 'battery_used_percent' in metrics:
            battery_usage[algo] = metrics['battery_used_percent'].get('mean', float('inf'))
    if battery_usage:
        best_battery = min(battery_usage, key=battery_usage.get)
        print(f"  Most Battery Efficient: {best_battery} ({battery_usage[best_battery]:.1f}%)")
    
    # TEST 6: Visualization option
    print("\n" + "-" * 70)
    print("TEST 6: Visualization (Optional)")
    print("-" * 70)
    
    print("\n💡 To generate visual comparison plots:")
    print("   comparator.visualize_comparison()")
    print("   comparator.visualize_comparison(save_path='comparison.png')")
    
    print("\n" + "=" * 70)
    print("✅ ALGORITHM COMPARISON TESTS COMPLETE")
    print("=" * 70)


"""
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
    test_minimax()

"""

def main():
    print("\n")
    print("=" * 70)
    print(" DRONE VISUAL NAVIGATION PROJECT - TESTS")
    print("=" * 70)
    
    # Commentez les tests que vous ne voulez pas exécuter
    
    # PART 1: Core Components
    # env = test_environment()
    # drone_strategy, env_strategy = test_strategies()
    # payoff_func = test_payoff_function()
    # test_logger()
    
    # PART 2: Game Theory Algorithms
    # test_minimax()
    test_nash()
    # test_bayesian()
    
    # PART 4: Simulation Components
    # test_sensor()
    # test_engine()
    # test_compare()
    
    print("\n" + "=" * 70)
    print(" TOUS LES TESTS TERMINÉS")
    print("=" * 70)


if __name__ == "__main__":
    main()
