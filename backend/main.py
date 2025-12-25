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
    # Utilisation de poids plus agressifs pour encourager le mouvement vers l'objectif
    payoff_func = PayoffFunction(w1=0.7, w2=0.05, w3=0.15, w4=0.1, stay_penalty_factor=0.4)
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
    max_steps = 100
    
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
    
    # 10. Test 6: NOUVEAU ENVIRONNEMENT - Labyrinthe complexe
    print("\n" + "-" * 70)
    print("TEST 6: Navigation dans un Labyrinthe Complexe")
    print("-" * 70)
    
    # Créer un environnement plus difficile avec un labyrinthe
    env_maze = Environment(width=25, height=25, start_pos=(2, 2), goal_pos=(22, 22))
    
    # Créer un labyrinthe en forme de couloirs
    maze_obstacles = [
        # Mur vertical gauche
        *[(5, y) for y in range(5, 20)],
        # Mur horizontal haut
        *[(x, 10) for x in range(5, 15)],
        # Mur vertical central
        *[(15, y) for y in range(2, 15)],
        # Mur horizontal bas
        *[(x, 18) for x in range(10, 20)],
        # Obstacles dispersés
        (8, 5), (8, 6), (12, 12), (12, 13), (18, 8), (18, 9)
    ]
    env_maze.add_obstacles(maze_obstacles)
    
    print(f"\nEnvironnement Labyrinthe: {env_maze}")
    print(f"Start: {env_maze.start_pos}, Goal: {env_maze.goal_pos}")
    print(f"Obstacles: {len(maze_obstacles)} obstacles")
    print(f"Coût batterie par mouvement: 2 unités (nouveau)")
    
    # Créer drone avec plus de batterie pour le labyrinthe
    drone_maze = Drone(environment=env_maze, battery_capacity=150)
    max_steps = 150
    
    print(f"\nNavigation dans le labyrinthe avec Minimax")
    print(f"Position initiale: {drone_maze.get_position()}")
    print(f"Objectif: {env_maze.goal_pos}")
    print(f"Batterie initiale: {drone_maze.get_battery_level()} unités\n")
    
    trajectory = []  # Pour enregistrer la trajectoire
    
    for step in range(max_steps):
        current_pos = drone_maze.get_position()
        trajectory.append(current_pos)
        
        # State params
        state_params = {
            'current_pos': current_pos,
            'goal_pos': env_maze.goal_pos,
            'initial_distance': env_maze.distance_to_goal(env_maze.start_pos),
            'battery_used': drone_maze.battery_capacity - drone_maze.get_battery_level(),
            'total_battery': drone_maze.battery_capacity,
            'distance_to_nearest_obstacle': env_maze.get_nearest_obstacle_distance(current_pos),
            'explored_cells': len(drone_maze.explored_cells),
            'total_cells': env_maze.width * env_maze.height,
            'collision': False,
            'environment': env_maze
        }
        
        # Actions valides
        valid_actions = drone_maze.get_valid_actions()
        
        if not valid_actions:
            print("❌ Aucune action valide!")
            break
        
        # Minimax décision
        action = minimax_solver.solve(valid_actions, state_params, verbose=False)
        
        # Exécuter
        success = drone_maze.move(action)
        
        # Afficher tous les 10 pas ou aux points clés
        if (step + 1) % 10 == 0 or success == False or env_maze.is_goal_reached(current_pos):
            distance_to_goal = env_maze.distance_to_goal(drone_maze.get_position())
            print(f"Étape {step+1:3d}: {action.value:15s} → {drone_maze.get_position()} | "
                  f"Batt: {drone_maze.get_battery_percentage():5.1f}% | "
                  f"Dist: {distance_to_goal:5.2f} | "
                  f"Exploré: {len(drone_maze.explored_cells)}")
        
        # Check goal
        if env_maze.is_goal_reached(drone_maze.get_position()):
            print(f"\n Objectif atteint en {step+1} étapes!")
            print(f" Statistiques:")
            print(f"   - Étapes totales: {step+1}")
            print(f"   - Batterie consommée: {drone_maze.battery_capacity - drone_maze.get_battery_level()} / {drone_maze.battery_capacity}")
            print(f"   - Batterie restante: {drone_maze.get_battery_percentage():.1f}%")
            print(f"   - Cellules explorées: {len(drone_maze.explored_cells)}")
            print(f"   - Efficacité: {len(trajectory)} positions visitées")
            break
    else:
        print(f"\n Limite de {max_steps} étapes atteinte")
        print(f"Position finale: {drone_maze.get_position()}")
        print(f"Distance à l'objectif: {env_maze.distance_to_goal(drone_maze.get_position()):.2f}")
    
    print("\n" + "=" * 70)
    print("✅ TESTS MINIMAX TERMINÉS")
    print("=" * 70)


def test_minimax_mixed_strategies():
    """Test MINIMAX avec STRATÉGIES MIXTES"""
    print("\n" + "="*70)
    print("TEST: MINIMAX AVEC STRATÉGIES MIXTES")
    print("="*70)
    
    from backend.game_theory.minimax import Minimax
    from backend.game_theory.strategies import DroneStrategies, EnvironmentStrategies, DroneAction
    from backend.core.drone import Drone
    
    # Créer environnement
    env = Environment(width=20, height=20, start_pos=(2, 2), goal_pos=(18, 18))
    obstacles = [
        (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
        (10, 8), (11, 8), (12, 8), (13, 8),
        (15, 5), (15, 6), (15, 7)
    ]
    env.add_obstacles(obstacles)
    
    # Créer drone
    drone = Drone(environment=env, battery_capacity=100)
    
    # Créer payoff function
    payoff_func = PayoffFunction(w1=0.7, w2=0.05, w3=0.15, w4=0.1, stay_penalty_factor=0.4)
    minimax_solver = Minimax(payoff=payoff_func)
    
    # Définir les stratégies mixtes à tester
    drone_mixed_strategies = [
        ('Uniform', DroneStrategies.create_uniform_mixed_strategy()),
        ('Cautious', DroneStrategies.create_cautious_strategy()),
        ('Aggressive', DroneStrategies.create_aggressive_strategy()),
        ('Balanced', DroneStrategies.create_balanced_strategy())
    ]
    
    env_mixed_strategies = [
        ('Uniform', EnvironmentStrategies.create_uniform_mixed_strategy()),
        ('Typical', EnvironmentStrategies.create_typical_conditions()),
        ('Adversarial', EnvironmentStrategies.create_adversarial_conditions()),
        ('Favorable', EnvironmentStrategies.create_favorable_conditions())
    ]
    
    # État initial
    state_params = {
        'current_pos': drone.get_position(),
        'goal_pos': env.goal_pos,
        'initial_distance': env.distance_to_goal(env.start_pos),
        'battery_used': 0,
        'total_battery': drone.battery_capacity,
        'distance_to_nearest_obstacle': env.get_nearest_obstacle_distance(drone.get_position()),
        'explored_cells': len(drone.explored_cells),
        'total_cells': env.width * env.height,
        'collision': False,
        'environment': env
    }
    
    print(f"\n🗺️  Environnement: {env}")
    print(f"🎯 Objectif: {env.goal_pos}")
    print(f"📍 Position initiale: {drone.get_position()}")
    print(f"🔋 Batterie: {drone.get_battery_level()} unités")
    
    # Résoudre avec stratégies mixtes
    best_strategy_name, best_strategy, best_payoff = minimax_solver.solve_mixed_strategy(
        drone_mixed_strategies,
        env_mixed_strategies,
        state_params,
        verbose=True
    )
    
    print(f"\n{'='*70}")
    print("RÉSULTAT FINAL")
    print(f"{'='*70}")
    print(f"✅ Meilleure stratégie: {best_strategy_name}")
    print(f"📊 Payoff garanti: {best_payoff:.3f}")
    print(f"📋 Distribution de probabilités:")
    for action, prob in zip(best_strategy.strategies, best_strategy.probabilities):
        if prob > 0:
            print(f"   {action.value:15s}: {prob*100:5.1f}%")
    
    # Test: Simuler quelques étapes avec la meilleure stratégie
    print(f"\n{'='*70}")
    print("SIMULATION AVEC LA MEILLEURE STRATÉGIE MIXTE")
    print(f"{'='*70}")
    
    for step in range(100):
        current_pos = drone.get_position()
        
        # Échantillonner une action de la stratégie mixte
        action = best_strategy.sample()
        
        # Vérifier si l'action est valide
        valid_actions = drone.get_valid_actions()
        if action not in valid_actions:
            # Si l'action échantillonnée n'est pas valide, en choisir une valide
            action = valid_actions[0] if valid_actions else None
        
        if action is None:
            print("❌ Aucune action valide!")
            break
        
        # Exécuter
        success = drone.move(action)
        
        distance = env.distance_to_goal(drone.get_position())
        print(f"Étape {step+1:2d}: {action.value:15s} → {drone.get_position()} | "
              f"Dist: {distance:5.2f} | Batt: {drone.get_battery_percentage():5.1f}%")
        
        if env.is_goal_reached(drone.get_position()):
            print(f"\n🎯 Objectif atteint!")
            break
    
    # Test: Simuler avec réévaluation à chaque étape
    print(f"\n{'='*70}")
    print("SIMULATION INTELLIGENTE (Réévaluation continue)")
    print(f"{'='*70}")
    
    # Réinitialiser le drone
    drone2 = Drone(environment=env, battery_capacity=100)
    max_steps = 100
    
    for step in range(max_steps):
        current_pos = drone2.get_position()
        
        # IMPORTANT: Mettre à jour les state_params à chaque étape
        state_params_updated = {
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
        
        # Réévaluer les stratégies à chaque étape
        best_strategy_name_updated, best_strategy_updated, best_payoff_updated = minimax_solver.solve_mixed_strategy(
            drone_mixed_strategies,
            env_mixed_strategies,
            state_params_updated,
            verbose=False  # Pas besoin d'afficher à chaque fois
        )
        
        # Échantillonner depuis la NOUVELLE meilleure stratégie
        action = best_strategy_updated.sample()
        
        # Vérifier validité
        valid_actions = drone2.get_valid_actions()
        
        # Vérifier si on a des actions valides disponibles
        if not valid_actions:
            print(f"\n Batterie épuisée à l'étape {step+1}!")
            print(f"Position finale: {drone2.get_position()}")
            print(f"Distance finale: {env.distance_to_goal(drone2.get_position()):.2f}")
            print(f"Batterie restante: {drone2.get_battery_percentage():.1f}%")
            break
        
        if action not in valid_actions:
            # Choisir l'action avec la plus haute probabilité parmi les valides
            best_prob = 0
            best_valid_action = None
            for i, (strat_action, prob) in enumerate(zip(best_strategy_updated.strategies, best_strategy_updated.probabilities)):
                if strat_action in valid_actions and prob > best_prob:
                    best_prob = prob
                    best_valid_action = strat_action
            action = best_valid_action if best_valid_action else None
        
        if action is None:
            print(f"\n Aucune action valide disponible!")
            break
        
        # Exécuter
        success = drone2.move(action)
        
        distance = env.distance_to_goal(drone2.get_position())
        print(f"Étape {step+1:2d}: {action.value:15s} → {drone2.get_position()} | "
              f"Dist: {distance:5.2f} | Batt: {drone2.get_battery_percentage():5.1f}% | "
              f"Stratégie: {best_strategy_name_updated}")
        
        # Vérifier objectif
        if env.is_goal_reached(drone2.get_position()):
            print(f"\n🎯 Objectif atteint en {step+1} étapes!")
            print(f"📊 Batterie consommée: {drone2.battery_capacity - drone2.get_battery_level()}/{drone2.battery_capacity}")
            print(f"📊 Batterie restante: {drone2.get_battery_percentage():.1f}%")
            break
    else:
        print(f"\n⚠️ Limite de {max_steps} étapes atteinte sans atteindre l'objectif")
        print(f"Position finale: {drone2.get_position()}")
        print(f"Distance finale: {env.distance_to_goal(drone2.get_position()):.2f}")
    
    print("\n" + "="*70)
    print("✅ TEST MINIMAX STRATÉGIES MIXTES TERMINÉ")
    print("="*70)


# Dans main()
def main():
    print("\n")
    print("=" * 70)
    print(" DRONE VISUAL NAVIGATION PROJECT - PART 1 & 2")
    print("=" * 70)
    
    # Tests existants
    # test_environment()
    # test_strategies()
    # test_payoff_function()
    
    # Test Minimax avec stratégies pures
    # test_minimax()
    
    # NOUVEAU: Test Minimax avec stratégies mixtes
    test_minimax_mixed_strategies()
    
    print("\n" + "=" * 70)
    print(" TOUS LES TESTS TERMINÉS")
    print("=" * 70)
if __name__ == "__main__":
    main()