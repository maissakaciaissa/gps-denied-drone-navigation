"""
Main testing file for the Drone Visual Navigation Project
This file is in the root directory and imports from backend/core/
"""
from backend.core.env import Environment


def test_basic_environment():
    """Test basic environment creation and visualization"""
    print("=" * 50)
    print("TEST 1: Basic Environment Creation")
    print("=" * 50)
    
    # Create a 15x15 environment
    env = Environment(width=15, height=15)
    
    # Set start and goal
    env.set_start(1, 1)
    env.set_goal(13, 13)
    
    # Add obstacles to create a simple maze
    obstacles = [
        # Vertical wall
        (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
        # Horizontal wall
        (6, 7), (7, 7), (8, 7), (9, 7), (10, 7),
        # Another vertical wall
        (10, 8), (10, 9), (10, 10), (10, 11),
    ]
    env.add_obstacles(obstacles)
    
    # Visualize with drone at start position
    print("\nEnvironment with drone at START position:")
    env.visualize(drone_position=(1, 1))
    
    # Get environment state
    state = env.get_state()
    print(f"Environment Info:")
    print(f"  - Grid size: {state['width']}x{state['height']}")
    print(f"  - Total cells: {state['total_cells']}")
    print(f"  - Obstacles: {state['obstacle_count']}")
    print(f"  - Start: {state['start']}")
    print(f"  - Goal: {state['goal']}")


def test_validation_functions():
    """Test position validation functions"""
    print("\n" + "=" * 50)
    print("TEST 2: Position Validation")
    print("=" * 50)
    
    env = Environment(width=10, height=10)
    env.set_start(0, 0)
    env.set_goal(9, 9)
    env.add_obstacles([(3, 3), (3, 4), (5, 5)])
    
    test_positions = [
        (5, 5),   # On obstacle
        (3, 3),   # On obstacle
        (4, 4),   # Valid position
        (9, 9),   # Goal position
        (-1, 5),  # Out of bounds
        (10, 5),  # Out of bounds
        (5, 15),  # Out of bounds
    ]
    
    print("\nTesting various positions:")
    for x, y in test_positions:
        within = env.is_within_bounds(x, y)
        valid = env.is_valid_position(x, y)
        print(f"  Position ({x:2}, {y:2}): Within bounds={within}, Valid={valid}")


def test_distance_calculations():
    """Test distance calculation functions"""
    print("\n" + "=" * 50)
    print("TEST 3: Distance Calculations")
    print("=" * 50)
    
    env = Environment(width=20, height=20)
    env.set_start(2, 2)
    env.set_goal(18, 18)
    env.add_obstacles([
        (10, 10), (11, 10), (12, 10),
        (10, 11), (11, 11), (12, 11),
    ])
    
    test_positions = [
        (2, 2),   # Start
        (10, 10), # Middle
        (18, 18), # Goal
        (5, 5),   # Random position
    ]
    
    print(f"\nGoal is at: {env.goal}")
    print("\nDistance to goal from various positions:")
    for x, y in test_positions:
        dist = env.distance_to_goal(x, y)
        print(f"  From ({x:2}, {y:2}): {dist:.1f} cells")
    
    print("\nNearest obstacle from various positions:")
    for x, y in test_positions:
        nearest = env.get_nearest_obstacle(x, y)
        if nearest:
            obs_pos, obs_dist = nearest
            print(f"  From ({x:2}, {y:2}): Obstacle at {obs_pos}, distance={obs_dist:.1f}")


def test_drone_movement_simulation():
    """Simulate a simple drone path"""
    print("\n" + "=" * 50)
    print("TEST 4: Simulating Drone Movement")
    print("=" * 50)
    
    env = Environment(width=12, height=12)
    env.set_start(1, 1)
    env.set_goal(10, 10)
    
    # Create an obstacle course
    obstacles = [
        (4, 1), (4, 2), (4, 3), (4, 4),
        (7, 5), (7, 6), (7, 7), (7, 8),
    ]
    env.add_obstacles(obstacles)
    
    # Simulate drone moving along a path
    drone_path = [
        (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (3, 4),
        (3, 5), (4, 5), (5, 5), (6, 5), (6, 6), (6, 7),
        (6, 8), (6, 9), (7, 9), (8, 9), (9, 9), (10, 10)
    ]
    
    print("\nDrone path visualization:")
    for i, pos in enumerate(drone_path):
        print(f"\nStep {i}: Drone at {pos}")
        env.visualize(drone_position=pos)
        
        # Show distance to goal
        dist = env.distance_to_goal(pos[0], pos[1])
        print(f"Distance to goal: {dist:.1f} cells")
        
        # Show nearest obstacle
        nearest = env.get_nearest_obstacle(pos[0], pos[1])
        if nearest:
            obs_pos, obs_dist = nearest
            print(f"Nearest obstacle: {obs_pos} at distance {obs_dist:.1f}")
        
        # Only show first 3 and last position to avoid too much output
        if i >= 3 and i < len(drone_path) - 1:
            print("... (skipping intermediate steps) ...")
            continue


def test_custom_scenario():
    """Create your own custom test scenario"""
    print("\n" + "=" * 50)
    print("TEST 5: Custom Scenario")
    print("=" * 50)
    
    # TODO: You can customize this for your own testing
    env = Environment(width=20, height=20)
    env.set_start(1, 1)
    env.set_goal(18, 18)
    
    # Add your own obstacles here
    obstacles = [
        # Add obstacle coordinates as (x, y) tuples
    ]
    env.add_obstacles(obstacles)
    
    print("\nYour custom environment:")
    env.visualize(drone_position=(1, 1))


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 48 + "╗")
    print("║  DRONE VISUAL NAVIGATION - ENVIRONMENT TESTS  ║")
    print("╚" + "=" * 48 + "╝")
    
    # Run all tests
    test_basic_environment()
    test_validation_functions()
    test_distance_calculations()
    
    # Uncomment to see full movement simulation (lots of output!)
    # test_drone_movement_simulation()
    
    test_custom_scenario()
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()