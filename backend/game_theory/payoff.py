import numpy as np
from typing import Dict, Tuple, List, Optional
from backend.game_theory.strategies import DroneAction, EnvironmentCondition
from backend.core.environment import Environment


class PayoffFunction:
    """
    Calculates payoffs for drone actions considering multiple objectives.
    
    Payoff formula:
    u_drone(s_drone, s_env) = w1·mission_success - w2·energy_consumed 
                             - w3·collision_risk + w4·map_quality
    """
    
    def __init__(self, w1: float = 0.4, w2: float = 0.2, w3: float = 0.3, w4: float = 0.1,
                 collision_severity: str = "terminal", collision_penalty: float = 0.5):
        """
        Initialize payoff function with weights.
        
        Args:
            w1: Weight for mission_success (default 0.4)
            w2: Weight for energy_consumed (default 0.2)
            w3: Weight for collision_risk (default 0.3)
            w4: Weight for map_quality (default 0.1)
            collision_severity: "terminal" (mission ends, score=0) or "penalty" (deduct points, continue)
            collision_penalty: Penalty amount for collision in penalty mode (default 0.5)
            
        Constraint: w1 + w2 + w3 + w4 must equal 1.0
        """
        if not np.isclose(w1 + w2 + w3 + w4, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {w1 + w2 + w3 + w4}")
        
        if collision_severity not in ["terminal", "penalty"]:
            raise ValueError(f"collision_severity must be 'terminal' or 'penalty', got '{collision_severity}'")
        
        self.w1 = w1  # mission_success
        self.w2 = w2  # energy_consumed
        self.w3 = w3  # collision_risk
        self.w4 = w4  # map_quality
        self.collision_severity = collision_severity
        self.collision_penalty = collision_penalty
        
    def calculate_mission_success(self, current_pos: Tuple[int, int], 
                                  goal_pos: Tuple[int, int], 
                                  initial_distance: float,
                                  collision: bool = False,
                                  environment: Environment = None) -> float:
        """
        Calculate mission success component.
        
        Args:
            current_pos: Current drone position (x, y)
            goal_pos: Goal position (x, y)
            initial_distance: Initial distance from start to goal
            collision: Whether a collision occurred
            environment: Environment instance for distance calculations
            
        Returns:
            Mission success score:
            - 1.0 if reached goal
            - distance_to_goal / initial_distance (partial credit)
            - Terminal mode: 0 if collision (mission ends)
            - Penalty mode: progress - penalty if collision (mission continues)
        """
        # Check if goal reached
        if current_pos == goal_pos:
            return 1.0
        
        # Calculate current distance to goal
        current_distance = environment.distance_to_goal(current_pos)
        
        # Calculate progress
        progress = 0.0
        if initial_distance > 0:
            # Higher score when closer to goal
            progress = 1.0 - (current_distance / initial_distance)
            progress = max(0.0, progress)
        
        # Handle collision based on severity mode
        if collision:
            if self.collision_severity == "terminal":
                # Terminal mode: mission over, zero score
                return 0.0
            else:  # penalty mode
                # Penalty mode: deduct penalty but continue mission
                return max(0.0, progress - self.collision_penalty)
        
        return progress
    
    def calculate_energy_consumed(self, battery_used: float, total_battery: float) -> float:
        """
        Calculate energy consumption component.
        
        Args:
            battery_used: Amount of battery used
            total_battery: Total battery capacity
            
        Returns:
            Energy consumption ratio (0 to 1)
        """
        if total_battery <= 0:
            return 1.0
        
        return battery_used / total_battery
    
    def calculate_collision_risk(self, distance_to_nearest_obstacle: float) -> float:
        """
        Calculate collision risk component.
        
        Args:
            distance_to_nearest_obstacle: Distance to nearest obstacle
            
        Returns:
            Collision risk score (higher when close to obstacles)
        """
        if distance_to_nearest_obstacle is None or distance_to_nearest_obstacle <= 0:
            return 1.0  # Maximum risk if no obstacle info or touching obstacle
        
        # Risk increases as 1/distance
        # Use a small epsilon to prevent division by zero
        epsilon = 0.1
        risk = 1.0 / (distance_to_nearest_obstacle + epsilon)
        
        # Normalize to [0, 1] range (assuming max reasonable risk at distance 1)
        normalized_risk = min(1.0, risk / 10.0)
        
        return normalized_risk
    
    def calculate_map_quality(self, explored_cells: int, total_cells: int) -> float:
        """
        Calculate map quality/exploration component.
        
        Args:
            explored_cells: Number of cells explored
            total_cells: Total number of cells in the environment
            
        Returns:
            Map quality score (0 to 1)
        """
        if total_cells <= 0:
            return 0.0
        
        # Higher score for more exploration
        return explored_cells / total_cells
    
    def compute_payoff(self, 
                      drone_action: DroneAction,
                      env_condition: EnvironmentCondition,
                      current_pos: Tuple[int, int],
                      goal_pos: Tuple[int, int],
                      initial_distance: float,
                      battery_used: float,
                      total_battery: float,
                      distance_to_nearest_obstacle: float,
                      explored_cells: int,
                      total_cells: int,
                      collision: bool = False,
                      environment: Environment = None) -> float:
        """
        Compute the overall payoff for a drone action and environment condition.
        
        Args:
            drone_action: Action taken by the drone
            env_condition: Environmental condition
            current_pos: Current position
            goal_pos: Goal position
            initial_distance: Initial distance to goal
            battery_used: Battery consumed
            total_battery: Total battery capacity
            distance_to_nearest_obstacle: Distance to nearest obstacle
            explored_cells: Number of explored cells
            total_cells: Total cells in environment
            collision: Whether collision occurred
            environment: Environment instance for distance calculations
            
        Returns:
            Overall payoff score
        """
        # Simulate the result of taking this action under this condition
        x, y = current_pos
        new_pos = current_pos
        action_energy = 0
        action_exploration = 0
        action_collision = False
        
        # Map action to position change and energy cost
        if drone_action == DroneAction.MOVE_UP:
            new_pos = (x, y + 1)
            action_energy = 5
            action_exploration = 1
        elif drone_action == DroneAction.MOVE_DOWN:
            new_pos = (x, y - 1)
            action_energy = 5
            action_exploration = 1
        elif drone_action == DroneAction.MOVE_LEFT:
            new_pos = (x - 1, y)
            action_energy = 5
            action_exploration = 1
        elif drone_action == DroneAction.MOVE_RIGHT:
            new_pos = (x + 1, y)
            action_energy = 5
            action_exploration = 1
        elif drone_action == DroneAction.STAY:
            new_pos = current_pos
            action_energy = 1
            action_exploration = 0
        elif drone_action == DroneAction.ROTATE:
            new_pos = current_pos
            action_energy = 2
            action_exploration = 0
        
        # Environmental condition affects collision and energy
        collision_multiplier = 1.0
        energy_multiplier = 1.0
        
        if env_condition == EnvironmentCondition.CLEAR_PATH:
            collision_multiplier = 0.5  # Lower risk
            energy_multiplier = 1.0
        elif env_condition == EnvironmentCondition.OBSTACLE_AHEAD:
            collision_multiplier = 3.0  # Much higher risk
            energy_multiplier = 1.2  # More energy to navigate
        elif env_condition == EnvironmentCondition.LOW_VISIBILITY:
            collision_multiplier = 1.5
            energy_multiplier = 1.3  # Slower, more careful
        elif env_condition == EnvironmentCondition.SENSOR_NOISE:
            collision_multiplier = 1.8
            energy_multiplier = 1.1
        elif env_condition == EnvironmentCondition.LIGHTING_CHANGE:
            collision_multiplier = 1.2
            energy_multiplier = 1.0
        
        # Check if new position causes collision
        if environment is not None:
            if not environment.is_valid_position(new_pos):
                action_collision = True
                new_pos = current_pos  # Stay at current position if collision
        
        # Calculate new state values
        new_battery_used = battery_used + (action_energy * energy_multiplier)
        new_explored = explored_cells + action_exploration
        new_obstacle_dist = environment.get_nearest_obstacle_distance(new_pos) if environment else distance_to_nearest_obstacle
        
        # Apply collision multiplier to risk
        effective_collision_risk = self.calculate_collision_risk(new_obstacle_dist) * collision_multiplier
        
        # Calculate components with simulated new state
        mission_success = self.calculate_mission_success(
            new_pos, goal_pos, initial_distance, action_collision, environment
        )
        
        energy_consumed = self.calculate_energy_consumed(
            new_battery_used, total_battery
        )
        
        map_quality = self.calculate_map_quality(
            new_explored, total_cells
        )
        
        # Apply weights and compute final payoff
        payoff = (self.w1 * mission_success - 
                 self.w2 * energy_consumed - 
                 self.w3 * effective_collision_risk + 
                 self.w4 * map_quality)
        
        return payoff
    
    def generate_payoff_matrix(self,
                              drone_actions: List[DroneAction],
                              env_conditions: List[EnvironmentCondition],
                              state_params: Dict) -> np.ndarray:
        """
        Generate a complete payoff matrix for all action/condition combinations.
        
        Args:
            drone_actions: List of possible drone actions
            env_conditions: List of possible environment conditions
            state_params: Dictionary containing state information for payoff calculation
            
        Returns:
            2D numpy array where [i,j] is payoff for drone_actions[i] and env_conditions[j]
        """
        n_actions = len(drone_actions)
        n_conditions = len(env_conditions)
        
        payoff_matrix = np.zeros((n_actions, n_conditions))
        
        for i, action in enumerate(drone_actions):
            for j, condition in enumerate(env_conditions):
                payoff = self.compute_payoff(
                    action, condition,
                    state_params.get('current_pos'),
                    state_params.get('goal_pos'),
                    state_params.get('initial_distance'),
                    state_params.get('battery_used'),
                    state_params.get('total_battery'),
                    state_params.get('distance_to_nearest_obstacle'),
                    state_params.get('explored_cells'),
                    state_params.get('total_cells'),
                    state_params.get('collision', False),
                    state_params.get('environment', None)
                )
                payoff_matrix[i, j] = payoff
        
        return payoff_matrix
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get the current weight configuration.
        
        Returns:
            Dictionary of weight names and values
        """
        return {
            'mission_success': self.w1,
            'energy_consumed': self.w2,
            'collision_risk': self.w3,
            'map_quality': self.w4
        }
    
    def __repr__(self) -> str:
        """String representation of the payoff function."""
        return (f"PayoffFunction(mission={self.w1:.2f}, energy={self.w2:.2f}, "
                f"collision={self.w3:.2f}, map={self.w4:.2f})")
