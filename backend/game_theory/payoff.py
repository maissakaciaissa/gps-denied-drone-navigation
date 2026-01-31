import numpy as np
from typing import Dict, Tuple, List, Optional
from backend.game_theory.strategies import DroneAction, EnvironmentCondition
from backend.core.environment import Environment


class PayoffFunction:
    """
    Calculates payoffs for drone actions considering multiple objectives.
    Returns (drone_payoff, env_payoff) tuples for game theory analysis.
    
    Payoff formula:
    u_drone(s_drone, s_env) = w1·mission_success - w2·energy_consumed 
                             - w3·collision_risk + w4·map_quality
    u_env(s_drone, s_env) = opposite goals (environment prefers obstacles, delays)
    """
    
    def __init__(self, w1: float = 0.4, w2: float = 0.2, w3: float = 0.3, w4: float = 0.1,
                 collision_severity: str = "terminal", collision_penalty: float = 0.5,
                 payoff_scale: float = 10.0, penalize_stay: bool = True, stay_penalty_factor: float = 0.6):
        """
        Initialize payoff function with weights.
        
        Args:
            w1: Weight for mission_success (default 0.4)
            w2: Weight for energy_consumed (default 0.2)
            w3: Weight for collision_risk (default 0.3)
            w4: Weight for map_quality (default 0.1)
            collision_severity: "terminal" (mission ends, score=0) or "penalty" (deduct points, continue)
            collision_penalty: Penalty amount for collision in penalty mode (default 0.5)
            payoff_scale: Multiplier to make payoffs more dramatic (default 10.0)
            
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
        self.payoff_scale = payoff_scale  # Scale payoffs for more dramatic differences
        self.penalize_stay = penalize_stay
        self.stay_penalty_factor = stay_penalty_factor
        
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
                      environment: Environment = None) -> Tuple[float, float]:
        """
        Compute the overall payoff for a drone action and environment condition.
        Returns (drone_payoff, env_payoff) tuple.
        
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
            Tuple of (drone_payoff, environment_payoff)
        """
        # Simulate the result of taking this action under this condition
        x, y = current_pos
        new_pos = current_pos
        action_energy = 0
        action_exploration = 0
        action_collision = False
        
        # Map action to position change and energy cost (ACTION-SPECIFIC)
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
            action_energy = 1  # STAY costs much less energy
            action_exploration = 0
        elif drone_action == DroneAction.ROTATE:
            new_pos = current_pos
            action_energy = 2  # ROTATE costs less than moving
            action_exploration = 0
        
        # Check if new position causes collision
        if environment is not None:
            if not environment.is_valid_position(new_pos):
                action_collision = True
                new_pos = current_pos  # Stay at current position if collision
        
        # Calculate new state values FIRST (needed for collision multiplier logic)
        new_battery_used = battery_used + action_energy  # Will multiply by energy_multiplier later
        new_explored = explored_cells + action_exploration
        new_obstacle_dist = environment.get_nearest_obstacle_distance(new_pos) if environment else distance_to_nearest_obstacle
        
        # Environmental condition affects collision and energy (CONDITION-SPECIFIC)
        collision_multiplier = 1.0
        energy_multiplier = 1.0
        
        if env_condition == EnvironmentCondition.CLEAR_PATH:
            collision_multiplier = 0.2  # Very low risk
            energy_multiplier = 1.0
        elif env_condition == EnvironmentCondition.OBSTACLE_AHEAD:
            # Only apply high penalty if actually moving into danger
            # Check if this move would reduce obstacle distance (moving toward obstacle)
            if drone_action in [DroneAction.MOVE_UP, DroneAction.MOVE_DOWN, 
                               DroneAction.MOVE_LEFT, DroneAction.MOVE_RIGHT]:
                # Check if new position is closer to obstacle or is invalid
                if action_collision or (environment and new_obstacle_dist < distance_to_nearest_obstacle):
                    collision_multiplier = 5.0  # DRAMATIC: Actually moving toward/into obstacle
                else:
                    collision_multiplier = 1.5  # Moderate: obstacle exists but not moving toward it
            else:
                collision_multiplier = 1.0  # STAY/ROTATE is safe even with obstacle
            energy_multiplier = 1.5  # More energy to navigate
        elif env_condition == EnvironmentCondition.LOW_VISIBILITY:
            collision_multiplier = 2.0
            energy_multiplier = 1.3  # Slower, more careful
        elif env_condition == EnvironmentCondition.SENSOR_NOISE:
            collision_multiplier = 2.5
            energy_multiplier = 1.1
        elif env_condition == EnvironmentCondition.LIGHTING_CHANGE:
            collision_multiplier = 1.5
            energy_multiplier = 1.0
        
        # Apply energy multiplier to battery
        new_battery_used = battery_used + (action_energy * energy_multiplier)
        
        # Calculate if moving toward or away from goal (DIRECTIONAL AWARENESS)
        current_dist_to_goal = environment.distance_to_goal(current_pos) if environment else np.linalg.norm(np.array(goal_pos) - np.array(current_pos))
        new_dist_to_goal = environment.distance_to_goal(new_pos) if environment else np.linalg.norm(np.array(goal_pos) - np.array(new_pos))
        
        moving_toward_goal = new_dist_to_goal < current_dist_to_goal
        moving_away_from_goal = new_dist_to_goal > current_dist_to_goal
        
        # Apply collision multiplier to risk
        effective_collision_risk = self.calculate_collision_risk(new_obstacle_dist) * collision_multiplier
        
        # Calculate components with simulated new state
        mission_success = self.calculate_mission_success(
            new_pos, goal_pos, initial_distance, action_collision, environment
        )
        
        # Calculer la distance normalisée AVANT tous les blocs (utilisée dans plusieurs conditions)
        distance_ratio = new_dist_to_goal / initial_distance if initial_distance > 0 else 0
        
        # Pénaliser FORTEMENT l'immobilité quand on est loin de l'objectif
        if self.penalize_stay and drone_action in [DroneAction.STAY, DroneAction.ROTATE]:
            if new_pos != goal_pos:
                # Pénalité DRAMATIQUE : échelle de 0 à 3.0
                stay_penalty = distance_ratio * self.stay_penalty_factor * 3.0  # Multiplié par 3!
                mission_success -= stay_penalty
        
        # BONUS MASSIF pour se rapprocher de l'objectif
        if moving_toward_goal and drone_action in [DroneAction.MOVE_UP, DroneAction.MOVE_DOWN,
                                                    DroneAction.MOVE_LEFT, DroneAction.MOVE_RIGHT]:
            # Bonus proportionnel au progrès
            progress = (current_dist_to_goal - new_dist_to_goal) / initial_distance
            # AUGMENTATION MASSIVE: 0.5 → 10.0 (x20)
            # Plus on est proche, plus le bonus est élevé (facteur d'urgence)
            urgency_factor = 1.0 + (1.0 - distance_ratio)  # 1.0 (loin) à 2.0 (proche)
            approach_bonus = progress * 10.0 * urgency_factor  # Bonus jusqu'à 20.0!
            mission_success += approach_bonus
        
        # Pénaliser MOINS s'éloigner (permet exploration)
        if moving_away_from_goal and drone_action in [DroneAction.MOVE_UP, DroneAction.MOVE_DOWN,
                                                       DroneAction.MOVE_LEFT, DroneAction.MOVE_RIGHT]:
            # Réduction: 3.0 → 0.5 (x6 moins sévère)
            mission_success -= 0.5  # Pénalité modérée when moving away from goal         
        
        if self.penalize_stay and drone_action in [DroneAction.STAY, DroneAction.ROTATE]:
            if new_pos != goal_pos:
                # Calculer la distance normalisée (0 = objectif, 1 = très loin)
                distance_ratio = new_dist_to_goal / initial_distance if initial_distance > 0 else 0
                # Pénalité progressive : plus on est loin, plus c'est pénalisé
                # À l'objectif (ratio=0) : pas de pénalité
                # Très loin (ratio=1) : pénalité maximale (stay_penalty_factor)    
                stay_penalty = distance_ratio * self.stay_penalty_factor
                mission_success -= stay_penalty 

        # BONUS pour se rapprocher de l'objectif
        if moving_toward_goal and drone_action in [DroneAction.MOVE_UP, DroneAction.MOVE_DOWN,
                                                    DroneAction.MOVE_LEFT, DroneAction.MOVE_RIGHT]:
            # Bonus proportionnel au progrès
            progress = (current_dist_to_goal - new_dist_to_goal) / initial_distance
            approach_bonus = progress * 0.5  # Bonus jusqu'à 0.5
            mission_success += approach_bonus           

        energy_consumed = self.calculate_energy_consumed(
            new_battery_used, total_battery
        )
        
        map_quality = self.calculate_map_quality(
            new_explored, total_cells
        )
        
        # Apply weights and compute drone payoff
        drone_payoff = (self.w1 * mission_success - 
                       self.w2 * energy_consumed - 
                       self.w3 * effective_collision_risk + 
                       self.w4 * map_quality)
        
        # Scale for more dramatic differences
        drone_payoff *= self.payoff_scale
        
        # Calculate environment payoff (game-theoretic interpretation)
        # Environment "succeeds" when it creates challenges that the drone handles poorly
        # Success = high collision risk + obstacles that cause energy waste + preventing goal progress
        # But environment gets lower payoff if drone still succeeds despite challenges
        env_success_rate = effective_collision_risk * 0.5 + energy_consumed * 0.3
        env_challenge_effectiveness = 1.0 - mission_success  # How much did we prevent mission?
        
        env_payoff = (self.w1 * env_challenge_effectiveness +  # Prevented mission progress
                     self.w2 * energy_consumed +                # Caused energy drain
                     self.w3 * effective_collision_risk -       # Created collision risk
                     self.w4 * map_quality * 0.5)               # Reduced exploration (lower weight)
        
        # Scale environment payoff
        env_payoff *= self.payoff_scale
        
        return (drone_payoff, env_payoff)
    
    def generate_payoff_matrix(self,
                              drone_actions: List[DroneAction],
                              env_conditions: List[EnvironmentCondition],
                              state_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete payoff matrices for all action/condition combinations.
        
        Args:
            drone_actions: List of possible drone actions
            env_conditions: List of possible environment conditions
            state_params: Dictionary containing state information for payoff calculation
            
        Returns:
            Tuple of (drone_payoff_matrix, env_payoff_matrix)
            Each is a 2D numpy array where [i,j] is payoff for drone_actions[i] and env_conditions[j]
        """
        n_actions = len(drone_actions)
        n_conditions = len(env_conditions)
        
        drone_matrix = np.zeros((n_actions, n_conditions))
        env_matrix = np.zeros((n_actions, n_conditions))
        
        for i, action in enumerate(drone_actions):
            for j, condition in enumerate(env_conditions):
                drone_payoff, env_payoff = self.compute_payoff(
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
                drone_matrix[i, j] = drone_payoff
                env_matrix[i, j] = env_payoff
        
        return drone_matrix, env_matrix
    
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