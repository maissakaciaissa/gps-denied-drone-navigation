"""
PART 3 : Drone Sensor System
This file contains the DroneSensor class that simulates the drone's vision and sensing capabilities.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from backend.core.environment import Environment
from backend.game_theory.strategies import EnvironmentCondition


class DroneSensor:
    def __init__(self, detection_range: int = 5, initial_visibility: float = 1.0):
        """
        Initialize the drone sensor.
        
        Args:
            detection_range: How many grid cells away the drone can see (default: 5)
            initial_visibility: Visibility level from 0.0 to 1.0 (default: 1.0 = perfect)
                              0.0 = completely blind
                              0.5 = foggy/reduced visibility
                              1.0 = perfect clarity
        """
        if detection_range < 1:
            raise ValueError("Detection range must be at least 1")
        if not 0.0 <= initial_visibility <= 1.0:
            raise ValueError("Visibility must be between 0.0 and 1.0")
            
        self.detection_range = detection_range
        self.visibility = initial_visibility
        self.base_range = detection_range  # Store original range for reference
        
    def get_effective_range(self) -> int:
        """
        Calculate the effective detection range based on current visibility.
        
        Returns:
            Effective range in grid cells (reduced by poor visibility)
        """
        # Visibility affects how far we can actually see
        effective = int(self.detection_range * self.visibility)
        return max(1, effective)  # Always see at least 1 cell
    
    def scan_environment(self, environment: Environment, drone_position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Scan the environment and return all obstacles the drone can currently see.
        
        Args:
            environment: The Environment object to scan
            drone_position: Current (x, y) position of the drone
            
        Returns:
            List of (x, y) coordinates of visible obstacles
        """
        visible_obstacles = []
        effective_range = self.get_effective_range()
        drone_x, drone_y = drone_position
        
        # loop over all environment obstacles
        for obstacle_pos in environment.obstacles:
            obs_x, obs_y = obstacle_pos
            
            # Calculate distance from drone to this obstacle
            distance = np.sqrt((obs_x - drone_x)**2 + (obs_y - drone_y)**2)
            
            # If within the effective range, it's visible
            if distance <= effective_range:
                visible_obstacles.append(obstacle_pos)
        
        return visible_obstacles
    
    def detect_obstacles(self, environment: Environment, drone_position: Tuple[int, int]) -> Dict[str, Optional[int]]:
        """
        Check specific directions (up, down, left, right) and report obstacle distances.
        
        Args:
            environment: The Environment object to scan
            drone_position: Current (x, y) position of the drone
            
        Returns:
            Dictionary with directions as keys and distances as values:
            - distance (int): How many cells away the obstacle is
            - None: No obstacle in that direction within range
            
            Example: {'up': 3, 'down': None, 'left': 5, 'right': 1}
        """
        directions = ['up', 'down', 'left', 'right']
        obstacles_by_direction = {}
        effective_range = self.get_effective_range()
        drone_x, drone_y = drone_position
        
        for direction in directions:
            # Scan in each direction up to effective range
            obstacle_distance = None
            
            for distance in range(1, effective_range + 1):
                # Calculate position to check based on direction
                if direction == 'up':
                    check_pos = (drone_x, drone_y + distance)
                elif direction == 'down':
                    check_pos = (drone_x, drone_y - distance)
                elif direction == 'left':
                    check_pos = (drone_x - distance, drone_y)
                else:  # right
                    check_pos = (drone_x + distance, drone_y)
                
                # Check if out of bounds (treated as obstacle/wall)
                if not environment.is_within_bounds(check_pos):
                    obstacle_distance = distance
                    break
                
                # Check if there's an obstacle at this position
                if environment.is_obstacle(check_pos):
                    obstacle_distance = distance
                    break
            
            obstacles_by_direction[direction] = obstacle_distance
        
        return obstacles_by_direction
    
    def update_visibility(self, condition: EnvironmentCondition) -> None:
        """
        Update visibility based on environmental conditions.
        
        Args:
            condition: The current environmental condition affecting the sensor
        """
        if condition == EnvironmentCondition.CLEAR_PATH:
            # Perfect conditions - maximum visibility
            self.visibility = 1.0
            
        elif condition == EnvironmentCondition.LOW_VISIBILITY:
            # Fog/darkness - significantly reduced visibility
            self.visibility = 0.4
            
        elif condition == EnvironmentCondition.SENSOR_NOISE:
            # Interference - moderately reduced visibility
            self.visibility = 0.6
            
        elif condition == EnvironmentCondition.LIGHTING_CHANGE:
            # Sudden light change - temporarily reduced visibility
            self.visibility = 0.7
            
        elif condition == EnvironmentCondition.OBSTACLE_AHEAD:
            # Obstacle doesn't affect sensor visibility itself
            # (obstacle detection is handled separately)
            pass
        
        # Ensure visibility stays in valid range
        self.visibility = max(0.0, min(1.0, self.visibility))
    
    def get_observable_region(self, environment: Environment, drone_position: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """
        Get all grid coordinates the drone can currently observe.
        
        Args:
            environment: The Environment object
            drone_position: Current (x, y) position of the drone
            
        Returns:
            Set of (x, y) coordinates that are observable
        """
        observable_cells = set()
        effective_range = self.get_effective_range()
        drone_x, drone_y = drone_position
        
        # Check all cells within the effective range
        for x in range(drone_x - effective_range, drone_x + effective_range + 1):
            for y in range(drone_y - effective_range, drone_y + effective_range + 1):
                pos = (x, y)
                
                # Check if position is within bounds
                if not environment.is_within_bounds(pos):
                    continue
                
                # Calculate distance from drone
                distance = np.sqrt((x - drone_x)**2 + (y - drone_y)**2)
                
                # If within effective range, it's observable
                if distance <= effective_range:
                    observable_cells.add(pos)
        
        return observable_cells
    
    def sense_environment_condition(self, environment: Environment, drone_position: Tuple[int, int]) -> Dict[EnvironmentCondition, float]:
        """
        Determine the current environment condition based on sensor readings.
        Returns a probability distribution (mixed strategy) over possible conditions.
        
        The sensor detects obstacles and determines conditions based on:
        - Proximity to obstacles
        - Visibility and sensor noise
        - Environmental factors
        
        Args:
            environment: The Environment object to sense
            drone_position: Current (x, y) position of the drone
        
        Returns:
            Dictionary mapping EnvironmentCondition to probability (0.0 to 1.0)
            Probabilities sum to 1.0. Example: {CLEAR_PATH: 0.7, LOW_VISIBILITY: 0.3}
        """
        import random
        
        # Detect obstacles in all directions
        obstacles_by_direction = self.detect_obstacles(environment, drone_position)
        
        # Check if there's an obstacle immediately ahead (within 2 cells in any direction)
        close_obstacles = [dist for dist in obstacles_by_direction.values() if dist is not None and dist <= 2]
        
        if close_obstacles:
            # Obstacle is ahead
            return {
                EnvironmentCondition.OBSTACLE_AHEAD: 1.0,
            }
        
        # Get visible obstacles in sensor range
        visible_obstacles = self.scan_environment(environment, drone_position)
        
        # Calculate obstacle density in observable region
        observable_region = self.get_observable_region(environment, drone_position)
        obstacle_density = len(visible_obstacles) / max(len(observable_region), 1)
        
        if len(visible_obstacles) > 5:
            # Very dense obstacle environment - sensor struggles with complex scenes
            if self.visibility < 0.5:
                # Poor visibility + dense obstacles = extreme uncertainty
                return {
                    EnvironmentCondition.OBSTACLE_AHEAD: 0.40,
                    EnvironmentCondition.LOW_VISIBILITY: 0.30,
                    EnvironmentCondition.SENSOR_NOISE: 0.15,
                    EnvironmentCondition.CLEAR_PATH: 0.15
                }
            elif self.visibility < 0.8:
                # Moderate visibility + dense obstacles
                return {
                    EnvironmentCondition.OBSTACLE_AHEAD: 0.55,
                    EnvironmentCondition.LOW_VISIBILITY: 0.25,
                    EnvironmentCondition.SENSOR_NOISE: 0.10,
                    EnvironmentCondition.CLEAR_PATH: 0.10
                }
            else:
                # Good visibility but dense obstacles
                return {
                    EnvironmentCondition.OBSTACLE_AHEAD: 0.60,
                    EnvironmentCondition.CLEAR_PATH: 0.35,
                    EnvironmentCondition.LIGHTING_CHANGE: 0.05,
                }
        
        elif obstacle_density > 0.3 or len(visible_obstacles) > 2:
            # Moderate obstacle density
            if self.visibility < 0.5:
                # Poor visibility with moderate obstacles
                return {
                    EnvironmentCondition.LOW_VISIBILITY: 0.40,
                    EnvironmentCondition.OBSTACLE_AHEAD: 0.30,
                    EnvironmentCondition.CLEAR_PATH: 0.20,
                    EnvironmentCondition.SENSOR_NOISE: 0.10
                }
            elif self.visibility < 0.8:
                # Moderate visibility with moderate obstacles
                return {
                    EnvironmentCondition.OBSTACLE_AHEAD: 0.35,
                    EnvironmentCondition.CLEAR_PATH: 0.45,
                    EnvironmentCondition.LOW_VISIBILITY: 0.15,
                    EnvironmentCondition.SENSOR_NOISE: 0.05
                }
            else:
                # Good visibility with moderate obstacles
                return {
                    EnvironmentCondition.OBSTACLE_AHEAD: 0.45,
                    EnvironmentCondition.CLEAR_PATH: 0.40,
                    EnvironmentCondition.LIGHTING_CHANGE: 0.15
                }
        
        elif len(visible_obstacles) == 1:
            # Single obstacle - confidence depends on visibility
            if self.visibility < 0.6:
                # Poor visibility - uncertain if obstacle or sensor issue
                return {
                    EnvironmentCondition.LOW_VISIBILITY: 0.40,
                    EnvironmentCondition.OBSTACLE_AHEAD: 0.35,
                    EnvironmentCondition.SENSOR_NOISE: 0.25
                }
            else:
                # Good visibility - more confident about single obstacle
                return {
                    EnvironmentCondition.OBSTACLE_AHEAD: 0.65,
                    EnvironmentCondition.CLEAR_PATH: 0.30,
                    EnvironmentCondition.LIGHTING_CHANGE: 0.05
                }
        
        else:
            # No obstacles nearby
            if self.visibility < 0.5:
                # Can't see obstacles but poor visibility - uncertainty remains
                return {
                    EnvironmentCondition.LOW_VISIBILITY: 0.55,
                    EnvironmentCondition.CLEAR_PATH: 0.30,
                    EnvironmentCondition.SENSOR_NOISE: 0.15
                }
            elif self.visibility < 0.8:
                # Moderate visibility, no obstacles detected
                return {
                    EnvironmentCondition.CLEAR_PATH: 0.75,
                    EnvironmentCondition.LOW_VISIBILITY: 0.15,
                    EnvironmentCondition.SENSOR_NOISE: 0.10
                }
            else:
                # Good visibility, no obstacles - pure strategy (high confidence)
                return {
                    EnvironmentCondition.CLEAR_PATH: 1.0
                }
    
    def get_sensor_info(self) -> Dict:
        """
        Get current sensor state information.
        
        Returns:
            Dictionary with sensor status
        """
        return {
            'base_detection_range': self.base_range,
            'current_detection_range': self.detection_range,
            'visibility': self.visibility,
            'effective_range': self.get_effective_range(),
            'status': 'optimal' if self.visibility >= 0.8 else 'degraded' if self.visibility >= 0.5 else 'poor'
        }
    
    def reset_visibility(self) -> None:
        """
        Reset visibility to perfect conditions (1.0)
        """
        self.visibility = 1.0
    
    def __repr__(self) -> str:
        """String representation of the sensor"""
        effective = self.get_effective_range()
        status = self.get_sensor_info()['status']
        return f"DroneSensor(range={self.detection_range}, effective={effective}, visibility={self.visibility:.2f}, status={status})"
