"""
PART 1: Environment Representation
This file contains the Environment class that represents the world where the drone operates.
It is a game map aka grid with starting point , obstacales and destination.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class Environment:
    """
    Represents the grid-based environment where the drone operates.
    """
    
    def __init__(self, width: int, height: int, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]):
        """
        Initialize the environment with a grid system.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            start_pos: Starting position (x, y)
            goal_pos: Goal position (x, y)
        """
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles: List[Tuple[int, int]] = []
        self.grid = np.zeros((height, width), dtype=int)  # 0 = free, 1 = obstacle
        
    def add_obstacle(self, position: Tuple[int, int]) -> None:
        """
        Add an obstacle at the specified position.
        
        Args:
            position: (x, y) coordinates of the obstacle
        """
        x, y = position
        if self.is_within_bounds(position):
            self.obstacles.append(position)
            self.grid[y, x] = 1
            
    def add_obstacles(self, positions: List[Tuple[int, int]]) -> None:
        """
        Add multiple obstacles at once.
        
        Args:
            positions: List of (x, y) coordinates
        """
        for pos in positions:
            self.add_obstacle(pos)
            
    def is_within_bounds(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is within the grid boundaries.
        
        Args:
            position: (x, y) coordinates to check
            
        Returns:
            True if position is valid, False otherwise
        """
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_obstacle(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position contains an obstacle.
        
        Args:
            position: (x, y) coordinates to check
            
        Returns:
            True if position has an obstacle, False otherwise
        """
        if not self.is_within_bounds(position):
            return True  # Out of bounds treated as obstacle
        x, y = position
        return self.grid[y, x] == 1
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is valid (within bounds and no obstacle).
        
        Args:
            position: (x, y) coordinates to check
            
        Returns:
            True if position is safe to move to, False otherwise
        """
        return self.is_within_bounds(position) and not self.is_obstacle(position)
    
    def distance_to_goal(self, position: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance from a position to the goal.
        Diagonal movement: Measures straight-line distance "as the crow flies"
        Best for: Calculating actual shortest path when drone can move diagonally
        
        Args:
            position: (x, y) coordinates
            
        Returns:
            Euclidean distance to goal
        """
        x1, y1 = position
        x2, y2 = self.goal_pos
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    
    def manhattan_distance_to_goal(self, position: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance from a position to the goal.
        Grid movement: Measures distance when moving only horizontally/vertically (no diagonals)
        Best for: Pathfinding when drone can only move up/down/left/right
        
        Args:
            position: (x, y) coordinates
            
        Returns:
            Manhattan distance to goal
        """
        x1, y1 = position
        x2, y2 = self.goal_pos
        return abs(x2 - x1) + abs(y2 - y1)
    
    def get_nearest_obstacle_distance(self, position: Tuple[int, int]) -> Optional[float]:
        """
        Find the distance to the nearest obstacle from a given position.
        
        Args:
            position: (x, y) coordinates
            
        Returns:
            Distance to nearest obstacle, or None if no obstacles
        """
        if not self.obstacles:
            return None
            
        min_distance = float('inf')
        x1, y1 = position
        
        for obs_x, obs_y in self.obstacles:
            distance = np.sqrt((obs_x - x1)**2 + (obs_y - y1)**2)
            min_distance = min(min_distance, distance)
            
        return min_distance if min_distance != float('inf') else None
    
    def get_obstacles_in_direction(self, position: Tuple[int, int], direction: str, max_range: int = 5) -> Optional[Tuple[int, int]]:
        """
        Get the nearest obstacle in a specific direction.
        
        Args:
            position: Starting (x, y) coordinates
            direction: 'up', 'down', 'left', 'right'
            max_range: Maximum distance to check
            
        Returns:
            Position of nearest obstacle in that direction, or None
        """
        x, y = position
        
        for i in range(1, max_range + 1):
            if direction == 'up':
                check_pos = (x, y + i)
            elif direction == 'down':
                check_pos = (x, y - i)
            elif direction == 'left':
                check_pos = (x - i, y)
            elif direction == 'right':
                check_pos = (x + i, y)
            else:
                return None
                
            if not self.is_within_bounds(check_pos):
                return check_pos
            if self.is_obstacle(check_pos):
                return check_pos
                
        return None
    
    def get_state(self) -> Dict:
        """
        Get the current state of the environment .
        
        Returns:
            Dictionary containing environment state
        """
        return {
            'width': self.width,
            'height': self.height,
            'start_pos': self.start_pos,
            'goal_pos': self.goal_pos,
            'obstacles': self.obstacles.copy(),
            'num_obstacles': len(self.obstacles)
        }
    
    def is_goal_reached(self, position: Tuple[int, int]) -> bool:
        """
        Check if the drone has reached the goal.
        
        Args:
            position: Current (x, y) coordinates
            
        Returns:
            True if at goal, False otherwise
        """
        return position == self.goal_pos
    
    def __repr__(self) -> str:
        """String representation of the environment."""
        return f"Environment({self.width}x{self.height}, obstacles={len(self.obstacles)}, goal={self.goal_pos})"
