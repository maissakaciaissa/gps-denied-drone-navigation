import numpy as np
from typing import List, Tuple, Optional

class Environment:
    """
    Represents the grid world where the drone operates.
    This is like a chessboard with obstacles, a start position, and a goal.
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize the environment grid.
        
        Args:
            width: Number of columns in the grid
            height: Number of rows in the grid
        """
        self.width = width
        self.height = height
        self.obstacles = []  # List of (x, y) positions that are blocked
        self.goal = None     # (x, y) position of the goal
        self.start = None    # (x, y) starting position
        
    def add_obstacle(self, x: int, y: int):
        """
        Add an obstacle at the specified position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if self.is_within_bounds(x, y) and (x, y) not in self.obstacles:
            self.obstacles.append((x, y))
            
    def add_obstacles(self, obstacle_list: List[Tuple[int, int]]):
        """
        Add multiple obstacles at once.
        
        Args:
            obstacle_list: List of (x, y) tuples
        """
        for x, y in obstacle_list:
            self.add_obstacle(x, y)
            
    def set_goal(self, x: int, y: int):
        """
        Set the goal position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if self.is_within_bounds(x, y):
            self.goal = (x, y)
        else:
            raise ValueError(f"Goal position ({x}, {y}) is out of bounds")
            
    def set_start(self, x: int, y: int):
        """
        Set the starting position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if self.is_within_bounds(x, y):
            self.start = (x, y)
        else:
            raise ValueError(f"Start position ({x}, {y}) is out of bounds")
            
    def is_within_bounds(self, x: int, y: int) -> bool:
        """
        Check if a position is within the grid boundaries.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if position is within bounds, False otherwise
        """
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """
        Check if a position is valid (within bounds and not an obstacle).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if position is valid, False otherwise
        """
        return self.is_within_bounds(x, y) and (x, y) not in self.obstacles
    
    def distance_to_goal(self, x: int, y: int) -> float:
        """
        Calculate Manhattan distance from a position to the goal.
        Manhattan distance = |x1 - x2| + |y1 - y2|
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Distance to goal (float)
        """
        if self.goal is None:
            raise ValueError("Goal has not been set")
        return abs(x - self.goal[0]) + abs(y - self.goal[1])
    
    def get_nearest_obstacle(self, x: int, y: int) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        Find the nearest obstacle from a given position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Tuple of (obstacle_position, distance) or None if no obstacles
        """
        if not self.obstacles:
            return None
            
        min_dist = float('inf')
        nearest = None
        
        for obs_x, obs_y in self.obstacles:
            dist = abs(x - obs_x) + abs(y - obs_y)
            if dist < min_dist:
                min_dist = dist
                nearest = (obs_x, obs_y)
                
        return (nearest, min_dist) if nearest else None
    
    def get_state(self) -> dict:
        """
        Get the current state of the environment.
        
        Returns:
            Dictionary containing environment state
        """
        return {
            'width': self.width,
            'height': self.height,
            'obstacles': self.obstacles.copy(),
            'goal': self.goal,
            'start': self.start,
            'total_cells': self.width * self.height,
            'obstacle_count': len(self.obstacles)
        }
    
    def visualize(self, drone_position: Optional[Tuple[int, int]] = None):
        """
        Print a simple text visualization of the environment.
        
        Args:
            drone_position: Current position of drone (optional)
        """
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        
        # Mark obstacles
        for x, y in self.obstacles:
            grid[y][x] = '#'
            
        # Mark goal
        if self.goal:
            grid[self.goal[1]][self.goal[0]] = 'G'
            
        # Mark start
        if self.start:
            grid[self.start[1]][self.start[0]] = 'S'
            
        # Mark drone
        if drone_position:
            grid[drone_position[1]][drone_position[0]] = 'D'
            
        # Print grid
        print("\n" + "=" * (self.width + 2))
        for row in grid:
            print("|" + "".join(row) + "|")
        print("=" * (self.width + 2))
        print("Legend: D=Drone, S=Start, G=Goal, #=Obstacle, .=Empty\n")

