"""
PART 1: Game Theory Strategies
This file defines all possible actions (strategies) for both the drone and the environment.
"""

import numpy as np
from typing import List, Dict, Tuple
from enum import Enum


class DroneAction(Enum):
    """Pure strategies for the drone."""
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    STAY = "stay"
    ROTATE = "rotate"


class EnvironmentCondition(Enum):
    """
    Pure strategies for the environment.
    we treated uncertainty (weather, obstacles) as if "nature"
    is making moves against us. 
    This helps us plan for worst-case scenarios!
    """
    CLEAR_PATH = "clear_path"
    OBSTACLE_AHEAD = "obstacle_ahead"
    LOW_VISIBILITY = "low_visibility"
    SENSOR_NOISE = "sensor_noise"
    LIGHTING_CHANGE = "lighting_change"


class MixedStrategy:
    """
    Represents a mixed strategy (probability distribution over pure strategies).
    """
    
    def __init__(self, strategies: List, probabilities: List[float]):
        """
        Initialize a mixed strategy.
        
        Args:
            strategies: List of pure strategies (DroneAction or EnvironmentCondition)
            probabilities: List of probabilities for each strategy (must sum to 1)
        """
        if len(strategies) != len(probabilities):
            raise ValueError("Number of strategies must match number of probabilities")
        
        total = sum(probabilities)
        if not np.isclose(total, 1.0):
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")
        
        self.strategies = strategies
        self.probabilities = probabilities
        
    def sample(self):
        """
        Sample one action from the mixed strategy based on probabilities.
        
        Returns:
            A randomly selected strategy according to the probability distribution
        """
        return np.random.choice(self.strategies, p=self.probabilities)
    
    def get_probability(self, strategy) -> float:
        """
        Get the probability of a specific strategy.
        
        Args:
            strategy: The strategy to query
            
        Returns:
            Probability of that strategy
        """
        try:
            idx = self.strategies.index(strategy)
            return self.probabilities[idx]
        except ValueError:
            return 0.0
    
    def __repr__(self) -> str:
        """String representation of the mixed strategy."""
        items = [f"{s.value if hasattr(s, 'value') else s}: {p:.2f}" 
                for s, p in zip(self.strategies, self.probabilities)]
        return f"MixedStrategy({', '.join(items)})"


class DroneStrategies:
    """
    Container for drone strategy operations.
    """
    
    @staticmethod
    def get_all_pure_strategies() -> List[DroneAction]:
        """
        Get all pure strategies available to the drone.
        
        Returns:
            List of all DroneAction enum values
        """
        return list(DroneAction)
    
    @staticmethod
    def create_uniform_mixed_strategy() -> MixedStrategy:
        """
        Create a uniform mixed strategy (equal probability for all actions).
        
        Returns:
            MixedStrategy with equal probabilities
        """
        actions = DroneStrategies.get_all_pure_strategies()
        n = len(actions)
        probabilities = [1.0 / n] * n
        return MixedStrategy(actions, probabilities)
    
    @staticmethod
    def create_cautious_strategy() -> MixedStrategy:
        """
        Create a cautious mixed strategy (prefer staying and careful movements).
        
        Example: 60% stay, 40% distributed among moves
        
        Returns:
            MixedStrategy favoring cautious actions
        """
        return MixedStrategy(
            strategies=[DroneAction.STAY, DroneAction.MOVE_UP, 
                       DroneAction.MOVE_DOWN, DroneAction.MOVE_LEFT, 
                       DroneAction.MOVE_RIGHT, DroneAction.ROTATE],
            probabilities=[0.6, 0.1, 0.1, 0.05, 0.05, 0.1]
        )
    
    @staticmethod
    def create_aggressive_strategy() -> MixedStrategy:
        """
        Create an aggressive mixed strategy (prefer moving).
        
        Example: Focus on movement actions, minimal staying
        
        Returns:
            MixedStrategy favoring movement
        """
        return MixedStrategy(
            strategies=[DroneAction.MOVE_UP, DroneAction.MOVE_DOWN, 
                       DroneAction.MOVE_LEFT, DroneAction.MOVE_RIGHT, 
                       DroneAction.STAY, DroneAction.ROTATE],
            probabilities=[0.25, 0.25, 0.2, 0.2, 0.05, 0.05]
        )
    
    @staticmethod
    def create_balanced_strategy() -> MixedStrategy:
        """
        Create a balanced mixed strategy.
        
        Returns:
            MixedStrategy with balanced probabilities
        """
        return MixedStrategy(
            strategies=[DroneAction.MOVE_UP, DroneAction.MOVE_DOWN, 
                       DroneAction.MOVE_LEFT, DroneAction.MOVE_RIGHT, 
                       DroneAction.STAY, DroneAction.ROTATE],
            probabilities=[0.2, 0.2, 0.2, 0.2, 0.15, 0.05]
        )
    
    @staticmethod
    def create_goal_oriented_strategy() -> MixedStrategy:
        """
        Create a goal-oriented mixed strategy.
        Strongly favors movement over staying, designed to make progress toward goal.
        No time wasted on STAY or ROTATE actions.
        
        Returns:
            MixedStrategy heavily favoring movement actions
        """
        return MixedStrategy(
            strategies=[DroneAction.MOVE_UP, DroneAction.MOVE_DOWN, 
                       DroneAction.MOVE_LEFT, DroneAction.MOVE_RIGHT, 
                       DroneAction.STAY, DroneAction.ROTATE],
            probabilities=[0.30, 0.30, 0.20, 0.20, 0.0, 0.0]
        )
    
    @staticmethod
    def create_custom_strategy(probabilities: Dict[DroneAction, float]) -> MixedStrategy:
        """
        Create a custom mixed strategy from a probability dictionary.
        
        Args:
            probabilities: Dictionary mapping DroneAction to probabilities
            
        Returns:
            MixedStrategy with custom probabilities
        """
        strategies = list(probabilities.keys())
        probs = list(probabilities.values())
        return MixedStrategy(strategies, probs)


class EnvironmentStrategies:
    """
    Container for environment strategy operations.
    """
    
    @staticmethod
    def get_all_pure_strategies() -> List[EnvironmentCondition]:
        """
        Get all pure strategies available to the environment.
        
        Returns:
            List of all EnvironmentCondition enum values
        """
        return list(EnvironmentCondition)
    
    @staticmethod
    def create_uniform_mixed_strategy() -> MixedStrategy:
        """
        Create a uniform mixed strategy for environment.
        
        Returns:
            MixedStrategy with equal probabilities
        """
        conditions = EnvironmentStrategies.get_all_pure_strategies()
        n = len(conditions)
        probabilities = [1.0 / n] * n
        return MixedStrategy(conditions, probabilities)
    
    @staticmethod
    def create_typical_conditions() -> MixedStrategy:
        """
        Create a mixed strategy representing typical environmental conditions.
        
        Example: 70% clear, 20% obstacle, 10% other conditions
        
        Returns:
            MixedStrategy representing common conditions
        """
        return MixedStrategy(
            strategies=[EnvironmentCondition.CLEAR_PATH, 
                       EnvironmentCondition.OBSTACLE_AHEAD,
                       EnvironmentCondition.LOW_VISIBILITY, 
                       EnvironmentCondition.SENSOR_NOISE,
                       EnvironmentCondition.LIGHTING_CHANGE],
            probabilities=[0.7, 0.15, 0.1, 0.03, 0.02]
        )
    
    @staticmethod
    def create_adversarial_conditions() -> MixedStrategy:
        """
        Create a mixed strategy representing adversarial environment.
        
        Example: More obstacles and challenging conditions
        
        Returns:
            MixedStrategy representing hostile conditions
        """
        return MixedStrategy(
            strategies=[EnvironmentCondition.CLEAR_PATH, 
                       EnvironmentCondition.OBSTACLE_AHEAD,
                       EnvironmentCondition.LOW_VISIBILITY, 
                       EnvironmentCondition.SENSOR_NOISE,
                       EnvironmentCondition.LIGHTING_CHANGE],
            probabilities=[0.3, 0.4, 0.15, 0.1, 0.05]
        )
    
    @staticmethod
    def create_favorable_conditions() -> MixedStrategy:
        """
        Create a mixed strategy representing favorable environment.
        
        Example: Mostly clear paths
        
        Returns:
            MixedStrategy representing benign conditions
        """
        return MixedStrategy(
            strategies=[EnvironmentCondition.CLEAR_PATH, 
                       EnvironmentCondition.OBSTACLE_AHEAD,
                       EnvironmentCondition.LOW_VISIBILITY, 
                       EnvironmentCondition.SENSOR_NOISE,
                       EnvironmentCondition.LIGHTING_CHANGE],
            probabilities=[0.85, 0.05, 0.05, 0.03, 0.02]
        )
    
    @staticmethod
    def create_custom_strategy(probabilities: Dict[EnvironmentCondition, float]) -> MixedStrategy:
        """
        Create a custom mixed strategy from a probability dictionary.
        
        Args:
            probabilities: Dictionary mapping EnvironmentCondition to probabilities
            
        Returns:
            MixedStrategy with custom probabilities
        """
        strategies = list(probabilities.keys())
        probs = list(probabilities.values())
        return MixedStrategy(strategies, probs)
