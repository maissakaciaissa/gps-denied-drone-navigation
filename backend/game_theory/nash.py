"""
PART 3 : Nash Equilibrium Solver
This file contains the NashEquilibriumSolver class that finds stable strategy pairs.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from itertools import combinations
from backend.game_theory.payoff import PayoffFunction
from backend.game_theory.strategies import (
    DroneAction, EnvironmentCondition, MixedStrategy,
    DroneStrategies, EnvironmentStrategies
)


class NashEquilibriumSolver:
    """
    Implements Nash Equilibrium finding for drone navigation game.
    Finds stable strategy pairs where neither player benefits from unilateral deviation.
    """
    
    def __init__(self, payoff_function: PayoffFunction):
        """
        Initialize the Nash Equilibrium solver.
        
        Args:
            payoff_function: PayoffFunction instance for calculating payoffs
        """
        self.payoff_function = payoff_function
        self.last_nash_drone = None
        self.last_nash_env = None
        
    def find_nash_equilibrium(self,
        drone_payoff_matrix: np.ndarray, env_payoff_matrix: np.ndarray,
        drone_actions: List[DroneAction], env_conditions: List[EnvironmentCondition]
    ) -> Tuple[MixedStrategy, MixedStrategy]:
        """
        Find Nash equilibrium given payoff matrices using linear programming.
        
        This uses the support enumeration method combined with linear programming
        to find mixed strategy Nash equilibrium.
        
        Args:
            drone_payoff_matrix: Payoff matrix for drone (rows=drone actions, cols=env conditions)
            env_payoff_matrix: Payoff matrix for environment
            drone_actions: List of drone actions corresponding to matrix rows
            env_conditions: List of environment conditions corresponding to matrix columns
            
        Returns:
            Tuple of (drone_mixed_strategy, env_mixed_strategy) at Nash equilibrium
        """
        n_drone = len(drone_actions)
        n_env = len(env_conditions)
        
        # Try to find Nash equilibrium using support enumeration
        nash_drone_probs, nash_env_probs = self.support_enumeration(
            drone_payoff_matrix, env_payoff_matrix
        )
        
        # If support enumeration fails, try iterative best response
        if nash_drone_probs is None or nash_env_probs is None:
            nash_drone_probs, nash_env_probs = self.iterative_best_response(
                drone_payoff_matrix, env_payoff_matrix, max_iterations=100
            )
        
        # Create mixed strategies from probability distributions
        drone_strategy = MixedStrategy(drone_actions, nash_drone_probs)
        env_strategy = MixedStrategy(env_conditions, nash_env_probs)
        
        # Store for later reference
        self.last_nash_drone = drone_strategy
        self.last_nash_env = env_strategy
        
        return drone_strategy, env_strategy
    
    def support_enumeration(self, 
        drone_payoff_matrix: np.ndarray, env_payoff_matrix: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find Nash equilibrium using support enumeration method.
        
        This method tries different combinations of action supports (subsets of actions
        that could be played with positive probability) and solves for mixed strategies
        that satisfy the indifference condition on those supports.
        
        Args:
            drone_payoff_matrix: Drone's payoff matrix
            env_payoff_matrix: Environment's payoff matrix
            
        Returns:
            Tuple of (drone_probabilities, env_probabilities) or (None, None) if failed
        """
        n_drone, n_env = drone_payoff_matrix.shape
        
        try:
            # Try different support sizes, starting with smaller ones (more likely to be Nash)
            for drone_support_size in range(1, n_drone + 1):
                for env_support_size in range(1, n_env + 1):
                    
                    # Try all combinations of actions in the support
                    for drone_support in combinations(range(n_drone), drone_support_size):
                        for env_support in combinations(range(n_env), env_support_size):
                            
                            # Try to solve for Nash with this support
                            result = self._solve_for_support(
                                drone_support, env_support,
                                drone_payoff_matrix, env_payoff_matrix
                            )
                            
                            if result is not None:
                                drone_probs, env_probs = result
                                
                                # Verify it's actually a Nash equilibrium
                                if self._verify_nash(
                                    drone_probs, env_probs,
                                    drone_payoff_matrix, env_payoff_matrix
                                ):
                                    return drone_probs, env_probs
            
            return None, None
            
        except Exception as e:
            # If support enumeration fails, return None to trigger alternative method
            return None, None
    
    def _solve_for_support(self,
        drone_support: tuple, env_support: tuple,
        drone_payoff_matrix: np.ndarray, env_payoff_matrix: np.ndarray,
        tolerance: float = 1e-6
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Solve for mixed strategies given fixed supports.
        
        For a Nash equilibrium with given supports:
        - All actions in support must give equal expected payoff
        - Actions outside support must not give higher payoff
        - Probabilities must sum to 1 and be non-negative
        
        Args:
            drone_support: Indices of drone actions in support
            env_support: Indices of environment conditions in support
            drone_payoff_matrix: Drone's payoff matrix
            env_payoff_matrix: Environment's payoff matrix
            tolerance: Numerical tolerance
            
        Returns:
            (drone_probs, env_probs) if valid Nash found, None otherwise
        """
        n_drone, n_env = drone_payoff_matrix.shape
        drone_support = list(drone_support)
        env_support = list(env_support)
        
        try:
            # For pure strategy Nash (support size 1 for both)
            if len(drone_support) == 1 and len(env_support) == 1:
                drone_probs = np.zeros(n_drone)
                env_probs = np.zeros(n_env)
                drone_probs[drone_support[0]] = 1.0
                env_probs[env_support[0]] = 1.0
                return drone_probs, env_probs
            
            # For mixed strategies, solve using indifference conditions
            # Drone must be indifferent over its support given env's strategy
            # Environment must be indifferent over its support given drone's strategy
            
            # Use linear programming to solve the indifference equations
            # For drone: payoff_matrix[i, :] @ env_probs should be equal for all i in drone_support
            # For env: drone_probs @ payoff_matrix[:, j] should be equal for all j in env_support
            
            # Build system of equations for environment's probabilities
            # Indifference: drone_payoff[i1, :] @ q = drone_payoff[i2, :] @ q for all i1, i2 in support
            if len(env_support) == len(drone_support):
                # Square system - can solve directly
                A = []
                for i in range(len(drone_support) - 1):
                    # Indifference equation: payoff[i] @ q = payoff[i+1] @ q
                    # => (payoff[i] - payoff[i+1]) @ q = 0
                    row = drone_payoff_matrix[drone_support[i], env_support] - \
                          drone_payoff_matrix[drone_support[i+1], env_support]
                    A.append(row)
                
                # Probability constraint: sum of probs in support = 1
                A.append(np.ones(len(env_support)))
                A = np.array(A)
                
                b = np.zeros(len(A))
                b[-1] = 1.0  # Sum to 1
                
                # Solve for probabilities in support
                try:
                    support_probs = np.linalg.solve(A, b)
                except:
                    return None
                
                # Check if probabilities are valid (non-negative)
                if np.any(support_probs < -tolerance) or np.any(support_probs > 1 + tolerance):
                    return None
                
                # Clip to valid range
                support_probs = np.clip(support_probs, 0, 1)
                support_probs = support_probs / np.sum(support_probs)  # Renormalize
                
                # Build full probability vector
                env_probs = np.zeros(n_env)
                env_probs[env_support] = support_probs
                
                # Now solve for drone's probabilities using environment's strategy
                A_drone = []
                for j in range(len(env_support) - 1):
                    row = env_payoff_matrix[drone_support, env_support[j]] - \
                          env_payoff_matrix[drone_support, env_support[j+1]]
                    A_drone.append(row)
                
                A_drone.append(np.ones(len(drone_support)))
                A_drone = np.array(A_drone)
                
                b_drone = np.zeros(len(A_drone))
                b_drone[-1] = 1.0
                
                try:
                    drone_support_probs = np.linalg.solve(A_drone, b_drone)
                except:
                    return None
                
                if np.any(drone_support_probs < -tolerance) or np.any(drone_support_probs > 1 + tolerance):
                    return None
                
                drone_support_probs = np.clip(drone_support_probs, 0, 1)
                drone_support_probs = drone_support_probs / np.sum(drone_support_probs)
                
                drone_probs = np.zeros(n_drone)
                drone_probs[drone_support] = drone_support_probs
                
                return drone_probs, env_probs
            
            return None
            
        except Exception as e:
            return None
    
    def _verify_nash(self,
        drone_probs: np.ndarray, env_probs: np.ndarray,
        drone_payoff_matrix: np.ndarray, env_payoff_matrix: np.ndarray,
        tolerance: float = 1e-4
    ) -> bool:
        """
        Verify if probability vectors form a Nash equilibrium.
        
        Args:
            drone_probs: Drone's probability distribution
            env_probs: Environment's probability distribution
            drone_payoff_matrix: Drone's payoff matrix
            env_payoff_matrix: Environment's payoff matrix
            tolerance: Numerical tolerance
            
        Returns:
            True if the strategy pair is a Nash equilibrium
        """
        # Calculate expected payoffs
        drone_expected = drone_payoff_matrix @ env_probs
        env_expected = drone_probs @ env_payoff_matrix
        
        # --- Check drone's strategy ---
        drone_support = np.where(drone_probs > tolerance)[0]
        if len(drone_support) > 0:
            support_payoffs = drone_expected[drone_support]
            max_support = np.max(support_payoffs)
            
            # Indifference on support
            if np.max(support_payoffs) - np.min(support_payoffs) > tolerance:
                return False
            
            # No profitable deviation
            if np.any(drone_expected > max_support + tolerance):
                return False
        
        # --- Check environment's strategy ---
        env_support = np.where(env_probs > tolerance)[0]
        if len(env_support) > 0:
            support_payoffs = env_expected[env_support]
            max_support = np.max(support_payoffs)
            
            # Indifference on support
            if np.max(support_payoffs) - np.min(support_payoffs) > tolerance:
                return False
            
            # No profitable deviation
            if np.any(env_expected > max_support + tolerance):
                return False
        
        return True
    
    def iterative_best_response(self, 
        drone_payoff_matrix: np.ndarray, env_payoff_matrix: np.ndarray,
        max_iterations: int = 100, tolerance: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find Nash equilibrium using iterative best response dynamics.
        
        Args:
            drone_payoff_matrix: Drone's payoff matrix
            env_payoff_matrix: Environment's payoff matrix
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (drone_probabilities, env_probabilities)
        """
        n_drone, n_env = drone_payoff_matrix.shape
        
        # Initialize with uniform strategies
        drone_probs = np.ones(n_drone) / n_drone
        env_probs = np.ones(n_env) / n_env
        
        for iteration in range(max_iterations):
            # Store old probabilities
            old_drone_probs = drone_probs.copy()
            old_env_probs = env_probs.copy()
            
            # Drone's best response to environment's strategy
            drone_probs = self._best_response_drone_numeric(
                env_probs, drone_payoff_matrix
            )
            
            # Environment's best response to drone's strategy
            env_probs = self._best_response_env_numeric(
                drone_probs, env_payoff_matrix
            )
            
            # Check convergence
            drone_change = np.linalg.norm(drone_probs - old_drone_probs)
            env_change = np.linalg.norm(env_probs - old_env_probs)
            
            if drone_change < tolerance and env_change < tolerance:
                break
        
        return drone_probs, env_probs
    
    def _best_response_drone_numeric(self, 
        env_probs: np.ndarray, drone_payoff_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate drone's best response to environment's mixed strategy (numeric version).
        
        Args:
            env_probs: Environment's probability distribution
            drone_payoff_matrix: Drone's payoff matrix
            
        Returns:
            Drone's best response probability distribution
        """
        # Calculate expected payoff for each drone action
        expected_payoffs = drone_payoff_matrix @ env_probs
        
        # Find actions with maximum expected payoff
        max_payoff = np.max(expected_payoffs)
        best_actions = np.where(np.abs(expected_payoffs - max_payoff) < 1e-8)[0]
        
        # Create uniform distribution over best actions
        best_response = np.zeros(len(expected_payoffs))
        best_response[best_actions] = 1.0 / len(best_actions)
        
        return best_response
    
    def _best_response_env_numeric(self, 
        drone_probs: np.ndarray, env_payoff_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate environment's best response to drone's mixed strategy (numeric version).
        
        Args:
            drone_probs: Drone's probability distribution
            env_payoff_matrix: Environment's payoff matrix
            
        Returns:
            Environment's best response probability distribution
        """
        # Calculate expected payoff for each environment condition
        expected_payoffs = drone_probs @ env_payoff_matrix
        
        # Find conditions with maximum expected payoff
        max_payoff = np.max(expected_payoffs)
        best_conditions = np.where(np.abs(expected_payoffs - max_payoff) < 1e-8)[0]
        
        # Create uniform distribution over best conditions
        best_response = np.zeros(len(expected_payoffs))
        best_response[best_conditions] = 1.0 / len(best_conditions)
        
        return best_response
    
    def best_response_drone(self, 
        env_strategy: MixedStrategy, drone_actions: List[DroneAction],
        drone_payoff_matrix: np.ndarray
    ) -> DroneAction:
        """
        Find the best drone action given the environment's mixed strategy.
        
        Args:
            env_strategy: Environment's mixed strategy
            drone_actions: List of possible drone actions
            drone_payoff_matrix: Drone's payoff matrix
            
        Returns:
            Best drone action
        """
        n_drone, n_env = drone_payoff_matrix.shape
        
        # Get environment probability vector
        env_probs = np.array(env_strategy.probabilities)
        
        # Calculate expected payoff for each action
        expected_payoffs = drone_payoff_matrix @ env_probs
        
        # Return action with highest expected payoff
        best_idx = np.argmax(expected_payoffs)
        return drone_actions[best_idx]
    
    def best_response_env(self, 
        drone_strategy: MixedStrategy, env_conditions: List[EnvironmentCondition],
        env_payoff_matrix: np.ndarray
    ) -> EnvironmentCondition:
        """
        Find the best environment condition given the drone's mixed strategy.
        
        Args:
            drone_strategy: Drone's mixed strategy
            env_conditions: List of possible environment conditions
            env_payoff_matrix: Environment's payoff matrix
            
        Returns:
            Best environment condition
        """
        n_drone, n_env = env_payoff_matrix.shape
        
        # Get drone probability vector
        drone_probs = np.array(drone_strategy.probabilities)
        
        # Calculate expected payoff for each condition
        expected_payoffs = drone_probs @ env_payoff_matrix
        
        # Return condition with highest expected payoff
        best_idx = np.argmax(expected_payoffs)
        return env_conditions[best_idx]
    
    def solve(self, 
        state_params: Dict,
        drone_actions: Optional[List[DroneAction]] = None,
        env_conditions: Optional[List[EnvironmentCondition]] = None
    ) -> DroneAction:
        """
        Solve for Nash equilibrium and return a recommended action.
        
        Args:
            state_params: Current state parameters for payoff calculation
            drone_actions: List of available drone actions (default: all actions)
            env_conditions: List of possible environment conditions (default: all conditions)
            
        Returns:
            Recommended drone action sampled from Nash equilibrium strategy
        """
        # Use all actions if not specified
        if drone_actions is None:
            drone_actions = DroneStrategies.get_all_pure_strategies()
        if env_conditions is None:
            env_conditions = EnvironmentStrategies.get_all_pure_strategies()
        
        # Generate payoff matrices
        drone_matrix, env_matrix = self.payoff_function.generate_payoff_matrix(
            drone_actions, env_conditions, state_params
        )
        
        # Find Nash equilibrium
        nash_drone, nash_env = self.find_nash_equilibrium(
            drone_matrix, env_matrix, drone_actions, env_conditions
        )
        
        # Sample action from Nash equilibrium strategy
        action = nash_drone.sample()
        
        return action
    
    def get_nash_strategies(self) -> Tuple[Optional[MixedStrategy], Optional[MixedStrategy]]:
        """
        Get the last computed Nash equilibrium strategies.
        
        Returns:
            Tuple of (drone_strategy, env_strategy) or (None, None) if not computed yet
        """
        return self.last_nash_drone, self.last_nash_env
    
    def calculate_expected_payoff(self, 
        drone_strategy: MixedStrategy, env_strategy: MixedStrategy,
        drone_payoff_matrix: np.ndarray
    ) -> float:
        """
        Calculate expected payoff for drone given both mixed strategies.
        
        Args:
            drone_strategy: Drone's mixed strategy
            env_strategy: Environment's mixed strategy
            drone_payoff_matrix: Drone's payoff matrix
            
        Returns:
            Expected payoff value
        """
        n_drone, n_env = drone_payoff_matrix.shape
        
        # Get probabilities directly from the strategies
        drone_probs = np.array(drone_strategy.probabilities)
        env_probs = np.array(env_strategy.probabilities)
        
        return drone_probs @ drone_payoff_matrix @ env_probs
    
    def __repr__(self) -> str:
        """String representation of the solver."""
        return f"NashEquilibriumSolver(payoff={self.payoff_function})"
