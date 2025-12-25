"""
PART 4: Bayesian Game Solver
This file handles uncertainty and incomplete information using Bayesian reasoning.
The drone maintains beliefs about the environment type and updates them as it learns.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from backend.game_theory.payoff import PayoffFunction
from backend.game_theory.strategies import (
    DroneAction, EnvironmentCondition, MixedStrategy,
    DroneStrategies, EnvironmentStrategies
)


class EnvironmentType:
    """Represents different types of environments the drone might face."""
    ADVERSARIAL = "adversarial"    # Environment actively opposes drone
    NEUTRAL = "neutral"            # Environment is random/indifferent
    FAVORABLE = "favorable"        # Environment tends to help drone
    
    @staticmethod
    def get_all_types() -> List[str]:
        """Get all possible environment types."""
        return [EnvironmentType.ADVERSARIAL, EnvironmentType.NEUTRAL, EnvironmentType.FAVORABLE]


class BayesianGameSolver:
    """
    Implements Bayesian game theory for drone navigation under uncertainty.
    
    The drone doesn't know what type of environment it's in initially.
    It maintains beliefs (probabilities) about each environment type and
    updates these beliefs as it observes outcomes (Bayesian learning).
    """
    
    def __init__(self, payoff_function: PayoffFunction, use_mixed_strategy: bool = True):
        """
        Initialize the Bayesian game solver.
        
        Args:
            payoff_function: PayoffFunction instance for calculating payoffs
            use_mixed_strategy: If True, use probabilistic action selection based on utilities.
                               If False, always select the action with highest expected utility.
        """
        self.payoff_function = payoff_function
        
        # Belief state: probability distribution over environment types
        self.beliefs = {}
        
        # History of observations for learning
        self.observation_history = []
        
        # Prior parameters (can be adjusted based on domain knowledge)
        self.prior_adversarial = 0.3
        self.prior_neutral = 0.5
        self.prior_favorable = 0.2
        
        # Strategy mode: mixed (probabilistic) or pure (deterministic)
        self.use_mixed_strategy = use_mixed_strategy
        
    def initialize_beliefs(self, 
        prior_adversarial: Optional[float] = None, 
        prior_neutral: Optional[float] = None,
        prior_favorable: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Initialize belief distribution over environment types.
        
        Sets the drone's initial beliefs before any observations.
        
        Args:
            prior_adversarial: Initial belief that environment is adversarial
            prior_neutral: Initial belief that environment is neutral
            prior_favorable: Initial belief that environment is favorable
            
        Returns:
            Dictionary mapping environment types to belief probabilities
            
        Example:
            >>> solver.initialize_beliefs()
            {'adversarial': 0.3, 'neutral': 0.5, 'favorable': 0.2}
        """
        # Use provided priors or defaults
        p_adversarial = prior_adversarial if prior_adversarial is not None else self.prior_adversarial
        p_neutral = prior_neutral if prior_neutral is not None else self.prior_neutral
        p_favorable = prior_favorable if prior_favorable is not None else self.prior_favorable
        
        # Normalize to ensure they sum to 1.0
        total = p_adversarial + p_neutral + p_favorable
        
        self.beliefs = {
            EnvironmentType.ADVERSARIAL: p_adversarial / total,
            EnvironmentType.NEUTRAL: p_neutral / total,
            EnvironmentType.FAVORABLE: p_favorable / total
        }
        
        # Reset observation history
        self.observation_history = []
        
        return self.beliefs.copy()
    
    def get_environment_strategy(self, env_type: str) -> MixedStrategy:
        """
        Get the expected strategy for a given environment type.
        
        Different environment types have different behavioral patterns:
        - Adversarial: Prefers obstacles and difficult conditions
        - Neutral: Random/uniform distribution
        - Favorable: Prefers clear paths and good conditions
        
        Args:
            env_type: Type of environment (adversarial, neutral, or favorable)
            
        Returns:
            MixedStrategy representing that environment type's behavior
        """
        all_conditions = EnvironmentStrategies.get_all_pure_strategies()
        
        if env_type == EnvironmentType.ADVERSARIAL:
            # Adversarial: High probability of obstacles and difficult conditions
            return EnvironmentStrategies.create_adversarial_conditions()
        
        elif env_type == EnvironmentType.NEUTRAL:
            # Neutral: Uniform/realistic distribution
            return EnvironmentStrategies.create_typical_conditions()
        
        elif env_type == EnvironmentType.FAVORABLE:
            # Favorable: High probability of clear paths
            return EnvironmentStrategies.create_favorable_conditions()
        
        else:
            # Default to neutral if unknown type
            return EnvironmentStrategies.create_uniform_mixed_strategy()
    
    def calculate_likelihood(self, 
        action: DroneAction, observed_condition: EnvironmentCondition, env_type: str
    ) -> float:
        """
        Calculate P(observation | environment_type) - the likelihood.
        
        This is the probability of observing a particular environmental
        condition given that the environment is of a specific type.
        
        Args:
            action: The action the drone took
            observed_condition: The environmental condition that was observed
            env_type: The hypothesized environment type
            
        Returns:
            Likelihood probability P(observation | env_type)
        """
        # Get the strategy for this environment type
        env_strategy = self.get_environment_strategy(env_type)
        
        # The likelihood is simply the probability that this environment type
        # would produce the observed condition
        likelihood = env_strategy.get_probability(observed_condition)
        
        return likelihood
    
    def update_beliefs(self,
        action: DroneAction, observed_condition: EnvironmentCondition, observed_payoff: float
    ) -> Dict[str, float]:
        """
        Update beliefs using Bayes' theorem after observing an outcome.
        
        Bayes' theorem: P(env_type | observation) ∝ P(observation | env_type) × P(env_type)
        
        If the drone moves forward and encounters an obstacle, this increases
        the belief that the environment is adversarial. If it encounters clear
        paths repeatedly, belief shifts toward favorable/neutral.
        
        Args:
            action: The action the drone took
            observed_condition: The environmental condition that actually occurred
            observed_payoff: The payoff that resulted (used for validation)
            
        Returns:
            Updated belief distribution
            
        Example:
            Initial beliefs: {adversarial: 0.3, neutral: 0.5, favorable: 0.2}
            After observing obstacle: {adversarial: 0.5, neutral: 0.4, favorable: 0.1}
        """
        if not self.beliefs:
            self.initialize_beliefs()
        
        # Store observation in history
        self.observation_history.append({
            'action': action,
            'condition': observed_condition,
            'payoff': observed_payoff
        })
        
        # Bayesian update for each environment type
        posterior = {}
        
        for env_type in EnvironmentType.get_all_types():
            # Prior: P(env_type)
            prior = self.beliefs[env_type]
            
            # Likelihood: P(observation | env_type)
            likelihood = self.calculate_likelihood(action, observed_condition, env_type)
            
            # Posterior (unnormalized): P(env_type | observation) ∝ P(observation | env_type) × P(env_type)
            posterior[env_type] = likelihood * prior
        
        # Normalize to make probabilities sum to 1.0
        total = sum(posterior.values())
        
        if total > 0:
            for env_type in posterior:
                posterior[env_type] /= total
        else:
            # If all likelihoods are 0 (shouldn't happen), keep previous beliefs
            posterior = self.beliefs.copy()
        
        # Update stored beliefs
        self.beliefs = posterior
        
        return self.beliefs.copy()
    
    def expected_utility(self,
        action: DroneAction, state_params: Dict
    ) -> float:
        """
        Calculate expected utility of an action over all environment types.
        
        E[U(action)] = Σ P(env_type) × U(action, env_type)
        
        This averages the utility of the action across all possible environment
        types, weighted by the current beliefs about each type.
        
        Args:
            action: The drone action to evaluate
            state_params: Current state parameters for payoff calculation
            
        Returns:
            Expected utility (payoff) of the action
        """
        if not self.beliefs:
            self.initialize_beliefs()
        
        expected_util = 0.0
        
        # For each environment type, calculate expected payoff and weight by belief
        for env_type, belief_prob in self.beliefs.items():
            # Get the strategy this environment type would use
            env_strategy = self.get_environment_strategy(env_type)
            
            # Calculate expected payoff against this environment strategy
            env_expected = 0.0
            
            for condition, prob in zip(env_strategy.strategies, env_strategy.probabilities):
                # Calculate payoff for this (action, condition) pair
                drone_payoff, _ = self.payoff_function.compute_payoff(
                    action,
                    condition,
                    state_params['current_pos'],
                    state_params['goal_pos'],
                    state_params['initial_distance'],
                    state_params['battery_used'],
                    state_params['total_battery'],
                    state_params['distance_to_nearest_obstacle'],
                    state_params['explored_cells'],
                    state_params['total_cells'],
                    state_params.get('collision', False),
                    state_params.get('environment', None)
                )
                
                # Weight by probability of this condition under this environment type
                env_expected += prob * drone_payoff
            
            # Weight by belief in this environment type
            expected_util += belief_prob * env_expected
        
        return expected_util
    
    def bayesian_decision(self,
        available_actions: List[DroneAction], state_params: Dict, verbose: bool = False
    ) -> Tuple[DroneAction, float, Dict]:
        """
        Choose action based on expected utilities and current beliefs.
        
        Two modes:
        - Pure strategy (deterministic): Always picks action with highest expected utility
        - Mixed strategy (probabilistic): Samples action proportional to utilities (softmax)
        
        Args:
            available_actions: List of valid drone actions
            state_params: Current state parameters
            verbose: If True, print detailed analysis
            
        Returns:
            Tuple of (best_action, expected_utility, analysis_details)
        """
        if not available_actions:
            raise ValueError("available_actions must be a non-empty list")
        
        if not self.beliefs:
            self.initialize_beliefs()
        
        analysis = {}
        utilities = []
        
        if verbose:
            print("\n" + "="*70)
            print("BAYESIAN DECISION ANALYSIS")
            print("="*70)
            print(f"\nMode: {'Mixed Strategy (Probabilistic)' if self.use_mixed_strategy else 'Pure Strategy (Deterministic)'}")
            print("\nCurrent Beliefs:")
            for env_type, prob in self.beliefs.items():
                print(f"  {env_type:15s}: {prob:6.2%}")
            print()
        
        # Evaluate each action
        for action in available_actions:
            expected_util = self.expected_utility(action, state_params)
            utilities.append(expected_util)
            
            analysis[action] = {
                'expected_utility': expected_util,
                'beliefs': self.beliefs.copy()
            }
            
            if verbose:
                print(f"Action: {action.value:15s} → Expected Utility: {expected_util:7.2f}")
        
        # Select action based on strategy mode
        if self.use_mixed_strategy:
            # Convert utilities to probabilities using softmax
            utilities_array = np.array(utilities)
            
            exp_utilities = np.exp(utilities_array)
            probabilities = exp_utilities / np.sum(exp_utilities)
            
            # Sample action according to probabilities
            selected_idx = np.random.choice(len(available_actions), p=probabilities)
            selected_action = available_actions[selected_idx]
            selected_utility = utilities[selected_idx]
            
            if verbose:
                print("\n" + "-"*70)
                print("Mixed Strategy Probabilities:")
                for action, prob in zip(available_actions, probabilities):
                    print(f"  {action.value:15s}: {prob:6.2%}")
                print("\n" + "-"*70)
                print(f"SELECTED ACTION (sampled): {selected_action.value}")
                print(f"Expected Utility: {selected_utility:.2f}")
                print("="*70)
        else:
            # Pure strategy: always pick best action
            best_idx = np.argmax(utilities)
            selected_action = available_actions[best_idx]
            selected_utility = utilities[best_idx]
            
            if verbose:
                print("\n" + "-"*70)
                print(f"SELECTED ACTION (best): {selected_action.value}")
                print(f"Expected Utility: {selected_utility:.2f}")
                print("="*70)
        
        return selected_action, selected_utility, analysis
    
    def solve(self,
        available_actions: List[DroneAction], state_params: Dict, verbose: bool = False
    ) -> DroneAction:
        """
        Main interface for Bayesian decision-making.
        
        This is the simple wrapper that external code should call.
        
        Args:
            available_actions: List of valid drone actions
            state_params: Current state parameters
            verbose: If True, print analysis
            
        Returns:
            Recommended drone action
        """
        action, _, _ = self.bayesian_decision(available_actions, state_params, verbose)
        return action
    
    def get_beliefs(self) -> Dict[str, float]:
        """
        Get the current belief distribution.
        
        Returns:
            Dictionary mapping environment types to belief probabilities
        """
        if not self.beliefs:
            return self.initialize_beliefs()
        return self.beliefs.copy()
    
    def get_observation_count(self) -> int:
        """
        Get the number of observations collected so far.
        
        Returns:
            Count of observations in history
        """
        return len(self.observation_history)
    
    def set_strategy_mode(self, use_mixed_strategy: bool):
        """
        Switch between pure and mixed strategy modes.
        
        Args:
            use_mixed_strategy: True for probabilistic (mixed), False for deterministic (pure)
        """
        self.use_mixed_strategy = use_mixed_strategy
    
    def get_strategy_mode(self) -> str:
        """
        Get current strategy mode.
        
        Returns:
            String describing current mode: "mixed" or "pure"
        """
        return "mixed" if self.use_mixed_strategy else "pure"
    
    def reset(self):
        """
        Reset the solver to initial state.
        
        Clears all beliefs and observation history.
        """
        self.beliefs = self.initialize_beliefs()
        self.observation_history = []
    
    def __repr__(self) -> str:
        """String representation of the solver."""
        mode = "mixed" if self.use_mixed_strategy else "pure"
        if self.beliefs:
            belief_str = ", ".join([f"{t}: {p:.2%}" for t, p in self.beliefs.items()])
            return f"BayesianGameSolver(mode={mode}, beliefs=[{belief_str}], observations={len(self.observation_history)})"
        return f"BayesianGameSolver(mode={mode}, payoff={self.payoff_function})"
