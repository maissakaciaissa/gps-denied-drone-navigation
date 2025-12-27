from typing import List, Tuple, Dict, Optional
from backend.game_theory.payoff import PayoffFunction
from backend.game_theory.strategies import DroneAction, EnvironmentCondition, MixedStrategy



class Minimax: 

    def __init__(self, payoff: PayoffFunction):
        self.payoff = payoff

    def evaluate_action(self, drone_action: DroneAction,state_params: Dict) -> Tuple[float, EnvironmentCondition]:
        """
        Pour une action donnée, trouve le pire scénario.
        
        Args:
            drone_action: L'action à évaluer
            state_params: État actuel (position, batterie, etc.)
        
        Returns:
            (pire_payoff, condition_qui_cause_ce_pire)
        """
        all_env_condition = list(EnvironmentCondition)
        worst_payoff = float('inf')
        worst_condition = None
        
        # 4. Pour chaque condition environnementale:
        #    a) Calculer le payoff avec self.payoff_function.compute_payoff()
        #       Paramètres nécessaires:
        #       - drone_action
        #       - env_condition
        #       - state_params['current_pos']
        #       - state_params['goal_pos']
        #       - state_params['initial_distance']
        #       - state_params['battery_used']
        #       - state_params['total_battery']
        #       - state_params['distance_to_nearest_obstacle']
        #       - state_params['explored_cells']
        #       - state_params['total_cells']
        #       - state_params.get('collision', False)
        #       - state_params.get('environment', None)
        #    
        #    b) compute_payoff retourne (drone_payoff, env_payoff)
        #       On ne garde que drone_payoff
        #    
        #    c) Si ce payoff < worst_payoff:
        #       Mettre à jour worst_payoff et worst_condition
        for env_condition in all_env_condition:
            drone_payoff, env_payoff = self.payoff.compute_payoff(
                drone_action, 
                env_condition, 
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
        
            if drone_payoff < worst_payoff : 
                worst_payoff = drone_payoff
                worst_condition = env_condition
        
        return worst_payoff, worst_condition


    def get_worst_case_payoff(self, drone_action : DroneAction, state_params: Dict) -> float:
        worst_payoff, _ = self.evaluate_action(drone_action, state_params)
        return worst_payoff


    def minimax_decision(self, available_actions: List[DroneAction], state_params: Dict, verbose: bool = False) -> Tuple[DroneAction, float, Dict]:
        """
        Choisit l'action avec le meilleur pire cas.
        
        Args:
            available_actions: Actions valides du drone
            state_params: État actuel
            verbose: Si True, affiche l'analyse
        
        Returns:
            (meilleure_action, son_pire_cas, détails_analyse)
        """

        if not available_actions:
            raise ValueError("available_actions doit être une liste non vide")

        best_action = None
        best_worst_case = float("-inf")
        analysis = {}
        
        if verbose:
            print("\n" + "="*70)
            print("ANALYSE MINIMAX")
            print("="*70)            

        
        # 4. Pour chaque action dans available_actions:
        #    a) Appeler evaluate_action() pour cette action
        #    b) Récupérer (worst_payoff, worst_condition)
        #    
        #    c) Stocker dans analysis:
        #       analysis[action] = {
        #           'worst_case_payoff': worst_payoff,
        #           'worst_condition': worst_condition
        #       }
        #    
        #    d) Si verbose, afficher l'action et son pire cas
        #    
        #    e) Si worst_payoff > best_worst_case:
        #       Mettre à jour best_worst_case et best_action


        for action in available_actions: 
            worst_payoff, worst_condition = self.evaluate_action(action, state_params)
            analysis[action] = {
                'worst_case_payoff' : worst_payoff,
                'worst_condition' : worst_condition
            }

            if verbose: 
                print(f"Action: {action}")
                print(f"Worst case payoff: {worst_payoff}")
                print(f"Worst condition: {worst_condition}")

            if worst_payoff > best_worst_case:
                best_worst_case = worst_payoff
                best_action = action

        if verbose:
            print("\n" + "-"*70)
            print(f"DÉCISION MINIMAX: {best_action.value}")
            print(f"Payoff garanti minimum: {best_worst_case:.2f}")
            print("="*70)
    
        return best_action, best_worst_case, analysis


    def solve(self,
            available_actions: List[DroneAction],
            state_params: Dict,
            verbose: bool = False) -> DroneAction:
        """
        Point d'entrée principal pour obtenir une décision minimax.
        
        Args:
            available_actions: Actions valides
            state_params: État actuel
            verbose: Afficher l'analyse
        
        Returns:
            L'action optimale selon minimax
        """
        

        best_action, _, _ = self.minimax_decision(available_actions, state_params, verbose)

        return best_action
    
    
    def evaluate_mixed_strategy(self, 
                                mixed_strategy: MixedStrategy, 
                                state_params: Dict) -> Tuple[float, EnvironmentCondition]:
        """
        Évalue une stratégie mixte et trouve le pire scénario environnemental.
        
        Pour chaque condition environnementale, on calcule le payoff ATTENDU 
        de la stratégie mixte, puis on trouve la condition qui donne le pire payoff attendu.
        
        Args:
            mixed_strategy: Stratégie mixte du drone (distribution de probabilité sur les actions)
            state_params: État actuel (position, batterie, etc.)
        
        Returns:
            (pire_payoff_attendu, condition_qui_cause_ce_pire)
        """
        all_env_conditions = list(EnvironmentCondition)
        worst_expected_payoff = float('inf')
        worst_condition = None
        
        # Pour chaque condition environnementale possible
        for env_condition in all_env_conditions:
            # Calculer le payoff ATTENDU de la stratégie mixte contre cette condition
            expected_payoff = 0.0
            
            # Somme pondérée sur toutes les actions de la stratégie mixte
            for action, prob in zip(mixed_strategy.strategies, mixed_strategy.probabilities):
                # Calculer le payoff pour cette action contre cette condition
                drone_payoff, _ = self.payoff.compute_payoff(
                    action,
                    env_condition,
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
                
                # Ajouter à l'espérance (payoff × probabilité)
                expected_payoff += prob * drone_payoff
            
            # Garder le pire cas (payoff attendu minimum)
            if expected_payoff < worst_expected_payoff:
                worst_expected_payoff = expected_payoff
                worst_condition = env_condition
        
        return worst_expected_payoff, worst_condition
    
    
    def solve_mixed_strategy(self,
                           drone_strategies: List[Tuple[str, MixedStrategy]],
                           env_strategies: List[Tuple[str, MixedStrategy]],
                           state_params: Dict,
                           verbose: bool = False) -> Tuple[str, MixedStrategy, float]:
        """
        Trouve la meilleure stratégie mixte du drone qui maximise le payoff garanti
        contre toutes les stratégies environnementales possibles.
        
        Args:
            drone_strategies: Liste de (nom, stratégie_mixte) pour le drone
            env_strategies: Liste de (nom, stratégie_mixte) pour l'environnement  
            state_params: État actuel
            verbose: Si True, affiche l'analyse détaillée
        
        Returns:
            (nom_meilleure_stratégie, meilleure_stratégie, payoff_garanti)
        """
        if verbose:
            print("\n" + "="*70)
            print("ANALYSE MINIMAX AVEC STRATÉGIES MIXTES")
            print("="*70)
            print(f"Nombre de stratégies drone à évaluer: {len(drone_strategies)}")
            print(f"Nombre de stratégies environnementales: {len(env_strategies)}")
        
        best_strategy_name = None
        best_strategy = None
        best_guaranteed_payoff = float('-inf')
        analysis = {}
        
        # Pour chaque stratégie mixte du drone
        for drone_name, drone_strategy in drone_strategies:
            worst_payoff_for_this_strategy = float('inf')
            worst_env_strategy_name = None
            
            if verbose:
                print(f"\n--- Évaluation: {drone_name} ---")
            
            # Contre chaque stratégie mixte de l'environnement
            for env_name, env_strategy in env_strategies:
                # Calculer le payoff attendu pour cette combinaison de stratégies mixtes
                expected_payoff = 0.0
                
                # Double somme: sur toutes les paires (action drone, condition env)
                for drone_action, drone_prob in zip(drone_strategy.strategies, drone_strategy.probabilities):
                    for env_condition, env_prob in zip(env_strategy.strategies, env_strategy.probabilities):
                        # Calculer le payoff pour cette paire
                        drone_payoff, _ = self.payoff.compute_payoff(
                            drone_action,
                            env_condition,
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
                        
                        # Ajouter à l'espérance (payoff × prob_drone × prob_env)
                        expected_payoff += drone_prob * env_prob * drone_payoff
                
                if verbose:
                    print(f"  vs {env_name:15s}: payoff attendu = {expected_payoff:7.3f}")
                
                # Garder le pire cas contre toutes les stratégies environnementales
                if expected_payoff < worst_payoff_for_this_strategy:
                    worst_payoff_for_this_strategy = expected_payoff
                    worst_env_strategy_name = env_name
            
            # Stocker l'analyse
            analysis[drone_name] = {
                'worst_payoff': worst_payoff_for_this_strategy,
                'worst_against': worst_env_strategy_name
            }
            
            if verbose:
                print(f"  → Pire cas: {worst_payoff_for_this_strategy:.3f} (contre {worst_env_strategy_name})")
            
            # Choisir la stratégie avec le meilleur pire cas (minimax)
            if worst_payoff_for_this_strategy > best_guaranteed_payoff:
                best_guaranteed_payoff = worst_payoff_for_this_strategy
                best_strategy_name = drone_name
                best_strategy = drone_strategy
        
        if verbose:
            print("\n" + "-"*70)
            print(f"DÉCISION MINIMAX (STRATÉGIES MIXTES)")
            print("-"*70)
            print(f"Meilleure stratégie: {best_strategy_name}")
            print(f"Payoff garanti: {best_guaranteed_payoff:.3f}")
            print(f"Distribution des actions:")
            for action, prob in zip(best_strategy.strategies, best_strategy.probabilities):
                print(f"  {action.value:15s}: {prob*100:5.1f}%")
            print("="*70)
        
        return best_strategy_name, best_strategy, best_guaranteed_payoff
    
    
    def evaluate_pure_action_vs_mixed_env(self,
                                          drone_action: DroneAction,
                                          env_strategy: MixedStrategy,
                                          state_params: Dict) -> float:
        """
        Évalue une action pure du drone contre une stratégie mixte de l'environnement.
        Calcule le payoff attendu en pondérant par les probabilités environnementales.
        
        Formule: E[u(a, σe)] = Σ σe(c) × u(a, c)
        
        Args:
            drone_action: Action pure du drone (ex: MOVE_UP)
            env_strategy: Stratégie mixte de l'environnement (ex: Typical: 70% CLEAR, 15% OBSTACLE...)
            state_params: État actuel (position, batterie, etc.)
        
        Returns:
            Payoff attendu (float)
        """
        expected_payoff = 0.0
        
        # Somme pondérée sur toutes les conditions environnementales
        for env_condition, env_prob in zip(env_strategy.strategies, env_strategy.probabilities):
            # Calculer le payoff pour cette paire (action pure, condition)
            drone_payoff, _ = self.payoff.compute_payoff(
                drone_action,
                env_condition,
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
            
            # Ajouter à l'espérance (payoff × probabilité)
            expected_payoff += env_prob * drone_payoff
        
        return expected_payoff
    
    
    def minimax_pure_vs_fixed_env(self,
                                  available_actions: List[DroneAction],
                                  env_strategy: MixedStrategy,
                                  env_name: str,
                                  state_params: Dict,
                                  verbose: bool = False) -> Tuple[DroneAction, float, Dict]:
        """
        Évalue les actions pures du drone contre UNE SEULE stratégie environnementale FIXE.
        Contrairement à minimax_pure_vs_mixed qui évalue contre TOUTES les stratégies,
        cette méthode utilise une distribution de probabilité environnementale constante.
        
        Usage: Quand l'environnement a une distribution fixe (ex: "Typical" tout le temps)
        
        Formule: max_a E[u(a, σe_fixe)]
        
        Args:
            available_actions: Liste d'actions pures disponibles au drone
            env_strategy: Stratégie mixte FIXE de l'environnement (ex: Typical)
            env_name: Nom de la stratégie (ex: "Typical")
            state_params: État actuel
            verbose: Si True, affiche l'analyse détaillée
        
        Returns:
            (meilleure_action, payoff_attendu, analyse_détaillée)
        """
        if not available_actions:
            raise ValueError("available_actions ne peut pas être vide")
        
        best_action = None
        best_expected_payoff = float('-inf')
        analysis = {}
        
        if verbose:
            print("\n" + "="*70)
            print(f"DÉCISION CONTRE ENVIRONNEMENT FIXE: {env_name}")
            print("="*70)
            print(f"Nombre d'actions à évaluer: {len(available_actions)}")
            print(f"Stratégie environnementale: {env_name} (FIXE)")
        
        # Pour chaque action pure du drone
        for drone_action in available_actions:
            # Calculer le payoff attendu contre la stratégie fixe
            expected_payoff = self.evaluate_pure_action_vs_mixed_env(
                drone_action,
                env_strategy,
                state_params
            )
            
            # Stocker l'analyse pour cette action
            analysis[drone_action] = {
                'expected_payoff': expected_payoff,
                'env_strategy': env_name
            }
            
            if verbose:
                print(f"  {drone_action.value:15s}: payoff attendu = {expected_payoff:7.3f}")
            
            # Choisir l'action avec le meilleur payoff attendu (maximisation)
            if expected_payoff > best_expected_payoff:
                best_expected_payoff = expected_payoff
                best_action = drone_action
        
        if verbose:
            print("\n" + "-"*70)
            print(f"MEILLEURE ACTION contre {env_name}")
            print("-"*70)
            print(f"Action choisie: {best_action.value}")
            print(f"Payoff attendu: {best_expected_payoff:.3f}")
            print("="*70)
        
        return best_action, best_expected_payoff, analysis
    
    