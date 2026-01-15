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
    
    