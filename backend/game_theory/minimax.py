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
                                  verbose: bool = False,
                                  sensor_distribution: Optional[Dict[EnvironmentCondition, float]] = None) -> Tuple[DroneAction, float, Dict]:
        """
        Évalue les actions pures du drone contre UNE stratégie environnementale.
        
        NOUVEAU: Peut maintenant utiliser la distribution des capteurs !
        - Si sensor_distribution est fourni: utilise les probabilités détectées par les capteurs
        - Sinon: utilise env_strategy (stratégie prédéfinie)
        
        Usage avec capteurs:
            sensor_dist = drone_sensor.sense_environment_condition(env, position)
            action, payoff, _ = minimax.minimax_pure_vs_fixed_env(
                actions, None, "Détecté", state_params, 
                sensor_distribution=sensor_dist
            )
        
        Args:
            available_actions: Actions pures disponibles
            env_strategy: Stratégie mixte de l'environnement (ignoré si sensor_distribution fourni)
            env_name: Nom de la stratégie (ex: "Typical" ou "Détecté par capteurs")
            state_params: État actuel
            verbose: Affichage détaillé
            sensor_distribution: Distribution {EnvironmentCondition: probability} des capteurs (optionnel)
        
        Returns:
            (meilleure_action, payoff_attendu, analyse)
        """
        if not available_actions:
            raise ValueError("available_actions ne peut pas être vide")
        
        # 🔍 NOUVEAU: Choisir la source de probabilités
        if sensor_distribution is not None:
            # Utiliser les capteurs
            use_sensor = True
            # Vérifier que c'est une distribution valide
            total_prob = sum(sensor_distribution.values())
            if not (0.99 <= total_prob <= 1.01):
                raise ValueError(f"sensor_distribution doit sommer à 1.0 (actuel: {total_prob})")
        else:
            # Utiliser la stratégie prédéfinie
            use_sensor = False
            if env_strategy is None:
                raise ValueError("env_strategy ou sensor_distribution doit être fourni")
        
        best_action = None
        best_expected_payoff = float('-inf')
        analysis = {}
        
        if verbose:
            print("\n" + "="*70)
            if use_sensor:
                print(f"DÉCISION AVEC CAPTEURS: {env_name}")
                print("="*70)
                print("Source: Distribution détectée par les capteurs du drone")
                print("\nProbabilités détectées:")
                for condition, prob in sensor_distribution.items():
                    if prob > 0.0:
                        print(f"  {condition.value:25s}: {prob*100:5.1f}%")
            else:
                print(f"DÉCISION CONTRE ENVIRONNEMENT FIXE: {env_name}")
                print("="*70)
                print("Source: Stratégie environnementale prédéfinie")
            print(f"\nNombre d'actions à évaluer: {len(available_actions)}")
        
        # Pour chaque action pure du drone
        for drone_action in available_actions:
            expected_payoff = 0.0
            action_details = {}
            
            # 🔍 Calculer le payoff attendu selon la source
            if use_sensor:
                # Utiliser les probabilités des capteurs
                for env_condition, probability in sensor_distribution.items():
                    if probability > 0.0:
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
                        expected_payoff += probability * drone_payoff
                        
                        action_details[env_condition] = {
                            'payoff': drone_payoff,
                            'probability': probability,
                            'contribution': probability * drone_payoff
                        }
            else:
                # Utiliser la stratégie prédéfinie (code existant)
                expected_payoff = self.evaluate_pure_action_vs_mixed_env(
                    drone_action,
                    env_strategy,
                    state_params
                )
            
            # Stocker l'analyse
            analysis[drone_action] = {
                'expected_payoff': expected_payoff,
                'env_strategy': env_name,
                'source': 'sensor' if use_sensor else 'predefined',
                'details': action_details if use_sensor else None
            }
            
            if verbose:
                print(f"  {drone_action.value:15s}: payoff attendu = {expected_payoff:7.3f}")
                if use_sensor and action_details:
                    for condition, details in action_details.items():
                        print(f"    └─ {condition.value:20s}: {details['payoff']:6.2f} × {details['probability']:.2f} = {details['contribution']:6.2f}")
            
            # Choisir la meilleure action
            if expected_payoff > best_expected_payoff:
                best_expected_payoff = expected_payoff
                best_action = drone_action
        
        if verbose:
            print("\n" + "-"*70)
            print(f"MEILLEURE ACTION {'(détectée par capteurs)' if use_sensor else f'contre {env_name}'}")
            print("-"*70)
            print(f"Action choisie: {best_action.value}")
            print(f"Payoff attendu: {best_expected_payoff:.3f}")
            print("="*70)
        
        return best_action, best_expected_payoff, analysis
    
    
    def evaluate_pure_action_vs_mixed_env(self,
                                          drone_action: DroneAction,
                                          env_strategy: MixedStrategy,
                                          state_params: Dict) -> float:
        """
        Calcule le payoff attendu d'une action pure contre une stratégie mixte.
        """
        expected_payoff = 0.0
        
        for env_condition, probability in zip(env_strategy.strategies, env_strategy.probabilities):
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
            expected_payoff += probability * drone_payoff
        
        return expected_payoff
