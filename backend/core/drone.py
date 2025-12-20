from typing import List, Tuple, Set, Dict, Optional
from backend.core.environment import Environment
from backend.game_theory.strategies import DroneAction

class Drone:
    def __init__(self, environment: Environment, battery_capacity: float = 100):
        self.environment = environment
        self.position = environment.start_pos
        self.battery_capacity = battery_capacity #capacite max
        self.current_battery = battery_capacity #capacite actuelle
        self.path = [environment.start_pos]
        self.explored_cells = {environment.start_pos}

    def move(self, action: DroneAction) -> bool:
        """
        Exécute une action et met à jour l'état du drone.
        
        Retourne True si le mouvement a réussi, False sinon.
        """
        # 1. Calculer la nouvelle position selon l'action
        new_pos = self._calculate_new_position(action)
        
        # 2. UTILISER L'ENVIRONNEMENT pour valider
        if not self.environment.is_valid_position(new_pos):
            return False  # Mouvement invalide (obstacle ou hors limites)
        
        # 3. Calculer le coût énergétique
        energy_cost = self._get_energy_cost(action)
        
        # 4. Vérifier si assez de batterie
        if self.current_battery < energy_cost:
            return False  # Pas assez de batterie
        
        # 5. Exécuter le mouvement
        self.position = new_pos
        self.current_battery -= energy_cost
        self.path.append(new_pos)
        self.explored_cells.add(new_pos)
        
        return True


    def get_valid_actions(self) -> List[DroneAction]:
        """
        Retourne les actions possibles depuis la position actuelle.
        UTILISE l'environnement pour vérifier chaque action.
        """
        valid_actions = []
        
        for action in DroneAction:
            new_pos = self._calculate_new_position(action)
            
            # CONNEXION AVEC ENVIRONNEMENT
            if self.environment.is_valid_position(new_pos):
                energy_cost = self._get_energy_cost(action)
                if self.current_battery >= energy_cost:
                    valid_actions.append(action)
        
        return valid_actions


    def _calculate_new_position(self, action: DroneAction) -> Tuple[int, int]:
        """Calcule la nouvelle position selon l'action."""
        x, y = self.position
        
        if action == DroneAction.MOVE_UP:
            return (x, y + 1)
        elif action == DroneAction.MOVE_DOWN:
            return (x, y - 1)
        elif action == DroneAction.MOVE_LEFT:
            return (x - 1, y)
        elif action == DroneAction.MOVE_RIGHT:
            return (x + 1, y)
        else:  # STAY ou ROTATE
            return (x, y)


    def _get_energy_cost(self, action: DroneAction) -> float:
        """Retourne le coût énergétique d'une action."""
        costs = {
            DroneAction.MOVE_UP: 5,
            DroneAction.MOVE_DOWN: 5,
            DroneAction.MOVE_LEFT: 5,
            DroneAction.MOVE_RIGHT: 5,
            DroneAction.STAY: 1,
            DroneAction.ROTATE: 2
        }
        return costs.get(action, 0)


    def is_alive(self) -> bool:
        """Vérifie si le drone a encore de la batterie."""
        return self.current_battery > 0

    def get_position(self) -> Tuple[int, int]:
        """Retourne la position actuelle."""
        return self.position

    def get_battery_level(self) -> float:
        """Retourne le niveau de batterie actuel."""
        return self.current_battery

    def get_battery_percentage(self) -> float:
        """Retourne le pourcentage de batterie."""
        return (self.current_battery / self.battery_capacity) * 100