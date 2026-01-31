"""
PART 1: Simulation Logger
This file handles recording and saving simulation data.
"""

import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


class SimulationLogger:
    """
    Logger class that keeps track of what happens during the simulation.
    """
    
    def __init__(self, log_name: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            log_name: Optional name for the log file
        """
        self.log_name = log_name or f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.history: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            'start_time': datetime.now().isoformat(),
            'log_name': self.log_name
        }
        
    def log_step(self, 
                step_number: int,
                action: str,
                position: tuple,
                battery_level: float,
                payoff: float,
                distance_to_goal: float,
                additional_data: Optional[Dict] = None) -> None:
        """
        Log a single simulation step.
        
        Args:
            step_number: Current step number
            action: Action taken by the drone
            position: Current position (x, y)
            battery_level: Current battery level (0-100)
            payoff: Payoff received for this action
            distance_to_goal: Current distance to goal
            additional_data: Optional dictionary of additional data to log
        """
        entry = {
            'step': step_number,
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'position': position,
            'battery_level': battery_level,
            'payoff': payoff,
            'distance_to_goal': distance_to_goal
        }
        
        if additional_data:
            entry.update(additional_data)
        
        self.history.append(entry)
    
    def log_event(self, event_type: str, description: str, data: Optional[Dict] = None) -> None:
        """
        Log a special event (e.g., collision, goal reached, battery depleted).
        
        Args:
            event_type: Type of event (e.g., 'collision', 'goal_reached')
            description: Human-readable description
            data: Optional additional data
        """
        entry = {
            'type': 'event',
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'description': description
        }
        
        if data:
            entry.update(data)
        
        self.history.append(entry)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata for the simulation.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete history log.
        
        Returns:
            List of all logged entries
        """
        return self.history.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the simulation.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.history:
            return {
                'total_steps': 0,
                'status': 'No data logged'
            }
        
        # Filter out events to get only step logs
        step_logs = [entry for entry in self.history if entry.get('type') != 'event']
        
        if not step_logs:
            return {
                'total_steps': 0,
                'events': len(self.history),
                'status': 'Only events logged'
            }
        
        total_payoff = sum(entry.get('payoff', 0) for entry in step_logs)
        avg_payoff = total_payoff / len(step_logs) if step_logs else 0
        
        battery_levels = [entry.get('battery_level', 0) for entry in step_logs if 'battery_level' in entry]
        
        return {
            'total_steps': len(step_logs),
            'total_events': len(self.history) - len(step_logs),
            'total_payoff': total_payoff,
            'average_payoff': avg_payoff,
            'final_battery': battery_levels[-1] if battery_levels else None,
            'battery_consumed': battery_levels[0] - battery_levels[-1] if len(battery_levels) > 1 else 0,
            'start_time': self.metadata.get('start_time'),
            'end_time': datetime.now().isoformat()
        }
    
    def save_to_json(self, filepath: Optional[str] = None) -> str:
        """
        Save the log to a JSON file.
        
        Args:
            filepath: Optional custom filepath. If not provided, uses default naming
            
        Returns:
            Path to the saved file
        """
        if filepath is None:
            filepath = f"logs/{self.log_name}.json"
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': self.metadata,
            'summary': self.get_summary(),
            'history': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def save_to_csv(self, filepath: Optional[str] = None) -> str:
        """
        Save the log to a CSV file.
        
        Args:
            filepath: Optional custom filepath. If not provided, uses default naming
            
        Returns:
            Path to the saved file
        """
        if filepath is None:
            filepath = f"logs/{self.log_name}.csv"
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Filter step logs only (exclude events)
        step_logs = [entry for entry in self.history if entry.get('type') != 'event']
        
        if not step_logs:
            # Save empty file or just metadata
            with open(filepath, 'w') as f:
                f.write("No step data to save\n")
            return filepath
        
        # Get all keys from the first entry to determine columns
        fieldnames = list(step_logs[0].keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(step_logs)
        
        return filepath
    
    def clear(self) -> None:
        """Clear all logged data."""
        self.history.clear()
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'log_name': self.log_name
        }
    
    def __repr__(self) -> str:
        """String representation of the logger."""
        return f"SimulationLogger({self.log_name}, steps={len(self.history)})"
    
    def __len__(self) -> int:
        """Return the number of logged entries."""
        return len(self.history)
