"""
PART 6: Algorithm Comparator
Compares all three algorithms (Minimax, Nash, Bayesian) against each other.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from backend.core.environment import Environment
from backend.core.drone import Drone
from backend.game_theory.payoff import PayoffFunction
from backend.simulation.engine import SimulationEngine
from backend.simulation.metrics import PerformanceMetrics


class AlgorithmComparator:
    """
    Compares all three algorithms against each other.
    Runs multiple trials and performs statistical analysis.
    """
    
    def __init__(self, environment: Environment, drone: Drone, payoff_function: PayoffFunction):
        """
        Initialize the algorithm comparator.
        
        Args:
            environment: Environment instance to test in
            drone: Drone instance to use
            payoff_function: PayoffFunction instance
        """
        self.environment = environment
        self.drone = drone
        self.payoff_function = payoff_function
        
        # Storage for results
        self.results = {
            'minimax': [],
            'nash': [],
            'bayesian': []
        }
        
        # Create engine for running simulations
        self.engine = SimulationEngine(environment, drone, payoff_function)
        
        # Metrics calculator
        self.metrics = PerformanceMetrics(environment, drone)
    
    @staticmethod
    def compare_algorithms(results: List[Dict]) -> Dict:
        """
        Compare multiple algorithm results and compute statistics.
        
        Args:
            results: List of result dictionaries from different algorithms
            
        Returns:
            Dictionary with statistical comparisons
        """
        if not results:
            return {}
        
        # Metrics to compare
        metrics_to_compare = [
            'path_length',
            'battery_used_percent',
            'collisions',
            'cells_explored',
            'exploration_rate',
            'path_efficiency',
            'computation_time',
            'final_distance_to_goal'
        ]
        
        comparison = {}
        
        # Calculate statistics for each metric
        for metric in metrics_to_compare:
            values = [r.get(metric, 0) for r in results if metric in r]
            
            if values:
                comparison[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Success rate
        successes = [r.get('success', False) for r in results]
        comparison['success_rate'] = sum(successes) / len(successes) if successes else 0.0
        
        # Best algorithm for each metric
        comparison['best_by_metric'] = {}
        
        for metric in ['path_length', 'battery_used_percent', 'collisions', 'computation_time']:
            if any(metric in r for r in results):
                best_result = min(results, key=lambda x: x.get(metric, float('inf')))
                comparison['best_by_metric'][metric] = best_result.get('algorithm', 'Unknown')
        
        for metric in ['path_efficiency', 'exploration_rate']:
            if any(metric in r for r in results):
                best_result = max(results, key=lambda x: x.get(metric, 0))
                comparison['best_by_metric'][metric] = best_result.get('algorithm', 'Unknown')
        
        return comparison
    
    def run_all_algorithms(self, 
        trials: int = 10, max_steps: int = 1000, verbose: bool = False
    ) -> Dict[str, List[Dict]]:
        """
        Run all three algorithms multiple times and store results.
        
        Args:
            trials: Number of trials to run for each algorithm
            max_steps: Maximum steps per simulation
            verbose: If True, print progress
            
        Returns:
            Dictionary with algorithm names as keys and lists of results as values
        """
        algorithms = ['minimax', 'nash', 'bayesian']
        
        if verbose:
            print("="*70)
            print("RUNNING ALGORITHM COMPARISON")
            print("="*70)
            print(f"Trials per algorithm: {trials}")
            print(f"Max steps per trial: {max_steps}")
            print()
        
        for algorithm in algorithms:
            if verbose:
                print(f"Running {algorithm.upper()}...")
            
            for trial in range(trials):
                # Reset for each trial
                self.engine.reset()
                
                # Run simulation
                result = self.engine.run_simulation(
                    max_steps=max_steps,
                    algorithm=algorithm,
                    verbose=False
                )
                
                # Store result
                self.results[algorithm].append(result)
                
                if verbose:
                    success_str = "[OK]" if result['success'] else "[FAIL]"
                    print(
                        f"  Trial {trial + 1}/{trials}: {success_str} "
                        f"Steps: {result['path_length']}, "
                        f"Battery: {result['battery_used_percent']:.1f}%"
                    )
            
            if verbose:
                print()
        
        if verbose:
            print("="*70)
            print("COMPARISON COMPLETE")
            print("="*70)
            print()
        
        return self.results
    
    def compare_metrics(self) -> Dict:
        """
        Perform statistical analysis on the results.
        
        Calculates mean, standard deviation, min, max for each metric across algorithms.
        
        Returns:
            Dictionary with statistical comparisons for each algorithm
        """
        if not any(self.results.values()):
            raise ValueError("No results to compare. Run run_all_algorithms() first.")
        
        comparison = {}
        
        for algorithm, results in self.results.items():
            if results:
                # Use the compare_algorithms static method
                stats = self.compare_algorithms(results)
                comparison[algorithm] = stats
        
        return comparison
    
    def visualize_comparison(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Create visualization plots comparing algorithm performance.
        
        Generates 4 subplots:
        1. Success Rate comparison (bar chart)
        2. Path Length comparison (box plot)
        3. Battery Usage comparison (box plot)
        4. Computation Time comparison (box plot)
        
        Args:
            save_path: Optional path to save the figure. If None, figure is not saved.
            show: Whether to display the plot. Default is True.
        """
        if not any(self.results.values()):
            print("No results available. Run run_all_algorithms() first.")
            return
        
        # Prepare data
        algorithms = ['minimax', 'nash', 'bayesian']
        colors = {'minimax': '#FF6B6B', 'nash': '#4ECDC4', 'bayesian': '#95E1D3'}
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Success Rate (Bar Chart)
        ax1 = axes[0, 0]
        success_rates = []
        for alg in algorithms:
            if self.results[alg]:
                successes = [r.get('success', False) for r in self.results[alg]]
                success_rates.append(sum(successes) / len(successes) * 100)
            else:
                success_rates.append(0)
        
        bars = ax1.bar(algorithms, success_rates, color=[colors[alg] for alg in algorithms])
        ax1.set_ylabel('Success Rate (%)', fontweight='bold')
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Path Length (Box Plot)
        ax2 = axes[0, 1]
        path_data = []
        for alg in algorithms:
            if self.results[alg]:
                paths = [r.get('path_length', 0) for r in self.results[alg]]
                path_data.append(paths)
            else:
                path_data.append([0])
        
        bp1 = ax2.boxplot(path_data, labels=algorithms, patch_artist=True, widths=0.7)
        for patch, alg in zip(bp1['boxes'], algorithms):
            patch.set_facecolor(colors[alg])
            patch.set_alpha(0.8)
            patch.set_linewidth(2.5)
        # Make all elements thicker
        for element in ['whiskers', 'fliers', 'means', 'caps']:
            if element in bp1:
                plt.setp(bp1[element], linewidth=2)
        plt.setp(bp1['medians'], linewidth=3, color='darkorange')
        ax2.set_ylabel('Path Length (steps)', fontweight='bold', fontsize=11)
        ax2.set_title('Path Length Distribution', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        # Better y-axis scaling with minimum range
        all_path_values = [v for sublist in path_data for v in sublist]
        if all_path_values:
            y_min, y_max = min(all_path_values), max(all_path_values)
            y_range = max(y_max - y_min, 5)  # Minimum range of 5
            ax2.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)
        
        # 3. Battery Usage (Box Plot)
        ax3 = axes[1, 0]
        battery_data = []
        for alg in algorithms:
            if self.results[alg]:
                battery = [r.get('battery_used_percent', 0) for r in self.results[alg]]
                battery_data.append(battery)
            else:
                battery_data.append([0])
        
        bp2 = ax3.boxplot(battery_data, labels=algorithms, patch_artist=True, widths=0.7)
        for patch, alg in zip(bp2['boxes'], algorithms):
            patch.set_facecolor(colors[alg])
            patch.set_alpha(0.8)
            patch.set_linewidth(2.5)
        for element in ['whiskers', 'fliers', 'means', 'caps']:
            if element in bp2:
                plt.setp(bp2[element], linewidth=2)
        plt.setp(bp2['medians'], linewidth=3, color='darkorange')
        ax3.set_ylabel('Battery Used (%)', fontweight='bold', fontsize=11)
        ax3.set_title('Battery Usage Distribution', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        # Better y-axis with minimum range
        all_battery_values = [v for sublist in battery_data for v in sublist]
        if all_battery_values:
            y_min, y_max = min(all_battery_values), max(all_battery_values)
            y_range = max(y_max - y_min, 3)  # Minimum range of 3
            ax3.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)
        
        # 4. Computation Time (Box Plot)
        ax4 = axes[1, 1]
        time_data = []
        for alg in algorithms:
            if self.results[alg]:
                times = [r.get('computation_time', 0) for r in self.results[alg]]
                time_data.append(times)
            else:
                time_data.append([0])
        
        bp3 = ax4.boxplot(time_data, labels=algorithms, patch_artist=True, widths=0.7)
        for patch, alg in zip(bp3['boxes'], algorithms):
            patch.set_facecolor(colors[alg])
            patch.set_alpha(0.8)
            patch.set_linewidth(2.5)
        for element in ['whiskers', 'fliers', 'means', 'caps']:
            if element in bp3:
                plt.setp(bp3[element], linewidth=2)
        plt.setp(bp3['medians'], linewidth=3, color='darkorange')
        ax4.set_ylabel('Computation Time (seconds)', fontweight='bold', fontsize=11)
        ax4.set_title('Computation Time Distribution', fontsize=12)
        ax4.grid(axis='y', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        # Show plot
        if show:
            plt.show()
        else:
            plt.close()

    def generate_comparison_report(self) -> str:
        """
        Generate a comprehensive comparison report.
        
        Answers: Which algorithm was best overall? Which was fastest?
        Which used least battery? Which was safest?
        
        Returns:
            Formatted string report
        """
        if not any(self.results.values()):
            return "No results available. Run run_all_algorithms() first."
        
        comparison = self.compare_metrics()
        
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("ALGORITHM COMPARISON REPORT")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Summary for each algorithm
        for algorithm in ['minimax', 'nash', 'bayesian']:
            if algorithm in comparison and comparison[algorithm]:
                stats = comparison[algorithm]
                report_lines.append(f"--- {algorithm.upper()} ---")
                report_lines.append(f"Success Rate: {stats.get('success_rate', 0):.1%}")
                
                if 'path_length' in stats:
                    pl = stats['path_length']
                    report_lines.append(
                        f"Path Length: {pl['mean']:.1f} ± {pl['std']:.1f} "
                        f"(min: {pl['min']:.0f}, max: {pl['max']:.0f})"
                    )
                
                if 'battery_used_percent' in stats:
                    bat = stats['battery_used_percent']
                    report_lines.append(
                        f"Battery Used: {bat['mean']:.1f}% ± {bat['std']:.1f}% "
                        f"(min: {bat['min']:.1f}%, max: {bat['max']:.1f}%)"
                    )
                
                if 'collisions' in stats:
                    col = stats['collisions']
                    report_lines.append(
                        f"Collisions: {col['mean']:.1f} ± {col['std']:.1f} "
                        f"(min: {col['min']:.0f}, max: {col['max']:.0f})"
                    )
                
                if 'computation_time' in stats:
                    time = stats['computation_time']
                    report_lines.append(
                        f"Computation Time: {time['mean']:.3f}s ± {time['std']:.3f}s "
                        f"(min: {time['min']:.3f}s, max: {time['max']:.3f}s)"
                    )
                
                if 'path_efficiency' in stats:
                    eff = stats['path_efficiency']
                    report_lines.append(f"Path Efficiency: {eff['mean']:.1%} ± {eff['std']:.1%}")
                
                report_lines.append("")
        
        # Best by category
        report_lines.append("="*70)
        report_lines.append("WINNERS BY CATEGORY")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Best success rate
        best_success = max(
            comparison.items(), 
            key=lambda x: x[1].get('success_rate', 0)
        )
        report_lines.append(
            f"[WINNER] Best Success Rate: {best_success[0].upper()} "
            f"({best_success[1].get('success_rate', 0):.1%})"
        )
        
        # Shortest path (lowest mean)
        path_algorithms = {
            alg: stats.get('path_length', {}).get('mean', float('inf'))
            for alg, stats in comparison.items()
        }
        best_path = min(path_algorithms.items(), key=lambda x: x[1])
        if best_path[1] != float('inf'):
            report_lines.append(
                f"[WINNER] Shortest Path: {best_path[0].upper()} "
                f"({best_path[1]:.1f} steps)"
            )
        
        # Least battery (lowest mean)
        battery_algorithms = {
            alg: stats.get('battery_used_percent', {}).get('mean', float('inf'))
            for alg, stats in comparison.items()
        }
        best_battery = min(battery_algorithms.items(), key=lambda x: x[1])
        if best_battery[1] != float('inf'):
            report_lines.append(
                f"[WINNER] Least Battery Usage: {best_battery[0].upper()} "
                f"({best_battery[1]:.1f}%)"
            )
        
        # Fastest (lowest computation time)
        time_algorithms = {
            alg: stats.get('computation_time', {}).get('mean', float('inf'))
            for alg, stats in comparison.items()
        }
        fastest = min(time_algorithms.items(), key=lambda x: x[1])
        if fastest[1] != float('inf'):
            report_lines.append(
                f"[WINNER] Fastest Computation: {fastest[0].upper()} "
                f"({fastest[1]:.3f}s)"
            )
        
        # Safest (fewest collisions)
        collision_algorithms = {
            alg: stats.get('collisions', {}).get('mean', float('inf'))
            for alg, stats in comparison.items()
        }
        safest = min(collision_algorithms.items(), key=lambda x: x[1])
        if safest[1] != float('inf'):
            report_lines.append(
                f"[WINNER] Safest (Fewest Collisions): {safest[0].upper()} "
                f"({safest[1]:.1f} collisions)"
            )
        
        # Most efficient path
        efficiency_algorithms = {
            alg: stats.get('path_efficiency', {}).get('mean', 0)
            for alg, stats in comparison.items()
        }
        most_efficient = max(efficiency_algorithms.items(), key=lambda x: x[1])
        if most_efficient[1] > 0:
            report_lines.append(
                f"[WINNER] Most Efficient Path: {most_efficient[0].upper()} "
                f"({most_efficient[1]:.1%} efficiency)"
            )
        
        report_lines.append("")
        report_lines.append("="*70)
        
        # Overall winner (based on success rate as primary metric)
        report_lines.append("")
        report_lines.append(f"*** OVERALL WINNER: {best_success[0].upper()} ***")
        report_lines.append(f"    (Highest success rate: {best_success[1].get('success_rate', 0):.1%})")
        report_lines.append("")
        report_lines.append("="*70)
        
        return "\n".join(report_lines)
    
    def get_results(self) -> Dict[str, List[Dict]]:
        """
        Get all stored results.
        
        Returns:
            Dictionary with algorithm names as keys and lists of results as values
        """
        return self.results
    
    def reset_results(self):
        """
        Clear all stored results.
        """
        self.results = {
            'minimax': [],
            'nash': [],
            'bayesian': []
        }
        
    def __repr__(self) -> str:
        """String representation of the comparator."""
        trial_counts = {alg: len(results) for alg, results in self.results.items()}
        return f"AlgorithmComparator(trials={trial_counts})"
