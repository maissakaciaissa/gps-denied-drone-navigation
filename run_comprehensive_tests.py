"""
Comprehensive Algorithm Comparison Tests
Runs multiple scenarios with different environment configurations
Generates detailed reports and visualizations for the LaTeX document
"""

import numpy as np
import matplotlib.pyplot as plt
from backend.core.environment import Environment
from backend.core.drone import Drone
from backend.game_theory.payoff import PayoffFunction
from backend.evaluation.compare import AlgorithmComparator
import time
import json


def create_test_scenarios():
    """Create different test scenarios for comprehensive comparison"""
    scenarios = []
    
    # Scenario 1: Simple environment (sparse obstacles)
    scenarios.append({
        'name': 'Simple Environment',
        'description': 'Sparse obstacles, easy navigation',
        'width': 20,
        'height': 20,
        'start_pos': (2, 2),
        'goal_pos': (17, 17),
        'obstacles': [(7, 7), (8, 7), (7, 8), (12, 12), (13, 12)],
        'battery': 200
    })
    
    # Scenario 2: Medium complexity (moderate obstacles)
    scenarios.append({
        'name': 'Medium Complexity',
        'description': 'Moderate obstacle density',
        'width': 25,
        'height': 25,
        'start_pos': (2, 2),
        'goal_pos': (22, 22),
        'obstacles': [
            (8, 8), (9, 8), (10, 8), (11, 8),
            (11, 9), (11, 10), (11, 11),
            (15, 15), (16, 15), (15, 16),
            (5, 12), (6, 12), (7, 12),
            (18, 8), (18, 9), (18, 10)
        ],
        'battery': 300
    })
    
    # Scenario 3: High complexity (dense obstacles)
    scenarios.append({
        'name': 'High Complexity',
        'description': 'Dense obstacles, challenging navigation',
        'width': 30,
        'height': 30,
        'start_pos': (2, 2),
        'goal_pos': (27, 27),
        'obstacles': [
            # Vertical wall
            (10, i) for i in range(5, 20)
        ] + [
            # Horizontal wall
            (i, 20) for i in range(11, 25)
        ] + [
            # Random obstacles
            (5, 10), (6, 10), (7, 10),
            (20, 8), (20, 9), (20, 10),
            (15, 15), (16, 16), (17, 17),
            (8, 25), (9, 25), (10, 25)
        ],
        'battery': 400
    })
    
    # Scenario 4: Narrow passage
    scenarios.append({
        'name': 'Narrow Passage',
        'description': 'Requires finding a narrow passage through obstacles',
        'width': 20,
        'height': 20,
        'start_pos': (2, 10),
        'goal_pos': (17, 10),
        'obstacles': [
            # Create walls with small gap
            (10, i) for i in range(0, 8)
        ] + [
            (10, i) for i in range(12, 20)
        ] + [
            (11, i) for i in range(0, 8)
        ] + [
            (11, i) for i in range(12, 20)
        ],
        'battery': 250
    })
    
    return scenarios


def run_scenario_tests(scenario, trials=20, max_steps=500):
    """Run comparison tests for a specific scenario"""
    print(f"\n{'='*70}")
    print(f"TESTING SCENARIO: {scenario['name']}")
    print(f"{'='*70}")
    print(f"Description: {scenario['description']}")
    print(f"Environment: {scenario['width']}x{scenario['height']}")
    print(f"Obstacles: {len(scenario['obstacles'])}")
    print(f"Trials per algorithm: {trials}")
    print(f"Max steps: {max_steps}")
    
    # Create environment
    env = Environment(
        width=scenario['width'],
        height=scenario['height'],
        start_pos=scenario['start_pos'],
        goal_pos=scenario['goal_pos']
    )
    env.add_obstacles(scenario['obstacles'])
    
    # Create drone and payoff function
    drone = Drone(environment=env, battery_capacity=scenario['battery'])
    payoff_func = PayoffFunction()
    
    # Create comparator
    comparator = AlgorithmComparator(env, drone, payoff_func)
    
    # Run comparison
    start_time = time.time()
    results = comparator.run_all_algorithms(
        trials=trials,
        max_steps=max_steps,
        verbose=True
    )
    total_time = time.time() - start_time
    
    # Generate comparison
    comparison = comparator.compare_metrics()
    report = comparator.generate_comparison_report()
    
    # Save visualization
    viz_path = f"reports/comparison_{scenario['name'].replace(' ', '_').lower()}.png"
    comparator.visualize_comparison(save_path=viz_path, show=False)
    
    print(f"\n✅ Scenario completed in {total_time:.2f}s")
    print(f"📊 Visualization saved to: {viz_path}")
    
    return {
        'scenario': scenario,
        'results': results,
        'comparison': comparison,
        'report': report,
        'total_time': total_time,
        'visualization_path': viz_path
    }


def generate_summary_statistics(all_scenario_results):
    """Generate overall summary statistics across all scenarios"""
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY ACROSS ALL SCENARIOS")
    print(f"{'='*70}\n")
    
    algorithms = ['minimax', 'nash', 'bayesian']
    
    # Aggregate statistics
    summary = {alg: {
        'total_trials': 0,
        'total_successes': 0,
        'avg_path_length': [],
        'avg_battery': [],
        'avg_time': [],
        'avg_collisions': []
    } for alg in algorithms}
    
    for scenario_result in all_scenario_results:
        comparison = scenario_result['comparison']
        for alg in algorithms:
            if alg in comparison:
                stats = comparison[alg]
                results = scenario_result['results'][alg]
                
                summary[alg]['total_trials'] += len(results)
                summary[alg]['total_successes'] += sum(1 for r in results if r.get('success', False))
                
                if 'path_length' in stats:
                    summary[alg]['avg_path_length'].append(stats['path_length']['mean'])
                if 'battery_used_percent' in stats:
                    summary[alg]['avg_battery'].append(stats['battery_used_percent']['mean'])
                if 'computation_time' in stats:
                    summary[alg]['avg_time'].append(stats['computation_time']['mean'])
                if 'collisions' in stats:
                    summary[alg]['avg_collisions'].append(stats['collisions']['mean'])
    
    # Print summary
    print("Algorithm Performance Summary:")
    print("-" * 70)
    
    for alg in algorithms:
        s = summary[alg]
        success_rate = (s['total_successes'] / s['total_trials'] * 100) if s['total_trials'] > 0 else 0
        
        print(f"\n{alg.upper()}:")
        print(f"  Overall Success Rate: {success_rate:.1f}% ({s['total_successes']}/{s['total_trials']})")
        
        if s['avg_path_length']:
            print(f"  Avg Path Length: {np.mean(s['avg_path_length']):.1f} steps")
        if s['avg_battery']:
            print(f"  Avg Battery Usage: {np.mean(s['avg_battery']):.1f}%")
        if s['avg_time']:
            print(f"  Avg Computation Time: {np.mean(s['avg_time']):.4f}s")
        if s['avg_collisions']:
            print(f"  Avg Collisions: {np.mean(s['avg_collisions']):.2f}")
    
    return summary


def create_aggregate_comparison_plot(all_scenario_results):
    """Create aggregate comparison plots across all scenarios"""
    scenarios_names = [r['scenario']['name'] for r in all_scenario_results]
    algorithms = ['minimax', 'nash', 'bayesian']
    colors = {'minimax': '#FF6B6B', 'nash': '#4ECDC4', 'bayesian': '#95E1D3'}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Algorithm Performance Across All Scenarios', fontsize=16, fontweight='bold')
    
    # 1. Success Rate by Scenario
    ax1 = axes[0, 0]
    x = np.arange(len(scenarios_names))
    width = 0.25
    
    for i, alg in enumerate(algorithms):
        success_rates = []
        for result in all_scenario_results:
            results = result['results'][alg]
            successes = sum(1 for r in results if r.get('success', False))
            success_rates.append(successes / len(results) * 100 if results else 0)
        
        ax1.bar(x + i * width, success_rates, width, label=alg.upper(), color=colors[alg])
    
    ax1.set_ylabel('Success Rate (%)', fontweight='bold')
    ax1.set_title('Success Rate by Scenario')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(scenarios_names, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 110)
    
    # 2. Average Path Length by Scenario
    ax2 = axes[0, 1]
    for i, alg in enumerate(algorithms):
        path_lengths = []
        for result in all_scenario_results:
            comparison = result['comparison']
            if alg in comparison and 'path_length' in comparison[alg]:
                path_lengths.append(comparison[alg]['path_length']['mean'])
            else:
                path_lengths.append(0)
        
        ax2.bar(x + i * width, path_lengths, width, label=alg.upper(), color=colors[alg])
    
    ax2.set_ylabel('Path Length (steps)', fontweight='bold')
    ax2.set_title('Average Path Length by Scenario')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(scenarios_names, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Average Battery Usage by Scenario
    ax3 = axes[1, 0]
    for i, alg in enumerate(algorithms):
        battery_usage = []
        for result in all_scenario_results:
            comparison = result['comparison']
            if alg in comparison and 'battery_used_percent' in comparison[alg]:
                battery_usage.append(comparison[alg]['battery_used_percent']['mean'])
            else:
                battery_usage.append(0)
        
        ax3.bar(x + i * width, battery_usage, width, label=alg.upper(), color=colors[alg])
    
    ax3.set_ylabel('Battery Usage (%)', fontweight='bold')
    ax3.set_title('Average Battery Usage by Scenario')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(scenarios_names, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Average Computation Time by Scenario
    ax4 = axes[1, 1]
    for i, alg in enumerate(algorithms):
        comp_times = []
        for result in all_scenario_results:
            comparison = result['comparison']
            if alg in comparison and 'computation_time' in comparison[alg]:
                comp_times.append(comparison[alg]['computation_time']['mean'])
            else:
                comp_times.append(0)
        
        ax4.bar(x + i * width, comp_times, width, label=alg.upper(), color=colors[alg])
    
    ax4.set_ylabel('Computation Time (s)', fontweight='bold')
    ax4.set_title('Average Computation Time by Scenario')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(scenarios_names, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = 'reports/aggregate_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Aggregate comparison plot saved to: {save_path}")
    return save_path


def save_results_to_file(all_scenario_results, summary):
    """Save results to JSON and text files for LaTeX report"""
    
    # Save JSON data
    json_data = {
        'scenarios': [],
        'summary': summary
    }
    
    for result in all_scenario_results:
        scenario_data = {
            'name': result['scenario']['name'],
            'description': result['scenario']['description'],
            'comparison': {}
        }
        
        for alg, stats in result['comparison'].items():
            # Convert to serializable format
            scenario_data['comparison'][alg] = {
                'success_rate': stats.get('success_rate', 0),
                'path_length_mean': stats.get('path_length', {}).get('mean', 0),
                'path_length_std': stats.get('path_length', {}).get('std', 0),
                'battery_mean': stats.get('battery_used_percent', {}).get('mean', 0),
                'battery_std': stats.get('battery_used_percent', {}).get('std', 0),
                'computation_time_mean': stats.get('computation_time', {}).get('mean', 0),
                'computation_time_std': stats.get('computation_time', {}).get('std', 0),
                'collisions_mean': stats.get('collisions', {}).get('mean', 0)
            }
        
        json_data['scenarios'].append(scenario_data)
    
    with open('reports/test_results.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print("💾 Results saved to: reports/test_results.json")
    
    # Save detailed text report
    with open('reports/detailed_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE ALGORITHM COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        
        for result in all_scenario_results:
            f.write(result['report'])
            f.write("\n\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        for alg, stats in summary.items():
            f.write(f"{alg.upper()}:\n")
            f.write(f"  Total Trials: {stats['total_trials']}\n")
            f.write(f"  Total Successes: {stats['total_successes']}\n")
            if stats['avg_path_length']:
                f.write(f"  Avg Path Length: {np.mean(stats['avg_path_length']):.1f}\n")
            if stats['avg_battery']:
                f.write(f"  Avg Battery: {np.mean(stats['avg_battery']):.1f}%\n")
            if stats['avg_time']:
                f.write(f"  Avg Time: {np.mean(stats['avg_time']):.4f}s\n")
            f.write("\n")
    
    print("💾 Detailed report saved to: reports/detailed_report.txt")


def main():
    """Main function to run comprehensive tests"""
    print("\n" + "="*70)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("Testing GPS-Denied Drone Navigation Algorithms")
    print("="*70)
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    print(f"\nTotal scenarios to test: {len(scenarios)}")
    
    # Run tests for each scenario
    all_results = []
    for scenario in scenarios:
        result = run_scenario_tests(scenario, trials=20, max_steps=500)
        all_results.append(result)
    
    # Generate summary
    summary = generate_summary_statistics(all_results)
    
    # Create aggregate plots
    create_aggregate_comparison_plot(all_results)
    
    # Save results
    save_results_to_file(all_results, summary)
    
    print("\n" + "="*70)
    print("✅ ALL COMPREHENSIVE TESTS COMPLETED")
    print("="*70)
    print("\nGenerated files for LaTeX report:")
    print("  - reports/comparison_*.png (individual scenario plots)")
    print("  - reports/aggregate_comparison.png (overall comparison)")
    print("  - reports/test_results.json (numerical data)")
    print("  - reports/detailed_report.txt (text report)")
    print("\nYou can now use these results in your LaTeX document!")


if __name__ == "__main__":
    main()
