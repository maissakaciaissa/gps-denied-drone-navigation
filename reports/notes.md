# Drone Visual Navigation Project

## Backend Structure Overview

---

## PART 1: Environment Representation

**What it is:** The world where the drone operates

**What you need to code:**

- Grid/Map system (2D or 3D coordinates)
- Obstacles placement
- Goal/destination location
- Drone starting position

---

## PART 2: Drone Model

**What it is:** The drone's properties and capabilities

**What you need to code:**

- Current position
- Battery level
- Movement capabilities (speed, range)
- Sensor range (how far it can "see")

---

## PART 3: Game Theory Components

### Players

- **Drone** (decision-maker)
- **Environment** (nature/adversary)

### Strategies/Actions

#### DRONE STRATEGIES

**Pure strategies:**

```
{move_up, move_down, move_left, move_right, stay, rotate}
```

**Mixed strategy (σ_drone):**
Probability distribution over pure strategies

_Example:_

```
σ = (0.4·move_forward, 0.3·move_left, 0.2·move_right, 0.1·stay)
```

#### ENVIRONMENT STRATEGIES

**Pure strategies:**

```
{clear_path, obstacle_ahead, low_visibility, sensor_noise, lighting_change}
```

**Mixed strategy (σ_env):**
Probability distribution representing environmental uncertainty

_Example:_

```
σ = (0.5·clear, 0.3·obstacle, 0.2·low_vis)
```

### Payoff Function

```
u_drone(s_drone, s_env) = w1·mission_success
                        - w2·energy_consumed
                        - w3·collision_risk
                        + w4·map_quality
```

**Constraint:** `w1 + w2 + w3 + w4 = 1` (weights sum to 1)

#### Detailed Component Calculations

**mission_success:**

- `1.0` if reached goal
- `distance_to_goal / initial_distance` (partial credit)
- `0` if collision

**energy_consumed:**

- `battery_used / total_battery`
- Each move costs energy

**collision_risk:**

- `distance_to_nearest_obstacle^(-1)`
- Higher when close to obstacles

**map_quality:**

- `unexplored_area / total_area`
- Rewards exploration

#### Example Payoff Matrix (2×2)

|             | **Clear** | **Obstacle** |
| ----------- | --------- | ------------ |
| **Forward** | (10, -5)  | (-20, 5)     |
| **Stay**    | (2, 0)    | (2, 0)       |

Format: `(drone_payoff, env_payoff)`

---

## PART 4: Decision-Making Algorithms

### Algorithm 1: Minimax (Adversarial Environment)

**Assumptions:**

1. Adversarial environment (worst-case scenario)
2. Drone minimizes the maximum loss the environment can cause
3. Related to dominant strategies

**Mathematical formulation:**

```
s*_drone = arg max    min    u_drone(s_drone, s_env)
           s_drone  s_env ∈ S_env
```

**Translation:** "Choose the action that gives the best payoff even if environment picks the worst response"

### Algorithm 2: Nash Equilibrium

For finding stable strategies

### Algorithm 3: Bayesian Game

For uncertainty handling and incomplete information

---

## HYBRID STRATEGY

### Phase 1: Initial Navigation (high uncertainty)

- Use **BAYESIAN GAME** with cautious priors
- Gather sensor information

### Phase 2: Known Dangerous Areas

- Switch to **MINIMAX** (adversarial)
- Guarantee safety

### Phase 3: Open Areas

- Use **NASH EQUILIBRIUM** (mixed)
- Balance speed vs. safety optimally

### Phase 4: Near Goal

- Pure strategy (dominant)
- Direct path if clear

---

## PART 5: Simulation Engine

Main simulation loop implementation

---

## PART 6: Performance Metrics

**What it is:** Measuring how well your solution works

### Metrics Available

#### Pareto Optimality

A solution is Pareto optimal if no other solution improves one objective without worsening another

_Check:_ Can we reach goal faster WITHOUT using more battery?

#### Security Level

```
min_security_level = min    u_drone(s_drone, s_env)
                     s_env
```

Worst-case guaranteed payoff

#### Nash Equilibrium Quality

```
Efficiency = u(Nash) / u(Optimal)
```

Compare: Payoff at Nash vs. Optimal possible payoff

#### Basic Metrics

- Success rate (did it reach goal?)
- Path efficiency (shortest path vs actual path)
- Battery usage
- Collision count
- Time taken

**Example metrics object:**

```python
metrics = {
    'success': True/False,
    'path_length': 15,
    'battery_used': 45,
    'collisions': 0,
    'computation_time': 2.5  # seconds
}
```

---

## PART 7: Comparison

Compare approaches with and without game theory:

```python
results_with_game_theory = run_simulation(use_minimax=True)
results_without = run_simulation(use_minimax=False)
```

---

## Important Remarks

### Algorithm Selection Guide

| Algorithm            | Environment Type   | Use Case                                                                    |
| -------------------- | ------------------ | --------------------------------------------------------------------------- |
| **MINIMAX**          | Adversarial        | Worst-case thinking, environment actively opposes drone                     |
| **NASH EQUILIBRIUM** | Neutral/Stochastic | Realistic scenarios, environment doesn't "try" to hurt drone                |
| **BAYESIAN**         | Unknown type       | Incomplete information, drone doesn't know true environment state initially |

---

## Project File Structure

```
backend/
│
├── core/
│   ├── environment.py      # PART 1: Grid, obstacles
│   ├── drone.py            # PART 2: Drone model
│   └── sensor.py           # Drone vision simulation
│
├── game_theory/
│   ├── payoff.py           # PART 3C: Payoff function
│   ├── minimax.py          # PART 4: Minimax algorithm
│   ├── nash.py             # PART 4: Nash equilibrium
│   └── strategies.py       # PART 3B: All strategies
│
├── simulation/
│   ├── engine.py           # PART 5: Main simulation loop
│   ├── metrics.py          # PART 6: Performance tracking
│   └── logger.py           # Data logging
│
├── evaluation/
│   └── compare.py          # PART 7: Compare approaches
│
└── main.py                 # Run everything
```
