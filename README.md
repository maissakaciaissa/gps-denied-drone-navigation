# GPS-Denied Drone Navigation - PART 1

Game-theoretic approach to drone navigation in GPS-denied environments.

## 🚀 Quick Start

```bash
cd gps-denied-drone-navigation
python backend/main.py
```

## ✅ Implemented Components

### 1. **Environment** (`backend/core/environment.py`)

- 20×20 grid system with obstacles
- Position validation and distance calculations
- Goal detection

### 2. **Strategies** (`backend/game_theory/strategies.py`)

- **Drone:** MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ROTATE
- **Environment:** CLEAR_PATH, OBSTACLE_AHEAD, LOW_VISIBILITY, SENSOR_NOISE, LIGHTING_CHANGE
- Mixed strategies: Cautious, Aggressive, Balanced, Typical, Adversarial, Favorable

### 3. **Payoff Function** (`backend/game_theory/payoff.py`)

- Returns `(drone_payoff, env_payoff)` tuples
- Four components: mission success, energy, collision risk, map quality
- Action-specific costs: MOVE=5, STAY=1, ROTATE=2
- Directional awareness: 70% penalty for moving away from goal
- Intelligent collision multipliers

### 4. **Logger** (`backend/simulation/logger.py`)

- Step logging: action, position, battery, payoff
- Event logging with metadata
- Export to JSON and CSV

### 5. **Simulation** (`backend/simulation/simulation.py`)

- Pure strategy testing (150 combinations)
- Mixed strategy testing (125 combinations)
- Full payoff matrix display
- Safe probability lookup

## 📊 Results

Successfully tested **275 strategy combinations** with verified correctness:

- Best: Cautious vs Favorable (+2.25 payoff)
- Worst: Custom_Exploration vs Custom_DangerZone (-0.60 payoff)
