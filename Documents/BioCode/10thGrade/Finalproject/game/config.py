# conditions for sim
GRID_SIZE = 4
MAX_TURNS = 500

# -------------------- Predator params -----------------------
PREDATOR_REPRODUCTION_THRESHOLD = 3
PREDATOR_STARVATION_LIMIT = 75
PREDATOR_HUNTING_REST = 2
PREDATOR_PROXIMITY_SCALE = 0.1          # reward per cell you close toward nearest prey

# -------------------- Combat & advantage -----------------------
# 0  --> no advantage at all (pure 50-50 coin-flip funny)
# 1  --> slight advantage
# 2+ --> strong advantage
PREDATOR_BASE_ADVANTAGE = 1


# -------------------- Prey params -----------------------
PREY_STARVATION_TICKS = 60
PREY_START_ENERGY = 3
PREY_REPRODUCTION_THRESHOLD = 2
PREY_EATING_REST = 2

# General params
AGGRESSION_THRESHOLD = 1.25
STEP_LIVING_COST = 0.01

# Grid rendering settings (for the renderer)
CELL_SIZE = 100
MARGIN = 2
GRID_WIDTH = GRID_SIZE * CELL_SIZE + (GRID_SIZE + 1) * MARGIN
GRID_SIZE_OPTIONS = [GRID_SIZE - 2, GRID_SIZE, GRID_SIZE + 2]
