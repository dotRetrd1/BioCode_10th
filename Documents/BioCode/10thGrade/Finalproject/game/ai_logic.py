import random
from .config   import GRID_SIZE, PREY_REPRODUCTION_THRESHOLD, PREDATOR_REPRODUCTION_THRESHOLD
from game.predator import Predator   

# ───────────────────────── helpers ────────────────────────────────────
def get_adjacent_positions(pos):
    x, y = pos
    return [
        ((x + 1) % GRID_SIZE, y),
        ((x - 1) % GRID_SIZE, y),
        (x, (y + 1) % GRID_SIZE),
        (x, (y - 1) % GRID_SIZE),
    ]

def toroidal_manhattan(a, b):
    dx = min(abs(a[0] - b[0]), GRID_SIZE - abs(a[0] - b[0]))
    dy = min(abs(a[1] - b[1]), GRID_SIZE - abs(a[1] - b[1]))
    return dx + dy

def predator_at(world, pos):
    return any(isinstance(a, Predator) and a.position == pos
               for a in world.agents)

# ───────────────────── scripted-prey policy ───────────────────────────
def prey_ai(agent, world):
    # 1. mandatory rest
    if agent.rest_turns > 0:
        return ("rest", agent.position)

    # 2. eat immediately if standing on food
    if world.is_food_at(agent.position):
        return ("eat", agent.position)

    # 3. reproduce as soon as the threshold is hit
    if agent.meals_eaten >= PREY_REPRODUCTION_THRESHOLD:
        return ("reproduce", agent.position)

    # 4. move toward the food while dodging predator
    target = world.food                      # (x, y) or None on first tick
    if target is not None:
        best_dist, best_steps = None, []
        for nxt in get_adjacent_positions(agent.position):
            if predator_at(world, nxt):      # never step onto a predator
                continue
            d = toroidal_manhattan(nxt, target)
            if best_dist is None or d < best_dist:
                best_dist, best_steps = d, [nxt]
            elif d == best_dist:
                best_steps.append(nxt)
        if best_steps:                       
            return ("move", random.choice(best_steps))

    # 5. fallback: random safe step, or truly random if boxed in
    safe = [p for p in get_adjacent_positions(agent.position)
            if not predator_at(world, p)]
    return ("move", random.choice(safe) if safe
            else random.choice(get_adjacent_positions(agent.position)))

# ───────────────────── scripted-predator policy ─────────────────────────
def predator_ai(agent, world):
    if agent.rest_turns > 0:
        return ("rest", agent.position)

    # If on prey → attack
    prey_here = world.get_prey_at(agent.position)
    if prey_here:
        return ("hunt", prey_here)

    # Look for nearby prey
    for pos in get_adjacent_positions(agent.position):
        prey = world.get_prey_at(pos)
        if prey:
            return ("move", pos)

    # Reproduce if enough kills
    if agent.kills >= PREDATOR_REPRODUCTION_THRESHOLD:
        return ("reproduce", agent.position)

    # Otherwise, move randomly
    return ("move", random.choice(get_adjacent_positions(agent.position)))