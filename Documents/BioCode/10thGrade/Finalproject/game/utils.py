import random
from game.config import GRID_SIZE

def random_position(exclude=[]):
    while True:
        pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if pos not in exclude:
            return pos