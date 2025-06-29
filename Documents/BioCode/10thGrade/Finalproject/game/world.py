# world.py

from game.utils import random_position
from game.config import GRID_SIZE
from game.prey import Prey
from game.predator import Predator
from model.network import PreyPolicyNet

class World:
    def __init__(self):
        self.turn = 0
        self.food = None
        self.agents = []
        self.next_id = 0
        self.grid_size = GRID_SIZE
        self.last_obs = None  

    def get_all_positions(self):
        return [agent.position for agent in self.agents]

    def generate_prey(self, scripted=False, position=None):
        if position is None:
            position = random_position(self.get_all_positions())
        prey = Prey(position, self.next_id, use_scripted_ai=scripted)
        self.next_id += 1
        self.add_agent(prey)
        
        if not scripted:
            from model.network import PreyPolicyNet
            #Normally assign a new model if needed
            prey.rl_model = PreyPolicyNet()
            self.rl_agent = prey
        return prey


    def generate_predator(self, scripted=False, position=None):
        if position is None:
            position = random_position(self.get_all_positions())
        predator = Predator(position, self.next_id, use_scripted_ai=scripted)
        self.next_id += 1
        self.add_agent(predator)
        if not scripted:
            predator.rl_model = PreyPolicyNet()  # assign a new RL model
        return predator

    def manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def add_agent(self, agent):
        self.agents.append(agent)

    def update_food(self):
        self.food = random_position(exclude=self.get_all_positions())

    def is_food_at(self, position): #check if food is at a certain position
        return self.food == position

    def get_prey_at(self, position):    #check if prey is at a certain position
        for agent in self.agents:
            if isinstance(agent, Prey) and agent.position == position:
                return agent
        return None

    def get_other_predator_at(self, position, self_predator):   #check if other predator is at a certain position
        for agent in self.agents:
            if (
                isinstance(agent, Predator)
                and agent.position == position
                and agent != self_predator
            ):
                return agent
        return None

    def move_agent(self, agent, new_pos):
        x, y = new_pos
        x %= GRID_SIZE
        y %= GRID_SIZE
        agent.position = (x, y)

    def remove_agent(self, agent):
        if agent in self.agents:
            self.agents.remove(agent)
