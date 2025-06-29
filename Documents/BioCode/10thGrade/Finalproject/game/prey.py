from game.agentBase import Agent
from model.network import PreyPolicyNet   # RL model
import torch

class Prey(Agent):
    def __init__(self, position, agent_id, use_scripted_ai=False):
        super().__init__(position, agent_id)
        self.use_scripted_ai = use_scripted_ai
        self.meals_eaten = 0
        self.steps_since_last_meal = 0
        self.rl_model = None            # set later for RL‑controlled prey

    def decide_action(self, world):
        if self.use_scripted_ai:
            # fixed import path (relative to game package)
            from game.ai_logic import prey_ai
            return prey_ai(self, world)

        # RL‑controlled branch
        if self.rl_model is not None:
            state = torch.tensor(world.get_observation(), dtype=torch.float32)
            probs = self.rl_model(state)
            return torch.argmax(probs).item()

        raise NotImplementedError("RL agent does not use scripted logic.")
