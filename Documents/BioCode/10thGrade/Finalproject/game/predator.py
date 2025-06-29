from game.agentBase import Agent
import torch
import random
from model.network import PreyPolicyNet          # reuse same net
from model.ppo_agent import PPONet               # for PPO actor-critic

from game.config import PREDATOR_BASE_ADVANTAGE

class Predator(Agent):
    def __init__(self, position, agent_id, use_scripted_ai=False, advantage=None):
        super().__init__(position, agent_id)
        self.use_scripted_ai = use_scripted_ai
        self.rest_turns = 0
        self.steps_since_last_meal = 0
        self.rl_model = None
        self.kills = 0
        # static combat advantage
        self.advantage = PREDATOR_BASE_ADVANTAGE if advantage is None else advantage

    # -----------------------------------------------------------
    # rl or scripted decisions
    # -----------------------------------------------------------
    def decide_action(self, world):
        if not self.use_scripted_ai and self.rl_model is not None:
            # fetch the last observation
            state = torch.tensor(world.last_obs, dtype=torch.float32)

            # the model may be a PPO actor-critic returning (logits, value)
            out = self.rl_model(state)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            # pick the highest-scoring action
            action = torch.argmax(logits).item()

            move_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            if action < 4:
                action_type = "move"
                target = (
                    (self.position[0] + move_map[action][0]) % world.grid_size,
                    (self.position[1] + move_map[action][1]) % world.grid_size
                )
            else:
                # 4 = rest (no reproduce in predator)
                action_type = "rest"
                target = self.position

            return action_type, target

        else:
            from .ai_logic import predator_ai
            return predator_ai(self, world)

    # -----------------------------------------------------------
    # 1-on-1 fight helper
    # winner stays on board, loser is removed from world
    # -----------------------------------------------------------
    def fight(self, other, world):
        assert isinstance(other, Predator)
        total = self.advantage + other.advantage
        prob_self = 0.5 if total == 0 else self.advantage / total
        winner = self if random.random() < prob_self else other
        loser = other if winner is self else self
        world.remove_agent(loser)
        return winner