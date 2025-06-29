import numpy as np
import random
import torch

from game.world import World
from game.prey import Prey
from game.predator import Predator
from game.config import (
    GRID_SIZE,
    PREY_STARVATION_TICKS,
    PREY_REPRODUCTION_THRESHOLD,
    PREY_EATING_REST,
    PREDATOR_HUNTING_REST,
)
from renderer import draw


class PreyEnv:
    def __init__(self, render=False):
        self.grid_size = GRID_SIZE
        self.render_enabled = render
        self.max_steps = 100
        self.current_step = 0
        self.world = None
        self.agent = None         # the prey we train on
        self.action_log = ""
        self._prey_counter = 0 


    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.world = World()
        self.world.generate_predator(scripted=True)

        # spawn one original rl prey
        self.world.generate_prey(scripted=False)
        self.world.update_food()

        self.agent = next(a for a in self.world.agents if isinstance(a, Prey))
        self._tag_prey(self.agent)
        self.world.rl_agent = self.agent
        self.current_step = 0
        self.action_log = ""
        return self.get_observation()

    def step(self, action):
        self.action_log = ""
        self.current_step += 1
        reward = 0.0
        done = False
        reproduced = False
        end_reason = None

        # ---- store current observation so RL-predators can read it ----
        curr_obs = self.get_observation()
        self.world.last_obs = curr_obs
        
        # -------- main prey acts (self.agent) --------
        if self.agent.rest_turns > 0:
            action = 4
            self.agent.rest_turns -= 1
            self._log("prey forced to rest")

        old_pos = self.agent.position
        old_dist = self._manhattan(old_pos, self.world.food)

        self._apply_action(self.agent, action)
        if action < 4:
            reward += 0.05
        else:
            reward -= 0.1

        new_dist = self._manhattan(self.agent.position, self.world.food)
        if new_dist < old_dist:
            reward += 0.4
        elif new_dist > old_dist:
            reward -= 0.05

        # -------- all other rl preys act --------
        for prey in [p for p in self.world.agents if isinstance(p, Prey) and p is not self.agent]:
            if prey.rest_turns > 0:
                prey.rest_turns -= 1
                continue
            other_act = self._choose_action(prey)
            self._apply_action(prey, other_act)

        # -------- predator turn (scripted) --------
        for predator in [a for a in self.world.agents if isinstance(a, Predator)]:
            act_type, target = predator.decide_action(self.world)
            if act_type == "move":
                self.world.move_agent(predator, target)
                self._log(f"predator moved to {target}")
            elif act_type == "hunt" and target in self.world.agents:
                self.world.remove_agent(target)
                predator.rest_turns = PREDATOR_HUNTING_REST
                self._log(f"predator hunted prey at {target.position}")
            elif act_type == "reproduce":
                self.world.add_agent(predator)
                predator.rest_turns = 2
                self._log("predator reproduced")
            elif act_type == "rest":
                predator.rest_turns -= 1
                self._log("predator rested")

        # -------- eating for main prey --------
        if self.world.is_food_at(self.agent.position):
            reward += 2.5
            self.agent.meals_eaten += 1
            self.agent.steps_since_last_meal = 0
            self.agent.rest_turns = PREY_EATING_REST
            self.world.update_food()
            self._log(f"prey ate (meals {self.agent.meals_eaten})")
        else:
            self.agent.steps_since_last_meal += 1

        # -------- reproduction --------
        if self.agent.meals_eaten >= PREY_REPRODUCTION_THRESHOLD:
            self.agent.meals_eaten = 0
            self.agent.rest_turns = 2
            reproduced = True
            reward += 7.5
            self._log("prey reproduced")
            newborn = self.world.generate_prey(
                scripted=False, position=self.agent.position
            )
            self._tag_prey(newborn)
            newborn.rl_model = self.agent.rl_model        # shared policy
            self.world.rl_agent = newborn
            self.agent = newborn                           # training now follows newborn

        # -------- end conditions --------
        if self.agent not in self.world.agents:
            reward = -5.0
            done = True
            end_reason = "killed"
            self._log("prey killed")
        elif self.agent.steps_since_last_meal >= PREY_STARVATION_TICKS:
            reward = -5.0
            done = True
            end_reason = "starvation"
            self._log("prey starved")
        elif self.current_step >= self.max_steps:
            done = True
            end_reason = "max steps"
            self._log("max steps reached")
        elif not any(isinstance(a, Prey) for a in self.world.agents):
            done = True
            end_reason = "all prey dead"
            self._log("all prey eliminated")

        obs = self.get_observation()
        return obs, reward, done, {
            "reproduced": reproduced,
            "done_reason": end_reason,
            "action_log": self.action_log,
        }

    # ---------- funcs ----------

    def _choose_action(self, prey):
        obs = self._get_observation_for(prey)
        with torch.no_grad():
            logits = prey.rl_model(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            )
        return int(torch.argmax(logits, dim=1).item())

    def _apply_action(self, prey, action):
        label = getattr(prey, "eid", "?")
        if action >= 4:
            self._log(f"prey#{id(prey)%1000} rested")
            return
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        x, y = prey.position
        new_pos = ((x + dx) % self.grid_size, (y + dy) % self.grid_size)
        self.world.move_agent(prey, new_pos)
        prey.rest_turns = 0
        self._log(f"prey{label} moved to {new_pos}")

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _log(self, msg):
        self.action_log += f"[{self.current_step}] {msg}\n"

    def _tag_prey(self, prey):
        prey.eid = f" no.{self._prey_counter}"
        self._prey_counter += 1

    # ---------- obs ----------

    def _get_observation_for(self, prey):
        view = 1
        obs = np.zeros((3, 3, 3), dtype=np.int8)
        px, py = prey.position
        for dx in range(-view, view + 1):
            for dy in range(-view, view + 1):
                x = (px + dx) % self.grid_size
                y = (py + dy) % self.grid_size
                lx, ly = dx + 1, dy + 1
                if self.world.food == (x, y):
                    obs[lx, ly, 0] = 1
                for a in self.world.agents:
                    if isinstance(a, Predator) and a.position == (x, y):
                        obs[lx, ly, 2] = 1
        obs[1, 1, 1] = 1
        return obs.transpose((2, 0, 1)).flatten()

    def get_observation(self):
        return self._get_observation_for(self.agent)

    # ---------- rendering ----------

    def render(self):
        if not self.render_enabled:
            return
        predator_count = sum(
            isinstance(a, Predator) for a in self.world.agents
        )
        prey_count = sum(isinstance(a, Prey) for a in self.world.agents)
        draw(
            self.world,
            predator_count,
            prey_count,
            self.current_step,
            self.action_log,
        )


    
