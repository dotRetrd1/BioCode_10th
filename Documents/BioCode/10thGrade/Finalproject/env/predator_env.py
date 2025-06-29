import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import random
import torch
from game.world import World
from game.predator import Predator
from game.prey import Prey
from game.config import (
    GRID_SIZE,
    PREDATOR_HUNTING_REST,
    PREDATOR_STARVATION_LIMIT,
    PREDATOR_REPRODUCTION_THRESHOLD,   
    PREDATOR_PROXIMITY_SCALE as PROXIMITY_SCALE,                   
    STEP_LIVING_COST,                  
)

from game.ai_logic import prey_ai          

actionLog = ""
doneReason = ""


class PredatorEnv:
    def __init__(self, render=False):
        self.grid_size = GRID_SIZE
        self.world = None
        self.agent = None  
        self.max_steps = 100
        self.current_step = 0
        self.render_enabled = render
        # track repeated direction
        self.last_move_dir = None
        self.same_dir_count = 0
        # keep a persistent log per episode
        self.action_log = ""

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.world = World()

        for _ in range(2):      #spwwn initial prey
            self.world.generate_prey(scripted=True)
        
        self.world.generate_predator(scripted=False)    #spawn original rl predator

        self.world.update_food()  
        self.agent = [a for a in self.world.agents if isinstance(a, Predator) and not a.use_scripted_ai][0]
        self.world.rl_agent = self.agent
        self.agent.steps_since_last_meal = 0
        self.current_step = 0
        self.last_move_dir = None
        self.same_dir_count = 0
        self.action_log = ""
        return self.get_observation()

    def _nearest_prey_dist(self):
        if not any(isinstance(a, Prey) for a in self.world.agents):
            return self.grid_size
        px, py = self.agent.position
        best = self.grid_size
        for prey in [a for a in self.world.agents if isinstance(a, Prey)]:
            x, y = prey.position
            dx = min((px - x) % self.grid_size, (x - px) % self.grid_size)
            dy = min((py - y) % self.grid_size, (y - py) % self.grid_size)
            best = min(best, dx + dy)
        return best

    def get_observation(self):
        view_range = 1
        prey_layer = np.zeros((3, 3), dtype=np.int8)
        self_layer = np.zeros((3, 3), dtype=np.int8)
        predator_layer = np.zeros((3, 3), dtype=np.int8)
        x0, y0 = self.agent.position
        for dx in range(-view_range, view_range+1):
            for dy in range(-view_range, view_range+1):
                x = (x0 + dx) % self.grid_size
                y = (y0 + dy) % self.grid_size
                local_x, local_y = dx + 1, dy + 1
                for p in [a for a in self.world.agents if isinstance(a, Prey)]:
                    if p.position == (x, y):
                        prey_layer[local_y][local_x] = 1
        self_layer[1][1] = 1
        obs = np.stack([prey_layer, self_layer, predator_layer], axis=0)
        return obs.flatten()

    def step(self, action):
        global actionLog  
        global doneReason
        doneReason = ""
        actionLog = ""
        self.action_log = ""
        self.current_step += 1
        self.agent.steps_since_last_meal += 1
        dist_before = self._nearest_prey_dist()
        move_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        reward = 0.0

        #actionmap: 0=up, 1=down, 2=left, 3=right, 4=rest, 5=reproduce
        if action < 4:
            dx, dy = move_map[action]
            x, y = self.agent.position
            new_pos = ((x + dx) % self.grid_size, (y + dy) % self.grid_size)
            self.world.move_agent(self.agent, new_pos)
            actionLog += f"[{self.current_step}] Predator moved to {new_pos} (action: {action}).\n"


        elif action == 5:            #reproduce
            if self.agent.kills >= PREDATOR_REPRODUCTION_THRESHOLD:
                child = self.world.generate_predator(
                    scripted=False,
                    position=self.agent.position
                )
                #share the same policy network so the child acts
                child.rl_model = self.agent.rl_model
                #switch training focus to the live child
                self.world.rl_agent = child
                self.agent = child

                actionLog += f"[{self.current_step}] Predator reproduced at {child.position}.\n"
                reward += 15.0
                child.kills = 0                   
            else:   #failed to reproduce (I cant be dammed to actually make it so it cant do this)
                reward -= 2.0
                actionLog += f"[{self.current_step}] Predator failed to reproduce (kills: {self.agent.kills}).\n"
                self.same_dir_count = 0
        #rest
        else:
            reward -= 3.0
            actionLog += f"[{self.current_step}] Predator rested.\n"
            self.same_dir_count = 0

        # repeated‑direction penalty
            if self.last_move_dir == action:    
                self.same_dir_count += 1
                if self.same_dir_count >= 4:
                    reward -= 3.5              #punishment for 3‑in‑a‑row 😠
                    actionLog += f"[{self.current_step}] Repeated direction penalty.\n"
                else:
                    self.same_dir_count = 1

        #reset last move var
        self.last_move_dir = action

        #scripted prey logic for each prey 
        for prey in [a for a in self.world.agents if isinstance(a, Prey)]:
            if prey.use_scripted_ai:
                act = prey.decide_action(self.world)
                kind, target = act
                if kind == "move":
                    self.world.move_agent(prey, target)
                    actionLog += f"[{self.current_step}] Prey moved to {target}.\n"
                elif kind == "eat":
                    prey.meals_eaten += 1
                    self.world.remove_food_at(target)
                    actionLog += f"[{self.current_step}] Prey ate food at {target}.\n"
                elif kind == "reproduce":
                    self.world.generate_prey(scripted=True, position=target)
                    prey.meals_eaten = 0
                    actionLog += f"[{self.current_step}] Prey reproduced at {target}.\n"

        self.world.update_food()

        # Check if the predator is on the same position as any prey and immediately hunt it (didnt want to waste logic on this so just forced ^^)
        for prey in [a for a in self.world.agents if isinstance(a, Prey)]:
            if prey.position == self.agent.position:
                actionLog += f"[{self.current_step}] Predator hunted prey at {prey.position}.\n"
                reward += 10.0
                self.agent.kills += 1
                self.world.remove_agent(prey)
                self.agent.rest_turns = PREDATOR_HUNTING_REST
                self.agent.steps_since_last_meal = 0

        # predator vs predator fight logic
        pos_map = {}
        for p in [a for a in self.world.agents if isinstance(a, Predator)]:
            pos_map.setdefault(p.position, []).append(p)

        for preds in pos_map.values():
            while len(preds) > 1:
                p1, p2 = preds.pop(), preds.pop()
                winner = p1.fight(p2, self.world)
                self._log(f"predator fight at {winner.position} – {winner.agent_id} survived")
                preds.append(winner)


        #predator forced to rest
        if self.agent.rest_turns > 0:
            actionLog += f"[{self.current_step}] Predator forced to rest.\n"
            reward -= 0.1
            self.agent.rest_turns -= 1  # decrement the rest cooldown

        #proximity to prey reward
        dist_after = self._nearest_prey_dist()
        if(dist_after < dist_before):
            reward += 2.75

        #per step penalty
        reward -= STEP_LIVING_COST

        #check for done conditions
        if (self.agent.steps_since_last_meal > PREDATOR_STARVATION_LIMIT):
            done = True
            doneReason = f"[{self.current_step}] Predator starved.\n"
        elif (self.current_step >= self.max_steps):
            done = True
            doneReason = f"[{self.current_step}] Predator reached max steps.\n"
        elif (not any(isinstance(a, Prey) for a in self.world.agents)):
            done = True
            doneReason = f"[{self.current_step}] Predator won (no prey left).\n"
        else:
            done = False

        self.action_log = actionLog  
        obs = self.get_observation()
        return obs, reward, done, {"action_log": self.action_log, "done_reason": doneReason}

    def render(self):
        if not self.render_enabled:
            return
        from renderer import draw
        from game.prey import Prey
        from game.predator import Predator
        predator_count = sum(isinstance(a, Predator) for a in self.world.agents)
        prey_count = sum(isinstance(a, Prey) for a in self.world.agents)
        draw(self.world, predator_count, prey_count, self.current_step, self.action_log)