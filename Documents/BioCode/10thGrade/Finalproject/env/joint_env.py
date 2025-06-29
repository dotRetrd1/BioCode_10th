import numpy as np, random, math, torch
from game.world     import World
from game.prey      import Prey
from game.predator  import Predator
from game.config    import (
    GRID_SIZE,
    STEP_LIVING_COST,
    PREY_EATING_REST,
    PREY_STARVATION_TICKS,
    PREY_REPRODUCTION_THRESHOLD,
    PREDATOR_HUNTING_REST,
    PREDATOR_STARVATION_LIMIT,
    PREDATOR_REPRODUCTION_THRESHOLD
)
from renderer import draw

# ─── hyperparams ─────────────────────────────────────────────────────
VIEW_RANGE             = 3        # local view radius
ACTION_REPEAT          = 3        # epeat each policy action this many env steps (google stuff say its good idk lolololol)
EPS_START, EPS_END     = 0.10, 0.01
EPS_DECAY              = 3e-4
ENTROPY_COEFF          = 0.03
SURVIVAL_BONUS         = 0.01
FOOD_DIST_SCALE        = 50
POTENTIAL_GAMMA        = 0.995
REPEAT_ACTION_PENALTY  = 15
REST_PENALTY           = 15
BASE_DIST              = 8
MAX_STEPS_DEFAULT      = 400

# ─── actions ──────────────────────────────────────────────────────────
MOVE_MAP = [(0,-1),(0,1),(-1,0),(1,0)]
REST      = 4
REPRO     = 5


class JointEnv:
    def __init__(self, render=False, max_steps=MAX_STEPS_DEFAULT):
        self.render_enabled = render
        self.max_steps = max_steps
        self.grid = GRID_SIZE
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        # new world
        self.world = World()
        # spawn one RL prey & one RL predator
        p0 = self.world.generate_prey(scripted=True)
        self.rl_preys = [p0]
        self.pred    = self.world.generate_predator(scripted=False)
        self.world.update_food()

        # initialize count-based exploration
        self.obs_counts = {}

        # trackers for prey
        self.prey       = p0
        self.prev_dir_p = None

        # trackers per-prey
        self.prey_prev_dir   = {p0: None}
        self.prey_same_count = {p0: 0}

        # trackers for predator
        self.prev_dir_r = None
        self.same_r     = 0

        # time & ε
        self.t       = 0
        self.epsilon = EPS_START

        self.action_log = ""
        # cache obs-template for dead-prey fallback
        obs_p0 = self._obs(p0, None)
        obs_r0 = self._obs(self.pred, None)
        self._obs_template_p = obs_p0.copy()
        return obs_p0, obs_r0


    def step(self, act_p, act_r):
        total_r_p = 0.0
        total_r_r = 0.0
        done = False
        reason = ""
        self.action_log = ""

        # run each policy action multiple times for smoother dynamics
        for _ in range(ACTION_REPEAT):
            self.t += 1

            # ε-greedy exploration
            if random.random() < self.epsilon:
                act_p = random.randrange(4)   # prey: only moves 0–3
            if random.random() < self.epsilon:
                act_r = random.randrange(6)

            # --- all RL-prey act ---
            comp_p_tot = {"move":0,"food":0,"potential":0,"repeat":0}
            r_p_step = 0.0
            for prey in list(self.rl_preys):
                if prey not in self.world.agents:
                    self.rl_preys.remove(prey)
                    continue

                obs = self._obs(prey, self.prey_prev_dir[prey])
                logits,_ = prey.rl_model(torch.tensor(obs, dtype=torch.float32))
                chosen = torch.argmax(logits).item()

                # bind for _apply_prey
                old_self, old_prev = self.prey, self.prev_dir_p
                self.prey, self.prev_dir_p = prey, self.prey_prev_dir[prey]

                rp, comp = self._apply_prey(chosen)
                r_p_step += rp
                for k,v in comp.items():
                    comp_p_tot[k] += v

                # store back trackers
                self.prey_prev_dir[prey]   = self.prev_dir_p
                self.prey = old_self
                self.prev_dir_p = old_prev

            total_r_p += r_p_step

            # --- predator acts ---
            r_r_step, comp_r = self._apply_pred(act_r)
            total_r_r += r_r_step

            # --- terminal checks ---
            if not self.rl_preys:
                total_r_r += 10
                reason = "all_prey_dead"
                done = True

            for prey in list(self.rl_preys):
                if prey.steps_since_last_meal >= PREY_STARVATION_TICKS:
                    self._log(f"Prey#{prey.agent_id} starved")
                    self.world.remove_agent(prey)
                    self.rl_preys.remove(prey)
                    total_r_r += 10
                    reason = "prey_starved"
                    done = True
                    break

            if self.pred.steps_since_last_meal > PREDATOR_STARVATION_LIMIT:
                total_r_r -= 5
                reason = "pred_starved"
                done = True

            if done or self.t >= self.max_steps:
                if self.t >= self.max_steps and not done:
                    reason = "max_steps"
                    done = True
                break

        # decay ε
        self.epsilon = max(EPS_END,
                           EPS_START / math.sqrt(1 + EPS_DECAY*self.t))

        if done and reason:
            self._log(f"*** Episode ended: {reason} ***")

        # --- count-based exploration bonus for prey ---
        # next‐state key is first-prey observation
        if self.rl_preys:
            obs_p0 = self._obs(self.rl_preys[0],
                               self.prey_prev_dir[self.rl_preys[0]])
        else:
            obs_p0 = np.zeros_like(self._obs_template_p)
        key = tuple(obs_p0.tolist())
        cnt = self.obs_counts.get(key, 0) + 1
        self.obs_counts[key] = cnt
        bonus = 1.0 / math.sqrt(cnt)
        total_r_p += bonus
        comp_p_tot["count_bonus"] = bonus

        # predator obs
        obs_r0 = self._obs(self.pred, self.prev_dir_r)

        info = dict(
            done_reason      = reason,
            action_log       = self.action_log,
            entropy_coeff    = ENTROPY_COEFF,
            reward_breakdown = {
                "prey": comp_p_tot,
                "pred": comp_r
            }
        )

        if self.render_enabled:
            self._render()
        return obs_p0, obs_r0, total_r_p, total_r_r, done, info


    def _apply_prey(self, act):
        comp = {"move":0,"food":0,"potential":0,"repeat":0}
        base = -STEP_LIVING_COST + SURVIVAL_BONUS

        # rest-cooldown
        if self.prey.rest_turns:
            self.prey.rest_turns -= 1
            base -= 0.5
            self._log(f"Prey#{self.prey.agent_id} rest cooldown")
            return base, comp

        # clamp action
        act = act % 4

        # move
        dx,dy = MOVE_MAP[act]
        old = self.prey.position
        new = ((old[0]+dx)%self.grid, (old[1]+dy)%self.grid)
        self.world.move_agent(self.prey, new)
        comp["move"] = 0.01; base += 0.01

        # food‐distance shaping
        d_old = abs(old[0]-self.world.food[0]) + abs(old[1]-self.world.food[1])
        d_new = abs(new[0]-self.world.food[0]) + abs(new[1]-self.world.food[1])
        bonus = FOOD_DIST_SCALE * (d_old - d_new) * 0.25
        comp["food"] = bonus; base += bonus

        # potential‐based shaping
        phi_old = -d_old; phi_new = -d_new
        pot = POTENTIAL_GAMMA*phi_new - phi_old
        comp["potential"] = pot; base += pot

        # repeat‐action penalty
        if self.prev_dir_p == act:
            comp["repeat"] = -REPEAT_ACTION_PENALTY
            base -= REPEAT_ACTION_PENALTY
        self.prev_dir_p = act

        self._log(f"Prey#{self.prey.agent_id} moved to {new}")

        # eat & reproduce
        if self.world.is_food_at(new):
            base += 10; comp["food"] += 10
            self.prey.meals_eaten += 1
            self.prey.steps_since_last_meal = 0
            self.prey.rest_turns = PREY_EATING_REST
            self.world.update_food()
            self._log(f"Prey#{self.prey.agent_id} ate at {new}")
        else:
            self.prey.steps_since_last_meal += 1

        if self.prey.meals_eaten >= PREY_REPRODUCTION_THRESHOLD:
            self.prey.meals_eaten = 0
            self.prey.rest_turns   = 2
            base += 50
            self._log(f"Prey#{self.prey.agent_id} reproduced at {new}")
            baby = self.world.generate_prey(scripted=False, position=new)
            baby.rl_model = self.prey.rl_model
            self.rl_preys.append(baby)
            self.prey_prev_dir[baby]   = None
            self.prey_same_count[baby] = 0

        return base, comp


    def _apply_pred(self, act):
        comp = {"move":0,"hunt":0,"dist_delta":0,"repeat":0}
        base = -STEP_LIVING_COST

        # rest‐cooldown
        if self.pred.rest_turns:
            self.pred.rest_turns -= 1
            base -= 0.5
            self._log(f"Predator#{self.pred.agent_id} rest cooldown")
            return base, comp

        # --- compute old distance to nearest prey for shaping ---
        prey_list = [q for q in self.world.agents if isinstance(q, Prey)]
        if prey_list:
            d_old = min(
                abs(self.pred.position[0] - q.position[0]) +
                abs(self.pred.position[1] - q.position[1])
                for q in prey_list
            )
        else:
            d_old = self.grid

        # --- existing move / rest / reproduce logic ---
        if act < 4:
            dx, dy = MOVE_MAP[act]
            old = self.pred.position
            new = ((old[0]+dx)%self.grid, (old[1]+dy)%self.grid)
            self.world.move_agent(self.pred, new)
            comp["move"] = 0
            self._log(f"Predator#{self.pred.agent_id} moved to {new}")
            if self.prev_dir_r == act:
                comp["repeat"] = -REPEAT_ACTION_PENALTY
                base   -= REPEAT_ACTION_PENALTY
            self.prev_dir_r = act

        elif act == REST:
            base -= REST_PENALTY
            self._log(f"Predator#{self.pred.agent_id} rested")
            self.prev_dir_r = None

        else:  # reproduce
            if self.pred.kills >= PREDATOR_REPRODUCTION_THRESHOLD:
                baby = self.world.generate_predator(
                    scripted=False,
                    position=self.pred.position
                )
                baby.rl_model = self.pred.rl_model
                self.pred.kills = 0
                base += 15
                self._log(f"Predator#{self.pred.agent_id} reproduced at {self.pred.position}")
            else:
                base -= 50
                self._log(f"Predator#{self.pred.agent_id} repro fail (kills={self.pred.kills})")
            self.prev_dir_r = None

        # --- hunting logic---
        for prey in [a for a in self.world.agents if isinstance(a, Prey)]:
            if prey.position == self.pred.position:
                self.world.remove_agent(prey)
                self.pred.kills += 1
                comp["hunt"] = 10
                base += 10
                self.pred.rest_turns = PREDATOR_HUNTING_REST
                self.pred.steps_since_last_meal = 0
                self._log(f"Predator#{self.pred.agent_id} hunted Prey#{prey.agent_id} at {prey.position}")
                break

        # --- dynamic distance shaping: reward decrease in distance ---
        if prey_list:
            d_new = min(
                abs(self.pred.position[0] - q.position[0]) +
                abs(self.pred.position[1] - q.position[1])
                for q in prey_list
            )
        else:
            d_new = self.grid

        # shaping bonus: positive when moving closer, negative when farther
        dist_delta = FOOD_DIST_SCALE * (d_old - d_new)
        comp["dist_delta"] = dist_delta
        base += dist_delta

        self.pred.steps_since_last_meal += 1
        return base, comp



    def _obs(self, agent, last_dir):
        size   = VIEW_RANGE*2 + 1
        layers = 4  # food, prey, predator, self
        arr    = np.zeros((layers+4, size, size), dtype=np.int8)

        ax, ay = agent.position
        for dx in range(-VIEW_RANGE, VIEW_RANGE+1):
            for dy in range(-VIEW_RANGE, VIEW_RANGE+1):
                x = (ax+dx)%self.grid
                y = (ay+dy)%self.grid
                ix, iy = dx+VIEW_RANGE, dy+VIEW_RANGE
                if self.world.food==(x,y):      arr[0,ix,iy] = 1
                for a in self.world.agents:
                    if isinstance(a, Prey) and a.position==(x,y):     arr[1,ix,iy] = 1
                    if isinstance(a, Predator) and a.position==(x,y): arr[2,ix,iy] = 1
        arr[3,VIEW_RANGE,VIEW_RANGE] = 1  # self

        return arr.flatten()


    def _log(self, msg):
        self.action_log += f"[{int(self.t - (2 * (self.t/3)))}] {msg}\n"

    def _render(self):
        pred_count = sum(isinstance(a,Predator) for a in self.world.agents)
        prey_count = sum(isinstance(a,Prey)     for a in self.world.agents)
        draw(self.world, pred_count, prey_count, self.t, self.action_log)
