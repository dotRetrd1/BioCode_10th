
#yes I had to do this in a seperate file 💔
import json, time, os, random
import torch, pygame
from env.joint_env import JointEnv
from model.ppo_agent import PPONet
from net_utils import AdaptNet

RESULTS_PATH = "run_stats.json"  # path for cached episodes

# ───────────────────────── size helper ─────────────────────────

def ensure_size(net: PPONet, obs_len: int) -> PPONet:
    if hasattr(net, "fc1") and net.fc1.in_features == obs_len:
        return net
    return AdaptNet(net)

# ───────────────────────── episode runner ──────────────────────

def run_episode(env: JointEnv, prey_net: PPONet, pred_net: PPONet):
    obs_p, obs_r = env.reset()
    for p in env.rl_preys:
        p.rl_model = prey_net
    env.pred.rl_model = pred_net

    ticks = 0
    log_all = ""                     
    while True:
        a_p = int(torch.argmax(prey_net(torch.tensor(obs_p, dtype=torch.float32))[0]))
        a_r = int(torch.argmax(pred_net(torch.tensor(obs_r, dtype=torch.float32))[0]))
        obs_p, obs_r, rew_p, rew_r, done, info = env.step(a_p, a_r)
        ticks += 1
        log_all += env.action_log     
        if done:
            break

    kills = log_all.count("hunted")
    reps  = log_all.count("Prey") and log_all.count("reproduced")
    food  = log_all.count(" ate ")

    pred_act = {
        "move":  log_all.count("Predator") and log_all.count("moved"),
        "rest":  log_all.count("Predator") and log_all.count("rested"),
        "repro": log_all.count("Predator") and log_all.count("reproduced at"),
        "hunt":  kills,
    }
    prey_act = {
        "move":  log_all.count("Prey") and log_all.count("moved"),
        "rest":  log_all.count("Prey") and log_all.count("rest cooldown"),
        "repro": reps,
        "eat":   food,
    }

    return dict(
        ticks=ticks,
        reps=reps,
        kills=kills,
        food=food,
        pred_act=pred_act,
        prey_act=prey_act,
        win="prey" if rew_p > rew_r else "pred",
        reason=info.get("done_reason", "")
    )

# ───────────────────────── batch collector ─────────────────────

def collect_stats(prey_net: PPONet, pred_net: PPONet, runs: int = 100, seed_base: int | None = None):
    # Run `runs` simulations and write raw stats to RESULTS_PATH.
    env = JointEnv(render=False)
    stats_list = []
    base_seed = seed_base if seed_base is not None else int(time.time())

    for i in range(runs):
        seed = base_seed + i
        random.seed(seed)
        torch.manual_seed(seed)

        # adapt nets to current observation length without retraining
        obs_p, obs_r = env.reset(seed=seed)
        prey_net_wrapped = ensure_size(prey_net,  len(obs_p))
        pred_net_wrapped = ensure_size(pred_net, len(obs_r))

        pygame.init(); pygame.display.set_mode((1, 1))  # headless surface
        stats = run_episode(env, prey_net_wrapped, pred_net_wrapped)
        pygame.quit()
        pygame.display.quit()
        pygame.font.quit()
        stats_list.append(stats)

    with open(RESULTS_PATH, "w", encoding="utf-8") as fp:
        json.dump(stats_list, fp, indent=2)
    print(f"[collector] Saved {len(stats_list)} episodes → {RESULTS_PATH}")
    return RESULTS_PATH

# ───────────────────────── CLI entry ───────────────────────────
if __name__ == "__main__":
    import sys
    from showcase import best_weight

    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    # fresh nets sized to JointEnv obs template
    base_env = JointEnv(render=False)
    op, or_ = base_env.reset()
    prey_net = PPONet(input_dim=len(op), act_dim=5)
    pred_net = PPONet(input_dim=len(or_), act_dim=6)

    # load best checkpoints if they exist
    prey_path = best_weight("prey_policy_ep*.pth")
    pred_path = best_weight("predator_policy_ep*.pth")
    if prey_path:
        prey_net.load_state_dict(torch.load(prey_path))
    if pred_path:
        pred_net.load_state_dict(torch.load(pred_path))

    collect_stats(prey_net, pred_net, runs)
