# main.py

#for showcase
import sys
from showcase import run_showcase
#

import os, re, glob, time, random, numpy as np, torch, pygame
import torch.nn.functional as F
from torch.distributions import Categorical

from env.joint_env    import JointEnv
from env.prey_env     import PreyEnv
from env.predator_env import PredatorEnv
from game.predator    import Predator
from model.network    import PreyPolicyNet   # didnt have the heart to just delete this (..)
from model.ppo_agent  import PPONet

# ═══════════ utilities funcs ════════════════════════════════════════════════
_EP_RE = re.compile(r"_ep(\d+)\.pth$")
def episode_num(p):
    m = _EP_RE.search(p)
    return int(m.group(1)) if m else 0

def best_weight(pat):
    files = glob.glob(pat)
    return None if not files else max(files, key=lambda x:(episode_num(x), os.path.getmtime(x)))

def ask(txt, default="y"):
    tag = "[Y/n]" if default.lower()=="y" else "[y/N]"
    ans = input(f"{txt} {tag} ").strip().lower()
    return (ans=="" and default=="y") or ans.startswith("y")

def save_and_replace(label, net, old_path, add_eps):
    old = episode_num(old_path) if old_path else 0
    new = old + add_eps
    fname = f"{label.lower()}_policy_ep{new}.pth"
    if old_path and os.path.exists(old_path):
        os.remove(old_path)
        print(" removed", old_path)
    torch.save(net.state_dict(), fname)
    print(" saved -->", fname)
    return fname

def ensure_size(net, obs_len: int):
    if hasattr(net, "fc1") and net.fc1.in_features == obs_len:
        return net
    print(f" Re-initialising network for input size {obs_len}")
    return PPONet(input_dim=obs_len)

# ═══════════ single-species PPO trainer ════════════════════════════════════
def train_species(label, Env, episodes, net=None, old=None):
    if old:
        try:
            net.load_state_dict(torch.load(old))
            print(f"Loaded {label} weights --> {old}")
        except Exception as e:
            print(f"Could not load {old}, starting again({e})")

    env = Env(render=False)
    obs0 = env.reset()
    net = ensure_size(net, len(obs0))
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    gamma, eps_clip, epochs = 0.99, 0.5, 4      #PPO hyperparams
    entropy_coeff = 0.5                         #entropy go boom

    for ep in range(1, episodes+1):
        #parameter space noise
        for p in net.parameters():
            p.data += torch.randn_like(p) * 0.005

        traj, obs, done = [], env.reset(), False
        while not done:
            st = torch.tensor(obs, dtype=torch.float32)
            logits, val = net(st)
            dist = Categorical(logits=logits)
            act  = dist.sample()
            logp = dist.log_prob(act)
            obs2, rew, done, _ = env.step(act.item())
            traj.append((st, act, logp.detach(), val.squeeze(-1), rew))
            obs = obs2

        #returns & advantages
        R, G = [], 0
        for *_, r in reversed(traj):
            G = r + gamma*G
            R.insert(0, G)
        returns = torch.tensor(R, dtype=torch.float32)
        vals = torch.stack([t[3] for t in traj])
        advs = returns - vals.detach()

        S  = torch.stack([t[0] for t in traj])
        A  = torch.stack([t[1] for t in traj])
        L0 = torch.stack([t[2] for t in traj])

        for _ in range(epochs):
            logits, vals = net(S)
            dist = Categorical(logits=logits)
            logps = dist.log_prob(A)

            ratios = torch.exp(logps - L0)
            s1 = ratios * advs
            s2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advs
            loss_pi = -torch.min(s1, s2).mean()

            vals = vals.view(-1)
            loss_v = F.mse_loss(vals, returns)
            loss_e = -entropy_coeff * dist.entropy().mean()

            loss = loss_pi + 0.5*loss_v + loss_e
            opt.zero_grad(); loss.backward(); opt.step()

        if ep % 50 == 0:
            total_r = sum(t[4] for t in traj)
            print(f"{label} PPO ep {ep:4d}  Σreward {total_r:6.1f}")    #I love the sigma thingie sm lolol

    save_and_replace(label, net, old, episodes)
    return net


# ═══════════ joint trainer ════════════════════════════════════════════
def joint_train(prey_net, pred_net, episodes, prey_path, pred_path,
                freeze_prey=False, freeze_pred=False):
    env = JointEnv(render=False)
    obs_p0, obs_r0 = env.reset()
    prey_net = ensure_size(prey_net, len(obs_p0))
    pred_net = ensure_size(pred_net, len(obs_r0))

    opt_p = torch.optim.Adam(prey_net.parameters(), lr=3e-4)
    opt_r = torch.optim.Adam(pred_net.parameters(), lr=3e-4)
    gamma, eps_clip, epochs = 0.999, 0.2, 4
    entropy_coeff = 0.25

    for ep in range(1, episodes+1):
        #parameter-space noise
        for p in prey_net.parameters():
            p.data += torch.randn_like(p) * 0.005
        for p in pred_net.parameters():
            p.data += torch.randn_like(p) * 0.005

        traj_p, traj_r = [], []
        obs_p, obs_r = env.reset()
        done = False

        #inject models
        for p in env.rl_preys: p.rl_model = prey_net
        env.pred.rl_model = pred_net

        while not done:
            sp = torch.tensor(obs_p, dtype=torch.float32)
            logits_p, val_p = prey_net(sp)
            dist_p = Categorical(logits=logits_p)
            act_p  = dist_p.sample(); lp = dist_p.log_prob(act_p)

            sr = torch.tensor(obs_r, dtype=torch.float32)
            logits_r, val_r = pred_net(sr)
            dist_r = Categorical(logits=logits_r)
            act_r  = dist_r.sample(); lr = dist_r.log_prob(act_r)

            obs_p2, obs_r2, rp, rr, done, _ = env.step(act_p.item(), act_r.item())
            traj_p.append((sp,   act_p, lp.detach(), val_p.squeeze(-1), rp))
            traj_r.append((sr,   act_r, lr.detach(), val_r.squeeze(-1), rr))
            obs_p, obs_r = obs_p2, obs_r2

        # compute returns & advs 
        def comp(traj):
            R, G = [], 0
            for *_, r in reversed(traj):
                G = r + gamma*G
                R.insert(0, G)
            R = torch.tensor(R, dtype=torch.float32)
            V = torch.stack([t[3] for t in traj])
            A = R - V.detach()
            return R, A

        R_p, A_p = comp(traj_p)
        R_r, A_r = comp(traj_r)

        S_p = torch.stack([t[0] for t in traj_p])
        A_p_ct, L0_p = torch.stack([t[1] for t in traj_p]), torch.stack([t[2] for t in traj_p])
        S_r = torch.stack([t[0] for t in traj_r])
        A_r_ct, L0_r = torch.stack([t[1] for t in traj_r]), torch.stack([t[2] for t in traj_r])

        if not freeze_prey:
            for _ in range(epochs):
                logits, vals = prey_net(S_p)
                dist = Categorical(logits=logits)
                logps = dist.log_prob(A_p_ct)
                ratios = torch.exp(logps - L0_p)
                s1 = ratios * A_p
                s2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * A_p
                loss = (
                    -torch.min(s1, s2).mean()
                    + 0.5*F.mse_loss(vals.view(-1), R_p)
                    - entropy_coeff * dist.entropy().mean()
                )
                opt_p.zero_grad(); loss.backward(); opt_p.step()

        if not freeze_pred:
            for _ in range(epochs):
                logits, vals = pred_net(S_r)
                dist = Categorical(logits=logits)
                logps = dist.log_prob(A_r_ct)
                ratios = torch.exp(logps - L0_r)
                s1 = ratios * A_r
                s2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * A_r
                loss = (
                    -torch.min(s1, s2).mean()
                    + 0.5*F.mse_loss(vals.view(-1), R_r)
                    - entropy_coeff * dist.entropy().mean()
                )
                opt_r.zero_grad(); loss.backward(); opt_r.step()

        if ep % 10 == 0:
            sum_p = sum(t[4] for t in traj_p)
            sum_r = sum(t[4] for t in traj_r)
            print(f"joint PPO ep {ep:4d}  prey Σ{sum_p:6.1f}  pred Σ{sum_r:6.1f}")

    if not freeze_prey:
        save_and_replace("prey", prey_net, prey_path, episodes)
    if not freeze_pred:
        save_and_replace("predator", pred_net, pred_path, episodes)


# ═════════ simulation demo ════════════════════════════
def demo(prey_net, pred_net, seed=None):
    pygame.init()
    ep = 0
    env = JointEnv(render=True)

    while True:
        ep += 1
        obs_p, obs_r = env.reset(seed=seed)
        for p in env.rl_preys: p.rl_model = prey_net
        env.pred.rl_model = pred_net

        print(f"\n[JointEnv] Episode {ep} — SPACE/--> step, ESC to quit")
        done = False
        while not done:
            lp, _ = prey_net(torch.tensor(obs_p, dtype=torch.float32))
            act_p = torch.argmax(lp).item()
            lr, _ = pred_net(torch.tensor(obs_r, dtype=torch.float32))
            act_r = torch.argmax(lr).item()

            obs_p, obs_r, _, _, done, _ = env.step(act_p, act_r)
            env._render()

            pause = True
            while pause:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or (ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE):
                        pygame.quit(); return
                    if ev.type==pygame.KEYDOWN and ev.key in (pygame.K_SPACE, pygame.K_RIGHT):
                        pause = False
                time.sleep(0.01)


# ═════════════════ menu ════════════════════════════════════════════════
if __name__=="__main__":
    print("=== BioCode DnD ===")
    print(" 1 simulation demo\n 2 train prey\n 3 train predator\n 4 joint-train both\n 5 two-phase (pred→prey) \n 6 Showcase")
    choice = input("Enter 1-6 [1] ").strip() or "1"

    prey_path = best_weight("prey_policy_ep*.pth")
    pred_path = best_weight("predator_policy_ep*.pth")

    joint = JointEnv(render=False)
    obs0_p, obs0_r = joint.reset()
    obs_dim_p = len(obs0_p)
    obs_dim_r = len(obs0_r)

    # prey: 5 actions (0–3 move, 4 rest)
    prey_net = PPONet(input_dim=obs_dim_p, act_dim=5)
    # pred: 6 actions (0–3 move, 4 rest, 5 reproduce)
    pred_net = PPONet(input_dim=obs_dim_r, act_dim=6)

    if prey_path:
        try:
            prey_net.load_state_dict(torch.load(prey_path))
        except Exception as e:
            print(f"Skipping load {prey_path}: {e}")
    if pred_path:
        try:
            pred_net.load_state_dict(torch.load(pred_path))
        except Exception as e:
            print(f"Skipping load {pred_path}: {e}")

    if choice=="2":
        n = int(input("Episodes to train PREY [default: 1000] ") or 1000)
        prey_net = train_species("Prey", PreyEnv, n, prey_net, prey_path)
    elif choice=="3":
        n = int(input("Episodes to train PREDATOR [default: 1000] ") or 1000)
        pred_net = train_species("Predator", PredatorEnv, n, pred_net, pred_path)
    elif choice=="4":
        n = int(input("Episodes for JOINT training [default: 500] ") or 500)
        joint_train(prey_net, pred_net, n, prey_path, pred_path)
    elif choice=="5":
        n_pred = int(input("Phase A: episodes for PREDATOR (prey frozen) [defualt: 1000] ") or 1000)
        n_prey = int(input("Phase B: episodes for PREY (pred frozen)   [defualt: 1000] ") or 1000)
        print("\nPhase A — training predator …")
        joint_train(prey_net, pred_net, n_pred, prey_path, pred_path,
                    freeze_prey=True,  freeze_pred=False)
        print("\nPhase B — training prey …")
        prey_path = best_weight("prey_policy_ep*.pth")
        pred_path = best_weight("predator_policy_ep*.pth")
        joint_train(prey_net, pred_net, n_prey, prey_path, pred_path,
                    freeze_prey=False, freeze_pred=True)
    elif choice=="6":
        print("Showcase mode: collecting and displaying results...")
        run_showcase(prey_net, pred_net, runs=1000)  
        sys.exit(0)
        

    if ask("Run simulation?", "y"):
        seed = int(input("Seed (blank=random): ") or "0") or None
        demo(prey_net, pred_net, seed)
