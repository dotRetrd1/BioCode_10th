#plan: (pseudocode be going crazy here >:) )  
#1. run 100 runs and collect average results, winning rates, most common action, win reasons, ticks till end, reproductin amount per run
#2. collect everything from the top runs with the highest reproduction amount, ticks till end, kills etc
#3. plot the results in fancy graphs 
#4. show the results in a sort of nice slides sort of way via the renderer: 
    #slide 1: title, description, author, date
    #slide 2: showcase graphs of average next to a timeloop of all the runs running very quickly 
    #slide 3: showcase the top runs also with a loop graphic visualization of them as a sim
import os
import time
import numpy as np
import pygame
import torch
import plotly.express as px
from collections import Counter
from env.joint_env import JointEnv
from model.ppo_agent import PPONet
from collector import collect_stats, RESULTS_PATH
from net_utils  import AdaptNet  

defaultRuns = 1000
STEPS_PER_TIMELAPSE = 100

# ────────────────────────────
#  Net helpers
# ────────────────────────────
class AdaptNet(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.in_dim = base.fc1.in_features

    def forward(self, x):
        if x.numel() != self.in_dim:
            if x.numel() > self.in_dim:             # truncate
                x = x[:self.in_dim]
            else:                                   # zero-pad
                pad = torch.zeros(
                    self.in_dim - x.numel(),
                    dtype=x.dtype,
                    device=x.device
                )
                x = torch.cat([x, pad])
        return self.base(x)


def ensure_size(net: PPONet, n: int) -> PPONet:
    """Wrap in AdaptNet if obs length mismatches, preserving weights."""
    if hasattr(net, "fc1") and net.fc1.in_features == n:
        return net
    return AdaptNet(net)

# ────────────────────────────
#  Data collection & metrics
# ────────────────────────────
def collect_runs(pre_net, pred_net, runs=defaultRuns):
    records = []
    for seed in range(runs):
        env = JointEnv(render=False)
        obs_p, obs_r = env.reset(seed=seed)

        pre_net  = ensure_size(pre_net,  len(obs_p))
        pred_net = ensure_size(pred_net, len(obs_r))
        for p in env.rl_preys:
            p.rl_model = pre_net
        env.pred.rl_model = pred_net

        episode = dict(ticks=0, reps=0, kills=0,
                       win=None, reason="", actions=[])
        done = False
        while not done:
            a_p = int(torch.argmax(pre_net(torch.tensor(obs_p, dtype=torch.float32))[0]))
            a_r = int(torch.argmax(pred_net(torch.tensor(obs_r, dtype=torch.float32))[0]))
            obs_p, obs_r, rp, rr, done, info = env.step(a_p, a_r)

            episode["ticks"] += 1
            episode["actions"].append(a_p)
            episode["reps"]  += info.get("reproduction_count", 0)
            episode["kills"] += info.get("kill", 0)

        episode["win"]    = "prey" if rp > rr else "pred"
        episode["reason"] = info.get("win_reason", "")
        records.append(episode)
    return records


def metrics(rs):
    pred_tot = Counter()
    prey_tot = Counter()

    for r in rs:
        pred_tot.update(r.get("pred_act", {}))
        prey_tot.update(r.get("prey_act", {}))

    return dict(
        ticks =[r["ticks"]  for r in rs],
        reps  =[r["reps"]   for r in rs],
        kills =[r["kills"]  for r in rs],
        food  =[r["food"]   for r in rs],
        win   =Counter(r["win"] for r in rs),
        pred_act = pred_tot,
        prey_act = prey_tot,
        acts  =Counter(   
            a for r in rs for a in r.get("actions", [])
        ).most_common(5)
    )

# ────────────────────────────
#  Plot generation
# ────────────────────────────

GRAPH_W, GRAPH_H = 3200/1.6, 2400/1.6
LAYOUT = dict(
    font=dict(size=70),
    title_font_size=100,
    margin=dict(l=80, r=80, t=250, b=80),  
    xaxis_visible=True,
    yaxis_visible=True
)
MARGIN = 15


def plot_metrics(m, out_dir="showcase_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # Pie – kills
    kills_ct = Counter(m["kills"]).most_common(defaultRuns)
    if kills_ct:
        lbl, val = zip(*kills_ct)
        fig = px.pie(values=val,
                     names=[f"{v} kills" for v in lbl],
                     title="Kills per run")
        fig.update_layout(**LAYOUT)
        p = os.path.join(out_dir, "kills_pie.png")
        fig.write_image(p, width=GRAPH_W, height=GRAPH_H)
        paths.append(p)

    # Pie – reproductions 
    rep_ct = Counter(m["reps"]).most_common(defaultRuns)
    if rep_ct:
        lbl, val = zip(*rep_ct)
        fig = px.pie(values=val,
                     names=[f"{v} reps" for v in lbl],
                     title="Reproductions per run")
        fig.update_layout(**LAYOUT)
        p = os.path.join(out_dir, "rep_pie.png")
        fig.write_image(p, width=GRAPH_W, height=GRAPH_H)
        paths.append(p)

    # bar - Win rate 
    fig = px.bar(x=list(m["win"].keys()),
                 y=list(m["win"].values()),
                 title="Wins")
    fig.update_layout(**LAYOUT)
    p = os.path.join(out_dir, "win_rates.png")
    fig.write_image(p, width=GRAPH_W, height=GRAPH_H)
    paths.append(p)

    # histogram - Ticks 
    fig = px.histogram(x=m["ticks"], nbins=25, title="Ticks per run (broken rn ಥ_ಥ) (I think)")
    fig.update_layout(font=dict(size=60),
                      title_font_size=100,
                      margin=dict(l=80, r=80, t=250, b=80),  
                      yaxis_visible=True, xaxis_visible=False)
    fig.update_xaxes(showgrid=True,
                     gridwidth=4,
                     gridcolor="rgba(255,255,255,0.2)",
                     griddash="dot")
    p = os.path.join(out_dir, "ticks_hist.png")
    fig.write_image(p, width=GRAPH_W, height=GRAPH_H)
    paths.append(p)

    # pie - food eaten
    food_ct = Counter(m["food"]).most_common(defaultRuns)
    if food_ct:     
        lbl, val = zip(*food_ct)
        fig = px.pie(values=val, names=[f"{v} food" for v in lbl],
                     title="Food eaten per run")
        fig.update_layout(**LAYOUT)
        p = os.path.join(out_dir, "food_pie.png")
        fig.write_image(p, width=GRAPH_W, height=GRAPH_H)
        paths.append(p)

    # Predator action counts
    if m["pred_act"]:
        fig = px.bar(x=list(m["pred_act"].keys()),
                     y=list(m["pred_act"].values()),
                     title="Predator action counts")
        fig.update_layout(**LAYOUT)
        p = os.path.join(out_dir, "pred_actions.png")
        fig.write_image(p, width=GRAPH_W, height=GRAPH_H)
        paths.append(p)

    # Prey action counts
    if m["prey_act"]:
        fig = px.bar(x=list(m["prey_act"].keys()),
                     y=list(m["prey_act"].values()),
                     title="Prey action counts")
        fig.update_layout(**LAYOUT)
        p = os.path.join(out_dir, "prey_actions.png")
        fig.write_image(p, width=GRAPH_W, height=GRAPH_H)
        paths.append(p)

    return paths

# ────────────────────────────
#  Pygame helpers
# ────────────────────────────

def scale_fit(surf, max_w, max_h):
    w, h = surf.get_size()
    s = min(max_w / w, max_h / h, 1)
    return pygame.transform.smoothscale(surf, (int(w * s), int(h * s)))


def load_surfs(paths, cw, ch):
    return [scale_fit(pygame.image.load(p), cw, ch) for p in paths]


# ─── frame recorder ──────────────────────────────
def record_frames(pre_net, pred_net, *, seed: int, steps: int, size):
    import renderer                        

    # ensure pygame + font + surface exist
    if not pygame.get_init():
        pygame.init()
    pygame.font.init()

    surf = pygame.display.get_surface()
    if surf is None or surf.get_size() != size:
        surf = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)

    renderer.small_font = pygame.font.SysFont(None, 18)
    renderer.font       = pygame.font.SysFont(None, 36)
    renderer.screen     = surf                              # share surface

    # build env and inject wrapped nets
    env = JointEnv(render=True)
    obs_p, obs_r = env.reset(seed=seed)

    pre  = ensure_size(pre_net,  len(obs_p))
    pred = ensure_size(pred_net, len(obs_r))
    for p in env.rl_preys:
        p.rl_model = pre
    env.pred.rl_model = pred

    env._render()                                           # first frame

    # capture exactly `steps` frames, restarting new episodes as needed
    frames, done, t = [], False, 0
    while t < steps:
        if done:                                           # episode ended
            obs_p, obs_r = env.reset()
            for p in env.rl_preys: p.rl_model = pre
            env.pred.rl_model = pred
            env._render()
            done = False

        a_p = int(torch.argmax(pre(torch.tensor(obs_p, dtype=torch.float32))[0]))
        a_r = int(torch.argmax(pred(torch.tensor(obs_r, dtype=torch.float32))[0]))

        obs_p, obs_r, _, _, done, _ = env.step(a_p, a_r)
        env._render()

        frames.append(scale_fit(surf.copy(), *size))
        t += 1

    return frames




# ────────────────────────────
#  Showcase
# ────────────────────────────
def run_showcase(pre_net: PPONet, pred_net: PPONet, runs: int = defaultRuns) -> None:
    # ── 1 make sure cached raw episodes ─────────────
    import json, os

    if not os.path.exists(RESULTS_PATH):
        print(f"[showcase] {RESULTS_PATH} missing → generating {runs} episodes …")
        collect_stats(pre_net, pred_net, runs)      # writes run_stats.json

    with open(RESULTS_PATH, "r", encoding="utf-8") as fp:
        raw = json.load(fp)
    print(f"[showcase] loaded {len(raw)} cached episodes")

    # ── 2  metrics & graphs  ────────────────
    m        = metrics(raw)
    graph_fp = plot_metrics(m)

    # ── 3 window & layout stuff ─────────────────
    pygame.init()
    screen = pygame.display.set_mode((0, 0))
    W, H   = screen.get_size()

    COLS, ROWS = 3, 2                       # 3 × 2 grid
    left_w = int(W * 0.69) - MARGIN * 2     # 69 % of width for graphs (¬‿¬)
    left_h = H - MARGIN * 2
    cell_w = (left_w - (COLS - 1) * MARGIN) // COLS
    cell_h = (left_h - (ROWS - 1) * MARGIN) // ROWS
    g_surfs = load_surfs(graph_fp[:COLS * ROWS], cell_w, cell_h)

    sim_size  = (5000, 5000)
    all_frames = record_frames(pre_net,  pred_net, seed=0,
                               steps=STEPS_PER_TIMELAPSE, size=sim_size)
    top_seed   = max(range(len(raw)), key=lambda i: raw[i]["reps"])
    top_frames = record_frames(pre_net,  pred_net, seed=top_seed,
                               steps=STEPS_PER_TIMELAPSE, size=sim_size)

    # ── 4 slide-show loop ───────────────────────────────────
    slide, i_all, i_top = 1, 0, 0
    font  = pygame.font.SysFont(None, 80)
    clock = pygame.time.Clock()

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                slide = slide % 3 + 1

        screen.fill((0, 0, 0))

        if slide == 1:
            for i, txt in enumerate(
                ["BioCode DND Showcase", "by ido (^_^)", "10th grade, 2024-2025"]
            ):
                surf = font.render(txt, True, (255, 255, 255))
                rect = surf.get_rect(center=(W // 2, H // 2 - 120 + i * 120))
                screen.blit(surf, rect)

        else:
            # graph grid
            for idx, surf in enumerate(g_surfs):
                r, c = divmod(idx, COLS)
                x = MARGIN + c * (cell_w + MARGIN)
                y = MARGIN * 2 + r * (cell_h + 10)      #10 is hardcoded for the rowgap, it doesnt even matter ok jstu chill is fine :p
                screen.blit(surf, (x, y))

            # simulation pane
            frame = (all_frames[i_all % len(all_frames)]
                     if slide == 2 else
                     top_frames[i_top % len(top_frames)])
            screen.blit(frame, (left_w + MARGIN * 2, MARGIN))
            if slide == 2:
                i_all += 1
            else:
                i_top += 1

        pygame.display.flip()
        clock.tick(15)

    pygame.quit()
