import sys
import os
import time
import torch
import pygame

# add project root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.predator_env import PredatorEnv
from model.network import PreyPolicyNet   

pygame.init()

def evaluate(model_path, render_delay=0.5):

    state_dict = torch.load(model_path, map_location="cpu")
    out_dim = state_dict["output.weight"].shape[0]  # auto‑detect: 5 or 6

    model = PreyPolicyNet(input_dim=27, hidden_dim=64, output_dim=out_dim)
    model.load_state_dict(state_dict)
    model.eval()

    env = PredatorEnv(render=True)

    #reset env (cant belive i added an option for a seed im so smart lololol)
    obs = env.reset(seed=None)
    total_reward = 0
    done = False

    env.agent.rl_model = model        #  attach the trained net
    env.world.rl_agent = env.agent    

    while not done:
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # get current state, pass it through the model and choose an action
        state = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            probs = model(state)
        action = torch.argmax(probs).item()   #greedy for demo; no sampling

        obs, reward, done, info = env.step(action)
        total_reward += reward
        last_info = info
        print(f"[step {env.current_step}] action: {action}, reward: {reward:.2f}")

        time.sleep(render_delay)
        # wait for any key to advance to next turn
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    waiting = False

    print(f"--- Episode ended due to: {last_info.get('done_reason')} ---")
    print(f"model {os.path.basename(model_path)}: total reward = {total_reward:.2f}")
    

    #wait for any keypress before moving on to the next model
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                waiting = False
                


if __name__ == "__main__":
    # list of model files to visualize lol✨
    EPISODES = [10000, 20000, 30000]  # saved models to view
    for ep in EPISODES:
        model_file = f"predator_policy_ep{ep}.pth"
        if not os.path.exists(model_file):
            print(f"model file not found: {model_file}")
            continue
        print(f"\n--- evaluating model from episode {ep} ---")
        evaluate(model_file, render_delay=0.5)
