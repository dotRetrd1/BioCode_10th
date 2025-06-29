import sys
import os
import time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.network import PreyPolicyNet
from env.prey_env import PreyEnv
import pygame
pygame.init()

EPISODES = [10000, 20000, 30000]  #saved models to view

def evaluate(model_path, render_delay=0.5):
    env = PreyEnv(render=True)

    state_dict = torch.load(model_path, map_location="cpu")
    out_dim = state_dict["output.weight"].shape[0]
    model = PreyPolicyNet(input_dim=27, hidden_dim=64, output_dim=out_dim)
    model.eval()
    model.load_state_dict(state_dict)

    obs = env.reset() 
    total_reward = 0
    done = False

    while not done:
        env.render()

        # choose action using policy
        state = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            probs = model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        obs, reward, done, info = env.step(action)
        print(f"Step {env.current_step}: Action {action}, Reward {reward}")
        if done:    #print end reaeson
            print(f"--- Episode ended due to: {info.get('done_reason')} ---")
        total_reward += reward

        #wait for any key to advance to next turn
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    waiting = False


    
    print(f"{os.path.basename(model_path)}: Total Reward = {total_reward}")

    
    def wait_for_key():
        font = pygame.font.SysFont("Arial", 24)
        screen = pygame.display.get_surface()
        message = font.render("Press any key for next model...", True, (255, 255, 255))
        rect = message.get_rect(center=(screen.get_width() // 2, screen.get_height() - 20))
        screen.blit(message, rect)
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    waiting = False
    wait_for_key()

if __name__ == "__main__":
    for ep in EPISODES:
        print(f"\n--- Evaluating model from Episode {ep} ---")
        model_file = f"prey_policy_ep{ep}.pth"
        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}")
            continue
        evaluate(model_file)

