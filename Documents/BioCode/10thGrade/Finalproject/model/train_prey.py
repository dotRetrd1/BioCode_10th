import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from network import PreyPolicyNet
from env.prey_env import PreyEnv
import numpy as np

def discount_rewards(rewards, gamma=0.99):
    if len(rewards) == 0:
        return [0.0]
    discounted = []
    r = 0
    for reward in reversed(rewards):
        r = reward + gamma * r
        discounted.insert(0, r)
    return discounted

def train(num_episodes=30000, render=False, save_every=10000):
    env = PreyEnv(render=render)
    model = PreyPolicyNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for episode in range(1, num_episodes + 1):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        log_probs = []
        rewards = []

        done = False
        while not done:
            probs = model(state)

            #check for NaNs
            if torch.isnan(probs).any():
                print("NaNs in action probabilities, skipping episode.")
                break

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)

            next_state, reward, done, _ = env.step(action.item())
            rewards.append(reward)
            state = torch.tensor(next_state, dtype=torch.float32)

        if len(log_probs) == 0 or len(rewards) == 0:
            continue

        #calculate discounted rewards
        discounted = discount_rewards(rewards)
        discounted = torch.tensor(discounted, dtype=torch.float32)

        #normalize rewards
        if len(discounted) > 1:
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-6)

        #policy loss
        loss = -torch.sum(torch.stack(log_probs) * discounted)

        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        #print progress
        total_reward = sum(rewards)
        print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Steps = {len(rewards)}")

        #save
        if episode % save_every == 0:
            torch.save(model.state_dict(), f"prey_policy_ep{episode}.pth")

    #torch.save(model.state_dict(), "prey_policy_final.pth")
    print("Training finished. Final model saved.")

if __name__ == "__main__":
    train()
