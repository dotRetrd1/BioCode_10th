import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
from model.network import PreyPolicyNet  
from env.predator_env import PredatorEnv
import numpy as np

def discount_rewards(rewards, gamma=0.99):
    discounted = []
    r = 0
    for reward in reversed(rewards):
        r = reward + gamma * r
        discounted.insert(0, r)
    return discounted

def train(num_episodes=30000, render=False, save_every=10000):
    env = PredatorEnv(render=render)
    #same network architecture, adjusting input_dim
    model = PreyPolicyNet(input_dim=27, hidden_dim=64, output_dim=6)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for episode in range(1, num_episodes + 1):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        log_probs = []
        rewards = []
        done = False
        
        while not done:
            probs = model(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            next_state, reward, done, info = env.step(action.item())
            rewards.append(reward)
            state = torch.tensor(next_state, dtype=torch.float32)
        
        discounted = discount_rewards(rewards)
        discounted = torch.tensor(discounted, dtype=torch.float32)
        if len(discounted) > 1:
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-6)
        loss = -torch.sum(torch.stack(log_probs) * discounted)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_reward = sum(rewards)
        print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Steps = {len(rewards)}")
        if episode % save_every == 0:
            torch.save(model.state_dict(), f"predator_policy_ep{episode}.pth")
        if episode % num_episodes ==  1:
            torch.save(model.state_dict(), f"predator_policy_ep{episode}.pth")
            print("Initial model saved.")
            
            
    #torch.save(model.state_dict(), "predator_policy_final.pth")
    print("Training finished saved final model.")

if __name__ == "__main__":
    train()
