import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

from rl_environment import LoadBalancerEnv

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

def validate(policy, env, steps=50):
    env.reset()
    rewards = []
    failed_counts = []
    actions_taken = []

    for step in range(steps):
        state = torch.FloatTensor(env.server_loads)
        with torch.no_grad():
            probs = policy(state)
        action = torch.argmax(probs).item()

        _, reward, failed = env.step(action)
        rewards.append(reward)
        failed_counts.append(len(failed))
        actions_taken.append(action)

        print(f"Step {step+1}: Action {action}, Reward: {reward:.4f}, Failed Servers: {failed}")

    # Plot validation reward
    plt.figure(figsize=(10, 4))
    plt.plot(range(steps), rewards)
    plt.title("Validation Rewards")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rl/plots/validation_rewards.png")
    plt.close()

    # Plot failed server count
    plt.figure(figsize=(10, 3))
    plt.plot(range(steps), failed_counts, color='red')
    plt.title("Failed Servers During Validation")
    plt.xlabel("Step")
    plt.ylabel("Failed Servers")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rl/plots/validation_failed_servers.png")
    plt.close()

    print("✅ Validation complete. Plots saved in rl/plots/")

def train(env, episodes=200, gamma=0.99, lr=0.01, entropy_beta=0.01):
    policy = PolicyNetwork(env.num_servers, env.num_servers)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_rewards = []

    for episode in range(episodes):
        env.reset()
        log_probs = []
        rewards = []
        entropies = []

        for step in range(50):
            state = torch.FloatTensor(env.server_loads)
            probs = policy(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            _, reward, _ = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, R, entropy in zip(log_probs, returns, entropies):
            loss -= log_prob * R + entropy_beta * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        all_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{episodes} | Total Reward: {episode_reward:.2f}")

    # Plot reward trend
    plt.figure(figsize=(10, 4))
    plt.plot(all_rewards)
    plt.title("Training Reward Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("rl/plots", exist_ok=True)
    plt.savefig("rl/plots/training_rewards.png")
    plt.close()
    torch.save(policy.state_dict(), "rl/policy_model.pth")
    print("✅ Training complete. Plot saved in rl/plots/")

    # Run validation after training
    print("\n🔍 Running validation...")
    validate(policy, env)

if __name__ == "__main__":
    env = LoadBalancerEnv()
    train(env)
