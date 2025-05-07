import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from collections import Counter

from rl_environment import LoadBalancerEnv
from train_agent import PolicyNetwork

# Load the environment and trained policy
env = LoadBalancerEnv()
policy = PolicyNetwork(env.num_servers, env.num_servers)

model_path = "C:\\Users\\laxmiprasanna\\Desktop\\dcs_project\\rl\\rl\\policy_model.pth"
assert os.path.exists(model_path), "Trained model not found. Please train the agent first."
policy.load_state_dict(torch.load(model_path))
policy.eval()

# Run test episode
env.reset()
rewards = []
failed_counts = []
actions_taken = []
server_loads_over_time = []

'''for step in range(50):
    state = torch.FloatTensor(env.server_loads)
    with torch.no_grad():
        probs = policy(state)
    action = torch.argmax(probs).item()  # Deterministic choice
    _, reward, failed = env.step(action)

    rewards.append(reward)
    failed_counts.append(len(failed))
    actions_taken.append(action)
    server_loads_over_time.append(env.server_loads.copy())

    print(f"Step {step+1}: Action {action}, Reward: {reward:.4f}, Failed Servers: {failed}")'''
for step in range(50):
    state = torch.FloatTensor(env.server_loads)
    with torch.no_grad():
        probs = policy(state)
    action = torch.multinomial(probs, 1).item()  # 👈 Stochastic choice

    _, reward, failed = env.step(action)

    rewards.append(reward)
    failed_counts.append(len(failed))
    actions_taken.append(action)
    server_loads_over_time.append(env.server_loads.copy())

    print(f"Step {step+1}: Action {action}, Reward: {reward:.4f}, Failed Servers: {failed}")


# Create output directory
os.makedirs("rl/plots", exist_ok=True)
os.makedirs("rl/outputs", exist_ok=True)

# Export test results to CSV
results_df = pd.DataFrame({
    "Step": list(range(1, 51)),
    "Action": actions_taken,
    "Reward": rewards,
    "FailedServers": failed_counts
})
results_df.to_csv("rl/outputs/test_results.csv", index=False)

# Plot: Rewards over time
plt.figure(figsize=(10, 4))
plt.plot(rewards, color='green')
plt.title("Test Rewards Over Time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("rl/plots/test_rewards.png")
plt.close()

# Plot: Failed server counts
plt.figure(figsize=(10, 3))
plt.plot(failed_counts, color='red')
plt.title("Failed Server Count During Test")
plt.xlabel("Step")
plt.ylabel("Failed Servers")
plt.grid(True)
plt.tight_layout()
plt.savefig("rl/plots/test_failed_servers.png")
plt.close()

# Plot: Actions taken
plt.figure(figsize=(10, 3))
plt.plot(actions_taken, color='blue')
plt.title("Actions Taken During Test")
plt.xlabel("Step")
plt.ylabel("Server Chosen")
plt.grid(True)
plt.tight_layout()
plt.savefig("rl/plots/test_actions.png")
plt.close()

# Analyze and visualize server usage
action_counts = Counter(actions_taken)

# Plot server selection frequency
plt.figure(figsize=(6, 4))
servers = list(action_counts.keys())
frequencies = list(action_counts.values())
plt.bar(servers, frequencies, color='orange')
plt.title("Server Usage Frequency")
plt.xlabel("Server ID")
plt.ylabel("Times Selected")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("rl/plots/server_usage_distribution.png")
plt.close()


# Plot: Server loads over time with unused server warning
server_loads_array = np.array(server_loads_over_time)
plt.figure(figsize=(10, 4))
for i in range(env.num_servers):
    if np.any(server_loads_array[:, i]):
        plt.plot(server_loads_array[:, i], label=f"Server {i}")
    else:
        print(f"⚠️ Server {i} was never used.")
plt.title("Server Loads During Test")
plt.xlabel("Step")
plt.ylabel("Load")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rl/plots/test_server_loads.png")
plt.close()

# Print action distribution
print("\n🔎 Action Distribution:", Counter(actions_taken))
print("\n✅ Test run complete. Plots saved in rl/plots/ and data exported to rl/outputs/test_results.csv")
