import numpy as np
import matplotlib.pyplot as plt
import os

class LoadBalancerEnv:
    def __init__(self, num_servers=4, threshold=1.0, decay=0.1):
        self.num_servers = num_servers
        #self.threshold = threshold
        #self.decay = decay
        self.threshold = 3.0
        self.decay = 0.2
        self.server_loads = np.zeros(num_servers)
        self.step_count = 0

        # ✅ Load LSTM predictions
        path = "C:\\Users\\laxmiprasanna\\Desktop\\dcs_project\\prediction\\prediction\\predicted_traffic.npy"
        if os.path.exists(path):
            self.predicted_traffic = np.load(path)
            print("✅ Loaded predicted traffic.")
        else:
            raise FileNotFoundError("Predicted traffic not found. Run LSTM predictor first.")
        
        self.history = {
            'step': [],
            'loads': [],
            'rewards': [],
            'failed_counts': [],
            'routed_to': []
        }

    def step(self, action):
        #load = self.predicted_traffic[self.step_count % len(self.predicted_traffic)] / 1000  # Normalize scale
        load = self.predicted_traffic[self.step_count % len(self.predicted_traffic)] / 1500
        self.server_loads *= (1 - self.decay)
        self.server_loads[action] += load

        failed_servers = {i for i, l in enumerate(self.server_loads) if l > self.threshold}
        reward = -np.std(self.server_loads)
        if action in failed_servers:
            reward -= 1.0

        # 📊 Logging
        self.history['step'].append(self.step_count)
        self.history['loads'].append(self.server_loads.copy())
        self.history['rewards'].append(reward)
        self.history['failed_counts'].append(len(failed_servers))
        self.history['routed_to'].append(action)

        self.step_count += 1
        return self.server_loads, reward, failed_servers

    def reset(self):
        self.server_loads = np.zeros(self.num_servers)
        self.step_count = 0
        self.history = {
            'step': [],
            'loads': [],
            'rewards': [],
            'failed_counts': [],
            'routed_to': []
        }

    def render_summary(self):
        os.makedirs("rl/plots", exist_ok=True)
        loads_arr = np.array(self.history['loads'])

        # Server Load
        plt.figure(figsize=(10, 4))
        for i in range(self.num_servers):
            plt.plot(self.history['step'], loads_arr[:, i], label=f"Server {i}")
        plt.title("Server Loads Over Time")
        plt.xlabel("Step")
        plt.ylabel("Load")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("rl/plots/server_loads.png")
        plt.close()

        # Rewards
        plt.figure(figsize=(10, 3))
        plt.plot(self.history['step'], self.history['rewards'], color='green')
        plt.title("Reward Over Time")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("rl/plots/rewards.png")
        plt.close()

        # Failed Server Count
        plt.figure(figsize=(10, 3))
        plt.plot(self.history['step'], self.history['failed_counts'], color='red')
        plt.title("Failed Server Count Over Time")
        plt.xlabel("Step")
        plt.ylabel("Failed Servers")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("rl/plots/failed_servers.png")
        plt.close()

        print("✅ Visualizations saved in rl/plots/")

# ✅ Quick test mode
if __name__ == "__main__":
    env = LoadBalancerEnv()
    env.reset()
    for _ in range(50):
        action = np.random.randint(env.num_servers)
        server_loads, reward, failed = env.step(action)
        print(f"Step {env.step_count}: Routed to Server {action} | Loads: {server_loads.round(2)} | Reward: {reward:.4f} | Failed: {failed}")
    env.render_summary()
