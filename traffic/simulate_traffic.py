import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_traffic(days=365, interval='5min'):
    # 1 year of 5-minute intervals = ~105,120 samples
    timestamps = pd.date_range(start="2023-01-01", periods=int(24 * (60 // 5) * days), freq=interval)

    # Simulate daily + weekly seasonality
    total_points = len(timestamps)
    t = np.arange(total_points)

    # Daily pattern (sin wave every 24h)
    daily = 50 * np.sin(2 * np.pi * t / (24 * 12))  # 12 samples/hour

    # Weekly pattern (sin wave every 7 days)
    weekly = 30 * np.sin(2 * np.pi * t / (24 * 12 * 7))

    # Long-term trend (optional)
    trend = 0.01 * t

    # Random noise
    noise = np.random.normal(0, 10, total_points)

    # Combine all components
    traffic = 150 + daily + weekly + trend + noise

    # Create and save DataFrame
    df = pd.DataFrame({'timestamp': timestamps, 'requests': traffic})
     # ✅ Create folder if it doesn't exist
    #os.makedirs("traffic", exist_ok=True)
    #df.to_csv("traffic/traffic_data.csv", index=False)
    folder = os.path.join(os.getcwd(), "traffic_visualization")
    os.makedirs(folder, exist_ok=True)

    df.to_csv(os.path.join(folder, "traffic_data.csv"), index=False)

    print("✅ Traffic data saved to traffic_visualization/traffic_data.csv")

    # Plot sample week for visualization
    sample = df.head(7 * 24 * 12)  # 1 week
    plt.figure(figsize=(14, 4))
    plt.plot(sample['timestamp'], sample['requests'], label="Synthetic Traffic (1 week)")
    plt.title("Simulated Web Traffic - Weekly View")
    plt.xlabel("Time")
    plt.ylabel("Requests per 5 minutes")
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    #plt.savefig("traffic/traffic_plot.png")
    plt.savefig(os.path.join(folder, "traffic_plot.png"))
    plt.show()

if __name__ == '__main__':
    generate_traffic()