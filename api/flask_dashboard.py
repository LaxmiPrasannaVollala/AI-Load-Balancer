from flask import Flask, render_template
import pandas as pd
import os
import shutil

# --- Setup ---
plot_dir = r"C:/Users/laxmiprasanna/Desktop/dcs_project/rl/rl/plots"
static_dir = r"C:/Users/laxmiprasanna/Desktop/dcs_project/dashboard/static"

# Ensure the static directory exists
os.makedirs(static_dir, exist_ok=True)

# Copy latest plots
plot_files = ["test_rewards.png", "test_failed_servers.png", "test_actions.png", "test_server_loads.png","server_usage_distribution.png"]
for f in plot_files:
    src = os.path.join(plot_dir, f)
    dst = os.path.join(static_dir, f)
    if os.path.exists(src):
        shutil.copy(src, dst)

# --- Flask App ---
app = Flask(__name__)

@app.route("/")
def index():
    csv_path = r"C:/Users/laxmiprasanna/Desktop/dcs_project/rl/rl/outputs/test_results.csv"
    print(f"CSV path: {csv_path}")
    df = pd.read_csv(csv_path)
    table_html = df.to_html(classes='data', index=False)
    return render_template("index.html", table_data=table_html)

# Only run this once when actually starting
if __name__ == "__main__":
    app.run(debug=True)
