import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import re
from sklearn.metrics import r2_score  # For R-squared calculation

# Path to the directory containing log files
log_dir = "results_predators_prey/logs/"

# Get all log files in the directory
log_files = glob.glob(os.path.join(log_dir, "*.txt"))  # Adjust the pattern if necessary

# Check if there are any log files
if not log_files:
    print("No log files found in the directory:", log_dir)
    exit()

# Initialize dictionaries to store grouped data
grouped_by_preys_mode = {}
grouped_by_MARLAlgorithm = {}

# Lists to store average scores for each run
average_scores = {}

# List to store all runs data
all_runs_data = []

# Process each log file
for log_file_path in log_files:
    filename = os.path.basename(log_file_path)
    print(f"Processing: {filename}")

    preys_mode = None
    MARLAlgorithm = None
    episodes = []
    scores = []

    # Read the log file
    with open(log_file_path, "r") as log_file:
        lines = log_file.readlines()

    # Parse Experiment Parameters
    for line in lines:
        if line.startswith("preys_mode:"):
            try:
                preys_mode = line.strip().split(":")[1].strip()
            except IndexError:
                preys_mode = "Unknown"
        elif line.startswith("MARLAlgorithm:"):
            try:
                MARLAlgorithm = line.strip().split(":")[1].strip()
            except IndexError:
                MARLAlgorithm = "Unknown"
        elif line.startswith("Episode"):
            # Start parsing episode data
            break  # Exit the loop to start parsing episodes

    # Validate extracted parameters
    if preys_mode is None or MARLAlgorithm is None:
        print(f"Failed to extract 'preys_mode' or 'MARLAlgorithm' from {filename}. Skipping this file.")
        continue

    # Parse Episode Data
    for line in lines:
        if line.startswith("Episode"):
            try:
                parts = line.strip().split(", ")
                episode = int(parts[0].split(" ")[1])
                score = float(parts[1].split(": ")[1])
                episodes.append(episode)
                scores.append(score)
            except (IndexError, ValueError):
                # Skip lines that don't match the expected format
                continue

    if not episodes:
        print(f"No valid episode data found in '{filename}'. Skipping.")
        continue

    # Compute the average score for this run
    avg_score = sum(scores) / len(scores)
    average_scores[log_file_path] = avg_score

    # Store run data
    run_data = {
        'filename': filename,
        'preys_mode': preys_mode,
        'MARLAlgorithm': MARLAlgorithm,
        'episodes': episodes,
        'scores': scores,
        'average_score': avg_score
    }
    all_runs_data.append(run_data)

    # Add to grouped dictionaries
    if preys_mode not in grouped_by_preys_mode:
        grouped_by_preys_mode[preys_mode] = []
    grouped_by_preys_mode[preys_mode].append(run_data)

    if MARLAlgorithm not in grouped_by_MARLAlgorithm:
        grouped_by_MARLAlgorithm[MARLAlgorithm] = []
    grouped_by_MARLAlgorithm[MARLAlgorithm].append(run_data)

# Check if any runs were processed
if not all_runs_data:
    print("No valid runs to plot.")
    exit()

# Print average scores for all runs
print("\nAverage Scores for Each Run:")
for run in all_runs_data:
    print(f"{run['filename']}: {run['average_score']:.2f}")

# Find and display the best run
best_run = max(all_runs_data, key=lambda x: x['average_score'])
print("\nBest Run:")
print(f"{best_run['filename']} with an average score of {best_run['average_score']:.2f}")

# ================== Original Plot: Scores Over Episodes ==================

plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(all_runs_data)))  # Assign distinct colors

for i, run in enumerate(all_runs_data):
    episodes = run['episodes']
    scores = run['scores']
    filename = run['filename']
    avg_score = run['average_score']
    color = colors[i]

    plt.plot(episodes, scores, marker="o", label=f"{filename} (Avg: {avg_score:.2f})", color=color)
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Scores Over Episodes for Multiple Runs")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to accommodate legend
plt.show()

# ================== New Plot: Lines of Best Fit ==================

plt.figure(figsize=(12, 8))
for i, run in enumerate(all_runs_data):
    episodes = run['episodes']
    scores = run['scores']
    filename = run['filename']
    avg_score = run['average_score']
    color = colors[i]

    if len(episodes) < 2:
        print(f"Not enough data points for {filename} to compute a best fit line.")
        continue

    # Convert lists to numpy arrays for polyfit
    episodes_np = np.array(episodes)
    scores_np = np.array(scores)

    # Perform linear regression (degree=1)
    slope, intercept = np.polyfit(episodes_np, scores_np, 1)
    best_fit = slope * episodes_np + intercept

    # Calculate R-squared
    r2 = r2_score(scores_np, best_fit)

    # Plot the best fit line
    plt.plot(episodes_np, best_fit, linestyle='--', color=color, label=f"{filename} Best Fit (R²={r2:.2f})")

plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Lines of Best Fit for Multiple Runs")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to accommodate legend
plt.show()

# ================== Plotting by preys_mode ==================

for preys_mode, runs in grouped_by_preys_mode.items():
    plt.figure(figsize=(12, 8))
    colors_pm = plt.cm.tab10(np.linspace(0, 1, len(runs)))  # Assign distinct colors

    for i, run in enumerate(runs):
        episodes = run['episodes']
        scores = run['scores']
        filename = run['filename']
        avg_score = run['average_score']
        color = colors_pm[i]

        # Plot actual scores
        plt.plot(episodes, scores, marker="o", label=f"{filename} (Avg: {avg_score:.2f})", color=color)

        # Compute and plot line of best fit
        if len(episodes) >= 2:
            episodes_np = np.array(episodes)
            scores_np = np.array(scores)
            slope, intercept = np.polyfit(episodes_np, scores_np, 1)
            best_fit = slope * episodes_np + intercept
            r2 = r2_score(scores_np, best_fit)
            plt.plot(episodes_np, best_fit, linestyle='--', color=color, label=f"{filename} Best Fit (R²={r2:.2f})")
        else:
            print(f"Not enough data points for {filename} to compute a best fit line.")

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(f"Scores Over Episodes for Preys Mode {preys_mode}")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to accommodate legend
    plt.show()

# ================== Plotting by MARLAlgorithm ==================

for algorithm, runs in grouped_by_MARLAlgorithm.items():
    plt.figure(figsize=(12, 8))
    colors_ma = plt.cm.tab10(np.linspace(0, 1, len(runs)))  # Assign distinct colors

    for i, run in enumerate(runs):
        episodes = run['episodes']
        scores = run['scores']
        filename = run['filename']
        avg_score = run['average_score']
        color = colors_ma[i]

        # Plot actual scores
        plt.plot(episodes, scores, marker="o", label=f"{filename} (Avg: {avg_score:.2f})", color=color)

        # Compute and plot line of best fit
        if len(episodes) >= 2:
            episodes_np = np.array(episodes)
            scores_np = np.array(scores)
            slope, intercept = np.polyfit(episodes_np, scores_np, 1)
            best_fit = slope * episodes_np + intercept
            r2 = r2_score(scores_np, best_fit)
            plt.plot(episodes_np, best_fit, linestyle='--', color=color, label=f"{filename} Best Fit (R²={r2:.2f})")
        else:
            print(f"Not enough data points for {filename} to compute a best fit line.")

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(f"Scores Over Episodes for MARL Algorithm: {algorithm}")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to accommodate legend
    plt.show()
