import os
import psutil  # To check memory and CPU usage dynamically
from concurrent.futures import ThreadPoolExecutor
import itertools
import subprocess


# Function to determine the number of CPU cores (compatible with Python 2.7)
def get_cpu_count():
    try:
        return psutil.cpu_count(logical=True)  # Logical cores
    except AttributeError:
        # Fallback method for older psutil versions
        return psutil.NUM_CPUS


# Function to determine the optimal number of workers
def determine_optimal_workers():
    cpu_count = get_cpu_count()  # Total number of CPU cores
    memory_info = psutil.virtual_memory()  # System memory info

    # Assume each worker requires 1GB of RAM
    available_memory_gb = memory_info.available / (1024 ** 3)
    memory_based_workers = int(available_memory_gb/0.33)  # One worker per GB of free memory

    # Use half the CPU cores for workers to avoid overloading the system
    cpu_based_workers = max(1, cpu_count // 2)

    # Final optimal worker count is the minimum of CPU and memory-based estimates
    optimal_workers = min(memory_based_workers, cpu_based_workers)

    print("Detected {} CPU cores and {:.2f} GB available memory.".format(cpu_count, available_memory_gb))
    print("Optimal number of workers based on system resources: {}".format(optimal_workers))

    return optimal_workers


# Define parameter options
parameters = {
    'learning_rate': [0.0001, 0.0005, 0.001],
    'optimizer': ['Adam', 'RMSProp'],
    'batch_size': [32, 64, 128],
    'agents_number': [2, 3],
    'grid_size': [5, 10],
    'gamma': [0.9, 0.95, 0.99],
    'reward_mode': [0, 1],
    'max_timestep': [100, 200],
    'MARLAlgorithm': ['IQL', 'VDN', 'QMIX']
}

# Create all combinations of parameters
keys, values = zip(*parameters.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]


# Function to run a single experiment
def run_experiment(params, index):
    print("Running experiment {}/{} with parameters: {}".format(index + 1, len(combinations), params))

    # Format arguments
    args = []
    for key, value in params.items():
        key_formatted = "--" + key.replace('_', '-')  # Replace '_' with '-' for argparse compatibility
        args.append(key_formatted + " " + str(value))
    args = ' '.join(args)

    # Construct the full command
    command = "python main.py " + args

    # Run the command
    try:
        subprocess.call(command, shell=True)
        print("Experiment {}/{} completed successfully.".format(index + 1, len(combinations)))
    except Exception as e:
        print("Experiment {} failed with error: {}".format(index + 1, e))


# Determine optimal number of workers
optimal_workers = determine_optimal_workers()

# Run experiments concurrently
with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
    for i, params in enumerate(combinations):
        executor.submit(run_experiment, params, i)

print("All experiments completed!")
