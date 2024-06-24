# 2024 - CSE3000 - Weighted Voting - RQ-4

# Network Latency Simulation

This project simulates network latencies and evaluates the performance of quorum-reaching algorithms under different attack strategies. The simulation compares normal latencies with enhanced security measures using Vivaldi coordinates.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Running the Simulation](#running-the-simulation)
  - [Command Line Options](#command-line-options)
- [Structure](#structure)
- [License](#license)

## Installation

To run this simulation, you need Python 3.x installed. Follow the steps below to set up the environment:

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Simulation

To run the simulation, use the following command:

```sh
python main.py --data <dataset> --out <output_directory> --multiprocess
```

### Command Line Options

- `--data` or `-d`: Specifies the dataset to be used for simulating latencies in the network. Possible values are `king`, `wonder`, and `planet`. Default is `wonder`.
- `--out` or `-o`: Specifies the output directory for the plots. Default is the current directory.
- `--multiprocess` or `-p`: Enables the experimental feature to run the program in a multiprocess fashion. Default is `False`.

Example command:
```sh
python main.py --data king --out ./output --multiprocess
```

## Structure

- **main.py**: The main entry point of the script. It defines the command-line interface and runs the simulation.
- **plots/simulation_average_quorum_time.py**: Contains the simulation logic and functions to run the simulation and generate plots.
- **data_generation/**: Directory containing utilities for data generation and handling network latency data.
- **vivaldi/**: Directory containing the implementation of Vivaldi coordinates and related attack strategies.
- **aware/**: Directory containing attack strategies specific to the AWARE protocol.

### Key Functions and Classes

- **simulate_normal_latencies**: Simulates normal latencies without enhanced security measures.
- **simulate_vivaldi_latencies**: Simulates latencies using Vivaldi coordinates for enhanced security.
- **plot_results**: Plots the results of the simulations.
- **run_one_simulation**: Runs a single simulation for a specific set of parameters.
- **run_simulation_over_all_behaviour**: Runs the simulation over all defined attack strategies and plots the results.

## License

This project is licensed under the MIT License. See the LICENSE file for details.