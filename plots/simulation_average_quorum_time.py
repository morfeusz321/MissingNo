import logging
import multiprocessing
from copy import deepcopy
import random
from enum import Enum
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from data_generation.util import get_latency_matrix
from data_generation.wonder_network import get_random_latency_matrix_of_size
from vivaldi import vivaldi_coordinates
from vivaldi.vivaldi_coordinates import NetworkNode, CoordinatesAttackStrategy
from aware import attacks, optimWeightsForLatency

logger = logging.getLogger(__name__)


def simulate_normal_latencies(n, f, delta, Mpropose, Mwrite, rounds, malicious, attack):
    # Simulated annealing for normal latencies
    return optimWeightsForLatency.run_simulation(n, f, delta, rounds, Mpropose, Mwrite, malicious, attack)


def simulate_vivaldi_latencies(coordinates, n, f, delta, real_latencies, Mwrite, rounds, is_malicious,
                               attack: CoordinatesAttackStrategy):
    # Simulated annealing adjusted for Vivaldi coordinates
    # Adjust coordinates in Mpropose and Mwrite based on Vivaldi distances
    # This is a placeholder: you need to adjust Mpropose and Mwrite based on Vivaldi coordinates

    # Stabilize Vivaldi coordinates
    for _ in range(100):  # Number of stabilization iterations
        for i in range(n):
            for j in range(n):
                if i != j:
                    coordinates[i] = coordinates[i].update(coordinates[j], real_latencies[i][j])
                    coordinates[j] = coordinates[j].update(coordinates[i], real_latencies[j][i])

    malicious_nodes = random.sample(range(n), f)
    for node in malicious_nodes:
        coordinates[node].set_strategy(attack)

    # print("Finished simulating vivaldi latencies")
    return optimWeightsForLatency.run_simulation_vivaldi(n, f, delta, rounds, coordinates, real_latencies,
                                                         malicious_nodes)


def plot_results(normal_times, vivaldi_times, title):
    rounds = len(vivaldi_times)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, rounds + 1), normal_times, label='AWARE with latency matrix')
    plt.plot(range(1, rounds + 1), vivaldi_times, label='AWARE with enhanced security')
    plt.xlabel('f')
    plt.ylabel('Average Time to Reach Quorum (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig(title + '.svg')
    plt.show()


class Dataset(str, Enum):
    KING = "king"
    WONDER = "wonder"
    PLANET = "planet"


def get_latency_data(dataset, n):
    data = None
    match dataset:
        case Dataset.PLANET:
            data = get_latency_matrix("data/planetLab.txt", n)
        case Dataset.KING:
            data = get_latency_matrix("data/matrix.txt", n)
        case Dataset.WONDER:
            data = get_random_latency_matrix_of_size(n)
    return data


def run_one_simulation(f, is_malicious, attack_on_aware, attack_on_vivaldi, dataset):
    delta = f
    n = 3 * f + 1 + delta
    rounds = 10  # simulation rounds
    Mpropose = get_latency_data(dataset, n)
    for row in Mpropose:
        print(row)
    Mwrite = Mpropose
    # Generate initial Vivaldi coordinates for simulation
    nodes = [NetworkNode() for _ in range(n)]
    for node in nodes:
        node.peers = nodes
    # # Simulate scenarios
    #
    vivaldi_times = simulate_vivaldi_latencies(nodes, n, f, delta, deepcopy(Mpropose), deepcopy(Mwrite), rounds,
                                               is_malicious, attack_on_vivaldi)
    normal_times = simulate_normal_latencies(n, f, delta, deepcopy(Mpropose), deepcopy(Mwrite), rounds, is_malicious,
                                             attack_on_aware)

    return vivaldi_times, normal_times


def run_simulation_over_all_behaviour(title, dataset, is_multiprocess):
    attacks_on_aware: List[attacks.AttackStrategy] = [attacks.NetworkPartitionAttack(),
                                                      attacks.InflationAttack(),
                                                      attacks.DeflationAttack(),
                                                      attacks.FrogBoilingAttack(),
                                                      attacks.OscillationAttack()
                                                      ]
    attacks_on_vivaldi: List[vivaldi_coordinates.CoordinatesAttackStrategy] = [
        vivaldi_coordinates.NetworkPartitionAttack(),
        vivaldi_coordinates.InflationAttack(),
        vivaldi_coordinates.DeflationAttack(),
        vivaldi_coordinates.FrogBoilingAttack(),
        vivaldi_coordinates.OscillationAttack()
    ]

    for attack_on_aware, attack_on_vivaldi in zip(attacks_on_aware, attacks_on_vivaldi):
        if is_multiprocess:
            run_average_over_f_simulation_multi(True, title + ' ' + type(attack_on_aware).__name__, attack_on_aware,
                                                attack_on_vivaldi, dataset)
        else:
            run_average_over_f_simulation_multi(True, title + ' ' + type(attack_on_aware).__name__, attack_on_aware,
                                                attack_on_vivaldi, dataset)


def run_one_simulation_wrapper(args):
    # Unpack arguments
    i, malicious, attack_on_aware, attack_on_vivaldi, dataset = args
    return run_one_simulation(i, malicious, attack_on_aware, attack_on_vivaldi,dataset)


def run_average_over_f_simulation_multi(malicious, title, attack_on_aware, attack_on_vivaldi, dataset):
    # Prepare arguments for each simulation
    args = [(i, malicious, attack_on_aware, attack_on_vivaldi, dataset) for i in range(1, 12)]

    # Number of processes
    num_processes = multiprocessing.cpu_count()

    # Create a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Execute simulations in parallel and collect results
        results = pool.map(run_one_simulation_wrapper, args)

    # Extract and average the results
    average_times_normal = [np.mean(result[1]) for result in results]
    average_times_vivaldi = [np.mean(result[0]) for result in results]

    # Plot the results
    plot_results(average_times_normal, average_times_vivaldi, title)


def run_average_over_f_simulation(malicious, title, attack_on_aware: attacks.AttackStrategy,
                                  attack_on_vivaldi: CoordinatesAttackStrategy, dataset):
    average_times_normal = []
    average_times_vivaldi = []
    for i in range(1, 12):
        vivaldi_times, normal_times = run_one_simulation(i, malicious, attack_on_aware, attack_on_vivaldi, dataset)
        average_times_normal.append(np.mean(normal_times))
        average_times_vivaldi.append(np.mean(vivaldi_times))
    plot_results(average_times_normal, average_times_vivaldi,
                 title)

# if __name__ == '__main__':
# run_simulation_over_all_behaviour('Comparison of Quorum Reaching Time')
