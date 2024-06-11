import logging
from copy import deepcopy
from datetime import timedelta
import random
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from vivaldi.vivaldi_coordinates import NetworkNode, CoordinatesAttackStrategy, OscillationAttack
from data_generation.wonder_network import get_random_latency_matrix_of_size
from attacks import AttackStrategy
import optimWeightsForLatency

logger = logging.getLogger(__name__)


def simulate_normal_latencies(n, f, delta, Mpropose, Mwrite, rounds, malicious=True):
    # Simulated annealing for normal latencies
    return optimWeightsForLatency.run_simulation(n, f, delta, rounds, Mpropose, Mwrite, malicious=malicious)


def simulate_vivaldi_latencies(coordinates, n, f, delta, Mpropose, Mwrite, rounds, is_malicious=True,
                               attack: CoordinatesAttackStrategy = OscillationAttack()):
    # Simulated annealing adjusted for Vivaldi coordinates
    # Adjust coordinates in Mpropose and Mwrite based on Vivaldi distances
    # This is a placeholder: you need to adjust Mpropose and Mwrite based on Vivaldi coordinates

    # Stabilize Vivaldi coordinates
    for _ in range(100):  # Number of stabilization iterations
        for i in range(n):
            for j in range(n):
                if i != j:
                    coordinates[i] = coordinates[i].update(coordinates[j], Mpropose[i][j])
                    coordinates[j] = coordinates[j].update(coordinates[i], Mpropose[j][i])
    if is_malicious:
        malicious_nodes = [random.randint(0, n - 1) for _ in range(f)]
        for node in malicious_nodes:
            coordinates[node].set_strategy(attack)

    # print("Finished simulating vivaldi latencies")
    return optimWeightsForLatency.run_simulation_vivaldi(n, f, delta, rounds, coordinates, Mpropose,
                                                         malicious=is_malicious)


def plot_results(normal_times, vivaldi_times, title):
    rounds = len(vivaldi_times)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, rounds + 1), normal_times, label='Normal Latencies')
    plt.plot(range(1, rounds + 1), vivaldi_times, label='Vivaldi Latencies')
    plt.xlabel('f')
    plt.ylabel('Average Time to Reach Quorum (ms)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(title + '.png')
    plt.show()


def run_one_simulation(f, is_malicious):
    delta = f
    n = 3 * f + 1 + delta
    rounds = 10  # simulation rounds
    Mpropose = get_random_latency_matrix_of_size(n)
    Mwrite = Mpropose
    # Generate initial Vivaldi coordinates for simulation
    nodes = [NetworkNode() for _ in range(n)]
    for node in nodes:
        node.peers = nodes
    # # Simulate scenarios
    #
    vivaldi_times = simulate_vivaldi_latencies(nodes, n, f, delta, deepcopy(Mpropose), deepcopy(Mwrite), rounds,
                                               is_malicious)
    normal_times = simulate_normal_latencies(n, f, delta, deepcopy(Mpropose), deepcopy(Mwrite), rounds, is_malicious)

    return vivaldi_times, normal_times
def run_simulation_over_all_behaviour(title):
    attacks_on_aware: List[AttackStrategy] = []


def run_average_over_f_simulation(malicious, title, attack: CoordinatesAttackStrategy = None):
    average_times_normal = []
    average_times_vivaldi = []
    for i in range(1, 3):
        vivaldi_times, normal_times = run_one_simulation(i, malicious)
        average_times_normal.append(np.mean(normal_times))
        average_times_vivaldi.append(np.mean(vivaldi_times))
    plot_results(average_times_normal, average_times_vivaldi,
                 title)


if __name__ == '__main__':
    # run_average_over_f_simulation(False, 'Comparison of Quorum Reaching Time with Normal Behaviour')
    run_average_over_f_simulation(True, 'Comparison of Quorum Reaching Time with Random Behaviour', )
