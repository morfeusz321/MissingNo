import time
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import vivaldi_coordinates
from data_generation.wonder_network import get_random_latency_matrix_of_size


def init_nodes(number_of_nodes):
    temp = []
    for index in range(number_of_nodes):
        temp.append(vivaldi_coordinates.NetworkNode(3))
        print(temp[index].coordinates.coordinates)
    return temp


def get_max_demenstions(replicas: List[vivaldi_coordinates.NetworkNode]) -> Tuple[int, int, int]:
    max_x = max([x.coordinates.coordinates[0] for x in replicas])
    max_y = max([x.coordinates.coordinates[1] for x in replicas])
    max_z = max([x.coordinates.coordinates[2] for x in replicas])
    return max_x, max_y, max_z


def generate_animation_data(number_of_nodes, number_of_iterations):
    nodes_over_time = []
    nodes = init_nodes(number_of_nodes)
    nodes_over_time.append(nodes)
    latency_matrix = get_random_latency_matrix_of_size(n)

    for _ in range(number_of_iterations):
        for i in range(n):
            for j in range(n - i):
                if i == j:
                    continue
                nodes[i].update(nodes[j], latency_matrix[i][j])
        nodes_over_time.append(nodes)
    nodes_postions_over_time = [map(lambda x: x.coordinates.coordinates, nodes_over_time)]
    return nodes_postions_over_time


if __name__ == '__main__':
    n = 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    nodes = init_nodes()
    x, y, z = get_max_demenstions(nodes)

    for node in nodes:
        xs = node.coordinates.coordinates[0]
        ys = node.coordinates.coordinates[1]
        zs = node.coordinates.coordinates[2]
        ax.scatter(xs, ys, zs, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    for i in range(20):
        for i in range(n):
            for j in range(n - i):
                if i == j:
                    continue
                nodes[i].update(nodes[j], latency_matrix[i][j])
        fig.clear()
        ax = fig.add_subplot(projection='3d')
        for node in nodes:
            xs = node.coordinates.coordinates[0]
            ys = node.coordinates.coordinates[1]
            zs = node.coordinates.coordinates[2]
            ax.scatter(xs, ys, zs, marker='o')
        time.sleep(100)
