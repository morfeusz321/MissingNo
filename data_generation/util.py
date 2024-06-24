import random
from datetime import timedelta


def read_latency_matrix(file_path: str):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    latency_matrix = []
    for line in lines:
        # Splitting each line by any whitespace, which handles spaces and tabs
        parsed_line = [float(value) if float(value) != -1 else 500000 for value in line.split()]
        latency_matrix.append(parsed_line)

    return latency_matrix


def get_processed_latencies(latency_matrix):
    processed_latencies = []
    for row in latency_matrix:
        # Convert microseconds to milliseconds and store as timedelta objects
        processed_row = [timedelta(milliseconds=float(value)) for value in row]
        processed_latencies.append(processed_row)

    return processed_latencies


def select_n_nodes(latency_matrix, n, random_selection=True):
    """ Selects n nodes from the latency matrix either randomly or the first n."""
    if random_selection:
        selected_indices = random.sample(range(len(latency_matrix)), n)
    else:
        selected_indices = list(range(n))  # Select the first n indices

    # Create a new matrix including only the selected indices
    reduced_matrix = []
    for i in selected_indices:
        reduced_row = [latency_matrix[i][j] for j in selected_indices]
        reduced_matrix.append(reduced_row)

    return reduced_matrix


def get_latency_matrix(file_path, n, random_selection=False):
    latency_matrix = read_latency_matrix(file_path)
    processed_latencies = get_processed_latencies(latency_matrix)
    n_processed = select_n_nodes(processed_latencies, n, random_selection)
    return n_processed


