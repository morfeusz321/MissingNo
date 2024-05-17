from matrix_utils import generate_random_matrix


def load_data():
    rounds = [generate_random_matrix(2000, 0, 1) for _ in range(5)]
    ip_to_id = {}
    discovered_nodes = 0

    with open('../data/king_data.txt', 'r') as file:
        for line in file:
            print(line)
            ip_source, _, id_destination, round, first_lookup, second_lookup = line.split()
            latency = int(first_lookup) - int(second_lookup)

            if ip_source not in ip_to_id:
                ip_to_id[ip_source] = discovered_nodes
                discovered_nodes += 1

            if id_destination not in ip_to_id:
                ip_to_id[id_destination] = discovered_nodes
                discovered_nodes += 1
            print(discovered_nodes)
            print(ip_to_id)
            rounds[int(round)-1][ip_to_id[ip_source]][ip_to_id[id_destination]] = latency

    return rounds, ip_to_id


if __name__ == '__main__':
    load_data()
