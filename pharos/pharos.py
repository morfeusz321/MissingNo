from datetime import timedelta
from typing import Tuple, List, Dict

from vivaldi.vivaldi_coordinates import NetworkCoordinate


class PharosCoordinate:
    def __init__(self, dimension: int):
        self.global_coord = NetworkCoordinate(dimension)
        self.local_coord = NetworkCoordinate(dimension)

    def estimated_rtt(self, other: 'PharosCoordinate', same_cluster: bool) -> timedelta:
        if same_cluster:
            return self.local_coord.estimated_rtt(other.local_coord)
        else:
            return self.global_coord.estimated_rtt(other.global_coord)

    def update(self, other: 'PharosCoordinate', rtt: timedelta, same_cluster: bool):
        if same_cluster:
            self.local_coord.update(other.local_coord, rtt)
            other.local_coord.update(self.local_coord, rtt)
        else:
            self.global_coord.update(other.global_coord, rtt)
            other.global_coord.update(self.global_coord, rtt)


def cluster_nodes(nodes: List[str], coordinates: Dict[str, PharosCoordinate], num_clusters: int) -> Dict[str, int]:
    # Compute pairwise distances
    distances = {
        (a, b): coordinates[a].global_coord.estimated_rtt(coordinates[b].global_coord).total_seconds()
        for a in nodes for b in nodes if a != b
    }
    sorted_pairs = sorted(distances.items(), key=lambda item: item[1])
    clusters = {node: -1 for node in nodes}
    cluster_id = 0
    while any(c == -1 for c in clusters.values()):
        for (a, b), _ in sorted_pairs:
            if clusters[a] == -1 and clusters[b] == -1:
                clusters[a] = cluster_id
                clusters[b] = cluster_id
                cluster_id = (cluster_id + 1) % num_clusters
            elif clusters[a] == -1:
                clusters[a] = clusters[b]
            elif clusters[b] == -1:
                clusters[b] = clusters[a]
    return clusters


def update_coordinates(coordinates: Dict[str, PharosCoordinate], measurements: List[Tuple[str, str, float]],
                       clusters: Dict[str, int], iterations: int = 10):
    for _ in range(iterations):
        for server1, server2, latency in measurements:
            rtt = timedelta(milliseconds=latency)
            same_cluster = clusters[server1] == clusters[server2]
            coordinates[server1].update(coordinates[server2], rtt, same_cluster)
            coordinates[server2].update(coordinates[server1], rtt, same_cluster)
