import heapq
import numpy as np


class Graph:
    """class to represent a weighted graph of the uncertainty of the relative poses."""
    def __init__(self):
        self.graph = dict()
        self.rel_cov = dict()

    def norm(self, cov: np.ndarray) -> float:
        """Calculate the det of the covariance matrix as the weight of the edge."""
        return np.linalg.det(cov)

    def add_edge(self, v1, v2, cov):
        """Add an edge to the graph."""
        weight = self.norm(cov)
        if v1 not in self.graph:
            self.graph[v1] = dict()
        if v2 not in self.graph:
            self.graph[v2] = dict()
        self.graph[v1][v2] = weight
        self.graph[v2][v1] = weight
        self.rel_cov[(v1, v2)] = cov
        self.rel_cov[(v2, v1)] = cov

    def remove_edge(self, v1, v2):
        """Remove an edge from the graph."""
        if v1 not in self.graph or v2 not in self.graph:
            return False
        if v2 not in self.graph[v1] or v1 not in self.graph[v2]:
            return False
        self.graph[v1].pop(v2)
        self.graph[v2].pop(v1)
        self.rel_cov.pop((v1, v2))
        self.rel_cov.pop((v2, v1))
        return True

    def update_edge(self, v1, v2, cov):
        """Update the weight of an edge."""
        self.add_edge(v1, v2, cov)

    def get_edge(self, v1, v2):
        """Get the weight of an edge."""
        if v1 not in self.graph or v2 not in self.graph[v1] or v2 not in self.graph[v1] or v1 not in self.graph[v2]:
           raise ValueError("Edge not found")
        return self.graph[v1][v2]

    def get_cov(self, v1, v2):
        """Get the covariance of an edge."""
        if (v1, v2) not in self.rel_cov:
            raise ValueError("Edge not found")
        return self.rel_cov[(v1, v2)]

    def dijkstra(self, start, end):
        """Find the shortest path between two nodes."""
        # Priority queue for exploring nodes
        pq = []
        heapq.heappush(pq, (0, start))

        # Distances and predecessors
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        predecessors = {node: None for node in self.graph}

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            # Early exit if we reach the destination
            if current_node == end:
                break

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in self.graph[current_node].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))

        # Reconstruct the shortest path
        path = []
        node = end
        while node is not None:
            path.insert(0, node)
            node = predecessors[node]

        if distances[end] == float('inf'):
            return None  # No path found

        return path

    def get_shortest_path(self, v1, v2):
        """Get the shortest path between two nodes."""
        if v1 == v2:
            return [v1]
        return self.dijkstra(v1, v2)

    def get_path_cov(self, path):
        """Get the covariance of a path."""
        cov = None
        for i in range(len(path) - 1):
            if cov is None:
                cov = self.get_cov(path[i], path[i+1])
            else:
                cov += self.get_cov(path[i], path[i+1])
        return cov
