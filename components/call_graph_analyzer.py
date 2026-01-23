import importlib.util
from collections import deque
from typing import Any


NETWORKX_AVAILABLE = importlib.util.find_spec("networkx") is not None
if NETWORKX_AVAILABLE:
    import networkx as nx
else:
    nx = None


class CallGraphAnalyzer:
    """
    Analyzes call graphs and computes graph metrics.
    """

    def __init__(self, call_graph: dict[str, Any]):
        self.call_graph = call_graph
        self.nodes = sorted(call_graph.get("nodes", {}).keys())
        self.edges = call_graph.get("edges", [])

    def analyze(self) -> dict[str, Any]:
        adjacency = {node: [] for node in self.nodes}
        for edge in self.edges:
            source = edge["source"]
            target = edge["target"]
            adjacency.setdefault(source, []).append(target)
            adjacency.setdefault(target, [])

        for node in adjacency:
            adjacency[node] = sorted(adjacency[node])

        in_degree = {node: 0 for node in adjacency}
        out_degree = {node: len(adjacency[node]) for node in adjacency}
        for node, targets in adjacency.items():
            for target in targets:
                in_degree[target] = in_degree.get(target, 0) + 1

        if NETWORKX_AVAILABLE and nx is not None:
            metrics = self._analyze_with_networkx(adjacency, in_degree, out_degree)
            cycles = [cycle for cycle in nx.simple_cycles(metrics["graph"])]
        else:
            betweenness = self._betweenness_centrality(adjacency)
            metrics = {
                "metrics": self._build_metrics(
                    in_degree, out_degree, betweenness
                ),
                "graph": None,
            }
            cycles = self._detect_cycles(adjacency)

        hotspots = self._identify_hotspots(metrics["metrics"], in_degree, out_degree)

        return {
            "metrics": metrics["metrics"],
            "cycles": cycles,
            "hotspots": hotspots,
        }

    def _analyze_with_networkx(
        self,
        adjacency: dict[str, list[str]],
        in_degree: dict[str, int],
        out_degree: dict[str, int],
    ) -> dict[str, Any]:
        graph = nx.DiGraph()
        for node in adjacency:
            graph.add_node(node)
        for node, targets in adjacency.items():
            for target in targets:
                graph.add_edge(node, target)

        betweenness = nx.betweenness_centrality(graph, normalized=True)
        metrics = self._build_metrics(in_degree, out_degree, betweenness)
        return {"metrics": metrics, "graph": graph}

    def _build_metrics(
        self,
        in_degree: dict[str, int],
        out_degree: dict[str, int],
        betweenness: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        metrics: dict[str, dict[str, Any]] = {}
        for node in sorted(in_degree.keys()):
            smell_count = len(
                self.call_graph["nodes"].get(node, {}).get("smells", [])
            )
            metrics[node] = {
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "smell_count": smell_count,
            }
        return metrics

    def _betweenness_centrality(
        self, adjacency: dict[str, list[str]]
    ) -> dict[str, float]:
        nodes = list(adjacency.keys())
        betweenness = {node: 0.0 for node in nodes}

        for source in nodes:
            stack: list[str] = []
            predecessors = {node: [] for node in nodes}
            sigma = {node: 0 for node in nodes}
            sigma[source] = 1
            distance = {node: -1 for node in nodes}
            distance[source] = 0

            queue = deque([source])
            while queue:
                node = queue.popleft()
                stack.append(node)
                for neighbor in adjacency.get(node, []):
                    if distance[neighbor] < 0:
                        queue.append(neighbor)
                        distance[neighbor] = distance[node] + 1
                    if distance[neighbor] == distance[node] + 1:
                        sigma[neighbor] += sigma[node]
                        predecessors[neighbor].append(node)

            delta = {node: 0.0 for node in nodes}
            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != source:
                    betweenness[w] += delta[w]

        normalization = 0.0
        n = len(nodes)
        if n > 2:
            normalization = 1 / ((n - 1) * (n - 2))

        if normalization > 0:
            for node in betweenness:
                betweenness[node] *= normalization

        return betweenness

    def _detect_cycles(self, adjacency: dict[str, list[str]]) -> list[list[str]]:
        nodes = sorted(adjacency.keys())
        index = {node: idx for idx, node in enumerate(nodes)}
        cycles_set: set[tuple[str, ...]] = set()

        def normalize_cycle(cycle: list[str]) -> tuple[str, ...]:
            cycle_nodes = cycle[:-1]
            min_node = min(cycle_nodes)
            min_index = cycle_nodes.index(min_node)
            rotated = cycle_nodes[min_index:] + cycle_nodes[:min_index]
            return tuple(rotated)

        def dfs(start: str, current: str, path: list[str]) -> None:
            for neighbor in adjacency.get(current, []):
                if index[neighbor] < index[start]:
                    continue
                if neighbor == start:
                    normalized = normalize_cycle(path + [start])
                    cycles_set.add(normalized)
                elif neighbor not in path:
                    dfs(start, neighbor, path + [neighbor])

        for start in nodes:
            dfs(start, start, [start])

        return [list(cycle) + [cycle[0]] for cycle in sorted(cycles_set)]

    def _identify_hotspots(
        self,
        metrics: dict[str, dict[str, Any]],
        in_degree: dict[str, int],
        out_degree: dict[str, int],
    ) -> dict[str, list[str]]:
        degree_scores = {
            node: in_degree.get(node, 0) + out_degree.get(node, 0)
            for node in metrics
        }
        max_degree = max(degree_scores.values(), default=0)
        max_centrality = max(
            (metric["betweenness_centrality"] for metric in metrics.values()),
            default=0.0,
        )
        highest_degree = sorted(
            [node for node, score in degree_scores.items() if score == max_degree]
        )
        highest_centrality = sorted(
            [
                node
                for node, metric in metrics.items()
                if metric["betweenness_centrality"] == max_centrality
            ]
        )
        return {
            "highest_degree": highest_degree,
            "highest_centrality": highest_centrality,
        }
