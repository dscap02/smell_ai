from components.call_graph_analyzer import CallGraphAnalyzer


def test_call_graph_analyzer_metrics_and_cycles():
    graph = {
        "nodes": {
            "module:A": {"smells": ["Smell1"]},
            "module:B": {"smells": []},
            "module:C": {"smells": ["Smell2", "Smell3"]},
        },
        "edges": [
            {"source": "module:A", "target": "module:B", "type": "internal"},
            {"source": "module:B", "target": "module:C", "type": "internal"},
            {"source": "module:C", "target": "module:A", "type": "internal"},
        ],
    }

    analyzer = CallGraphAnalyzer(graph)
    results = analyzer.analyze()

    metrics = results["metrics"]
    assert metrics["module:A"]["in_degree"] == 1
    assert metrics["module:A"]["out_degree"] == 1
    assert metrics["module:A"]["smell_count"] == 1
    assert metrics["module:C"]["smell_count"] == 2

    cycles = results["cycles"]
    assert cycles, "Expected at least one cycle"
    cycle_nodes = set(cycles[0])
    assert {"module:A", "module:B", "module:C"}.issubset(cycle_nodes)

    hotspots = results["hotspots"]
    assert "highest_degree" in hotspots
    assert "highest_centrality" in hotspots


def test_call_graph_analyzer_without_networkx(monkeypatch):
    """
    Force the analyzer to execute the fallback path
    (_betweenness_centrality and _detect_cycles).
    """

    import components.call_graph_analyzer as analyzer_module

    # Force NetworkX to be unavailable
    monkeypatch.setattr(analyzer_module, "NETWORKX_AVAILABLE", False)
    monkeypatch.setattr(analyzer_module, "nx", None)

    from components.call_graph_analyzer import CallGraphAnalyzer

    graph = {
        "nodes": {
            "A": {"smells": []},
            "B": {"smells": []},
            "C": {"smells": []},
        },
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "A"},  # cycle
        ],
    }

    analyzer = CallGraphAnalyzer(graph)
    result = analyzer.analyze()

    # --- assertions ---
    assert "metrics" in result
    assert "cycles" in result
    assert "hotspots" in result

    # Fallback cycle detection
    assert result["cycles"]
    assert {"A", "B", "C"}.issubset(set(result["cycles"][0]))

    # Fallback betweenness must exist
    for node, metric in result["metrics"].items():
        assert "betweenness_centrality" in metric


def test_identify_hotspots_and_metrics():
    from components.call_graph_analyzer import CallGraphAnalyzer

    graph = {
        "nodes": {
            "X": {"smells": ["s1"]},
            "Y": {"smells": []},
        },
        "edges": [],
    }

    analyzer = CallGraphAnalyzer(graph)

    in_degree = {"X": 2, "Y": 1}
    out_degree = {"X": 1, "Y": 0}
    betweenness = {"X": 0.5, "Y": 0.0}

    metrics = analyzer._build_metrics(in_degree, out_degree, betweenness)
    hotspots = analyzer._identify_hotspots(metrics, in_degree, out_degree)

    assert metrics["X"]["smell_count"] == 1
    assert metrics["X"]["in_degree"] == 2
    assert metrics["X"]["out_degree"] == 1

    assert "X" in hotspots["highest_degree"]
    assert "X" in hotspots["highest_centrality"]
