import os
import pandas as pd

from components.call_graph_generator import CallGraphGenerator


def test_call_graph_generator_builds_nodes_and_edges(tmp_path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    file_path = project_path / "sample.py"
    file_path.write_text(
        """
class Greeter:
    def hello(self):
        helper()
        Greeter.bye()
        unknown()

    def bye(self):
        pass

def helper():
    Greeter.bye()
    unknown()
"""
    )

    smell_data = pd.DataFrame(
        [
            {
                "filename": str(file_path),
                "function_name": "helper",
                "smell_name": "HelperSmell",
            },
            {
                "filename": str(file_path),
                "function_name": "hello",
                "smell_name": "HelloSmell",
            },
        ]
    )

    generator = CallGraphGenerator(
        project_root=str(project_path),
        smell_data=smell_data,
        files=[str(file_path)],
    )
    graph = generator.build()

    nodes = graph["nodes"]
    assert "sample:helper" in nodes
    assert "sample:Greeter.hello" in nodes
    assert "sample:Greeter.bye" in nodes
    assert "external:unknown" in nodes

    assert nodes["sample:helper"]["smells"] == ["HelperSmell"]
    assert nodes["sample:Greeter.hello"]["smells"] == ["HelloSmell"]

    edges = {(edge["source"], edge["target"]) for edge in graph["edges"]}
    assert ("sample:Greeter.hello", "sample:helper") in edges
    assert ("sample:Greeter.hello", "sample:Greeter.bye") in edges
    assert ("sample:helper", "sample:Greeter.bye") in edges
