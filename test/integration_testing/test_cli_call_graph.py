import os
from unittest.mock import Mock, patch

import pandas as pd

from cli.cli_runner import CodeSmileCLI


@patch("components.rule_checker.RuleChecker.rule_check")
def test_cli_generates_call_graph_outputs(mock_rule_check, tmp_path):
    mock_rule_check.return_value = pd.DataFrame(
        columns=[
            "filename",
            "function_name",
            "smell_name",
            "line",
            "description",
            "additional_info",
        ]
    )

    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    (input_path / "sample.py").write_text(
        """
def alpha():
    beta()

def beta():
    pass
"""
    )

    args = Mock(
        input=str(input_path),
        output=str(output_path),
        max_walkers=1,
        parallel=False,
        resume=False,
        multiple=False,
        call_graph=True,
        analyze_call_graph=False,
        visualize_call_graph=False,
    )

    cli = CodeSmileCLI(args)
    cli.execute()

    call_graph_json = os.path.join(
        str(output_path), "output", "call_graph.json"
    )
    call_graph_dot = os.path.join(
        str(output_path), "output", "call_graph.dot"
    )

    assert os.path.exists(call_graph_json)
    assert os.path.exists(call_graph_dot)
