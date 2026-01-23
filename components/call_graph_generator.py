import ast
import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from utils.file_utils import FileUtils
from components.call_graph_analyzer import CallGraphAnalyzer


@dataclass
class ModuleDefinitions:
    module_name: str
    functions: set[str]
    classes: dict[str, set[str]]


class CallGraphGenerator:
    """
    Builds a call graph from Python source files using AST parsing.
    """

    def __init__(
        self,
        project_root: str,
        smell_data: pd.DataFrame | None = None,
        files: list[str] | None = None,
    ):
        self.project_root = os.path.abspath(project_root)
        self.files = files or FileUtils.get_python_files(self.project_root)
        self.smell_data = smell_data if smell_data is not None else pd.DataFrame()
        self._graph: dict[str, Any] | None = None

    def build(self) -> dict[str, Any]:
        """
        Builds the call graph and returns the graph structure.
        """
        module_definitions, module_trees = self._collect_definitions()
        smell_index = self._build_smell_index(module_definitions)

        nodes: dict[str, dict[str, Any]] = {}
        edges: set[tuple[str, str, str]] = set()

        for definitions in module_definitions.values():
            for function_name in sorted(definitions.functions):
                fqn = f"{definitions.module_name}:{function_name}"
                nodes[fqn] = {
                    "smells": sorted(smell_index.get(fqn, [])),
                    "type": "function",
                }
            for class_name, methods in definitions.classes.items():
                for method_name in sorted(methods):
                    fqn = f"{definitions.module_name}:{class_name}.{method_name}"
                    nodes[fqn] = {
                        "smells": sorted(smell_index.get(fqn, [])),
                        "type": "method",
                    }

        for file_path, tree in module_trees.items():
            definitions = module_definitions[file_path]
            visitor = _CallCollector(definitions)
            visitor.visit(tree)
            for source, target, edge_type in visitor.edges:
                if target not in nodes:
                    nodes[target] = {"smells": [], "type": "external"}
                edges.add((source, target, edge_type))

        edge_list = [
            {"source": source, "target": target, "type": edge_type}
            for source, target, edge_type in sorted(edges)
        ]

        self._graph = {"nodes": nodes, "edges": edge_list}
        return self._graph

    def export_json(self, path: str) -> None:
        """
        Exports the call graph to a JSON file.
        """
        graph = self._graph or self.build()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(graph, json_file, indent=2, sort_keys=True)

    def export_dot(self, path: str) -> None:
        """
        Exports the call graph to a DOT file.
        """
        graph = self._graph or self.build()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        analyzer = CallGraphAnalyzer(graph)
        analysis = analyzer.analyze()
        hotspots = set(analysis.get("hotspots", {}).get("highest_degree", []))
        hotspots.update(
            analysis.get("hotspots", {}).get("highest_centrality", [])
        )

        lines = [
            "digraph call_graph {",
            "    graph [splines=true, rankdir=LR];",
            "    node [style=filled];",
        ]
        for node in sorted(graph["nodes"].keys()):
            node_data = graph["nodes"][node]
            node_type = node_data.get("type", "external")
            smell_count = len(node_data.get("smells", []))
            label = node
            if smell_count == 1:
                label = f"{node} (1 smell)"
            elif smell_count > 1:
                label = f"{node} ({smell_count} smells)"

            if smell_count > 0:
                fillcolor = "orange"
            elif node_type == "function":
                fillcolor = "lightblue"
            elif node_type == "method":
                fillcolor = "lightgreen"
            else:
                fillcolor = "lightgray"

            attributes = [f'label="{label}"', f"fillcolor={fillcolor}"]
            if node in hotspots:
                attributes.extend(["color=red", "penwidth=2"])
            lines.append(f'    "{node}" [{", ".join(attributes)}];')

        for edge in graph["edges"]:
            lines.append(
                f'    "{edge["source"]}" -> "{edge["target"]}";'
            )
        lines.append("}")
        with open(path, "w", encoding="utf-8") as dot_file:
            dot_file.write("\n".join(lines))

    def _collect_definitions(
        self,
    ) -> tuple[dict[str, ModuleDefinitions], dict[str, ast.AST]]:
        module_definitions: dict[str, ModuleDefinitions] = {}
        module_trees: dict[str, ast.AST] = {}

        for file_path in self.files:
            with open(file_path, "r", encoding="utf-8") as file:
                source = file.read()
            tree = ast.parse(source)
            module_name = self._module_name_from_path(file_path)
            collector = _DefinitionCollector(module_name)
            collector.visit(tree)
            module_definitions[file_path] = ModuleDefinitions(
                module_name=module_name,
                functions=collector.functions,
                classes=collector.classes,
            )
            module_trees[file_path] = tree

        return module_definitions, module_trees

    def _build_smell_index(
        self, module_definitions: dict[str, ModuleDefinitions]
    ) -> dict[str, list[str]]:
        smell_index: dict[str, list[str]] = {}
        if self.smell_data.empty:
            return smell_index

        definitions_by_path = {
            os.path.abspath(path): definitions
            for path, definitions in module_definitions.items()
        }

        for _, row in self.smell_data.iterrows():
            file_path = os.path.abspath(row.get("filename", ""))
            function_name = row.get("function_name", "")
            smell_name = row.get("smell_name", "")

            definitions = definitions_by_path.get(file_path)
            if not definitions or not function_name:
                continue

            module_name = definitions.module_name
            candidates: list[str] = []
            if function_name in definitions.functions:
                candidates.append(f"{module_name}:{function_name}")

            class_matches = [
                class_name
                for class_name, methods in definitions.classes.items()
                if function_name in methods
            ]
            for class_name in class_matches:
                candidates.append(
                    f"{module_name}:{class_name}.{function_name}"
                )

            if not candidates:
                candidates.append(f"{module_name}:{function_name}")

            for fqn in candidates:
                smell_index.setdefault(fqn, []).append(str(smell_name))

        return smell_index

    def _module_name_from_path(self, file_path: str) -> str:
        relative_path = os.path.relpath(file_path, self.project_root)
        module_path = os.path.splitext(relative_path)[0]
        module_path = module_path.replace(os.sep, ".")
        if module_path.endswith(".__init__"):
            module_path = module_path[: -len(".__init__")]
        return module_path if module_path else "root"


class _DefinitionCollector(ast.NodeVisitor):
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.functions: set[str] = set()
        self.classes: dict[str, set[str]] = {}
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._class_stack.append(node.name)
        self.classes.setdefault(node.name, set())
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._register_function(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._register_function(node.name)
        self.generic_visit(node)

    def _register_function(self, name: str) -> None:
        if self._class_stack:
            class_name = self._class_stack[-1]
            self.classes.setdefault(class_name, set()).add(name)
        else:
            self.functions.add(name)


class _CallCollector(ast.NodeVisitor):
    def __init__(self, definitions: ModuleDefinitions):
        self.definitions = definitions
        self._function_stack: list[str] = []
        self._class_stack: list[str] = []
        self.edges: list[tuple[str, str, str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._enter_function(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._enter_function(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Call(self, node: ast.Call) -> Any:
        if not self._function_stack:
            self.generic_visit(node)
            return

        source = self._current_fqn()
        target, edge_type = self._resolve_call(node.func)
        self.edges.append((source, target, edge_type))
        self.generic_visit(node)

    def _enter_function(self, function_name: str) -> None:
        self._function_stack.append(function_name)

    def _current_fqn(self) -> str:
        if self._class_stack:
            class_name = self._class_stack[-1]
            function_name = self._function_stack[-1]
            return (
                f"{self.definitions.module_name}:{class_name}.{function_name}"
            )
        function_name = self._function_stack[-1]
        return f"{self.definitions.module_name}:{function_name}"

    def _resolve_call(self, func: ast.AST) -> tuple[str, str]:
        module_name = self.definitions.module_name
        if isinstance(func, ast.Name):
            if func.id in self.definitions.functions:
                return f"{module_name}:{func.id}", "internal"
            return f"external:{func.id}", "external"
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            class_name = func.value.id
            method_name = func.attr
            if class_name in self.definitions.classes:
                if method_name in self.definitions.classes[class_name]:
                    return (
                        f"{module_name}:{class_name}.{method_name}",
                        "internal",
                    )
            return f"external:{class_name}.{method_name}", "external"
        return "external:unknown", "external"
