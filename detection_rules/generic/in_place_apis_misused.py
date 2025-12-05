import ast
from detection_rules.smell import Smell


class InPlaceAPIsMisusedSmell(Smell):
    """
    Detects misuse of non-in-place Pandas/NumPy APIs (ignored return values
    or misleading `inplace` usage).
    """

    # APIs known to return new objects (non-in-place)
    PANDAS_NONINPLACE_METHODS: set[str] = {
        "drop", "rename", "sort_values", "fillna", "replace",
        "drop_duplicates", "reset_index", "set_index",
        "groupby", "agg", "transform", "abs",
        "merge", "join", "concat",
        "pivot", "melt", "stack", "unstack",
    }

    # APIs supporting the `inplace` argument
    PANDAS_INPLACE_CAPABLE: set[str] = {
        "drop", "rename", "sort_values", "fillna", "replace",
        "drop_duplicates", "reset_index", "set_index",
    }

    # APIs used for inspection only (never treated as smells)
    PANDAS_INFO_METHODS: set[str] = {"head", "info", "describe"}

    # NumPy APIs returning new arrays
    NUMPY_NONINPLACE_FUNCTIONS: set[str] = {
        "copy", "reshape", "transpose", "flatten", "ravel",
        "concatenate", "stack", "split", "clip",
        "add", "subtract", "multiply", "divide", "power",
        "sqrt", "exp", "log",
        "mean", "median", "std", "var",
        "sum", "prod", "min", "max",
        "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    }

    # NumPy in-place ndarray mutators
    NUMPY_INPLACE_METHODS: set[str] = {"sort", "resize"}

    def __init__(self):
        super().__init__(
            name="in_place_apis_misused",
            description=(
                "Detects misuse of non-in-place APIs in Pandas and NumPy. "
                "Checks whether the result of the operation is actually used "
                "or whether developers incorrectly assume in-place behavior."
            ),
        )

    # ------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------
    def detect(
        self, ast_node: ast.AST, extracted_data: dict[str, any]
    ) -> list[dict[str, any]]:
        smells: list[dict[str, any]] = []

        libraries = extracted_data.get("libraries", {})
        dataframe_vars = extracted_data.get("dataframe_variables", [])

        # Map alias → library name (e.g., "pd" → "pandas")
        alias_to_lib = {alias: name for name, alias in libraries.items()}

        # Parent map is needed to understand whether a return value is used
        parent_map = self._build_parent_map(ast_node)

        # Scan all function/method calls
        for node in ast.walk(ast_node):
            if not isinstance(node, ast.Call):
                continue

            call_info = self._classify_call(
                node, dataframe_vars, libraries, alias_to_lib
            )
            if call_info is None:
                continue

            lib = call_info["library"]
            name = call_info["name"]

            inplace_flag = self._get_inplace_flag(node)
            value_used = self._is_value_used(node, parent_map)

            if lib == "pandas":
                self._handle_pandas_call(
                    node=node,
                    method_name=name,
                    inplace_flag=inplace_flag,
                    value_used=value_used,
                    smells=smells,
                )
            elif lib == "numpy":
                self._handle_numpy_call(
                    node=node,
                    func_name=name,
                    value_used=value_used,
                    smells=smells,
                )

        return smells

    # ------------------------------------------------------------
    # AST parent mapping
    # ------------------------------------------------------------
    def _build_parent_map(self, root: ast.AST) -> dict[ast.AST, ast.AST]:
        parent_map: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(root):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent
        return parent_map

    # ------------------------------------------------------------
    # Classify call as Pandas/NumPy API of interest
    # ------------------------------------------------------------
    def _classify_call(
        self,
        node: ast.Call,
        dataframe_vars: list[str],
        libraries: dict[str, str],
        alias_to_lib: dict[str, str],
    ) -> dict[str, str] | None:

        func = node.func

        # df.method(...)
        if isinstance(func, ast.Attribute):
            owner = func.value
            if isinstance(owner, ast.Name):
                owner_id = owner.id

                if owner_id in dataframe_vars:
                    return {"library": "pandas", "name": func.attr}

                # pd.xxx / np.xxx
                lib_name = alias_to_lib.get(owner_id)
                if lib_name:
                    if lib_name.startswith("pandas"):
                        return {"library": "pandas", "name": func.attr}
                    if lib_name.startswith("numpy"):
                        return {"library": "numpy", "name": func.attr}

        # direct function call: clip(...), merge(...)
        if isinstance(func, ast.Name):
            func_name = func.id
            for lib_name, alias in libraries.items():
                if alias == func_name or lib_name.split(".")[-1] == func_name:
                    if lib_name.startswith("pandas"):
                        return {"library": "pandas", "name": func_name}
                    if lib_name.startswith("numpy"):
                        return {"library": "numpy", "name": func_name}

        return None

    # ------------------------------------------------------------
    # Extract literal inplace flag
    # ------------------------------------------------------------
    def _get_inplace_flag(self, call: ast.Call) -> bool | None:
        for kw in getattr(call, "keywords", []):
            if kw.arg == "inplace":
                if (
                    isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, bool)
                ):
                    return kw.value.value
                return None
        return None

    # ------------------------------------------------------------
    # Determine whether the return value is actually used
    # ------------------------------------------------------------
    def _is_value_used(
        self, node: ast.AST, parent_map: dict[ast.AST, ast.AST]
    ) -> bool:

        current = node

        while True:
            parent = parent_map.get(current)
            if parent is None:
                return False

            # assignment (x = f(...), x, y = f(...))
            if isinstance(parent, ast.Assign):
                if parent.value is current:
                    return True
                if (
                    isinstance(parent.value, ast.Tuple)
                    and current in parent.value.elts
                ):
                    return True
                current = parent
                continue

            # augmented assignment (x += f(...))
            if isinstance(parent, ast.AugAssign) and parent.value is current:
                return True

            # returned from function
            if isinstance(parent, (ast.Return, ast.Yield, ast.YieldFrom)):
                return True

            # standalone expression
            if isinstance(parent, ast.Expr):
                return False

            # passed as argument
            if isinstance(parent, ast.Call):
                return True

            # chained call: f(...).g()
            if isinstance(parent, ast.Attribute) and parent.value is current:
                current = parent
                continue

            # used in expression
            if isinstance(
                parent,
                (
                    ast.BoolOp,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Compare,
                    ast.Subscript,
                    ast.Dict,
                    ast.List,
                    ast.Tuple,
                    ast.Set,
                ),
            ):
                return True

            current = parent

    # ------------------------------------------------------------
    # Pandas rules
    # ------------------------------------------------------------
    def _handle_pandas_call(
        self,
        node: ast.Call,
        method_name: str,
        inplace_flag: bool | None,
        value_used: bool,
        smells: list[dict[str, any]],
    ) -> None:

        if method_name in self.PANDAS_INFO_METHODS:
            return

        is_noninplace = method_name in self.PANDAS_NONINPLACE_METHODS
        is_inplace_capable = method_name in self.PANDAS_INPLACE_CAPABLE

        if not (is_noninplace or is_inplace_capable):
            return

        if inplace_flag is False:
            smells.append(
                self.format_smell(
                    line=node.lineno,
                    additional_info=(
                        f"Explicitly setting `inplace=False` for "
                        f"Pandas method `{method_name}` may be misleading. "
                        "Consider assigning the returned object to a variable "
                        "or using `inplace=True` if you intend to modify the "
                        "original DataFrame."
                    ),
                )
            )
            return

        if inplace_flag is True:
            return

        if is_noninplace and not value_used:
            smells.append(
                self.format_smell(
                    line=node.lineno,
                    additional_info=(
                        f"The result of Pandas method `{method_name}` is not "
                        "assigned to any variable nor otherwise used, and "
                        "`inplace` is not explicitly set. This suggests an "
                        "incorrect assumption of in-place behavior. Assign the "
                        "result to a variable or use `inplace=True` if "
                        "supported."
                    ),
                )
            )

    # ------------------------------------------------------------
    # NumPy rules
    # ------------------------------------------------------------
    def _handle_numpy_call(
        self,
        node: ast.Call,
        func_name: str,
        value_used: bool,
        smells: list[dict[str, any]],
    ) -> None:

        if func_name not in self.NUMPY_NONINPLACE_FUNCTIONS:
            return

        if not value_used:
            smells.append(
                self.format_smell(
                    line=node.lineno,
                    additional_info=(
                        f"The result of NumPy function `{func_name}` is not "
                        "assigned to a variable or otherwise used. NumPy "
                        "functions like this return a new array/value and do "
                        "not modify inputs in-place. You may be incorrectly "
                        "assuming in-place behavior."
                    ),
                )
            )
