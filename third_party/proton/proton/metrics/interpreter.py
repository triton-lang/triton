import re
import pandas as pd

def interpret_statements(statements, gf, metric_mappings=None):
    """
    Evaluate assignment statements and add new columns to gf.dataframe.
    Each statement is a tuple: ("assign", var_name, expr_ast).

    Parameters
    ----------
    statements : list of tuples
        List of ("assign", var_name, expr_ast).
    gf : object
        An object that holds a pandas DataFrame (gf.dataframe) and
        a list of incremented metrics (gf.inc_metrics).
    metric_mappings : dict, optional
        Mapping of metric names to actual DataFrame column names.

    Returns
    -------
    list
        The list of newly created column names.
    """
    created_columns = []
    var_map = {}  # Maps user variables to actual DataFrame column names
    evaluation_stack = set()

    # Build a dict from variable → expr_ast for all assignments
    assignment_map = {}
    for statement_type, var_name, expr_ast in statements:
        if statement_type != "assign":
            raise ValueError(f"Unknown statement type {statement_type}")
        assignment_map[var_name] = expr_ast

    # Evaluate each assignment in order
    for statement_type, var_name, expr_ast in statements:
        if var_name in var_map:
            # Already evaluated
            continue

        col_data = _evaluate_expression(
            expr_ast,
            gf,
            var_map,
            metric_mappings,
            evaluation_stack,
            assignment_map
        )
        col_data = _ensure_series(col_data, gf)
        gf.dataframe[var_name] = col_data
        gf.inc_metrics.append(var_name)
        var_map[var_name] = var_name
        created_columns.append(var_name)

    return created_columns


def _evaluate_expression(expr, gf, var_map, metric_mappings,
                        evaluation_stack, assignment_map):
    """
    Recursively evaluate an expression against the DataFrame
    and already-processed variables.

    Parameters
    ----------
    expr : tuple or scalar
        The expression AST node (tuple) or a numeric scalar.
    gf : object
        Object containing a pandas DataFrame and list of metrics.
    var_map : dict
        Mapping of normalized variable names to DataFrame column names.
    metric_mappings : dict
        Optional mapping of metric names to actual DataFrame columns.
    evaluation_stack : set
        Used to detect circular dependencies.
    assignment_map : dict
        Mapping of normalized variable names to their AST expressions.

    Returns
    -------
    pd.Series or scalar
        Result of evaluating the expression.
    """
    if isinstance(expr, tuple):
        tag = expr[0]

        if tag == "var_ref":
            raw_var_name = expr[1]
            norm_var_name = "@" + raw_var_name

            # Detect circular dependency
            if norm_var_name in evaluation_stack:
                raise ValueError("Circular dependency detected.")

            # If we already have it in var_map, return its column data
            if norm_var_name in var_map:
                return gf.dataframe[var_map[norm_var_name]]

            # Otherwise, check if there's an assignment for it
            expr_ast = assignment_map.get(norm_var_name)
            if expr_ast is None:
                raise ValueError(f"Variable {norm_var_name} not found.")

            evaluation_stack.add(norm_var_name)
            try:
                val = _evaluate_expression(
                    expr_ast,
                    gf,
                    var_map,
                    metric_mappings,
                    evaluation_stack,
                    assignment_map
                )
                return val
            finally:
                evaluation_stack.remove(norm_var_name)

        elif tag == "metric_ref":
            metric = expr[1]
            if metric_mappings and metric in metric_mappings:
                metric = metric_mappings[metric]
            if metric in gf.inc_metrics:
                # Already assigned metric in DataFrame
                return gf.dataframe[metric]
            elif f"{metric} (inc)" in gf.dataframe.columns:
                # Possibly stored under a column with "(inc)"
                return gf.dataframe[f"{metric} (inc)"]
            return gf.dataframe[metric]

        elif tag == "agg_call":
            agg_name, subexpr = expr[1], expr[2]
            subval = _evaluate_expression(
                subexpr,
                gf,
                var_map,
                metric_mappings,
                evaluation_stack,
                assignment_map
            )
            subval = _ensure_series(subval, gf)

            if agg_name == "sum":
                return subval.sum()
            elif agg_name == "avg":
                return subval.mean()
            elif agg_name == "min":
                return subval.min()
            elif agg_name == "max":
                return subval.max()
            else:
                raise ValueError(f"Unknown aggregator '{agg_name}'")

        elif tag == "func_call":
            func_name, args = expr[1], expr[2]
            if len(args) != 2:
                raise ValueError(
                    f"Function {func_name} expects 2 arguments, got {len(args)}"
                )
            left = _evaluate_expression(
                args[0],
                gf,
                var_map,
                metric_mappings,
                evaluation_stack,
                assignment_map
            )
            right = _evaluate_expression(
                args[1],
                gf,
                var_map,
                metric_mappings,
                evaluation_stack,
                assignment_map
            )
            return _apply_func(func_name, left, right)

        else:
            raise ValueError(f"Unknown expression tag: {tag}")

    elif isinstance(expr, (int, float)):
        # Numeric literal
        return expr

    else:
        raise ValueError(f"Invalid AST node: {expr}")


def _apply_func(func_name, left, right):
    """
    Apply an arithmetic function (ADD, SUB, MULT, DIV) to left and right operands,
    converting scalars to pd.Series as needed.
    """
    left, right = _ensure_series_pair(left, right)

    if func_name == "ADD":
        return left + right
    elif func_name == "SUB":
        return left - right
    elif func_name == "MULT":
        return left * right
    elif func_name == "DIV":
        return left / right
    else:
        raise ValueError(f"Unknown function {func_name}")


def _ensure_series(value, gf):
    """
    Ensure that `value` is a pd.Series. If it is a scalar, convert it
    to a Series with the same index as gf.dataframe.
    """
    if not isinstance(value, pd.Series):
        value = pd.Series([value] * len(gf.dataframe), index=gf.dataframe.index)
    return value


def _ensure_series_pair(left, right):
    """
    Convert operands to pd.Series if one is a scalar and the other is a Series.
    """
    if isinstance(left, pd.Series) and not isinstance(right, pd.Series):
        right = pd.Series([right] * len(left), index=left.index)
    elif isinstance(right, pd.Series) and not isinstance(left, pd.Series):
        left = pd.Series([left] * len(right), index=right.index)
    return left, right