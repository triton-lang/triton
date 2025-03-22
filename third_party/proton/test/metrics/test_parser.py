import pytest
from pyparsing import ParseException
import pandas as pd
from triton.profiler.metrics.parser import MetricExprParser
from triton.profiler.metrics.interpreter import interpret_statements

def test_simple_assignment():
    parser = MetricExprParser()
    ast = parser.parse("@myVar = 123")
    # We expect a list with one element: ("assign", "@myVar", 123.0)
    assert ast.as_list() == [("assign", "@myVar", 123.0)]

def test_assignment_with_metric_ref():
    parser = MetricExprParser()
    ast = parser.parse("@myVar = time")
    # We expect [("assign", "@myVar", ("metric_ref", "time"))]
    assert ast.as_list() == [("assign", "@myVar", ("metric_ref", "time"))]

def test_aggregation_call():
    parser = MetricExprParser()
    ast = parser.parse("@aggResult = sum(time)")
    # We expect [("assign", "@aggResult", ("agg_call", "sum", ("metric_ref", "time")))]
    assert ast.as_list() == [
        ("assign", "@aggResult", ("agg_call", "sum", ("metric_ref", "time")))
    ]

def test_function_call_add():
    parser = MetricExprParser()
    ast = parser.parse("@calc = ADD(123, 456)")
    # The second element is a tuple: ("func_call", "ADD", [123.0, 456.0])
    assert ast.as_list() == [
        ("assign", "@calc", ("func_call", "ADD", [123.0, 456.0]))
    ]

def test_function_call_nested():
    parser = MetricExprParser()
    # Example: @calc = MULT( ADD(1,2), 3 )
    expr = "@calc = MULT(ADD(1,2), 3)"
    ast = parser.parse(expr)
    # AST would be:
    # ("assign",
    #   "@calc",
    #   ("func_call", "MULT",
    #       [
    #         ("func_call", "ADD", [1.0, 2.0]),
    #         3.0
    #       ]
    #   )
    # )
    expected = [
        ("assign",
         "@calc",
         ("func_call", "MULT", [
             ("func_call", "ADD", [1.0, 2.0]),
             3.0
         ])
        )
    ]
    assert ast.as_list() == expected

def test_var_ref():
    parser = MetricExprParser()
    ast = parser.parse("@output = @inputVar")
    # Expect: [("assign", "@output", ("var_ref", "inputVar"))]
    assert ast.as_list() == [("assign", "@output", ("var_ref", "inputVar"))]

def test_multiple_statements():
    parser = MetricExprParser()
    expr = "@varA = 1; @varB = sum(time); @varC = MULT(@varA, @varB);"
    ast = parser.parse(expr)
    # We'll get a list of three assignments in order:
    # [
    #   ("assign", "@varA", 1.0),
    #   ("assign", "@varB", ("agg_call", "sum", ("metric_ref", "time"))),
    #   ("assign", "@varC", ("func_call", "MULT", [
    #       ("var_ref", "varA"),
    #       ("var_ref", "varB")
    #   ]))
    # ]
    expected = [
        ("assign", "@varA", 1.0),
        ("assign", "@varB", ("agg_call", "sum", ("metric_ref", "time"))),
        ("assign", "@varC", ("func_call", "MULT", [
            ("var_ref", "varA"),
            ("var_ref", "varB"),
        ]))
    ]
    assert ast.as_list() == expected

def test_parse_error():
    parser = MetricExprParser()
    # Missing right-hand side expression
    with pytest.raises(ValueError) as excinfo:
        parser.parse("@foo =")
    assert "Failed to parse metric expression" in str(excinfo.value)

def test_find_derivable_metrics_simple():
    parser = MetricExprParser("@A = time")
    # time is a metric reference, so find_derivable_metrics should return ["time"]
    metrics = parser.find_derivable_metrics()
    assert metrics == ["time"]

def test_find_derivable_metrics_multiple():
    parser = MetricExprParser("@A = sum(time); @B = MULT(@A, requests); @C = avg(latency)")
    # We expect "time", "requests", "latency" in sorted order
    metrics = parser.find_derivable_metrics()
    assert metrics == ["latency", "requests", "time"]

def test_find_derivable_metrics_none():
    parser = MetricExprParser("@X = 123; @Y = ADD(4,5)")
    # No metric references used, so result should be []
    metrics = parser.find_derivable_metrics()
    assert metrics == []

    import pandas as pd

class DummyGF:
    def __init__(self, data):
        # Create a DataFrame from the provided data.
        self.dataframe = pd.DataFrame(data)
        # A list to track which metrics were created/modified.
        self.inc_metrics = []

# Helper to create a DummyGF with a single column.
def dummy_gf_with_column(col_name, values):
    df = pd.DataFrame({col_name: values})
    gf = DummyGF(df)
    gf.inc_metrics.append(col_name)
    return gf


# ----------------------
# Interpretation Unit Tests
# ----------------------

def test_interpret_metric_ref():
    # Create a dummy gf with a column for "time/s".
    gf = dummy_gf_with_column("time", [1.0, 2.0, 3.0, 4.0])
    # Parse an assignment that sets @foo equal to the metric "time/s".
    parser = MetricExprParser("@foo = time")
    ast = parser.ast
    metric_mappings = {}
    created = interpret_statements(ast, gf, metric_mappings)
    # The new column @foo should equal the "time" column.
    print(gf.dataframe.head())
    pd.testing.assert_series_equal(gf.dataframe["@foo"], gf.dataframe["time"], check_names=False)
    # Also, the created column name should be returned.
    assert created == ["@foo"]

def test_interpret_agg_call():
    # Test aggregator: @foo = sum(@bar)
    # Create a gf that contains a metric "time/s".
    gf = dummy_gf_with_column("time/s", [1.0, 2.0, 3.0, 4.0])
    # In this test, assign @bar to reference "time/s" and then aggregate it.
    parser = MetricExprParser("@bar = time/s; @foo = sum(@bar)")
    ast = parser.ast
    metric_mappings = {}
    created = interpret_statements(ast, gf, metric_mappings)
    # The sum of [1, 2, 3, 4] is 10.0. Since _ensure_series converts scalars to a Series,
    # we expect @foo to be a Series with 10.0 repeated.
    expected_series = pd.Series([10.0] * len(gf.dataframe), index=gf.dataframe.index)
    pd.testing.assert_series_equal(gf.dataframe["@foo"], expected_series, check_names=False)
    # Check that both assignments were processed.
    assert created == ["@bar", "@foo"]

def test_interpret_func_call():
    # Test an arithmetic function call: @foo = ADD(2, 3)
    gf = DummyGF(pd.DataFrame({"dummy": [0, 0, 0, 0]}))
    parser = MetricExprParser("@foo = ADD(2, 3)")
    ast = parser.ast
    metric_mappings = {}
    created = interpret_statements(ast, gf, metric_mappings)
    expected_series = pd.Series([5.0] * len(gf.dataframe), index=gf.dataframe.index)
    pd.testing.assert_series_equal(gf.dataframe["@foo"], expected_series, check_names=False)
    assert created == ["@foo"]

def test_interpret_func_call_several():
    gf = DummyGF(pd.DataFrame({"dummy": [1, 2, 3, 4]}))
    parser = MetricExprParser("@foo = MULT(dummy, dummy);@bar = ADD(@foo, dummy)")
    ast = parser.ast
    metric_mappings = {}
    created = interpret_statements(ast, gf, metric_mappings)
    expected_series = pd.Series([2, 6, 12, 20], index=gf.dataframe.index)
    pd.testing.assert_series_equal(gf.dataframe["@bar"], expected_series, check_names=False)
    assert created == ["@foo", "@bar"]

def test_circular_dependency():
    # Test that a circular dependency is detected.
    # Here, @foo depends on @bar and @bar depends on @foo.
    parser = MetricExprParser("@foo = @bar; @bar = @foo")
    ast = parser.ast
    gf = DummyGF(pd.DataFrame({"dummy": [0, 0, 0, 0]}))
    metric_mappings = {}
    with pytest.raises(ValueError, match="Circular dependency detected"):
        interpret_statements(ast, gf, metric_mappings)

def test_invalid_func_args():
    # Test that a function call with an invalid number of arguments raises an error.
    parser = MetricExprParser("@foo = ADD(2)")
    ast = parser.ast
    gf = DummyGF(pd.DataFrame({"dummy": [0, 0, 0, 0]}))
    metric_mappings = {}
    with pytest.raises(ValueError, match="expects 2 arguments"):
        interpret_statements(ast, gf, metric_mappings)