from pyparsing import (
    Word, alphas, alphanums, oneOf, Forward,
    Suppress, ZeroOrMore, Optional, ParseException, ParserElement,
    nums
)

class MetricExprParser:
    def __init__(self, expr_string=None):
        """
        Initialize the parser and optionally parse an expression string.
        """
        ParserElement.setDefaultWhitespaceChars(" \t\r\n")

        # 1) Numeric literal
        number = Word(nums + ".-").setParseAction(lambda t: float(t[0]))

        # 2) Reserved function keywords
        func_kw = oneOf("ADD SUB MULT DIV", caseless=False)
        agg_kw = oneOf("sum avg min max", caseless=False)

        # 3) Variables must start with '@' and contain only [a-zA-Z0-9_].
        #    We'll capture the part AFTER '@' as the var name.
        var_start = Suppress("@")
        var_body = Word(alphas + "_", alphanums + "_")
        def var_ref_action(toks):
            # e.g. if the token is ["myVar"], produce ("var_ref", "myVar")
            return ("var_ref", toks[0])

        var_ref = (var_start + var_body).setParseAction(var_ref_action)

        # 4) Metric references are anything that:
        #    - does NOT start with '@'
        #    - isn't recognized as a reserved function or aggregator
        # We’ll parse them with the same Word(...) logic, but we place them
        # after we check for var_ref and function calls in the grammar.
        # For now, just match the same character set as var_body plus maybe
        # additional chars if your metrics can contain '/', etc.:
        metric_identifier = Word(alphas + "_/", alphas + "_/" + alphanums)
        def metric_ref_action(toks):
            # e.g. "time/s" => ("metric_ref", "time/s")
            return ("metric_ref", toks[0])
        metric_ref = metric_identifier.setParseAction(metric_ref_action)

        # 5) Build a Forward for expressions
        expr = Forward()

        # 6) Aggregation calls, e.g.: sum(expr), avg(expr)
        def agg_call_action(toks):
            # toks = [agg_name, sub_expr]
            return ("agg_call", toks[0], toks[1])
        agg_call = (
            agg_kw
            + Suppress("(")
            + expr
            + Suppress(")")
        ).setParseAction(agg_call_action)

        # 7) Arithmetic function calls, e.g. DIV(expr, expr)
        def func_call_action(toks):
            # toks = [FUNC_NAME, expr1, expr2, ...]
            func_name = toks[0]
            args = toks[1:]
            return ("func_call", func_name, args)

        func_call = (
            func_kw
            + Suppress("(")
            + expr
            + ZeroOrMore(Suppress(",") + expr)
            + Suppress(")")
        ).setParseAction(func_call_action)

        # 8) The order in which we combine these matters: we want to try
        #    agg_call and func_call first, then var_ref, then metric_ref, then number.
        #    Because if you parse `@foo` as var_ref, that should not become a metric_ref.
        atom = agg_call | func_call | var_ref | metric_ref | number
        expr <<= atom

        # 9) Assignment statements: @varname = expr
        #    We parse the left side explicitly as an '@'-prefixed name.
        lhs_var = Suppress("@") + var_body
        def assign_action(toks):
            # toks.var_name => 'myVar', so store as "@myVar"
            return ("assign", f"@{toks.var_name[0]}", toks.expression)
        assignment = (
            lhs_var("var_name")
            + Suppress("=")
            + expr("expression")
        ).setParseAction(assign_action)

        # 10) Multiple statements separated by semicolons
        statements = assignment + ZeroOrMore(Suppress(";") + assignment) + Optional(Suppress(";"))
        statements.setParseAction(lambda toks: list(toks))

        # Attach grammar to self
        self.parser = statements
        self.expr_str = None
        self.ast = None

        # Optionally parse right away
        if expr_string is not None:
            self.parse(expr_string)

    def parse(self, expr_string):
        try:
            self.ast = self.parser.parseString(expr_string, parseAll=True)
            self.expr_str = expr_string
            return self.ast
        except ParseException as pe:
            raise ValueError(f"Failed to parse metric expression: {pe}") from None

    def find_derivable_metrics(self):
        """
        Find all distinct metric names (i.e., 'metric_ref') used in the expression.
        Return them as a list.
        """
        needed = set()

        def walk(node):
            if isinstance(node, tuple):
                tag = node[0]
                if tag == "metric_ref":
                    needed.add(node[1])
                else:
                    # Recurse through child elements
                    for child in node[1:]:
                        if isinstance(child, tuple):
                            walk(child)
                        elif isinstance(child, list):
                            for c in child:
                                walk(c)

        if not self.ast:
            return []

        for stmt in self.ast:
            walk(stmt)

        return sorted(needed)