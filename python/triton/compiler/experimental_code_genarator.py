"""
    TODO:
        Builtin function calls:
            fix _generator thing (needed for reductions)
        
        Function return values.
            
        Calling user defined functions!
        
        IR code location stuff (`_set_insertion_point_and_loc`)
            
        Error checking:
            Loop carried variable verification
        
        Function Specializations!
            to_tensor
        
        Support older python versions(< 3.8), check code_generator.py for reference.
"""

import textwrap
import inspect
import ast
from typing import List, Tuple, Dict, Type, Optional

from .._C.libtriton import ir
from ..runtime import JITFunction
from .. import language as tl
from .errors import (
    CompilationError,
    CompileTimeAssertionFailure,
    UnsupportedLanguageConstruct,
)


class GenScope:
    """
    Scope of a function at IR generation time.
    """

    def __init__(self, builder, parent, block=None):
        self.builder = builder
        self.parent = parent
        self.defines = {}
        self.redefines = set()
        if not block:
            block = self.builder.create_block()
        self.block = block

    def resolve(self, name):
        if name in self.defines:
            return self.defines[name]

        if self.parent:
            return self.parent.resolve(name)

    def define(self, name, value):
        if self.parent and self.parent.resolve(name):
            self.redefines.add(name)
        self.defines[name] = value

    def __enter__(self):
        self.builder.set_insertion_point_to_end(self.block)
        return self

    def __exit__(self, *args, **kwargs):
        if self.parent:
            self.builder.set_insertion_point_to_end(self.parent.block)


# Non SCF If, used for if blocks containing returns.
class TopLevelIfOp:
    def __init__(self, cond, builder, parent):
        self.builder = builder
        self.parent = parent
        self.cond = cond

        self.if_scope = GenScope(builder, parent)
        self.else_scope = GenScope(builder, parent)

    def If(self):
        pass

    def Else(self):
        pass
    
    def finalize(self):
        pass

class IfOp:
    def __init__(self, cond, builder, parent):
        self.builder = builder
        self.parent = parent
        self.cond = cond

        self.if_scope = GenScope(builder, parent)
        self.else_scope = GenScope(builder, parent)

    def If(self):
        return self.if_scope

    def Else(self):
        return self.else_scope

    def finalize(self):
        redef_names = set(self.if_scope.redefines)
        redef_names.update(self.else_scope.redefines)

        if_values = [self.if_scope.resolve(redef) for redef in redef_names]
        else_values = [self.else_scope.resolve(redef) for redef in redef_names]

        if_op = self.builder.create_if_op(
            [y.type.to_ir(self.builder) for y in if_values], self.cond.handle, True
        )
        self.builder.set_insertion_point_to_end(self.if_scope.block)
        if len(if_values) > 0:
            self.builder.create_yield_op([y.handle for y in if_values])

        self.builder.set_insertion_point_to_end(self.else_scope.block)
        if len(else_values) > 0:
            self.builder.create_yield_op([y.handle for y in else_values])
        self.builder.set_insertion_point_to_end(self.parent.block)

        self.if_scope.block.merge_block_before(if_op.get_then_block())
        self.else_scope.block.merge_block_before(if_op.get_else_block())

        for i, name in enumerate(redef_names):
            new_tensor = tl.core.tensor(if_op.get_result(i), if_values[i].type)
            self.parent.define(name, new_tensor)


class WhileOp:
    def __init__(self, builder, parent: GenScope, loop_carried: List[str]):
        self.builder = builder
        self.parent = parent
        self.loop_carried = loop_carried
        self.loop_init = [parent.resolve(name) for name in loop_carried]

        self.while_op = self.builder.create_while_op(
            [val.type.to_ir(self.builder) for val in self.loop_init],
            [arg.handle for arg in self.loop_init],
        )

        after_block = self.builder.create_block_with_parent(
            self.while_op.get_after(),
            [val.type.to_ir(self.builder) for val in self.loop_init],
        )
        self.body_scope = GenScope(builder, parent, block=after_block)

        before_block = self.builder.create_block_with_parent(
            self.while_op.get_before(),
            [val.type.to_ir(self.builder) for val in self.loop_init],
        )
        self.cond_scope = GenScope(builder, parent, block=before_block)

        for i, (name, val) in enumerate(zip(loop_carried, self.loop_init)):
            self.body_scope.define(name, tl.core.tensor(after_block.arg(i), val.type))
            self.cond_scope.define(name, tl.core.tensor(before_block.arg(i), val.type))

    def condition(self, test):
        self.builder.create_condition_op(
            test.handle,
            [self.cond_scope.block.arg(i) for i in range(len(self.loop_init))],
        )

    def Cond(self):
        return self.cond_scope

    def Body(self):
        return self.body_scope

    def finalize(self):
        # Insert the yield op for the while loop.
        with self.body_scope:
            self.builder.create_yield_op(
                [self.body_scope.resolve(name).handle for name in self.loop_carried]
            )

        # Merge with parent.
        for i, name in enumerate(self.loop_carried):
            new_def = tl.core.tensor(
                self.while_op.get_result(i), self.loop_init[i].type
            )
            self.parent.define(name, new_def)


class ForOp:
    def __init__(
        self,
        builder,
        parent: GenScope,
        it_range,
        index_name: str,
        loop_carried: List[str],
    ):
        self.builder = builder
        self.parent = parent
        self.loop_carried = loop_carried
        self.index_name = index_name
        self.loop_init = [parent.resolve(name) for name in loop_carried]

        num_stages = None
        loop_unroll_factor = None
        lb = None
        ub = None
        step = None

        if isinstance(it_range, range):
            lb = it_range.start
            ub = it_range.stop
            step = it_range.step
        elif isinstance(it_range, tl.range):
            lb = it_range.start
            ub = it_range.end
            step = it_range.step
            num_stages = it_range.num_stages
            loop_unroll_factor = it_range.loop_unroll_factor

        # handle negative constant step (not supported by scf.for in MLIR)
        negative_step = False
        if isinstance(step, int) and step < 0:
            step = -step
            negative_step = True
            lb, ub = ub, lb

        lb = tl.semantic.to_tensor(lb, self.builder)
        ub = tl.semantic.to_tensor(ub, self.builder)
        step = tl.semantic.to_tensor(step, self.builder)

        if not lb.dtype.is_int() or not ub.dtype.is_int() or not step.dtype.is_int():
            raise TypeError(
                f"For loop bounds and step must all be ints, are ({lb.dtype}, {ub.dtype}, {step.dtype})"
            )

        iv_type = tl.semantic.integer_promote_impl(lb.dtype, ub.dtype)
        iv_type = tl.semantic.integer_promote_impl(iv_type, step.dtype)
        iv_ir_type = iv_type.to_ir(self.builder)
        iv_is_signed = iv_type.int_signedness == tl.core.dtype.SIGNEDNESS.SIGNED

        lb = lb.handle
        ub = ub.handle
        step = step.handle

        lb = self.builder.create_int_cast(lb, iv_ir_type, iv_is_signed)
        ub = self.builder.create_int_cast(ub, iv_ir_type, iv_is_signed)
        step = self.builder.create_int_cast(step, iv_ir_type, iv_is_signed)

        self.for_op = self.builder.create_for_op(
            lb, ub, step, [arg.handle for arg in self.loop_init]
        )

        if num_stages is not None:
            self.for_op.set_attr(
                "tt.num_stages", self.builder.get_int32_attr(num_stages)
            )
        if loop_unroll_factor is not None:
            self.for_op.set_attr(
                "tt.loop_unroll_factor", self.builder.get_int32_attr(loop_unroll_factor)
            )

        self.iv = self.builder.create_poison(iv_ir_type)

        self.body_scope = GenScope(builder, parent, block=self.for_op.get_body(0))
        self.body_scope.define(index_name, tl.core.tensor(self.iv, iv_type))

    def Body(self):
        return self.body_scope

    def finalize(self):
        # For loops might not need yield ops.
        if len(self.loop_carried) == 0:
            return

        with self.body_scope:
            self.builder.create_yield_op(
                [self.body_scope.resolve(name).handle for name in self.loop_carried]
            )

        # Merge with parent.
        for i, name in enumerate(self.loop_carried):
            new_def = tl.core.tensor(
                self.for_op.get_result(i), self.loop_init[i].type
            )
            self.parent.define(name, new_def)


class GenInterface:
    def __init__(self, builder):
        self.builder = builder

    def call_JitFunction(self, fn: JITFunction, args, kwargs):
        raise NotImplementedError("TODO: Implement function call interface")

class GenGenScope:
    """
    Used during generator generation time. Needed for correct SSA construction of loops and correct break, continue statement usage.
    """

    def __init__(self, is_triton: bool, parent=None):
        self.defined = set()
        self.redefined = set()
        self.is_triton = is_triton
        self.parent = parent

    def define(self, name):
        if not self.is_triton:
            self.parent.define(name)

        self.defined.add(name)
        if self.parent and self.parent.resolve(name):
            self.redefined.add(name)

    def resolve(self, name):
        if not self.is_triton:
            self.parent.resolve(name)

        if name in self.defined:
            return True
        if not self.parent:
            return False
        return self.parent.resolve(name)

    def merge(self):
        """
        Propagate redefined variables upwards
        """
        if not self.parent:
            return
        for name in self.redefined:
            if name not in self.parent.defined:
                self.parent.redefined.add(name)


class InterpreterResult:
    def __init__(self, success, value=None):
        self.success = success
        self.value = value

class PseudoInterpreter(ast.NodeVisitor):
    """
    This is used to resolve triton functions.

    consider the case:

    ```python
    ...
    from triton import language as tl

    @jit
    def kernel():
        ...
        tl.sstore( .... ) # We need to be able to resolve this and if it's a triton builtin at code generator generation time.
        ...
    ```
    """

    def __init__(self, global_scope):
        self.global_scope = global_scope
        self.cache = {}

    def visit(self, node):
        if node in self.cache:
            return self.cache[node]
        result = super().visit(node)
        self.cache[node] = result

        return result

    def visit_Name(self, node):
        # NOTE: We assume the function won't redefine globals.

        name = node.id
        if name in self.global_scope:
            return InterpreterResult(True, self.global_scope[name])
        return InterpreterResult(False, None)

    def visit_Attribute(self, node):
        dep = self.visit(node.value)
        if not dep or not dep.success:
            return dep

        value = getattr(dep.value, node.attr)
        return InterpreterResult(value is not None, value)


class ExpressionAnalyser(ast.NodeVisitor):
    """
    Determine if an expression is a Triton expression.
    """

    def __init__(self, interpreter, triton_variables):
        self.cache = {}
        self.interpreter = interpreter
        self.triton_variables = triton_variables

    def visit(self, node):
        if node in self.cache:
            return self.cache[node]
        result = super().visit(node)
        self.cache[node] = result
        return result

    def visit_Constant(self, node):
        # FIXME: Support older python versions.
        return False

    def visit_Name(self, node):
        return node.id in self.triton_variables

    def visit_Attribute(self, node):
        # Attributes of a triton tensor are considered also triton values.
        if self.visit(node.value):
            return True

        result = self.interpreter.visit(node)
        if result is None or not result.success:
            return False

        return tl.core.is_builtin(result.value)

    def visit_Subscript(self, node):
        return self.visit(node.value)

    def visit_BinOp(self, node):
        return self.visit(node.left) or self.visit(node.right)

    def visit_UnaryOp(self, node):
        return self.visit(node.operand)

    def visit_Compare(self, node):
        if self.visit(node.left):
            return True

        for val in node.comparators:
            if self.visit(val):
                return True

    def visit_BoolOp(self, node):
        for val in node.values:
            if self.visit(val):
                return True
        return False

    def visit_Call(self, node):
        if self.visit(node.func):
            return True

        # For overriden functions we need to check function arguments to choose if we want to use Python or Triton versions.
        # TODO: What about `print`, it has runtime side effects.

        _overriden_functions = {
            "print": tl.core.device_print,  # NOTE: Print is a special case because it has side effects.
            "min": tl.minimum,
            "max": tl.maximum,
        }
        if isinstance(node.func, ast.Name) and node.func.id in _overriden_functions:
            for arg in node.args:
                if self.visit(arg):
                    return True

            for arg in node.keywords:
                if self.visit(arg.value):
                    return True
        return False


class TritonCodeGeneratorGenerator(ast.NodeTransformer):
    """
    Convert expressions to python code that build trtion IR.
    """

    def __init__(self, code_generator_generator):
        self.code_generator_generator = code_generator_generator
        self.interpreter = code_generator_generator.interpreter
        self.expression_analyser = code_generator_generator.expression_analyser

    def visit_Name(self, node):
        if node.id in self.code_generator_generator.triton_variables:
            return self.code_generator_generator.triton_lookup_variable(node.id)
        return node

    def visit_Call(self, node):
        #TODO: tensor attributes can be jit functions or builtins!
        func = self.interpreter.visit(node.func)

        if isinstance(func, JITFunction):
            raise NotImplementedError("TODO: JITFunction")

        if func.success:
            sig = inspect.signature(func.value)

            if "_generator" in sig.parameters:
                raise NotImplementedError("TODO: Generator interface")

        for i in range(len(node.args)):
            node.args[i] = self.visit(node.args[i])

        for i in range(len(node.keywords)):
            node.keywords[i] = self.visit(node.keywords[i])

        node.keywords.append(
            ast.keyword(arg="_builder", value=ast.Name(id="builder", ctx=ast.Load()))
        )

        return node

    def convert_to_triton(self, node):
        """
        Generate code to convert the obj to triton object.
        """
        return ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name("tl", ctx=ast.Load()), attr="core", ctx=ast.Load()
                ),
                attr="to_tensor",
                ctx=ast.Load(),
            ),
            args=[node],
            keywords=[ast.keyword("_builder", ast.Name("builder", ctx=ast.Load()))],
        )

    def visit_UnaryOp(self, node):
        _method_name_for_unary_op: Dict[Type[ast.unaryop], str] = {
            ast.USub: "__neg__",
            ast.UAdd: "__pos__",
            ast.Not: "__not__",
            ast.Invert: "__invert__",
        }

        value = node.operand
        if not self.expression_analyser.visit(value):
            value = self.convert_to_triton(self.visit(value))
        else:
            value = self.visit(value)

        op_type = _method_name_for_unary_op[type(node.op)]

        result = ast.Call(
            func=ast.Attribute(value=value, attr=op_type, ctx=ast.Load()),
            args=[],
            keywords=[ast.keyword("_builder", ast.Name("builder", ctx=ast.Load()))],
        )
        return result

    def visit_BinOp(self, node):
        _method_name_for_bin_op: Dict[Type[ast.operator], str] = {
            ast.Add: "__add__",
            ast.Sub: "__sub__",
            ast.Mult: "__mul__",
            ast.Div: "__truediv__",
            ast.FloorDiv: "__floordiv__",
            ast.Mod: "__mod__",
            ast.Pow: "__pow__",
            ast.LShift: "__lshift__",
            ast.RShift: "__rshift__",
            ast.BitAnd: "__and__",
            ast.BitOr: "__or__",
            ast.BitXor: "__xor__",
        }
        left = node.left

        if not self.expression_analyser.visit(left):
            left = self.convert_to_triton(self.visit(left))
        else:
            left = self.visit(node.left)

        right = self.visit(node.right)
        op_type = _method_name_for_bin_op[type(node.op)]
        result = ast.Call(
            func=ast.Attribute(value=left, attr=op_type, ctx=ast.Load()),
            args=[right],
            keywords=[ast.keyword("_builder", ast.Name("builder", ctx=ast.Load()))],
        )
        return result

    def visit_Subscript(self, node):
        def _visit(sub):
            if isinstance(sub, ast.Tuple):
                for i, e in enumerate(sub.elts):
                    sub.elts[i] = _visit(e)
                return sub
            elif isinstance(sub, ast.Slice):
                return ast.Call(
                    func=ast.Name(id="slice", ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(arg="lower", value=sub.lower if sub.lower else ast.Constant(None)),
                        ast.keyword(arg="upper", value=sub.upper if sub.upper else ast.Constant(None)),
                        ast.keyword(arg="step", value=sub.upper if sub.step else ast.Constant(None)),
                    ],
                )
            else:
                return self.visit(sub)
        slice_expression = _visit(node.slice)
        return ast.Call(func=ast.Attribute(value=node.value, attr="__getitem__", ctx=ast.Load()), args=[slice_expression], keywords=[])

    def visit_Compare(self, node):
        _method_name_for_comp_op: Dict[Type[ast.cmpop], str] = {
            ast.Eq: "__eq__",
            ast.NotEq: "__ne__",
            ast.Lt: "__lt__",
            ast.LtE: "__le__",
            ast.Gt: "__gt__",
            ast.GtE: "__ge__",
        }
        assert len(node.comparators) == 1
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op_type = _method_name_for_comp_op[type(node.ops[0])]
        result = ast.Call(
            func=ast.Attribute(value=left, attr=op_type, ctx=ast.Load()),
            args=[right],
            keywords=[ast.keyword("_builder", ast.Name("builder", ctx=ast.Load()))],
        )
        return result


class LoopInfo:
    def __init__(self, is_triton: bool, loop_op_creation: Optional[ast.Call] = None):
        self.is_triton = is_triton
        self.loop_op_creation = loop_op_creation

    def set_loop_carried(self, variables: List[str]):
        """
        Set the loop carried variables for a triton loop op.
        """
        assert self.is_triton

        expr = ast.List(elts=[ast.Constant(var) for var in variables], ctx=ast.Load())
        # Loop carried variables must always be the last element.
        self.loop_op_creation.args.append(expr)


class CodeGeneratorGenerator(ast.NodeTransformer):
    def __init__(self, global_scope):

        # Information regarding loops, triton or python.
        self.loop_stack = []

        # AST Node body stack, used to insert expressions to the current node body.
        self.body_stack = []

        # Tracking python and triton stacks.
        self.gen_gen_scopes = [GenGenScope(True)]

        # Code generation time scope tracking.
        # TODO: Merge this with gen_gen_scope
        self.scope_count = 0
        self.scope_stack = ["entry"]
        self.op_counts = 0

        # Variables marked as triton.
        self.triton_variables = set()

        # Used for naming variables.
        self.variable_counts = 0
        self.scope_counts = 0

        self.interpreter = PseudoInterpreter(global_scope)
        self.expression_analyser = ExpressionAnalyser(
            self.interpreter, self.triton_variables
        )
        self.triton_gen = TritonCodeGeneratorGenerator(self)

    def push_scope(self, is_triton: bool):
        if is_triton:
            name = f"__scope_{self.scope_count}"
            self.scope_stack.append(name)
            self.scope_count += 1
        self.gen_gen_scopes.append(GenGenScope(is_triton, self.gen_gen_scopes[-1]))

    def pop_scope(self):
        if self.gen_gen_scopes[-1].is_triton:
            self.scope_stack.pop()

        self.gen_gen_scopes[-1].merge()
        self.gen_gen_scopes.pop()

    def push_expr(self, node):
        """
        Push the expression to the last block.
        """
        self.body_stack[-1].append(node)

    def _visit_body(self, node, body_attr):
        """
        Visit/Replace nodes of a body while allowing insertion of expressions.
        """

        new_body = []
        self.body_stack.append(new_body)
        body = getattr(node, body_attr)
        for i, cnode in enumerate(body):
            replace = self.visit(cnode)
            if replace is not None:
                self.body_stack[-1].append(replace)

        setattr(node, body_attr, new_body)

        self.body_stack.pop()

    def visit_FunctionDef(self, node):
        # TODO: Attributes!
        node.decorator_list = []
        self.body_stack.append([])

        meta_args = []
        tensor_args = []
        for arg in node.args.args:
            if arg.annotation is not None:
                anno_result = self.interpreter.visit(arg.annotation)
                if anno_result.success and anno_result.value == tl.core.constexpr:
                    meta_args.append(arg)
                    continue

            tensor_args.append(arg.arg)
            self.triton_variables.add(arg.arg)
        
        fn_arguments = []

        for targ in tensor_args:
            fn_arguments.append(ast.Name(id=targ, ctx=ast.Load()))

        fn_type = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="function_type",
                ctx=ast.Load(),
            ),
            args=[
                ast.List(elts=[], ctx=ast.Load()),
                ast.List(elts=fn_arguments, ctx=ast.Load()),
            ],
            keywords=[],
        )

        self.push_expr(
            ast.Assign(
                targets=[ast.Name(id="_prototype", ctx=ast.Store())],
                value=fn_type
            )
        )


        # _ret_val = None
        self.push_expr(ast.Assign(targets=[ast.Name(id="_ret_val", ctx=ast.Store())], value=ast.Constant(None)))

        ir_fn_type = ast.Call(
            func=ast.Attribute(value=fn_type, attr="to_ir", ctx=ast.Load()),
            args=[ast.Name(id="builder", ctx=ast.Load())],
            keywords=[],
        )

        self.push_expr(
            ast.Assign(
                targets=[ast.Name(id="function", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        ast.Name(id="builder", ctx=ast.Load()),
                        attr="get_or_insert_function",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(id="module", ctx=ast.Load()),
                        ast.Constant(node.name),  # TODO: Use a mangled function name!
                        ir_fn_type,
                        ast.Constant("public"),
                        ast.Constant(False),
                    ],
                    keywords=[],
                ),
            )
        )

        self.push_expr(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="module", ctx=ast.Load()),
                        attr="push_back",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id="function", ctx=ast.Load())],
                    keywords=[],
                )
            )
        )

        self.push_expr(
            ast.Assign(
                targets=[ast.Name(id="entry", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="GenScope", ctx=ast.Load()),
                    args=[
                        ast.Name(id="builder", ctx=ast.Load()),
                        ast.Constant(value=None),
                    ],
                    keywords=[
                        ast.keyword(
                            arg="block",
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="function", ctx=ast.Load()),
                                    attr="add_entry_block",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[],
                            ),
                        )
                    ],
                ),
            )
        )

        entry_block = ast.With(
            items=[
                ast.withitem(context_expr=ast.Name("entry", ctx=ast.Load())),
            ],
            body=node.body,
        )

        self.body_stack.append([])
        for i, tensor_arg in enumerate(tensor_args):
            arg = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="function", ctx=ast.Load()),
                    attr="args",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(i)],
                keywords=[],
            )
            self.triton_define_variable(
                tensor_arg,
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="tl", ctx=ast.Load()),
                        attr="tensor",
                        ctx=ast.Load(),
                    ),
                    args=[arg, ast.Name(id=tensor_arg, ctx=ast.Load())],
                    keywords=[],
                ),
            )

        arg_setup = self.body_stack[-1]
        self.body_stack.pop()

        return_expr = []
        self.body_stack.append(return_expr)

        self.push_expr(
            ast.If(
                test=ast.UnaryOp(ast.Not(), ast.Name(id="_ret_val", ctx=ast.Load())),
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="builder", ctx=ast.Load()),
                                attr="ret",
                                ctx=ast.Load(),
                            ),
                            args=[ast.List(elts=[], ctx=ast.Load())],
                            keywords=[],
                        )
                    )
                ],
                orelse=[],
            )
        )


        self.body_stack.pop()

        self._visit_body(entry_block, "body")
        entry_block.body = arg_setup + entry_block.body + return_expr

        node.args.args = [ast.arg(arg="builder"), ast.arg(arg="module")] + node.args.args

        self.push_expr(entry_block)

        # function.reset_type(prototype.to_ir(builder))
        self.push_expr(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="function", ctx=ast.Load()),
                        attr="reset_type",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="_prototype", ctx=ast.Load()),
                                attr="to_ir",
                                ctx=ast.Load(),
                            ),
                            args=[ast.Name(id="builder", ctx=ast.Load())],
                            keywords=[],
                        )
                    ],
                    keywords=[],
                )
            )
        )
        
        self.push_expr(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="function", ctx=ast.Load()),
                        attr="finalize",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
            )
        )

        node.body = self.body_stack[-1]

        return node

    def triton_define_variable(self, name: str, value_exp):
        self.gen_gen_scopes[-1].define(name)

        # scope_{n}.define("name", ... )
        self.push_expr(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(self.scope_stack[-1], ctx=ast.Load()),
                        attr="define",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant(name), value_exp],
                    keywords=[],
                )
            )
        )

    def triton_lookup_variable(self, name):
        if isinstance(name, str):
            name = ast.Constant(name)

        # scope_{n}.resolve("name")
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(self.scope_stack[-1], ctx=ast.Load()),
                attr="resolve",
                ctx=ast.Load(),
            ),
            args=[name],
            keywords=[],
        )

    def visit_Assign(self, node):
        if not self.expression_analyser.visit(node.value) and (
            not isinstance(node.targets[0], ast.Name)
            or node.targets[0].id not in self.triton_variables
        ):
            return node

        assert len(node.targets) == 1, "only single assignment target is supported"
        assert isinstance(
            node.targets[0], ast.Name
        ), "only simple assignments are supported!"

        self.triton_variables.add(node.targets[0].id)

        self.triton_define_variable(
            node.targets[0].id, self.triton_gen.visit(node.value)
        )

    def visit_AugAssign(self, node):
        if (
            not isinstance(node.target, ast.Name)
            or node.target.id not in self.triton_variables
        ):
            return node

        lhs = ast.Name(id=node.target.id, ctx=ast.Load())
        rhs = ast.BinOp(lhs, node.op, node.value)
        assign = ast.Assign(targets=[node.target], value=rhs)
        self.visit(assign)

    def visit_AnnAssign(self, node):
        raise NotImplementedError("TODO")

    def visit_With(self, node):
        self._visit_body(node, "body")
        # No need to push a scope here.
        return node

    def visit_While(self, node):
        if not self.expression_analyser.visit(node.test):
            self.push_scope(False)
            self.loop_stack.append(LoopInfo(False))
            self._visit_body(node, "body")
            self._visit_body(node, "orelse")
            self.loop_stack.pop()
            self.pop_scope()
            return node

        assert (
            len(node.orelse) == 0
        ), "Triton while loops are not allowed to have else statements at the end."

        op_name = f"_while_op_{self.op_counts}"
        self.op_counts += 1

        loop_op_expr = ast.Call(
            func=ast.Name(id="WhileOp", ctx=ast.Load()),
            args=[
                ast.Name(id="builder", ctx=ast.Load()),
                ast.Name(id=self.scope_stack[-1], ctx=ast.Load()),
            ],
            keywords=[],
        )

        self.loop_stack.append(LoopInfo(True, loop_op_creation=loop_op_expr))
        self.push_expr(
            ast.Assign(
                targets=[ast.Name(id=op_name, ctx=ast.Store())], value=loop_op_expr
            )
        )
        self.push_scope(True)

        cond_body = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=op_name, ctx=ast.Load()),
                            attr="Cond",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                    optional_vars=ast.Name(id=self.scope_stack[-1], ctx=ast.Store()),
                )
            ],
            body=[
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=op_name, ctx=ast.Load()),
                            attr="condition",
                            ctx=ast.Load(),
                        ),
                        args=[self.triton_gen.visit(node.test)],
                        keywords=[],
                    )
                )
            ],
        )
        self.push_expr(cond_body)
        self.pop_scope()

        self.push_scope(True)

        # Process loop body.
        while_body = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=op_name, ctx=ast.Load()),
                            attr="Body",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                    optional_vars=ast.Name(id=self.scope_stack[-1], ctx=ast.Store()),
                )
            ],
            body=node.body,
        )
        self._visit_body(while_body, "body")
        self.push_expr(while_body)

        self.loop_stack[-1].set_loop_carried(self.gen_gen_scopes[-1].redefined)
        self.loop_stack.pop()
        self.pop_scope()

        self.push_expr(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=op_name, ctx=ast.Load()),
                        attr="finalize",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
            )
        )

    def visit_For(self, node):
        is_python = True

        # Triton only allows `for x in iterator()` style for loop declerations,
        # so anything else is Python. We might extend these a bit in the future if we want.
        if isinstance(node.iter, ast.Call):
            iterator = self.interpreter.visit(node.iter.func)
            if iterator and iterator.success:
                iterator = iterator.value
                if iterator is tl.range:
                    is_python = False

                elif iterator is range:
                    is_python = False

                elif iterator is tl.static_range:
                    # Rewrite static_range to range.
                    node.iter = ast.Name(id="range", ctx=ast.Load())

        if is_python:
            self.push_scope(False)
            self.loop_stack.append(LoopInfo(False))
            self._visit_body(node, "body")
            self._visit_body(node, "orelse")
            self.loop_stack.pop()
            self.pop_scope()
            return node

        assert (
            len(node.orelse) == 0
        ), "Triton for loops are not allowed to have else statements at the end."

        assert isinstance(
            node.target, ast.Name
        ), "For loop target must be simple variable"

        op_name = f"_for_op_{self.op_counts}"
        self.op_counts += 1

        loop_op_expr = ast.Call(
            func=ast.Name(id="ForOp", ctx=ast.Load()),
            args=[
                ast.Name(id="builder", ctx=ast.Load()),
                ast.Name(id=self.scope_stack[-1], ctx=ast.Load()),
                node.iter,
                ast.Constant(node.target.id),
            ],
            keywords=[],
        )
        # We will rewrite the loop creation op later to insert loop carried variables.
        self.loop_stack.append(LoopInfo(True, loop_op_creation=loop_op_expr))
        self.push_expr(
            ast.Assign(
                targets=[ast.Name(id=op_name, ctx=ast.Store())], value=loop_op_expr
            )
        )

        self.push_scope(True)

        # Process loop body.
        for_body = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=op_name, ctx=ast.Load()),
                            attr="Body",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                    optional_vars=ast.Name(id=self.scope_stack[-1], ctx=ast.Store()),
                )
            ],
            body=node.body,
        )

        self.gen_gen_scopes[-1].define(node.target.id)
        self.triton_variables.add(node.target.id)

        self._visit_body(for_body, "body")
        self.push_expr(for_body)

        self.loop_stack[-1].set_loop_carried(self.gen_gen_scopes[-1].redefined)
        self.loop_stack.pop()
        self.pop_scope()

        self.push_expr(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=op_name, ctx=ast.Load()),
                        attr="finalize",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
            )
        )

    def visit_If(self, node):
        if not self.expression_analyser.visit(node.test):
            self.push_scope(False)
            self._visit_body(node, "body")
            self._visit_body(node, "orelse")
            self.pop_socpe()
            return node

        op_name = f"_if_op_{self.op_counts}"
        self.op_counts += 1

        # if_op_{n} = IfOp(cond, builder, parent)
        self.push_expr(
            ast.Assign(
                targets=[ast.Name(id=op_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="IfOp", ctx=ast.Load()),
                    args=[
                        self.triton_gen.visit(node.test),
                        ast.Name(id="builder", ctx=ast.Load()),
                        ast.Name(id=self.scope_stack[-1], ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
            )
        )

        self.push_scope(True)

        if_block = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=op_name, ctx=ast.Load()),
                            attr="If",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                    optional_vars=ast.Name(id=self.scope_stack[-1], ctx=ast.Store()),
                )
            ],
            body=node.body,
        )
        self._visit_body(if_block, "body")
        self.push_expr(if_block)

        self.pop_scope()

        if len(node.orelse) > 0:
            self.push_scope(True)
            else_if_block = ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=op_name, ctx=ast.Load()),
                                attr="Else",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        ),
                        optional_vars=ast.Name(
                            id=self.scope_stack[-1], ctx=ast.Store()
                        ),
                    )
                ],
                body=node.orelse,
            )
            self._visit_body(else_if_block, "body")
            self.push_expr(else_if_block)

        self.push_expr(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=op_name, ctx=ast.Load()),
                        attr="finalize",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                )
            )
        )

    def visit_IfExp(self, node):
        if not (
            self.expression_analyser.visit(node.test)
            or self.expression_analyser.visit(node.body)
            or self.expression_analyser.visit(node.orelse)
        ):
            return node

        raise NotImplementedError("TODO: Implement If expressions!")

    def visit_Continue(self, node):
        if self.loop_stack[-1].is_triton:
            raise AssertionError(
                "Continue statements inside triton loops are not supported due to MLIR SCF limitations."
            )

        if self.gen_gen_scopes[-1].is_triton:
            raise AssertionError(
                "Can't use a Continue statement for a python loop while in a triton block."
            )

        return node

    def visit_Break(self, node):
        if self.loop_stack[-1].is_triton:
            raise AssertionError(
                "Break statements inside triton loops are not supported due to MLIR SCF limitations."
            )

        if self.gen_gen_scopes[-1].is_triton:
            raise AssertionError(
                "Can't use a Break statement for a python loop while in a triton block."
            )

        return node

    def visit_Return(self, node):
        ret_type = None
        if node.value is not None:
            ret_val = self.triton_gen.visit(node.value)

            self.push_expr(ast.Assign(
                targets = [ 
                    ast.Name("_ret_val", ctx=ast.Store())
                ],
                value = ret_val
            ))
            ret_type = ast.Attribute(value=ast.Name("_ret_val", ctx=ast.Load()), attr="type", ctx=ast.Load())
        else:
            ret_type = ast.Attribute(value=ast.Name("tl", ctx=ast.Load()), attr="void", ctx=ast.Load())

        # Update the return type of the function.
        #TODO: add an assertion to make sure the return types are consistent.
        self.push_expr(ast.Assign(
            targets = [
                ast.Attribute(
                    value = ast.Name(id="_prototype", ctx=ast.Load()),
                    attr = "ret_types",
                    ctx = ast.Store()
                )
            ],
            value = ast.List(elts = [ret_type], ctx=ast.Load())
        ))


        return_values = []
        if node.value is not None:
            return_values = [ast.Name("_ret_val", ctx=ast.Load())]
        
        self.push_expr(
            ast.Expr(value=
                ast.Call(
                    func = ast.Attribute(
                        ast.Name("builder", ctx=ast.Load()),
                        "ret",
                        ast.Load()
                    ),
                    args = [ast.List(elts = return_values, ctx = ast.Load())],
                    keywords = []
                )
            )
        )

    def visit_Call(self, node):
        if self.expression_analyser.visit(node):
            return self.triton_gen.visit(node)
        return self.generic_visit(node)


def generate_code_generator(func, debug=True):
    function = ast.parse(inspect.getsource(func))
    visitor = CodeGeneratorGenerator(func.__globals__)
    function = visitor.visit(function)
    function = ast.fix_missing_locations(function)

    if debug:
        print(ast.dump(function))
        print(ast.unparse(function))

    transformed_code = compile(function, filename="<ast>", mode="exec")
    local_namespace = {}

    # NOTE: Consider if it's ok to override the global scope like this. Maybe group them into a single thing like `_codegen.WhileOp`, `_codegen....`
    global_scope = dict(func.__globals__)
    global_scope["tl"] = tl
    global_scope["GenScope"] = GenScope
    global_scope["IfOp"] = IfOp
    global_scope["WhileOp"] = WhileOp
    global_scope["ForOp"] = ForOp

    exec(transformed_code, global_scope, local_namespace)
    return local_namespace[func.__name__]


def experimental_codegen_to_ttir(
    builder_fn, options, codegen_fns, context, args, module_map=None
):
    print(args)
    builder = ir.builder(context)
    builder.options = options
    builder.module_map = {} if module_map is None else module_map
    builder.codegen_fns = codegen_fns

    module = builder.create_module()

    builder_fn(builder, module, **args)
    module.context = context
    module.dump()
    return module