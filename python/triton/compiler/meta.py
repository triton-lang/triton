import triton
import triton.language as tl
import ast
import inspect
import copy
from functools import reduce
from contextlib import contextmanager

def replace_ast_name(root, name, new_node):
    # node visitor helper class
    class Visitor(ast.NodeTransformer):
        def __init__(self, old_name, new_node):
            self.old_name = old_name
            self.new_node = new_node
        
        def visit_Name(self, node):
            return self.new_node if node.id==self.old_name else node

    return Visitor(name, new_node).visit(root)


class meta_tuple:

    def __init__(self, *names):
        self.names = names
    
    def expand_args(self):
        return [ast.arg(arg=name) for name in self.names]


class meta_macro:
    
    def __init__(self, fn) -> None:
        self.fn = fn
    
    def __str__(self):
        return self.fn.__name__

    def expand_args(self):
        return []

    def __call__(self, scopes, node):
        func_name_node = ast.Name(id=str(self.fn.__name__), ctx=ast.Load())
        return ast.Call(func=func_name_node, args=node.args, keywords=node.keywords)
        

def _make_iterator(node):
    if isinstance(node, ast.Name):
        return IndexIterator(node)
    if isinstance(node, ast.Call):
        name = node.func.id
        if name == 'meta_zip':
            return zip(*[_make_iterator(arg) for arg in node.args])
        assert False
    assert False

def meta_zip(*args):
    pass

def meta_map(scopes, node):
    fn_args = node.args[1].args.args
    fn_body = node.args[1].body
    elts = [replace_ast_name(fn_body, fn_args[0].arg, new_node) for new_node in _make_iterator(node.args[0])]
    return ast.Tuple(elts=elts, ctx=ast.Load())

def meta_for_each(scopes, node):
    idx = next((i for i, v in enumerate(scopes[-1].body) if v == node or (isinstance(v, ast.Expr) and v.value==node)))
    scopes[-1].body.pop(idx)
    for i, iter in enumerate(_make_iterator(node.args[0])):
        body = copy.deepcopy(node.args[1].body)
        args = copy.deepcopy(node.args[1].args.args)
        for old_arg, new_arg in zip(args, iter):
            replace_ast_name(body, old_arg.arg, new_arg)
        scopes[-1].body.insert(idx+i, ast.copy_location(ast.Expr(body), node))
    return scopes[-1].body[idx]


class IndexIterator:

    def __init__(self, name: ast.Name):
        assert isinstance(name, ast.Name)
        self.name = name
        self.index = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= 1:
            raise StopIteration
        index_node = ast.Index(value=ast.Constant(self.index))
        self.index += 1  
        return ast.Subscript(value=self.name, slice=index_node, ctx=self.name.ctx)


class ScopedVisitor(ast.NodeTransformer):

    @contextmanager
    def scope(self, node):
        try:
            self.scopes.append(node)
            yield
        finally:
            self.scopes.pop()

    def __init__(self):
        self.scopes = []
    
class SpecializationVisitor(ScopedVisitor):
    def __init__(self, tuple_args, macro_args, global_scope):
        self.tuple_args = tuple_args
        self.macro_args = macro_args
        self.global_scope = global_scope
        self.tuples = dict()
        super().__init__()

    def visit_For(self, node):
        with self.scope(node):
            self.generic_visit(node)
        return node
    
    def visit_While(self, node):
        with self.scope(node):
            self.generic_visit(node)
        return node
    
    def visit_If(self, node):
        with self.scope(node):
            self.generic_visit(node)
        return node
    
    def _resolve_symbol(self, node):
        whitelist = {meta_zip, meta_map, meta_for_each}
        whitelist.update(self.macro_args.values())
        if isinstance(node, ast.Name) and self.global_scope.get(node.id, None) in whitelist:
            return self.global_scope[node.id]
        return None
        
    def _extend_arg(self, arg):
        obj = self.global_scope.get(arg.arg, None)
        return [arg] if obj is None else obj.expand_args()
    
    def visit_FunctionDef(self, node):
        with self.scope(node):
            args = [self._extend_arg(arg) for arg in node.args.args]
            node.args.args = reduce(list.__add__, args)
            for tuple_name, tuple_data in self.tuple_args.items():
                elts = [ast.Name(id=name) for name in tuple_data.names]
                value = ast.Tuple(elts=elts, ctx=ast.Load())
                targets = [ast.Name(id=tuple_name, ctx=ast.Store())]
                self.tuples[tuple_name] = value
                node.body.insert(0,  ast.Assign(targets=targets, value=value, lineno=None))
            self.generic_visit(node)
        return node

    
    def visit_Call(self, node):
        impl = self._resolve_symbol(node.func)
        if impl is None:
            return self.generic_visit(node)
        return ast.copy_location(impl(self.scopes, node), node)

    

def specialize(func, **kwargs):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    tuple_args = {k: v for k,v in kwargs.items() if isinstance(v, meta_tuple)}
    macro_args = {k: meta_macro(v) for k,v in kwargs.items() if callable(v)}
    global_scope = globals()
    global_scope.update(tuple_args)
    global_scope.update(macro_args)
    visitor = SpecializationVisitor(tuple_args=tuple_args, macro_args=macro_args, global_scope=global_scope)
    new_tree = visitor.visit(tree)
    # print(ast.dump(new_tree, indent=2))
    new_source = ast.unparse(new_tree)
    return new_source


# ---------------- Template -----------------

def normalize(Outs: meta_tuple, Ins: meta_tuple, 
              stride_om, stride_im, M, N, BLOCK_N: tl.constexpr,
              InitState: meta_macro,
              UpdateState: meta_macro,
              ApplyState: meta_macro):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    Ins = meta_map(Ins, lambda In: In + pid_m*stride_im)
    Outs = meta_map(Outs, lambda Out: Out + pid_m*stride_om)
    curr_state = InitState()
    for start_n in range(pid_n*BLOCK_N, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        inputs = meta_map(Ins, lambda In: tl.load(In + offs_n, mask=mask_n))
        curr_state = UpdateState(curr_state, inputs)
    final_state = curr_state
    for start_n in range(pid_n*BLOCK_N, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        inputs = meta_map(Ins, lambda In: tl.load(In + offs_n, mask=mask_n))
        outputs = ApplyState(final_state, inputs)
        meta_for_each(meta_zip(Outs, outputs), 
                         lambda Out, out: tl.store(Out + offs_n, out, mask=mask_n))

def softmax_init():
    return 0., float("-inf")

def softmax_update(state, value):
    m_ip1 = tl.maximum(state[1], tl.max(value, axis=0))
    d_ip1 = state[0] * tl.exp(state[1] - m_ip1) + tl.sum(tl.exp(value - m_ip1), 0)
    return d_ip1, m_ip1

def softmax_apply(state, value):
    return tl.exp(value - state[1]) / state[0]


softmax = specialize(normalize, 
                    Ins=meta_tuple('X'), 
                    Outs=meta_tuple('Y'),
                    InitState=softmax_init, 
                    UpdateState=softmax_update,
                    ApplyState=softmax_apply)
print(softmax)