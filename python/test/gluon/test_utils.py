from triton._C.libtriton import ir
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler import make_backend
from triton.experimental.gluon._runtime import GluonASTSource
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import MockTensor


class StubOptions:
    num_warps: int = 4
    warp_size: int = 32
    num_ctas: int = 1
    def __init__(self, **kwargs):
        self.num_warps = kwargs.get('num_warps', 4);
        self.num_ctas = kwargs.get('num_ctas', 1)
        self.warp_size = kwargs.get('warp_size', 32)

def from_ast_to_module(fn, stub_target, *args, grid, **kwargs):
    context = ir.context()
    ir.load_dialects(context)
    options = dict(sanitize_overflow=False)
    stub_backend = make_backend(stub_target)
    stub_backend.load_dialects(context)
    options = stub_backend.parse_options(options)
    codegen_fns = stub_backend.get_codegen_implementation(options)

    builder = ir.builder(context)
    module = builder.create_module()
    module.set_attr("ttg.threads-per-warp", builder.get_int32_attr(stub_target.warp_size))
    stub_options = StubOptions(**kwargs)
    module.set_attr("ttg.num-warps", builder.get_int32_attr(stub_options.num_warps))
    module.set_attr("ttg.num-ctas", builder.get_int32_attr(stub_options.num_ctas))
    module.set_attr("ttg.target", builder.get_string_attr("..."))
    wrapped_args = list(map(MockTensor.wrap_dtype, args))

    signature_for_ast = {}
    const_args = {}
    keys = list(fn.signature.parameters.keys())
    for i in range(len(keys)):
        param_name = keys[i]
        param = fn.params[i]
        arg = wrapped_args[i] if i < len(wrapped_args) else 'consexpr'

        if param.is_constexpr:
            signature_for_ast[param_name] = 'constexpr'
            const_args[(i,)] = arg
        else:
            # non-const type parameter is either tensor or keyword arguments used in stub_options
            assert arg.dtype is not None;
            signature_for_ast[param_name] = '*' + str(arg.dtype)

    src = GluonASTSource(fn=fn, signature=signature_for_ast, constexprs=const_args)
    module = ast_to_ttir(fn, src, context=context, options=stub_options, codegen_fns=codegen_fns, module_map=dict(), module=module)
    assert module.verify()
    return module
