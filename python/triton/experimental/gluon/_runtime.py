from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource
from triton.runtime.jit import JITFunction
from typing import TypeVar, Optional, Callable, Iterable, Union
from triton._C.libtriton import ir

T = TypeVar("T")


class GluonASTSource(ASTSource):

    def __init__(self, fn, signature, constexprs=None, attrs=None) -> None:
        super().__init__(fn, signature, constexprs, attrs)
        self.ext = "ttgir"
        self.run_ext_passes = False

    def make_ir(self, options, codegen_fns, module_map, context):
        module = ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                             module_map=module_map)
        builder = ir.builder(context)
        module.set_attr("ttg.num-warps", builder.get_int32_attr(options.num_warps))
        return module


class GluonJITFunction(JITFunction[T]):

    def create_binder(self):
        result = super().create_binder()
        self.ASTSource = GluonASTSource
        return result

    def is_gluon(self):
        return True


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[GluonJITFunction[T], Callable[[T], JITFunction[T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        return GluonJITFunction(
            fn,
            version=version,
            do_not_specialize=do_not_specialize,
            do_not_specialize_on_alignment=do_not_specialize_on_alignment,
            debug=debug,
            noinline=noinline,
            repr=repr,
            launch_metadata=launch_metadata,
        )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
