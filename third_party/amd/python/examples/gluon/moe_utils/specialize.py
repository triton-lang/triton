# Adapted from python/triton_kernels/triton_kernels/specialize.py
import inspect
import re
import textwrap
import types

import triton
from triton.experimental.gluon._runtime import GluonJITFunction

from triton_kernels.specialize import ClosureArg, FnSpecs


def define_kernel(src, module, attrs=None, is_gluon=False, **extra_globals):
    """
    Dynamically create a Triton function or kernel from a src string,
    linking any symbols in the kernel to objects specified by extra_globals.
    """

    # create templace function
    def _empty_fn():
        pass

    gdict = dict(**(_empty_fn.__globals__))
    gdict.update(extra_globals)
    f = types.FunctionType(_empty_fn.__code__, gdict)
    f.__module__ = module.__name__

    src = textwrap.dedent(src)
    src = src[src.find("def "):]

    stored_functions = []
    function_name = src[4:].split("(")[0].strip()

    exec_globals = gdict
    exec_globals.update({"stored_functions": stored_functions})
    exec(src + "\n\nstored_functions.append(" + function_name + ")\n", exec_globals)

    f.__signature__ = inspect.signature(stored_functions[0])
    f.__name__ = function_name
    f.__doc__ = stored_functions[0].__doc__

    if attrs is None:
        attrs = dict()
    if is_gluon:
        f = GluonJITFunction(f, **attrs)
    else:
        f = triton.JITFunction(f, **attrs)
    f._unsafe_update_src(src)
    return f


def specialize(fn, module, constants, tuples, name=None, do_not_specialize=tuple()):
    assert isinstance(fn, triton.runtime.jit.JITFunction)
    if name is None:
        name = f"{fn.__name__}"
    # Get original source code
    src = inspect.getsource(fn.fn)
    src = textwrap.dedent(src)
    lines = src.split("\n")
    # Skip decorator and def line
    def_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("def"))
    # separate header vs body LOC
    header_end = def_idx
    while not lines[header_end].rstrip().endswith(":"):
        header_end += 1
    body_lines = lines[header_end + 1:]
    header_lines = lines[def_idx:header_end + 1]
    # clean-up header
    header_clean = [
        l.split("#", 1)[0].strip()  # keep code, discard comment
        for l in header_lines
        if l.split("#", 1)[0].strip()  # skip blank‑after‑comment lines
    ]
    # decompose arguments
    header_src = " ".join(header_clean)  # turn it into a single line
    m = re.search(r"\((.*)\)\s*:", header_src)
    if not m:
        raise ValueError("Could not parse function header")
    args_str = m.group(1)
    args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
    non_specialized_args = []
    for arg in args:
        arg_key = arg.split(":")[0].split("=")[0].strip()
        new_args = tuples.get(arg_key, [arg])
        if arg_key not in constants:
            non_specialized_args += new_args
    # add global symbols
    spec_fns = {v.__name__: v for k, v in constants.items() if isinstance(v, triton.runtime.jit.JITFunction)}
    globals = spec_fns | fn.get_capture_scope()
    # build new source code and define kernel dynamically
    new_signature = f"def {name}({', '.join(non_specialized_args)}):"
    lang_module = "gl" if fn.is_gluon() else "tl"
    constexpr_lines = [
        f"    {key}: {lang_module}.constexpr = {value.__name__ if callable(value) else value}"
        for key, value in constants.items()
    ]
    tuple_lines = [
        f"    {key} = {'(' + ','.join(value) + (',' if len(value)>=1 else '') + ')'}" for key, value in tuples.items()
    ]
    new_src = "\n".join(["@gluon.jit" if fn.is_gluon() else "@triton.jit", new_signature] + constexpr_lines +
                        tuple_lines + body_lines)
    # Track how many logical lines precede the function body so we can adjust
    # the bookkeeping metadata to match the template definition.
    new_preamble_len = 1 + len(constexpr_lines) + len(tuple_lines)  # def + injected init lines
    original_preamble_len = len(header_lines)
    line_delta = new_preamble_len - original_preamble_len
    # find function parameters
    sig = inspect.signature(triton.runtime.jit.JITFunction.__init__)
    params = list(sig.parameters.values())[2:]
    attrs = {param.name: getattr(fn, param.name, param.default) for param in params}

    # make a new repr which appends the repr of the specialized functions.
    base_repr = attrs["repr"]

    def new_repr(specialization):
        ret = base_repr(specialization)
        for spec_fn in spec_fns.values():
            spec_repr = spec_fn.repr(None)
            if spec_repr:
                # Avoid dots in the appended repr so kernel name keeps the base kernel's name.
                spec_repr = spec_repr.rsplit(".", 1)[-1].strip("_")
            if spec_repr:
                ret += f"_{spec_repr}"
        return ret

    attrs["repr"] = new_repr

    if do_not_specialize:
        attrs["do_not_specialize"] = do_not_specialize
    ret = define_kernel(new_src, module, attrs, is_gluon=fn.is_gluon(), **globals)

    # Reuse the original kernel's metadata so that stack traces and other
    # source-based tooling report the correct file and line numbers.
    adjust_line_number = lambda line_num: max(1, line_num - line_delta)

    ret.raw_src = list(fn.raw_src)
    ret.starting_line_number = adjust_line_number(fn.starting_line_number)
    ret.def_file_line_number = adjust_line_number(fn.def_file_line_number)
    ret.def_file_col_number = fn.def_file_col_number

    orig_code = fn.fn.__code__
    ret.file_name = orig_code.co_filename
    ret.fn.__code__ = ret.fn.__code__.replace(
        co_filename=orig_code.co_filename,
        co_firstlineno=adjust_line_number(orig_code.co_firstlineno),
    )
    return ret


class SpecializationModule:

    def __init__(self, module_name: str, kernels: list[tuple[str, object]], closure_args: dict[str, ClosureArg]):
        self.module_name = module_name
        self.kernels = kernels
        self.closure_args = closure_args
        self._modules = dict()

    def get(self, **kwargs):
        import types
        import sys
        specs = [FnSpecs.default()] * len(self.closure_args)
        for key, value in kwargs.items():
            specs[list(self.closure_args.keys()).index(key)] = value
        key = tuple(spec.name for spec in specs)
        if key in self._modules:
            return self._modules[key]
        spec_constants = {arg.fn_name: spec.fn for arg, spec in zip(self.closure_args.values(), specs)}
        spec_tuples = {arg.fn_params_name: spec.fn_arg_names for arg, spec in zip(self.closure_args.values(), specs)}
        do_not_specialize = []
        for spec in specs:
            do_not_specialize.extend(spec.fn_arg_do_not_specialize)
        module = types.ModuleType(self.module_name + '_'.join(key))
        sys.modules[module.__name__] = module
        for kernel_name, kernel_fn in self.kernels:
            setattr(module, kernel_name,
                    specialize(kernel_fn, module, spec_constants, spec_tuples, do_not_specialize=do_not_specialize))
        self._modules[key] = module
        return module
