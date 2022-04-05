from dataclasses import dataclass
from typing import Mapping
import triton
from triton.code_gen import JITFunction, Kernel

# We don't have any actual GPU memory when doing AoT compilation.
# import triton._C.libtriton.triton as _triton

from ._types import NamedVariantsMap
from .abstract_values import AbstractValue
from .sig_annotation_dsl import sig_generator

from .c_codegen import make_kernel_dispatcher_header_source, KernelSignatureData


@dataclass
class TritonCompileConfig:
    device_idx: int
    num_warps: int = 4
    num_stages: int = 4
    force_nc_cache: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _to_python_ir(obj: AbstractValue):
    name = obj.tt_dtype
    if hasattr(obj, "data_ptr"):
        return "ptr", name
    return "scalar", name


def _handle_meta_constants(func: JITFunction, meta: Mapping):
    return {
        func.arg_names.index(name): triton.language.constexpr(value)
        for name, value in meta.items()
        if any([isinstance(value, int), isinstance(value, JITFunction)])
    }


@dataclass
class DispatcherCCode:
    name: str
    header: str
    soruce: str

    def output(self, dst_path: str):
        from pathlib import Path
        dst = Path(dst_path)
        h = dst / f"{self.name}.h"
        src = dst / f"{self.name}.c"

        h.write_text(self.header)
        src.write_text(self.soruce)


@dataclass
class Compiler:
    attr_var_size_scope: NamedVariantsMap
    conf: TritonCompileConfig

    # TODO: add default size var to scope for un-annotated attribs
    # TODO: Allow warp setting (in self.conf) per code_gen call and not as a compiler param

    def code_gen(self, func: JITFunction, sig_annotation: str, **meta) -> DispatcherCCode:
        sig_ = KernelSignatureData.parse_sig_dsl(sig_annotation, func.arg_names)
        name = func.__name__
        input_constants = _handle_meta_constants(func, meta)

        binaries = []
        attr_sizes = []
        print(f"Generating PTX for {name}...")
        for ker_id, wargs in enumerate(sig_generator(
            pointers=sig_.pointer_types,
            attributes=sig_.attribute_types,
            attr_vars=sig_.attr_size_vars,
            named_vars=self.attr_var_size_scope,
        )):
            tensor_idxs = [i for i, arg in enumerate(wargs) if hasattr(arg, "data_ptr")]
            for i, pos in enumerate(sorted(input_constants)):
                wargs.insert(pos + i, input_constants[pos])
            # attributes
            attributes = dict()
            for i, arg in enumerate(wargs):
                if i in func.do_not_specialize:
                    continue
                if isinstance(arg, int):
                    attributes[i] = Kernel.pow2_divisor(arg)
                elif i in tensor_idxs:
                    addr = arg.data_ptr()
                    # This line checks an actual address on GPU. Since this is an AoT compiler,
                    # we simply skip it.
                    # range_size = _triton.runtime.get_pointer_range_size(addr)
                    range_size = arg.data_ptr()
                    attributes[i] = min(
                        Kernel.pow2_divisor(addr), Kernel.pow2_divisor(range_size)
                    )
            # transforms ints whose value is one into constants for just-in-time compilation
            constants = {
                i: arg
                for i, arg in enumerate(wargs)
                if isinstance(arg, int) and arg == 1 and i not in func.do_not_specialize
            }
            constants.update(
                {
                    i: arg.value
                    for i, arg in enumerate(wargs)
                    if isinstance(arg, triton.language.constexpr)
                }
            )
            constants.update({i: None for i, arg in enumerate(wargs) if arg is None})
            arg_types = [
                _to_python_ir(arg) for i, arg in enumerate(wargs) if i not in constants
            ]
            compile = dict(
                arg_types=arg_types,
                device=self.conf.device_idx,
                attributes=attributes,
                constants=constants,
                num_warps=self.conf.num_warps,
                num_stages=self.conf.num_stages,
            )
            ker_attr_sizes = [
                Kernel.pow2_divisor(s.val)
                for s in wargs
                if isinstance(s, AbstractValue) and s.is_attr
            ]
            kname = f"{name} with " + ' '.join([f'{n}:{s}' for n, s in zip(sig_.attribute_names, ker_attr_sizes)])
            print(f"\t [ {ker_id+1} ] Compiling {kname}")

            bin_ = func._compile(**compile)            
            ptx = bin_.asm["ptx"]
            
            binaries.append(ptx)
            attr_sizes.append(ker_attr_sizes)

        h, src = make_kernel_dispatcher_header_source(name, binaries=binaries, attr_sizes=attr_sizes, sig_data=sig_)
        return DispatcherCCode(name, h,src)