from itertools import product
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple, Union
from triton.code_gen import JITFunction

from abstract_values import (
    AbstractInt,
    AbstractFloat,
    AbstractBool,
    AbstractPtr,
    DummyCudaDevice,
    TYPE_MAKERS
)

from _types import ModuleScope

def dict_product(d):
    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))

ValueMaker = Union[AbstractInt, AbstractFloat, AbstractBool]
MetaValue = Union[int, float, bool, str]
TypeVariant = str


@dataclass
class CompileParams:
    num_warps: int = 4
    num_stages: int = 4
    force_nc_cache: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class KernelSignitureConfig:
    name: str
    pointers: Sequence[AbstractPtr]
    abs_val_makers: Sequence[ValueMaker]
    meta: Mapping[str, MetaValue]
    named_variants: Mapping[str, Sequence[int]]
    variants: Sequence[TypeVariant]
    compile_params: CompileParams

    def signiture_iter(self):
        for prod_vals in dict_product(self.named_variants):
            abstract_vals = [
                f(v)
                for f, v in zip(
                    self.abs_val_makers, map(prod_vals.__getitem__, self.variants)
                )
            ]
            yield self.pointers[:] + abstract_vals


def parse_named_variants(d):
    """Turn string variants into integers"""
    # TODO: parse error messages
    return {k: [int(x) for x in d[k].split(",")] for k in d}


def parse_meta(meta_kwargs: Dict, module_scope: ModuleScope):
    if module_scope is None:
        module_scope = {}
    meta = {}
    errors = {}
    for k, v in meta_kwargs.items():
        if isinstance(v, str):
            if isinstance(module_scope.get(v), JITFunction):
                meta[k] = module_scope.get(v)
                continue
            errors[k] = v
        elif isinstance(v, (float, int)):
            meta[k] = v
        else:
            errors[k] = v
    if len(errors):
        # TODO: proper error msgs
        msg = "\t"
        for k, v in errors.items():
            msg += f"{k} is {type(v)}"
            if isinstance(v, str):
                msg += f" ({v} not in scope as a JITFunction)"
            msg += "\n"
        raise ValueError(
            f"only numbers and JITFunctions are allowed as Meta args.\n Following keys are not compatible:\n {msg} "
        )

    return meta


def _parse_type_size_variants(var_ident: Sequence[str], nv: Dict):
    """Build shared and private variants of current function"""
    private_var_count = 0
    variants = {}
    ordered_v_keys = []
    for v in var_ident:
        if v == "_":
            continue
        if v[-1] == "^":
            name = f"vv{private_var_count}"
            private_var_count += 1
            variants[name] = nv[v[:-1]]
            ordered_v_keys.append(name)
        elif v in nv:
            variants[v] = nv[v]
            ordered_v_keys.append(v)

    return ordered_v_keys, variants


def parse_func(
    name: str,
    fconf: Mapping,
    named_variants: Mapping[str, Sequence[int]],
    compile_conf: CompileParams,
    module_scope: ModuleScope = None,
) -> KernelSignitureConfig:
    tys = [v.strip() for v in fconf.get("types", "").split(",")]
    ty_vars = [v.strip() for v in fconf.get("type_variants", "").split(",")]
    pointers = []
    type_makers = []

    for ty in tys:
        # Pointer type (e.g. Tensor)
        if ty[-1] == "*":
            # TODO: device hard coded to 0, needs config
            dtype = AbstractPtr(ty[:-1], DummyCudaDevice(0))
            pointers.append(dtype)
        else:
            # create abstract Int/Float/Bool and leave the abstract value to be filled by type size variants
            ty_cls = TYPE_MAKERS[ty]
            type_makers.append(ty_cls)

    ordered_keys, variants = _parse_type_size_variants(ty_vars, named_variants)

    meta = parse_meta(fconf.get("meta", {}), module_scope)

    assert len(type_makers) == len(
        ordered_keys
    ), f"In {name} function config got {len(type_makers)} values and {len(ordered_keys)} variants. \n\t Values and Variants must be of same length"

    return KernelSignitureConfig(
        name=name,
        pointers=pointers,
        abs_val_makers=type_makers,
        variants=ordered_keys,
        meta=meta,
        named_variants=variants,
        compile_params=compile_conf,
    )


def parse_compilation_config(
    conf: Dict, module_scope: ModuleScope = None
) -> Mapping[str, KernelSignitureConfig]:
    
    named_variants = parse_named_variants(conf.get("named_type_variants", {}))
    compile_conf = CompileParams.from_dict(conf.get("compile_params", {}))

    return {
        func_name: parse_func(
            func_name,
            fconf,
            named_variants,
            compile_conf=compile_conf,
            module_scope=module_scope,
        )
        for func_name, fconf in conf.get("kernels", {}).items()
    }
