from ctypes import pointer
from curses import nonl
from dataclasses import dataclass
from itertools import product
from typing import Sequence, Iterator, Tuple

from ._types import NamedVariantsMap
from .abstract_values import TYPE_MAKERS, DummyCudaDevice, AbstractPtr, AbstractValue


class _ConfigKeys:
    POINTER_TYPE = "*"
    IGNORE_SYMBOL = "_"
    UNIQUE_VARIANT = "^"
    PTR = 'ptr' 
    ATTR = 'attr'


@dataclass
class SignatureTokens:
    var_types: Sequence[str]
    dtypes: Sequence[str]
    arg_names: Sequence[str]
    size_vars: Sequence[str]
    
    def _filter(self, var_type, seq: Sequence[str]):
        return [seq[idx] for idx, typ in enumerate(self.var_types) if typ == var_type]

    @property
    def pointers(self) -> Sequence[str]:
        return self._filter(_ConfigKeys.PTR, self.dtypes)

    @property
    def pointer_names(self) -> Sequence[str]:
        return self._filter(_ConfigKeys.PTR, self.arg_names)

    @property
    def attributes(self) -> Sequence[str]:
        return self._filter(_ConfigKeys.ATTR, self.dtypes)

    @property
    def attribute_names(self) -> Sequence[str]:
        return self._filter(_ConfigKeys.ATTR, self.arg_names)

    @property
    def attribute_sizes(self) -> Sequence[str]:
        return self._filter(_ConfigKeys.ATTR, self.size_vars)


def tokenize_signature_annotation(type_ann: str, arg_names: Sequence[str]) -> SignatureTokens:
    """
    For now the sig annotation is a ' ' separated string
    i.e. '*i32 *i64 u64 u64'
    """
    PTR = _ConfigKeys.PTR
    ATTR = _ConfigKeys.ATTR
    
    _vtype = []
    _dtype = []
    _name = []
    _size_var = []

    name_idx = 0

    def _get_arg_name(token):
        nonlocal name_idx
        if name_idx >= len(arg_names):
            raise ValueError(f"no argument name for {token}")
        arg_ = arg_names[name_idx]
        name_idx += 1
        return arg_


    def _build_pointer(token):
        tok = token[1:]
        if not tok in TYPE_MAKERS:
            # TODO: add position hint for error reporting
            raise ValueError(f"{token} is not a valid pointer annotation")
        _vtype.append(PTR)
        _dtype.append(tok)
        _name.append(_get_arg_name(token))
        _size_var.append(_ConfigKeys.IGNORE_SYMBOL)

    def _build_attr(token):
        if ':' not in token:
            tok = token
            size_var = _ConfigKeys.IGNORE_SYMBOL
        else:
            tok, size_var = token.split(':')
        
        if not tok in TYPE_MAKERS:
            # TODO: add position hint for error reporting
            raise ValueError(f"{token} is not a valid attribute annotation")
        _vtype.append(ATTR)
        _dtype.append(tok)
        _name.append(_get_arg_name(token))
        _size_var.append(size_var)


    for token in type_ann.split(' '):
        if token.startswith(_ConfigKeys.POINTER_TYPE):
            _build_pointer(token)
        elif token.startswith(_ConfigKeys.UNIQUE_VARIANT):
            _build_attr(token[1:])
        else:
            _build_attr(token)
        
    return SignatureTokens(var_types=_vtype, dtypes=_dtype, arg_names=_name, size_vars=_size_var)



def _duplicate_independet_type_variants(attr_vars, named_vars):
    """
    Type variants defined in a global scope (visible to all kernels).
    This function generates per kernel version of named variants.

    "^" suffix means use a unique named variant.
    E.g:
        A = 1, 8
        Types:          x: 32, y:i32, z:i32
        Type Variants:  A^,    A,     A
    y and z attributes will have coupled values, first attribute will get it's own values.
    in total |product([1,8], [1,8])| kernels will be compiled.
    """
    var_counts = {}
    new_named_vars_scope = {}
    new_attr_vars = []
    for v in attr_vars:
        if v[-1] == _ConfigKeys.UNIQUE_VARIANT:
            var_name = v[:-1]
            if var_name not in var_counts:
                var_counts[var_name] = 0
            var_counts[var_name] += 1
            vc = var_counts[var_name]
            # TODO: handle var_name not in scope error message
            new_var_name = f"{var_name}{vc}"
        else:
            new_var_name = v
            var_name = v
        new_named_vars_scope[new_var_name] = named_vars[var_name]
        new_attr_vars.append(new_var_name)

    return new_attr_vars, new_named_vars_scope


def dict_product(d):
    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))


def sig_generator(
    pointers: Sequence[str],
    attributes: Sequence[str],
    attr_vars: Sequence[str],
    named_vars: NamedVariantsMap,
    constants_scope: NamedVariantsMap = None
) -> Iterator[Sequence[AbstractValue]]:

    abstract_pointers = [AbstractPtr(p, DummyCudaDevice(0)) for p in pointers]
    abstract_attr_makers = [TYPE_MAKERS[ty] for ty in attributes]

    dup_attr_vars, dup_named_vars = _duplicate_independet_type_variants(
        attr_vars, named_vars
    )

    for concrete_val_dict in dict_product(dup_named_vars):
        concrete_vals = map(concrete_val_dict.__getitem__, dup_attr_vars)
        abstract_vals = [f(v) for f, v in zip(abstract_attr_makers, concrete_vals)]
        yield abstract_pointers[:] + abstract_vals
