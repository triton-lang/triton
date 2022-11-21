from __future__ import division, annotations

from typing import Optional, Sequence, List, Union, Any

from . import ir

from .base import (
    _to_tensor,
    _binary_op_type_checking_impl,
    tensor,
    dtype,
    block_type,
)


def dispatch(
    func,
    *,
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    ret_shape: Optional[Sequence[int]] = None,
    _builder: ir.builder = None,
) -> tensor:
    """
    Dispatch a function to a library

    :param func: the function to dispatch
    :param lib_name: the name of the library
    :param lib_path: the path of the library
    :param args: the arguments of the function
    :param arg_type_symbol_dict: the type of the arguments
    :param ret_shape: the shape of the return value
    :param _builder: the builder

    :return: the return value of the function
    """
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(
            f"length of input args does not match."
            f"Expect {len(args)}, got {num_args}"
        )

    arg_types_l: List[Union[type, dtype]] = []
    arg_list: List[Any] = []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types_l.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types_l.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types_l)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(
            f"input arg type does not match."
            f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}"
        )
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_type = arg_type_symbol_dict[arg_types][1]
        ret_type = (
            block_type(ret_type, ret_shape) if ret_shape is not None else ret_type
        )
        return tensor(
            func(lib_name, lib_path, symbol, arg_list, ret_type.to_ir(_builder)),
            ret_type,
        )


def elementwise(
    *,
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    _builder: ir.builder = None,
) -> tensor:
    """
    Dispatch an elementwise function to a library

    :param lib_name: the name of the library
    :param lib_path: the path of the library
    :param args: the arguments of the function
    :param arg_type_symbol_dict: the type of the arguments
    :param _builder: the builder

    :return: the return value of the function
    """
    dispatch_args = args.copy()
    all_scalar = True
    ret_shape = None
    for dispatch_arg in dispatch_args:
        if dispatch_arg.type.is_block():
            all_scalar = False
    if not all_scalar:
        if len(args) == 1:
            dispatch_args[0] = _to_tensor(dispatch_args[0], _builder)
            ret_shape = dispatch_args[0].shape
        elif len(args) == 2:
            dispatch_args[0] = _to_tensor(dispatch_args[0], _builder)
            dispatch_args[1] = _to_tensor(dispatch_args[1], _builder)
            dispatch_args[0], dispatch_args[1] = _binary_op_type_checking_impl(
                dispatch_args[0], dispatch_args[1], builder=_builder
            )
            ret_shape = dispatch_args[0].shape
        else:
            for i in range(len(dispatch_args)):
                dispatch_args[i] = _to_tensor(dispatch_args[i], _builder)
            broadcast_arg = dispatch_args[0]
            # Get the broadcast shape over all the arguments
            for i in range(len(dispatch_args)):
                _, broadcast_arg = _binary_op_type_checking_impl(
                    dispatch_args[i], broadcast_arg, builder=_builder
                )
            # Change the shape of each argument based on the broadcast shape
            for i in range(len(dispatch_args)):
                dispatch_args[i], _ = _binary_op_type_checking_impl(
                    dispatch_args[i], broadcast_arg, builder=_builder
                )
            ret_shape = broadcast_arg.shape
    func = getattr(_builder, "create_extern_elementwise")
    return dispatch(
        func,
        lib_name=lib_name,
        lib_path=lib_path,
        args=dispatch_args,
        arg_type_symbol_dict=arg_type_symbol_dict,
        ret_shape=ret_shape,
        _builder=_builder,
    )
