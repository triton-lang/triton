from __future__ import annotations  # remove after python 3.11

from . import core, semantic


def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, ret_shape: tuple, _builder=None):
    '''
        Dispatch a function to a library
        :param func: the function to dispatch
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param ret_shape: the shape of the return value
        :param _builder: the builder
        :return: the return value of the function
    '''
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, core.tensor):
            arg_types.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"input arg type does not match."
                         f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_type = arg_type_symbol_dict[arg_types][1]
        if ret_shape:
            ret_type = core.block_type(ret_type, ret_shape)
        return core.tensor(func(lib_name, lib_path, symbol, arg_list, ret_type.to_ir(_builder)), ret_type)


def elementwise(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, _builder=None):
    '''
        Dispatch an elementwise function to a library
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param _builder: the builder
        :return: the return value of the function
    '''
    dispatch_args = args.copy()
    all_scalar = True
    ret_shape = None
    for i in range(len(dispatch_args)):
        dispatch_args[i] = core._to_tensor(dispatch_args[i], _builder)
        if dispatch_args[i].type.is_block():
            all_scalar = False
    if not all_scalar:
        broadcast_arg = dispatch_args[0]
        # Get the broadcast shape over all the arguments
        for i, item in enumerate(dispatch_args):
            _, broadcast_arg = semantic.binary_op_type_checking_impl(
                item, broadcast_arg, _builder)
        # Change the shape of each argument based on the broadcast shape
        for i in range(len(dispatch_args)):
            dispatch_args[i], _ = semantic.binary_op_type_checking_impl(
                dispatch_args[i], broadcast_arg, _builder)
        ret_shape = broadcast_arg.shape
    func = getattr(_builder, "create_external_elementwise")
    return dispatch(func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, ret_shape, _builder)
