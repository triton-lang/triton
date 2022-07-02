from . import core, semantic


def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, ret_shape: tuple, _builder=None):
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
        ret_type = core.block_type(ret_type, ret_shape) if ret_shape is not None else ret_type
        return core.tensor(func(lib_name, lib_path, symbol, arg_list, ret_type.to_ir(_builder)), ret_type)


def elementwise(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, _builder=None):
    if len(args) == 1:
        ret_shape = args[0].shape
    elif len(args) == 2:
        args[0], args[1] = semantic.binary_op_type_checking_impl(args[0], args[1], _builder)
        ret_shape = args[0].shape
    else:
        return ValueError("elementwise takes 1 or 2 arguments")
    func = getattr(_builder, "create_extern_elementwise")
    return dispatch(func, lib_name, lib_path, args, arg_type_symbol_dict, ret_shape, _builder)
