from . import core


def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, _builder=None):
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    first_symbol = list(arg_type_symbol_dict.values())[0][0]
    if len(args) != len(first_symbol.arg_types):
        raise ValueError(f"length of input args does not match function {first_symbol.op_name}'s declaration."
                         f"Expect {len(args)}, got {len(first_symbol.arg_types)}")

    arg_types = []
    for i, arg in enumerate(args):
        if isinstance(arg, core.tensor):
            arg_types.append(arg.dtype)
            args[i] = core._to_tensor(arg, _builder)
        else:
            arg_types.append(type(arg))
            args[i] = core._constexpr_to_value(arg)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"input arg type does not match function {first_symbol.op_name}'s declaration."
                         f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_type = arg_type_symbol_dict[arg_types][1]
        return core.tensor(func(lib_name, lib_path, symbol, args, ret_type.to_ir(_builder)), ret_type)


def elementwise(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, _builder=None):
    func = getattr(_builder, "create_extern_elementwise")
    dispatch(func, lib_name, lib_path, args, arg_type_symbol_dict)
