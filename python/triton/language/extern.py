from . import core


def dispatch(func_name, path, args, arg_types, ret_type, _builder):
  if len(args) != len(arg_types):
    raise ValueError(f"length of input args does not match function {func_name}'s declaration."
                     f"Expect {len(args)}, got {len(arg_types)}")

  for i, arg in enumerate(args):
    match = True
    if type(arg) is core.tensor:
      match = arg.dtype.name == arg_types[i]
      args[i] = core._to_tensor(arg, _builder)
    else:
      match = isinstance(arg, arg_types[i])
      args[i] = core._constexpr_to_value(arg)
    if not match:
      raise ValueError(f"input arg type does not match function {func_name}'s declaration."
                       f"Expect {arg_types[i]}, got {arg.dtype}")

  func = getattr(_builder, func_name)
  return core.tensor(func(path, *args), ret_type)
