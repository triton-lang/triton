import os
import functools
from pathlib import Path
from triton.runtime.build import compile_module_from_src
from triton.backends.nvidia.driver import library_dirs, include_dirs


class ArgSpecializer:
    def __init__(self, specialize_extra):
        mod = compile_module_from_src(
            src=Path(os.path.join(os.path.dirname(__file__), "_specialize_simple.c")).read_text(),
            name="__triton_specialize_simple",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=[],
        )

        use_fallback = "HIP" in specialize_extra.__qualname__

        def _specialize_int_fallback(arg, specialize_value, align):
            key = specialize_extra(arg, "int", align=align) if specialize_value else None
            if arg == 1 and specialize_value:
                return ("constexpr", 1)
            elif -(2**31) <= arg and arg <= 2**31 - 1:
                return ("i32", key)
            elif 2**63 <= arg and arg <= 2**64 - 1:
                return ("u64", key)
            else:
                return ("i64", key)

        def _specialize_tensor_fallback(arg, specialize_value, align):
            return specialize_extra(arg, "tensor", align=align) if specialize_value else None

        def _specialize_tensor(arg, specialize_value, align):
            return mod.specialize_tensor(arg.data_ptr(), specialize_value, align)

        if use_fallback:
            self.specialize_int = _specialize_int_fallback
            self.specialize_tensor = _specialize_tensor_fallback
        else:
            self.specialize_int = mod.specialize_int
            self.specialize_tensor = _specialize_tensor
