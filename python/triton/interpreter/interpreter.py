import itertools
import random
from typing import Tuple

from .. import language as tl
# import .language.core as lcore
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
from .tl_lang import (TritonLangProxy, WrappedTensor, _primitive_to_tensor,
                      debugger_constexpr)

torch = torch_wrapper.torch
tl_method_backup = {}


def get_proxy_method(proxy, name):
    method = getattr(proxy, name)

    def fun(*args, **kwarg):
        return method(*args, **kwarg)

    return fun


def attach_triton(module, proxy):
    method_list = [func for func in dir(TritonLangProxy) if func[0] != "_"]
    for name in method_list:
        if hasattr(module, name):
            attr = getattr(module, name)
            tl_method_backup[name] = attr
            if callable(attr):
                setattr(module, name, get_proxy_method(proxy, name))
            else:
                setattr(module, name, getattr(proxy, name))


def detach_triton(module):
    for name, method in tl_method_backup.items():
        setattr(module, name, method)


def program_ids_from_grid(grid: Tuple[int, ...]) -> Tuple[int, ...]:
    # reverse the grid dimensions and generate the range for each dimension
    reversed_grid = reversed(grid)
    ranges_for_each_dimension = [range(dim) for dim in reversed_grid]

    # gen all combinations
    index_combinations = list(itertools.product(*ranges_for_each_dimension))
    random.shuffle(index_combinations)

    for index_combination in index_combinations:
        yield index_combination


class DebuggerFunction:
    def __init__(self, func, grid=(1,)):
        self.func = func
        self.grid = grid

    def _is_constexpr(self, name):
        return name in self.func.__annotations__ and self.func.__annotations__[name] is lcore.constexpr

    def _get_constexpr(self):
        result = []
        for name, annotation in self.func.__annotations__.items():
            if annotation is lcore.constexpr:
                result.append(name)
        return result

    def _assert_constexpr(self, **kwargs):
        constexp = self._get_constexpr()
        missing = [i for i in constexp if i not in kwargs.keys()]
        assert len(missing) == 0, f"You must specify constexpr {missing}"

    def _get_grid(self, **kwargs):
        if callable(self.grid):
            return self.grid(kwargs)
        else:
            return self.grid

    def __call__(self, *args, **kwargs):
        self._assert_constexpr(**kwargs)

        memory = MemoryMap()

        def convert_arg(v):
            name, arg = v
            if torch.is_tensor(arg):
                ptr = memory.add_tensor(arg)
                return WrappedTensor(torch.tensor([ptr], dtype=torch.int64, device="cuda"))
            if self._is_constexpr(name):
                return debugger_constexpr(arg)
            return WrappedTensor(_primitive_to_tensor(arg))

        new_args = tuple(map(convert_arg, zip(self.func.__code__.co_varnames, args)))
        new_kwargs = {k: convert_arg((k, v)) for (k, v) in kwargs.items() if k not in ["num_warps", "num_stages"]}

        grid = self._get_grid(**kwargs)
        for program_id in program_ids_from_grid(grid):
            proxy = TritonLangProxy(memory, ExecutionContext(program_id, grid))
            attach_triton(tl, proxy)
            self.func(*new_args, **new_kwargs)
            detach_triton(tl)


class GridSelector:
    """
    Entry point of the debugger
    """

    def __init__(self, func):
        version = torch.__version__
        assert version[0] == "2", f"Triton Debugger only supports torch >= 2.0, using {version}"
        self.func = func

    def __getitem__(self, grid):
        return DebuggerFunction(self.func, grid)

    def __call__(self, *args, **kwargs):
        return DebuggerFunction(self.func)(*args, **kwargs)


class AutotuneGridSelector:
    def __init__(self, func, autotune_params):
        self.func = func
        self.autotune_params = autotune_params

    def __getitem__(self, grid):
        return AutotuneRunner(self.func, self.autotune_params, grid)

    def __call__(self, *args, **kwargs):
        return AutotuneRunner(self.func, self.autotune_params)(*args, **kwargs)


class AutotuneRunner:
    def __init__(self, func, autotune_params, grid=None):
        self.func = func
        self.autotune_params = autotune_params
        self.grid = grid

    def __call__(self, *args, **kwargs):
        assert len(self.autotune_params["configs"]) >= 1

        for config in self.autotune_params["configs"][1:]:

            def convert_arg(v):
                if torch.is_tensor(v):
                    return torch.clone(v)
                return v

            new_args = tuple(map(convert_arg, args))
            new_kwargs = {k: convert_arg(v) for k, v in kwargs.items()}
            if self.grid:
                self.func[self.grid](*new_args, **new_kwargs, **config.kwargs)
            else:
                self.func(*new_args, **new_kwargs, **config.kwargs)

        main_config = self.autotune_params["configs"][0]
        if self.grid:
            self.func[self.grid](*args, **kwargs, **main_config.kwargs)
        else:
            self.func(*args, **kwargs, **main_config.kwargs)


def triton_debug_autotune(**kwars):
    def wrapper(func):
        return AutotuneGridSelector(func, kwars)

    return wrapper
