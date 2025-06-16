# Copyright © 2024 BAAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications:
# - 2025-06-03:
#   - init version: e9c7aa71832eb2f897a49ce787e42d5377404a72
#   - adapt torch_device_fn to ascend
#

import inspect
import sqlite3
import threading
import weakref
from typing import Dict

import triton
import torch
import torch_npu
torch_device_fn = torch.npu

from .code_cache import config_cache_dir

DEVICE_COUNT = torch_device_fn.device_count()
major_version = eval(triton.__version__.split(".")[0])


class LibTuner(triton.runtime.Autotuner):
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Dict = None,
        warmup=25,
        rep=100,
        use_cuda_graph=False,
    ):
        if major_version == 2:
            super().__init__(
                fn,
                arg_names,
                configs,
                key,
                reset_to_zero,
                restore_value,
                prune_configs_by,
                warmup,
                rep,
            )
            self.base_fn = fn
            while not inspect.isfunction(self.base_fn):
                self.base_fn = self.base_fn.fn
        else:
            super().__init__(
                fn,
                arg_names,
                configs,
                key,
                reset_to_zero,
                restore_value,
                pre_hook,
                post_hook,
                prune_configs_by,
                warmup,
                rep,
                use_cuda_graph,
            )
        self.__name__ = self.base_fn.__name__
        self.cache_path = config_cache_dir() / "TunedConfig.db"
        self.preload()
        weakref.finalize(self, self.store)

    def preload(self):
        connect = sqlite3.connect(self.cache_path)
        c = connect.cursor()
        c.execute(
            f"CREATE TABLE IF NOT EXISTS {self.__name__} (key TEXT PRIMARY KEY, config TEXT)"
        )
        cursor = c.execute(f"SELECT key, config from {self.__name__}")

        for row in cursor:
            key_str, config_str = row
            key = [eval(k) for k in key_str[1:-1].split(", ")]

            cfg_ls = [item.split(": ") for item in config_str.split(", ")]
            config = triton.Config({})
            attrs = -5 if major_version == 2 else -4
            for k, v in cfg_ls[:attrs]:
                config.kwargs[k] = eval(v)
            config.num_warps = eval(cfg_ls[attrs][1])
            config.num_ctas = eval(cfg_ls[attrs + 1][1])
            config.num_stages = eval(cfg_ls[attrs + 2][1])
            if major_version == 2:
                config.enable_warp_specialization = eval(cfg_ls[attrs + 3][1])
                config.enable_persistent = eval(cfg_ls[attrs + 4][1])
            else:
                config.maxnreg = eval(cfg_ls[attrs + 3][1])

            self.cache[tuple(key)] = config

        connect.close()
        self.volumn = len(self.cache)

    def store(self):
        if len(self.cache) == self.volumn:
            return
        connect = sqlite3.connect(self.cache_path)
        c = connect.cursor()
        c.execute(
            f"CREATE TABLE IF NOT EXISTS {self.__name__} (key TEXT PRIMARY KEY, config TEXT)"
        )
        for key, config in self.cache.items():
            c.execute(
                f"INSERT OR IGNORE INTO {self.__name__} (key, config) VALUES (?, ?)",
                (str(key), config.__str__()),
            )

        connect.commit()
        connect.close()


def libtuner(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=25,
    rep=100,
    use_cuda_graph=False,
):
    """
    Decorator for triton library autotuner.
    """

    def decorator(fn):
        return LibTuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
        )

    return decorator


class LibEntry(triton.KernelInterface):
    def __init__(
        self,
        fn,
    ):
        self.fn = fn
        self.arg_names = fn.arg_names
        self.divisibility = 16
        self.kernel_cache = tuple(dict() for _ in range(DEVICE_COUNT))

        while not isinstance(fn, triton.runtime.JITFunction):
            fn = fn.fn
        self.jit_function: triton.runtime.JITFunction = fn
        self.specialize_indices = [
            p.num
            for p in self.jit_function.params
            if not p.is_constexpr and not p.do_not_specialize
        ]
        self.do_not_specialize_indices = [
            p.num
            for p in self.jit_function.params
            if not p.is_constexpr and p.do_not_specialize
        ]
        self.lock = threading.Lock()

    def key(self, spec_args, dns_args, const_args):
        spec_key = [
            (arg.dtype, arg.data_ptr() % self.divisibility == 0)
            if hasattr(arg, "data_ptr")
            else (type(arg), arg)
            for arg in spec_args
        ]
        dns_key = [
            arg.dtype
            if hasattr(arg, "data_ptr")
            else type(arg)
            if not isinstance(arg, int)
            else "i32"
            if -(2**31) <= arg and arg <= 2**31 - 1
            else "u64"
            if 2**63 <= arg and arg <= 2**64 - 1
            else "i64"
            for arg in dns_args
        ]
        # const args passed by position
        return tuple(spec_key + dns_key + const_args)

    def run(self, *args, **kwargs):
        grid = kwargs["grid"]

        # collect all the arguments
        spec_args = []  # specialize arguments
        dns_args = []  # do not specialize arguments
        const_args = []  # constexpr arguments
        k_args = []  # kernel arguments
        for i, arg in enumerate(args):
            if i in self.specialize_indices:
                k_args.append(arg)
                spec_args.append(arg)
            elif i in self.do_not_specialize_indices:
                k_args.append(arg)
                dns_args.append(arg)
            else:
                const_args.append(arg)
        for p in self.jit_function.params[len(args) :]:
            if p.name in kwargs:
                val = kwargs[p.name]
            elif p.default is inspect._empty:
                continue
            else:
                val = p.default

            if p.is_constexpr:
                const_args.append(val)
            elif p.do_not_specialize:
                dns_args.append(val)
                k_args.append(val)
            else:
                spec_args.append(val)
                k_args.append(val)

        entry_key = self.key(spec_args, dns_args, const_args)
        device = torch_device_fn.current_device()
        cache = self.kernel_cache[device]
        while entry_key not in cache:
            # NOTE: we serialize the first run of a jit function regardless of which device to run on
            # because Triton runtime is currently not threadsafe.
            with self.lock:
                if entry_key in cache:
                    break
                kernel = self.fn.run(*args, **kwargs)
                fn = self.fn
                # collect constexpr arguments for grid computation
                constexprs = {}
                while not isinstance(fn, triton.runtime.JITFunction):
                    if isinstance(fn, triton.runtime.Autotuner):
                        config = fn.best_config
                        constexprs["num_warps"] = config.num_warps
                        constexprs["num_stages"] = config.num_stages
                        constexprs["num_ctas"] = config.num_ctas
                        constexprs = {**constexprs, **config.kwargs}
                    elif isinstance(fn, triton.runtime.Heuristics):
                        for v, heur in fn.values.items():
                            constexprs[v] = heur(
                                {
                                    **dict(zip(fn.arg_names, args)),
                                    **kwargs,
                                    **constexprs,
                                }
                            )
                    else:
                        raise RuntimeError("Invalid Runtime Function")
                    fn = fn.fn
                for p in self.jit_function.params:
                    if (
                        p.is_constexpr
                        and p.name not in constexprs
                        and (p.default is not inspect._empty)
                    ):
                        constexprs[p.name] = p.default
                cache[entry_key] = (kernel, constexprs)
            return kernel, constexprs

        kernel, constexprs = cache[entry_key]

        if callable(grid):
            # collect all arguments to the grid fn，ie:
            # 1. args,
            # 2. kwargs,
            # 3. all all other captured arguments in CompiledKernel from Autotunner & Heuristics
            # when kwargs & captured args conflict, captured args have higher priority
            meta = {**dict(zip(self.arg_names, args)), **kwargs, **constexprs}
            grid = grid(meta)
        grid = grid + (1, 1)

        kernel[grid[0:3]](*k_args)
        return kernel, constexprs


def libentry():
    """
    Decorator for triton library entries.
    """

    def decorator(fn):
        return LibEntry(fn)

    return decorator