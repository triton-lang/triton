from typing import Optional
from dataclasses import dataclass
import torch
# import triton


@dataclass(frozen=True)
class Config:
    max_shared: int = 0
    alloc_scratch: int = 262144
    alignment: int = 128


@dataclass
class State:
    grid_size: int = 1
    alignment: int = 128
    scratch_size: int = 0
    config: Config = Config()
    stream: Optional[int] = None
    profile_mem_cpu: Optional[torch.Tensor] = None


state: Optional[State] = None
global_scratch_mem: Optional[torch.Tensor] = None
activated: bool = False


def set_alloc_state(global_scratch: torch.Tensor, grid_size: int, scratch_size: int, alignment: int,
                    stream: Optional[int]):
    global state
    global global_scratch_mem
    global activated

    if not activated:
        return

    assert state, "profiler must be initialized"
    state.grid_size = grid_size
    state.scratch_size = scratch_size
    state.alignment = alignment
    state.stream = stream
    global_scratch_mem = global_scratch


def init(config=dict()):
    global state
    global activated

    if not activated:
        return

    state = State()
    # device = triton.runtime.driver.active.get_current_device()
    # shared_mem = triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]
    shared_mem = 220 * 1024
    args = {'max_shared': shared_mem}
    args.update({k: config[k] for k in Config.__dataclass_fields__.keys() if k in config})
    state.config = Config(**args)


def finalize() -> Optional[State]:
    global state
    global global_scratch_mem
    global activated

    if not activated:
        return None

    assert state, "profiler must be initialized"
    curr_state = state
    size = curr_state.grid_size * curr_state.config.alloc_scratch
    # TODO(fywkevin): copy profiling data to profile_mem_cpu, the offset depends on the alignment
    curr_state.profile_mem_cpu = torch.empty(size, device="cpu", dtype=torch.int8)

    state = None
    global_scratch_mem = None
    return curr_state


def _alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def activate():
    global activated
    activated = True
    # triton.set_allocator(_alloc_fn)


def deactivate():
    global activated
    activated = False
