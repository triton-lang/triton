from abc import ABCMeta, abstractmethod
import re
from typing import Callable, List, Protocol, Sequence

from triton._utils import find_paths_if
from triton._C.libtriton import make_tensordesc_args


def decompose_descriptor(arg):
    # Currently host-side tensor descriptors are passed as tensor desc + shape + strides.
    # We still need to pass shape/strides after descriptor lowering, so they appear twice.
    return [arg.base, *arg.shape, *arg.strides, arg.padding == "nan", arg.round_f32_to_tf32, *arg.shape, *arg.strides]


def _is_descriptor(arg):
    return isinstance(arg, str) and arg.startswith("tensordesc")


def wrap_handle_tensordesc_impl(launcher, signature, tensordesc_meta, make_tensordesc_arg):
    signature = tuple(signature.values())
    tensordesc_paths = find_paths_if(signature, lambda _, x: _is_descriptor(x))
    if len(tensordesc_paths) == 0:
        return launcher

    # Build a tree to speed up tensordesc type checking, e.g.
    # signature = ('tensordesc', 'i32', ('i32', 'tensordesc'))
    # relevant_paths = {0: {}, 2: {1: {}}}
    relevant_paths = {}
    for path in tensordesc_paths:
        cur = relevant_paths
        for step in path:
            cur = cur.setdefault(step, {})

    def inner(*args):
        base_args = args[:-1]
        kernel_args = args[-1]
        wrapped = make_tensordesc_args(
            kernel_args,
            signature,
            relevant_paths,
            tensordesc_meta,
            base_args,
            make_tensordesc_arg,
        )
        return launcher(*base_args, wrapped)

    return inner


def _parse_descriptor(descriptor):
    match = re.match(r"tensordesc(?:_im2col)?<([^[>]*)\[([^\]]*)\]", descriptor)
    assert match, f"Malformed tensor descriptor type: {descriptor}"

    dtype = match.group(1)
    block_shape = match.group(2)
    block_ndim = block_shape.count(",") + 1

    rank_match = re.search(r",input_rank=(\d+)", descriptor)
    ndim = int(rank_match.group(1)) if rank_match else block_ndim
    return (dtype, ndim)


def _expand_descriptor(descriptor, has_tensordesc_meta, descriptor_type):
    dtype, ndim = _parse_descriptor(descriptor)
    expanded = []

    # If there is no descriptor metadata, the descriptor was decomposed to:
    # base pointer, shape, strides, padding, round_f32_to_tf32.
    if not has_tensordesc_meta:
        expanded.append("*" + dtype)
        for _ in range(2 * ndim):
            expanded.append("i64")
        expanded.append("i1")
        expanded.append("i1")
    else:
        expanded.append(descriptor_type)

    for _ in range(ndim):
        expanded.append("i32")
    for _ in range(ndim):
        expanded.append("i64")
    return expanded


def expand_signature(signature, tensordesc_meta, descriptor_type):
    has_tensordesc_meta = bool(tensordesc_meta)

    result = []

    def visit(signature, result):
        if _is_descriptor(signature):
            result.extend(_expand_descriptor(signature, has_tensordesc_meta, descriptor_type))
            return
        elif isinstance(signature, tuple):
            inner = []
            for s in signature:
                visit(s, inner)
            result.append(tuple(inner))
        else:
            result.append(signature)

    result = []
    for s in signature:
        visit(s, result)
    return result


class Benchmarker(Protocol):

    def __call__(self, kernel_call: Callable, *, quantiles: List[float], **kwargs) -> Sequence[float]:
        pass


class DriverBase(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def is_active(self):
        pass

    @abstractmethod
    def map_python_to_cpp_type(self, ty: str) -> str:
        """
        Converts a Triton type string to its corresponding C++ type string for this backend.

        Args:
            ty (str): The Triton type string. e.g., 'i32', '*fp16', 'fp32'.

        Returns:
            str: The C++ type string.
        """
        pass

    @abstractmethod
    def get_current_target(self):
        pass

    @abstractmethod
    def get_active_torch_device(self):
        pass

    @abstractmethod
    def get_benchmarker(self) -> Benchmarker:
        """
        Return the benchmarking function that this backend should use by default.
        """
        raise NotImplementedError

    def __init__(self) -> None:
        pass


class GPUDriver(DriverBase):

    def __init__(self):
        # TODO: support other frameworks than torch
        import torch
        self.get_device_capability = torch.cuda.get_device_capability
        try:
            from torch._C import _cuda_getCurrentRawStream
            self.get_current_stream = _cuda_getCurrentRawStream
        except ImportError:
            self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device

    # TODO: remove once TMA is cleaned up
    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
