# type: ignore

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

from triton.tools.triton_to_gluon_translator.nvidia_helpers import *  # noqa: F401,F403

# ---- NVIDIA tensor descriptors (TMA requires sm90+) ----


@gluon.jit
def tl_make_tensor_descriptor(base, shape, strides, block_shape, padding_option: ttgl.constexpr = "zero"):
    ttgl.static_assert(False, "tensor descriptors require TMA (sm90+)")


@gluon.jit
def tl_store_tensor_descriptor(desc, offsets, value):
    ttgl.static_assert(False, "tensor descriptors require TMA (sm90+)")


@gluon.jit
def tl_load_tensor_descriptor(desc, offsets):
    ttgl.static_assert(False, "tensor descriptors require TMA (sm90+)")


@gluon.jit
def tl_obj_store(obj, offsets, value):
    ttgl.static_assert(not isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor),
                       "tensor descriptors require TMA (sm90+)")
    return obj.store(offsets, value)


@gluon.jit
def tl_obj_load(obj, offsets):
    ttgl.static_assert(not isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor),
                       "tensor descriptors require TMA (sm90+)")
    return obj.load(offsets)
