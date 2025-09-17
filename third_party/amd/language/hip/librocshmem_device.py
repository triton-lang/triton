################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
from triton.language import core
import triton.language as tl
from triton_dist.language.core import extern_call


@core.extern
def set_rocshmem_ctx(ctx, _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(ctx, tl.pointer_type(tl.void), _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void),): ("rocshmem_set_ctx", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


void_ptr = core.pointer_type(core.void)


@core.extern
def my_pe(_semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [],
        {(): ("rocshmem_my_pe_wrapper", (tl.int32))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def n_pes(_semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [],
        {(): ("rocshmem_n_pes_wrapper", (tl.int32))},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def int_p(dest, value, pe, _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(value, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.int32, tl.int32): ("rocshmem_int_p_wrapper", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def remote_ptr(local_ptr, pe, _semantic=None):
    return tl.cast(
        extern_call(
            "librocshmem_device",
            "",
            [
                tl.cast(local_ptr, tl.pointer_type(tl.void), _semantic=_semantic),
                tl.cast(pe, tl.int32, _semantic=_semantic),
            ],
            {
                (tl.pointer_type(tl.void), tl.int32): (
                    "rocshmem_ptr_wrapper",
                    tl.pointer_type(tl.void),
                ),
            },
            is_pure=False,
            _semantic=_semantic,
        ),
        local_ptr.dtype,
        _semantic=_semantic,
    )
