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

# TODO: add rocshmem


@core.extern
def my_pe(_builder=None):
    return core.extern_elementwise("librocshmem_device", "", [], {
        (): ("rocshmem_my_pe", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def n_pes(_builder=None):
    return core.extern_elementwise("librocshmem_device", "", [], {
        (): ("rocshmem_n_pes", core.dtype("int32")),
    }, is_pure=True, _builder=_builder)


@core.extern
def int_p(dest, value, pe, _builder=None):
    # force have a return value, even not used.
    return core.extern_elementwise(
        "librocshmem_device", "", [dest, value, pe], {
            (core.pointer_type(core.dtype("int32")), core.dtype("int32"), core.dtype("int32")):
            ("rocshmem_int_p", core.dtype("int32")),
        }, is_pure=False, _builder=_builder, check_args=False)


@core.extern
def remote_ptr(local_ptr, pe, _builder=None):
    return core.extern_elementwise(
        "librocshmem_device", "", [local_ptr, pe], {
            (core.pointer_type(core.dtype("int32")), core.dtype("int32")):
            ("rocshmem_ptr", core.pointer_type(core.dtype("int32"))),
        }, is_pure=False, _builder=_builder, check_args=False)
