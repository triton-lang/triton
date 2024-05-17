import sys
from io import StringIO

import triton
import triton.language as tl


def test_mma_remark():

    # 32, 32, 128
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
        a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                        block_shape=(32, 128), order=(1, 0))
        b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                        block_shape=(128, 32), order=(0, 1))
        c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                        block_shape=(32, 32), order=(1, 0))
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        c = tl.dot(a, b)
        tl.store(c_block_ptr, c)

    error_capture = StringIO()
    out_capture = StringIO()
    sys.stderr = error_capture
    sys.stdout = out_capture
    error_file = open('error.log', 'w')
    out_file = open('out.log', 'w')
    sys.stderr = error_file
    sys.stdout = out_file
    triton.compile(
        triton.compiler.ASTSource(
            fn=matmul_kernel, signature={
                0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9:
                'i32', 10: 'i32', 11: 'i32'
            }, constants={}))
    error_messages = error_capture.getvalue()
    out_messages = out_capture.getvalue()
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    print(error_messages)
    print(out_messages)

    try:
        assert "test_warning.py:24:16: remark: Warning: can't use MMA V3 for the dot op" in error_messages, "expect MMA V3 remark"
        assert "test_warning.py:24:16: note: see current operation:" in error_messages, "expect MMA V3 note"
    except AssertionError as assertion_err:
        raise assertion_err from error_messages
