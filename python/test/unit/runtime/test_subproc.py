import multiprocessing
import shutil
import tempfile

import triton
import triton.language as tl
from triton.backends.compiler import AttrsDescriptor
from triton.compiler import ASTSource, IRSource
from triton._C.libtriton import ir

target = triton.runtime.driver.active.get_current_target()


def compile_fn(attrs):

    @triton.jit
    def kernel_sub(a, b, o, N: tl.constexpr):
        idx = tl.arange(0, N)
        tl.store(o + idx, tl.load(a + idx) - tl.load(b + idx) * 777)

    src = ASTSource(
        fn=kernel_sub,
        constants={'N': 32},
        signature={'a': "*fp32", 'b': "*fp32", 'o': "*fp32"},
        attrs=attrs,
    )
    triton.compile(src=src, target=target)


def test_compile_in_subproc() -> None:
    config = AttrsDescriptor.from_hints({i: 16 for i in range(4)})
    multiprocessing.set_start_method('fork')
    proc = multiprocessing.Process(target=compile_fn, args=(config, ))
    proc.start()
    proc.join()
    assert proc.exitcode == 0


def compile_fn_dot(attrs):

    @triton.jit
    def kernel_dot(Z):
        offs = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
        z = tl.load(Z + offs)
        z = tl.dot(z, z)
        tl.store(Z + offs, z)

    src = ASTSource(fn=kernel_dot, signature={'Z': "*fp32"}, attrs=attrs, constants={})
    triton.compile(src=src, target=target)


def test_compile_in_forked_subproc(fresh_triton_cache) -> None:
    config = AttrsDescriptor.from_hints({0: 16})
    assert multiprocessing.get_start_method() == 'fork'
    proc = multiprocessing.Process(target=compile_fn_dot, args=(config, ))
    proc.start()
    proc.join()
    assert proc.exitcode == 0


def compile_empty_kernel_with_gc(attrs):

    @triton.jit
    def empty_kernel():
        pass

    import gc
    gc.collect()
    src = ASTSource(fn=empty_kernel, signature={}, attrs=attrs, constants={})
    triton.compile(src=src, target=target)


def test_compile_in_forked_subproc_with_forced_gc(fresh_triton_cache) -> None:
    '''
    Tests that compilation artifacts can safely live in forked process.

    Scenario being tested here ("p" stands for parent process, "c" is child process):
    1. p compiles a kernel 1, and produces compilation artifacts.
    2. p forks the process to create c.
    3. c deletes compilation artifacts inherited from p, compiles kernel 2, and terminates.
    3. p wait for c and join it.

    This is a regression test that ensures thread pool in MLIRContext is released
    safely after compilation.
    '''
    import gc
    old_gc_state = gc.isenabled()
    # disable GC to manage resources manually in the manner described in comment above
    gc.disable()

    # stage 1.p
    config = AttrsDescriptor.from_hints({0: 16})
    compile_empty_kernel_with_gc(config)

    # stage 2.p
    shutil.rmtree(fresh_triton_cache)
    assert multiprocessing.get_start_method() == 'fork'
    proc = multiprocessing.Process(target=compile_empty_kernel_with_gc, args=(config, ))

    # stage 3.c
    proc.start()
    # stage 3.p
    proc.join()

    # restore gc state
    if old_gc_state:
        gc.enable()
    assert proc.exitcode == 0


def test_mlir_attribute_parsing() -> None:
    '''
    Tests that MLIR attributes are parsed correctly from input ttir/ttgir.

    Checks for the following:
    1. Name and type signature are parsed correctly
    2. _get_num_warps_from_ir_str() works
    3. tt.nv_tma_desc attribute is parsed correctly
    '''

    sample_ttgir = r"""
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                %arg3: i32 {tt.divisibility = 16 : i32},
                                %arg4: i32 {tt.divisibility = 16 : i32},
                                %arg5: i32 {tt.divisibility = 16 : i32},
                                %arg6: i32 {tt.divisibility = 16 : i32},
                                %arg7: i32 {tt.divisibility = 16 : i32},
                                %arg8: i32 {tt.divisibility = 16 : i32},
                                %desc: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}) attributes {noinline = false} {
    tt.return
  }
}
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(sample_ttgir)
        f.flush()
        context = ir.context()
        ir.load_dialects(context)
        src = IRSource(f.name, context)

        # check name and type signature
        # should match ty_to_cpp(...)
        assert  src.signature == \
                    {0: "*f32", 1: "*f32", 2: "*f32", 3: "i32", \
                           4: "i32", 5: "i32", 6: "i32", 7: "i32", 8: "i32", 9: "nvTmaDesc"}
        assert src.name == "@matmul_kernel"

        # check num warps
        assert src.parse_options()['num_warps'] == 8
