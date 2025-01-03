import multiprocessing
import shutil

import triton
import triton.language as tl
from triton.compiler import ASTSource

target = triton.runtime.driver.active.get_current_target()
start_method = 'fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn'


def compile_fn():

    @triton.jit
    def kernel_sub(a, b, o, N: tl.constexpr):
        idx = tl.arange(0, N)
        tl.store(o + idx, tl.load(a + idx) - tl.load(b + idx) * 777)

    src = ASTSource(
        fn=kernel_sub,
        constexprs={'N': 32},
        signature={'a': "*fp32", 'b': "*fp32", 'o': "*fp32", 'N': 'constexpr'},
    )
    triton.compile(src=src, target=target)


def test_compile_in_subproc() -> None:
    mp_ctx = multiprocessing.get_context(start_method)
    proc = mp_ctx.Process(target=compile_fn)
    proc.start()
    proc.join()
    assert proc.exitcode == 0


def compile_fn_dot():

    @triton.jit
    def kernel_dot(Z):
        offs = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
        z = tl.load(Z + offs)
        z = tl.dot(z, z)
        tl.store(Z + offs, z)

    src = ASTSource(fn=kernel_dot, signature={'Z': "*fp32"})
    triton.compile(src=src, target=target)


def test_compile_in_forked_subproc(fresh_triton_cache) -> None:
    mp_ctx = multiprocessing.get_context(start_method)
    proc = mp_ctx.Process(target=compile_fn_dot)
    proc.start()
    proc.join()
    assert proc.exitcode == 0


def compile_empty_kernel_with_gc():

    @triton.jit
    def empty_kernel():
        pass

    import gc
    gc.collect()
    src = ASTSource(fn=empty_kernel, signature={})
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
    compile_empty_kernel_with_gc()

    # stage 2.p
    shutil.rmtree(fresh_triton_cache)
    mp_ctx = multiprocessing.get_context(start_method)
    proc = mp_ctx.Process(target=compile_empty_kernel_with_gc)

    # stage 3.c
    proc.start()
    # stage 3.p
    proc.join()

    # restore gc state
    if old_gc_state:
        gc.enable()
    assert proc.exitcode == 0
