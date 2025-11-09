import glob
import os
import pytest
import re
import subprocess
import sys
import tempfile

import numpy as np

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.driver import include_dirs, library_dirs
from triton._internal_testing import is_cuda, is_hip

kernel_utils_src = """
import triton

@triton.jit
def mul(x, y):
    return x * y
"""

kernel_src = """
import triton
import triton.language as tl
import kernel_utils

@triton.jit
def kernel(C, A, B, M, N, K,
          stride_cm, stride_cn,
          stride_am, stride_ak,
          stride_bk, stride_bn,
          BLOCK_M: tl.constexpr,
          BLOCK_N: tl.constexpr,
          BLOCK_K: tl.constexpr):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
  offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
  offs_k = tl.arange(0, BLOCK_K)
  a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

  accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  for k in range(0, tl.cdiv(K, BLOCK_K)):
      # Load the next block of A and B, generate a mask by checking the K dimension.
      # If it is out of bounds, set it to 0.
      a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
      b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
      # We accumulate along the K dimension.
      accumulator += tl.dot(a, b)
      # Advance the ptrs to the next K block.
      a_ptrs += BLOCK_K * stride_ak
      b_ptrs += BLOCK_K * stride_bk

  c = kernel_utils.mul(accumulator, accumulator)
  # Write back the block of the output matrix C with masks.
  offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  tl.store(c_ptrs, c)
"""

gluon_kernel_src = """
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def kernel(
    C,
    A,
    B,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    NUM_WARPS: gl.constexpr = 4
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    # Define layouts
    threads_per_elem_mk: gl.constexpr = triton.cdiv(
        BLOCK_M * BLOCK_K // (NUM_WARPS * 64), 16
    )
    threads_per_elem_kn: gl.constexpr = triton.cdiv(
        BLOCK_K * BLOCK_N // (NUM_WARPS * 64), 16
    )
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[threads_per_elem_mk, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, threads_per_elem_kn],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[NUM_WARPS // 2, 2],
    )

    shared_a: gl.constexpr = gl.SwizzledSharedLayout(

        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0, 1]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    # Allocate shared memory
    smem_a = gl.allocate_shared_memory(
        A.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        B.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b
    )

    # Create offsets for first block of A and B
    offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked_kn))
    offs_am = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk)
    )
    offs_bn = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kn)
    )

    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Load first block
    a = gl.amd.cdna3.buffer_load(
        ptr=A,
        offsets=offs_a,
        mask=(offs_ak[None, :] < K) & (offs_am[:, None] < M),
        cache=".ca",
    )
    b = gl.amd.cdna3.buffer_load(
        ptr=B,
        offsets=offs_b,
        mask=(offs_bk[:, None] < K) & (offs_bn[None, :] < N),
        cache=".ca",
    )
    smem_a.store(a)

    # Initialize accumulator
    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)
    zeros = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

    num_k_iter = gl.cdiv(K, BLOCK_K)

    for k in range(num_k_iter - 1):
        # Advance the offsets to the next K block
        offs_a += BLOCK_K * stride_ak
        offs_b += BLOCK_K * stride_bk

        # Load the next block of A and B
        a = gl.amd.cdna3.buffer_load(
            ptr=A,
            offsets=offs_a,
            mask=(offs_ak[None, :] < K - (k + 1) * BLOCK_K) & (offs_am[:, None] < M),
            cache=".ca",
        )

        # Store current B and load from shared memory
        smem_b.store(b)
        cur_a = smem_a.load(layout=dot_a_layout)

        b = gl.amd.cdna3.buffer_load(
            ptr=B,
            offsets=offs_b,
            mask=(offs_bk[:, None] < K - (k + 1) * BLOCK_K) & (offs_bn[None, :] < N),
            cache=".ca",
        )
        cur_b = smem_b.load(layout=dot_b_layout)

        # Perform MFMA
        mfma_out = gl.amd.cdna3.mfma(cur_a, cur_b, zeros)
        accumulator += mfma_out

        smem_a.store(a)

    # Epilogue - final iteration
    smem_b.store(b)
    cur_a = smem_a.load(layout=dot_a_layout)
    cur_b = smem_b.load(layout=dot_b_layout)

    mfma_out = gl.amd.cdna3.mfma(cur_a, cur_b, zeros)
    accumulator += mfma_out

    c = accumulator.to(gl.float16)

    # Write back the block of the output matrix C with masks
    offs_cm = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_cn = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    gl.amd.cdna3.buffer_store(
        stored_value=c, ptr=C, offsets=c_offs, mask=c_mask
    )
"""

test_utils_src = """
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "kernel.h"

static void write_buffer_to_csv(char *filename, int32_t *buffer, int size) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    for (int i = 0; i < size; i++) {
        fprintf(file, "%d", buffer[i]);
        if (i < size - 1) {
            fprintf(file, ",");
        }
    }
    fclose(file);
}

static void read_csv_to_buffer(char *filename, int16_t *buffer, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    int index = 0;
    while (fscanf(file, "%hd,", &buffer[index]) != EOF && index < size) {
        index++;
    }
    fclose(file);
}"""


def gen_kernel_library(dir, libname):
    c_files = glob.glob(os.path.join(dir, "*.c"))
    subprocess.run(
        ["gcc"] + c_files + ["-I", include_dirs[0], "-c", "-fPIC"],
        check=True,
        cwd=dir,
    )
    o_files = glob.glob(os.path.join(dir, "*.o"))

    command = ["gcc", *o_files, "-shared", "-o", libname]
    for lib_dir in library_dirs():
        command.extend(["-L", lib_dir])
    subprocess.run(command, check=True, cwd=dir)


def gen_test_bin(dir, M, N, K, exe="test", algo_id=0):
    test_src = f"""
int main(int argc, char **argv) {{
  int M = {M}, N = {N}, K = {K};

  // initialize CUDA handles
  CUdevice dev;
  CUcontext ctx;
  CUstream stream;
  CUdeviceptr A, B, C;
  CUresult err = 0;
  cuInit(0);
  cuDeviceGet(&dev, 0);
  cuCtxCreate(&ctx, 0, dev);
  cuMemAlloc(&A, M * K * 2);
  cuMemAlloc(&B, K * N * 2);
  cuMemAlloc(&C, M * N * 4);
  cuStreamCreate(&stream, 0);
  load_matmul_fp16();

  // initialize input data
  int16_t hA[M*K];
  int16_t hB[K*N];
  memset(hA, 0, M*K*2);
  memset(hB, 0, K*N*2);
  read_csv_to_buffer(argv[1], hA, M*K);
  read_csv_to_buffer(argv[2], hB, K*N);
  cuMemcpyHtoD(A, hA, M*K*2);
  cuMemcpyHtoD(B, hB, K*N*2);

  // launch kernel
  CUresult ret;
  int algo_id = {algo_id};
  if (algo_id == 0) {{
    ret = matmul_fp16_default(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1);
  }} else {{
    ret = matmul_fp16(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1, {algo_id});
  }}
  if (ret != 0) fprintf(stderr, "kernel launch failed\\n");
  assert(ret == 0);

  // read data
  int32_t hC[M*N];
  memset(hC, 0, M*N*4);
  cuMemcpyDtoH(hC, C, M*N*4);
  write_buffer_to_csv(argv[3], hC, M*N);

  // free cuda handles
  unload_matmul_fp16();
  cuMemFree(A);
  cuMemFree(B);
  cuMemFree(C);
  cuCtxDestroy(ctx);
}}
"""
    src = test_utils_src + test_src
    with open(os.path.join(dir, "test.c"), "w") as file:
        file.write(src)

    command = ["gcc", "test.c"]
    for inc_dir in include_dirs:
        command.extend(["-I", inc_dir])
    for lib_dir in library_dirs():
        command.extend(["-L", lib_dir])
    command.extend(["-l", "cuda", "-L", dir, "-l", "kernel", "-o", exe])
    subprocess.run(command, check=True, cwd=dir)


def write_triton_kernels(dir, src, util_src):
    kernel_path = os.path.join(dir, "kernel.py")
    with open(kernel_path, "w") as file:
        file.write(src)

    kernel_utils_path = os.path.join(dir, "kernel_utils.py")
    with open(kernel_utils_path, "w") as file:
        file.write(util_src)

    return kernel_path


def _compile_kernel(dir, signature, kernel_name, out_name, out_path, num_warps, grid, kernel_path):
    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

    subprocess.run(
        [
            sys.executable,
            compiler_path,
            "-n",
            kernel_name,
            "--signature",
            signature,
            "--out-name",
            out_name,
            "-o",
            out_path,
            "-w",
            str(num_warps),
            "-g",
            grid,
            kernel_path,
        ],
        check=True,
        cwd=dir,
    )


# Edge case kernel with no specialization
def compile_aot_kernel_no_specialization(dir, kernel_path, dtype, BM, BN, BK):
    # compile all desired configs
    sig = f"*fp32, *{dtype}, *{dtype}, i32, i32, i32, i32, i32, i32, i32, i32, i32, {BM}, {BN}, {BK}"
    name = f"matmul_{dtype}"
    grid = f"M/{BM}, N/{BN}, 1"
    _compile_kernel(
        dir=dir,
        signature=sig,
        kernel_name="kernel",
        out_name=name,
        out_path=name,
        num_warps=1,
        grid=grid,
        kernel_path=kernel_path,
    )


def compile_aot_kernels(dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints):
    # compile all desired configs
    for ha, hb in ha_hb_hints:
        sig = f"*fp32:16, *{dtype}:16, *{dtype}:16, i32, i32, i32, i32{ha}, i32:1, i32{hb}, i32:1, i32:16, i32:1, {BM}, {BN}, {BK}"
        name = f"matmul_{dtype}"
        grid = f"M/{BM}, N/{BN}, 1"
        _compile_kernel(
            dir=dir,
            signature=sig,
            kernel_name="kernel",
            out_name=name,
            out_path=name,
            num_warps=1,
            grid=grid,
            kernel_path=kernel_path,
        )


def link_aot_kernels(dir):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # link all desired configs
    h_files = glob.glob(os.path.join(dir, "*.h"))
    subprocess.run([sys.executable, linker_path] + h_files + ["-o", "kernel"], check=True, cwd=dir)


def generate_matmul_test_data(dir, M, N, K):
    a = np.random.randn(M * K).astype(np.float16).reshape((M, K))
    b = np.random.randn(M * K).astype(np.float16).reshape((K, N))
    a_path = os.path.join(dir, "a.csv")
    b_path = os.path.join(dir, "b.csv")
    c_path = os.path.join(dir, "c.csv")
    for x, path in [(a, a_path), (b, b_path)]:
        x.view(np.int16).ravel().tofile(path, sep=",")
    return a, b, a_path, b_path, c_path


def check_hasco_binary_str(tmp_dir: str, dtype: str):
    # Linking is not yet enabled on HIP backend so just check compilation for now.
    h_files = glob.glob(f"matmul_{dtype}.*.h", root_dir=tmp_dir)
    cpp_files = glob.glob(f"matmul_{dtype}.*.cpp", root_dir=tmp_dir)
    assert len(h_files) == 1, "Expected one .h file"
    assert len(cpp_files) == 1, "Expected one .cpp file"
    pattern = re.compile(r'HSACO_NAME\[(\d+)\]')
    with open(os.path.join(tmp_dir, cpp_files[0]), "r") as cpp_file:
        content = cpp_file.read()
        matches = pattern.findall(content)
        assert len(matches) == 1, "Expected one HSACO_NAME definition"
        assert int(matches[0]) > 16, "Expected valid HSACO object binary string"


# Test edge case where the provided kernel signature has no specializations
def test_compile_link_matmul_no_specialization():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernel_no_specialization(tmp_dir, kernel_path, dtype, BM, BN, BK)
        if is_hip():
            check_hasco_binary_str(tmp_dir, dtype)
            return

        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library(tmp_dir, "libkernel.so")
        gen_test_bin(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = tmp_dir
        subprocess.run(["./test", a_path, b_path, c_path], env=env, check=True, cwd=tmp_dir)

        # read data and compare against reference
        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((M, N)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


def test_compile_link_matmul():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernels(tmp_dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints=[(":16", ":16")])
        if is_hip():
            check_hasco_binary_str(tmp_dir, dtype)
            return
        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library(tmp_dir, "libkernel.so")
        gen_test_bin(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = tmp_dir
        subprocess.run(["./test", a_path, b_path, c_path], env=env, check=True, cwd=tmp_dir)

        # read data and compare against reference
        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((M, N)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


def test_launcher_has_no_available_kernel():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernels(tmp_dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints=[(":1", ":1")])
        if is_hip():
            check_hasco_binary_str(tmp_dir, dtype)
            return

        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library(tmp_dir, "libkernel.so")
        gen_test_bin(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = tmp_dir
        result = subprocess.run(
            ["./test", a_path, b_path, c_path],
            env=env,
            cwd=tmp_dir,
            capture_output=True,
            text=True,
        )

        # It should fail since the launcher requires all the strides be 1 while they are not.
        assert result.returncode == -6
        assert "kernel launch failed" in result.stderr


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_compile_link_autotune_matmul():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)

        tile_sizes = [
            [16, 16, 16],
            [64, 64, 32],
        ]

        for ts in tile_sizes:
            BM, BN, BK = ts[0], ts[1], ts[2]
            compile_aot_kernels(tmp_dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints=[(":16", ":16"), (":16", ""),
                                                                                      ("", ":16")])

        link_aot_kernels(tmp_dir)

        gen_kernel_library(tmp_dir, "libkernel.so")

        # compile test case
        M, N, K = 64, 64, 64
        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))

        for algo_id in range(len(tile_sizes)):
            # generate and run test case
            test_name = f"test_{algo_id}"
            gen_test_bin(tmp_dir, M, N, K, exe=test_name, algo_id=algo_id)

            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = tmp_dir
            subprocess.run(
                [f"./{test_name}", a_path, b_path, c_path],
                check=True,
                cwd=tmp_dir,
                env=env,
            )

            # read data and compare against reference
            c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
            c_tri = c.reshape((M, N)).view(np.float32)
            np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=1e-4)


def test_ttgir_to_asm():
    src = """
module attributes {{"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = {warp_size} : i32, "ttg.num-ctas" = 1 : i32}} {{
  tt.func public @sum_kernel_0d1d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {{
    tt.return
  }}
}}
"""
    target = GPUTarget("hip", "gfx942", 64) if is_hip() else GPUTarget("cuda", 80, 32)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = os.path.join(tmp_dir, "empty_kernel.ttgir")
        with open(kernel_path, "w") as fp:
            fp.write(src.format(warp_size=target.warp_size))
        k = triton.compile(kernel_path, target=target)
        if is_cuda():
            ptx = k.asm["ptx"]
            assert ".target sm_80" in ptx
            assert ".address_size 64" in ptx
        elif is_hip():
            amdgcn = k.asm["amdgcn"]
            assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"' in amdgcn
            assert '.wavefront_size: 64' in amdgcn


def test_gluon_kernel():
    if not is_hip():
        pytest.skip("Gluon kernel is only supported on HIP")
    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, gluon_kernel_src, kernel_utils_src)
        compile_aot_kernel_no_specialization(tmp_dir, kernel_path, dtype, BM, BN, BK)
        check_hasco_binary_str(tmp_dir, dtype)
