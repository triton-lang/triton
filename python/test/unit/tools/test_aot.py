import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

import triton
from triton.common import cuda_include_dir, libcuda_dirs

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
def kernel(C, A, B,
          stride_cm, stride_cn,
          stride_am, stride_ak,
          stride_bk, stride_bn,
          BLOCK_M: tl.constexpr,
          BLOCK_N: tl.constexpr,
          BLOCK_K: tl.constexpr):
  ms = tl.arange(0, BLOCK_M)
  ns = tl.arange(0, BLOCK_N)
  ks = tl.arange(0, BLOCK_K)
  a = tl.load(A + ms[:, None] * stride_am + ks[None, :] * stride_ak)
  b = tl.load(B + ks[:, None] * stride_bk + ns[None, :] * stride_bn)
  c = tl.dot(a, b)
  c = kernel_utils.mul(c, c)
  tl.store(C + ms[:, None] * stride_cm + ns[None, :] * stride_cn, c)
"""


def gen_test_bin(dir, M, N, K, BM, BN, BK):
    test_src = '''
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
}'''

    test_src += f'''
int main(int argc, char **argv) {{
  int M = {M}, N = {N}, K = {K};
  int BM = {M}, BN = {N}, BK = {K};

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
  load_matmul_fp16xfp16_16x16x16();

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
  cuStreamSynchronize(stream);
  CUresult ret = matmul_fp16xfp16_16x16x16(stream, M/BM, N/BN, 1, C, A, B, N, 1, K, 1, N, 1);
  if (ret != 0) fprintf(stderr, "kernel launch failed\\n");
  assert(ret == 0);

  cuStreamSynchronize(stream);

  // read data
  int32_t hC[M*N];
  memset(hC, 0, M*N*4);
  cuMemcpyDtoH(hC, C, M*N*4);
  write_buffer_to_csv(argv[3], hC, M*N);

  // free cuda handles
  unload_matmul_fp16xfp16_16x16x16();
  cuMemFree(A);
  cuMemFree(B);
  cuMemFree(C);
  cuCtxDestroy(ctx);
}}
'''

    with open(os.path.join(dir, "test.c"), "w") as file:
        file.write(test_src)
    c_files = glob.glob(os.path.join(dir, "*.c"))
    subprocess.run(["gcc"] + c_files + ["-I", cuda_include_dir(),
                                        "-L", libcuda_dirs()[0],
                                        "-l", "cuda",
                                        "-o", "test"], check=True, cwd=dir)


def generate_matmul_launcher(dir, dtype, BM, BN, BK, ha_hb_hints):
    kernel_path = os.path.join(dir, "kernel.py")
    with open(kernel_path, "w") as file:
        file.write(kernel_src)

    kernel_utils_path = os.path.join(dir, "kernel_utils.py")
    with open(kernel_utils_path, "w") as file:
        file.write(kernel_utils_src)

    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # compile all desired configs
    for ha in ha_hb_hints:
        for hb in ha_hb_hints:
            sig = f'*fp32:16, *{dtype}:16, *{dtype}:16, i32{ha}, i32:1, i32{hb}, i32:1, i32:16, i32:1, {BM}, {BN}, {BK}'
            name = f"matmul_{dtype}x{dtype}_{BM}x{BN}x{BK}"
            subprocess.run([sys.executable, compiler_path, "-n", "kernel", "--signature", sig, "--out-name", name, "-o", name, "-w", "1", kernel_path], check=True, cwd=dir)

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


def test_compile_link_matmul():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:

        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        generate_matmul_launcher(tmp_dir, dtype, BM, BN, BK, ha_hb_hints=["", ":16"])

        # compile test case
        M, N, K = 16, 16, 16
        gen_test_bin(tmp_dir, M, N, K, BM, BN, BK)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        subprocess.run(["./test", a_path, b_path, c_path], check=True, cwd=tmp_dir)

        # read data and compare against reference
        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((M, N)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.)


def test_launcher_has_no_available_kernel():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        generate_matmul_launcher(tmp_dir, dtype, BM, BN, BK, ha_hb_hints=[":1"])

        # compile test case
        M, N, K = 16, 16, 16
        gen_test_bin(tmp_dir, M, N, K, BM, BN, BK)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        result = subprocess.run(["./test", a_path, b_path, c_path], cwd=tmp_dir, capture_output=True, text=True)

        # It should fail since the launcher requires all the strides be 1 while they are not.
        assert result.returncode == -6
        assert "kernel launch failed" in result.stderr


def test_ttgir_to_ptx():
    src = """
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @sum_kernel_0d1d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
    tt.return
  }
}
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = os.path.join(tmp_dir, "empty_kernel.ttgir")
        with open(kernel_path, "w") as fp:
            fp.write(src)
        k = triton.compile(kernel_path, cc=80)
        ptx = k.asm["ptx"]
        assert ".target sm_80" in ptx
        assert ".address_size 64" in ptx
