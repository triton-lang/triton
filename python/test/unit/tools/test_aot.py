import glob
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import triton

kernel_src = """
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
  tl.store(C + ms[:, None] * stride_cm + ns[None, :] * stride_cn, c)
"""

test_src = """
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include "kernel.h"

static void read_csv_to_buffer(char *filename, int16_t *buffer, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf(\"Could not open file %s\\n\", filename);
        return;
    }
    int index = 0;
    while (fscanf(file, "%hd,", &buffer[index]) != EOF && index < size) {
        index++;
    }
    fclose(file);
}

int main(int argc, char **argv) {
  int M = 16, N = 16, K = 16;
  int BM = 16, BN = 16, BK = 16;

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
  load_kernel();

  // initialize input data
  int16_t hA[M*K];
  int16_t hB[K*N];
  read_csv_to_buffer(argv[0], hA, M*K);
  read_csv_to_buffer(argv[1], hB, K*N);
  cuMemcpyHtoD(A, hA, M*K*2);
  cuMemcpyHtoD(B, hB, K*N*2);

  // launch kernel
  int numWarps = 1;
  int gX = 1, gY = 1, gZ = 1;
  kernel(stream, M/BM, N/BN, 1, numWarps, C, A, B, N, 1, K, 1, N, 1);

  // read data
  int32_t hC[M*N];
  cuMemcpyDtoH(hC, C, M*N*4);
  printf("%f\\n", hC[0]);


  // free cuda handles
  unload_kernel();
  cuMemFree(A);
  cuMemFree(B);
  cuMemFree(C);
  cuCtxDestroy(ctx);
}
"""


def test_compile_link_matmul():
    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = os.path.join(tmp_dir, "kernel.py")
        with open(kernel_path, "w") as file:
            file.write(kernel_src)

        compile_path = Path(triton.tools.__path__[0]) / "aot" / "compile.py"
        dtype = "fp16"
        np_dtype = np.float16
        M, N, K = 16, 16, 16
        BM, BN, BK = 16, 16, 16
        # hints = [":16", ""]
        hints = [":16"]
        for ha in hints:
            for hb in hints:
                for hc in [""]:
                    sig = f'*fp32:16, *{dtype}:16, *{dtype}:16, i32{ha}, i32: 1, i32{hb}, i32: 1, i32{hc}, i32: 1, {BM}, {BN}, {BK}'
                    name = f"matmul_{dtype}x{dtype}_{BM}x{BN}x{BK}"
                    subprocess.run(["python", compile_path, "-n", "kernel", "--signature", sig, "--out-name", name, "-o", tmp_dir + "/" + name, kernel_path], check=True)

        link_path = Path(triton.tools.__path__[0]) / "aot" / "link.py"
        subprocess.run(["python", link_path] + glob.glob(tmp_dir + "/*.h") + ["-o", tmp_dir + "/kernel"], check=True)

        test_path = os.path.join(tmp_dir, "test.c")
        with open(test_path, "w") as file:
            file.write(test_src)
        subprocess.run(["gcc"] + glob.glob(tmp_dir + "/*.c") + ["-I", "/usr/local/cuda/include/"] + ["-o", tmp_dir + "/test", "-L", "/usr/lib/wsl/lib/", "-l", "cuda"], check=True)

        # create data
        a_path = os.path.join(tmp_dir, "a.csv")
        b_path = os.path.join(tmp_dir, "b.csv")
        for size, path in [(M * K, a_path), (K * N, b_path)]:
            x = np.random.randn(size).astype(np_dtype)
            np.savetxt(path, x, delimiter=",")

        subprocess.run([os.path.join(tmp_dir, "test"), a_path, b_path], check=True)
