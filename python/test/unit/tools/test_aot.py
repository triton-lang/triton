import os
import subprocess
import tempfile
from pathlib import Path

import triton

src = """
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

# with tempfile.TemporaryDirectory() as tmp_dir:
tmp_dir = tempfile.mkdtemp()
kernel_path = os.path.join(tmp_dir, "kernel.py")
with open(kernel_path, "w") as file:
    file.write(src)

compile_path = Path(triton.tools.__path__[0]) / "aot" / "compile.py"
dtype = "fp16"
BM, BN, BK = 16, 16, 16
hints = [":16", ""]
for ha in hints:
    for hb in hints:
        for hc in [""]:
            sig = f'*fp32:16, *{dtype}:16, *{dtype}:16, i32{ha}, i32: 1, i32{hb}, i32: 1, i32{hc}, i32: 1, {BM}, {BN}, {BK}'
            sa = ha.split(":")[-1]
            sb = hb.split(":")[-1]
            name = f"matmul_{dtype}_{BM}_{BN}_{BK}"
            subprocess.run(["python", compile_path, "-n", "kernel", "--signature", sig, "--out-name", name, "-o", tmp_dir + "/" + name, kernel_path], check=True)
print(tmp_dir)
