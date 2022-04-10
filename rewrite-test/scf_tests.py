import pytest

import triton
import triton.language as tl

import torch

def test_if():
  ref_ir = """module {
  func @only_if(%arg0: i32, %arg1: i32, %arg2: i32) {
    %cst = arith.constant -1.000000e+00 : f32
    %0 = arith.cmpi sgt, %arg2, %arg0 : i32
    %1 = scf.if %0 -> (f32) {
      %cst_0 = arith.constant 0.000000e+00 : f32
      scf.yield %cst_0 : f32
    } else {
      scf.yield %cst : f32
    }
    %2 = arith.addf %1, %1 : f32
    return
  }
}
"""

  @triton.jit
  def only_if(lb, ub, value):
    a = -1.0
    if value > lb:
      a = 0.0
    c = a + a

  mod, _ = only_if.compile_to_ttir(2, 3, 4, grid=(1,))
  generated_ir = mod.str()
  assert mod.verify()
  assert ref_ir == generated_ir

def test_if_else():
  ref_ir = """module {
  func @if_else(%arg0: i32, %arg1: i32, %arg2: i32) {
    %0 = arith.cmpi sgt, %arg2, %arg0 : i32
    %1 = scf.if %0 -> (f32) {
      %cst = arith.constant 0.000000e+00 : f32
      scf.yield %cst : f32
    } else {
      %cst = arith.constant 1.000000e+00 : f32
      scf.yield %cst : f32
    }
    %2 = arith.addf %1, %1 : f32
    return
  }
}
"""
  @triton.jit
  def if_else(lb, ub, value):
    if value > lb:
      a = 0.0
    else:
      a = 1.0
    c = a + a

  mod, _ = if_else.compile_to_ttir(2, 3, 4, grid=(1,))
  generated_ir = mod.str()
  assert mod.verify()
  assert ref_ir == generated_ir

def test_for():
  ref_ir = """module {
  func @for_loop(%arg0: i32) {
    %cst = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c0_i32 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.index_cast %c1_i32 : i32 to index
    %3 = scf.for %arg1 = %0 to %1 step %2 iter_args(%arg2 = %cst) -> (f32) {
      %cst_0 = arith.constant 1.000000e+00 : f32
      %4 = arith.addf %arg2, %cst_0 : f32
      scf.yield %4 : f32
    }
    return
  }
}
"""

  @triton.jit
  def for_loop(K):
    a = 1.0
    for k in range(0, K):
      a += 1.0

  mod, _ = for_loop.compile_to_ttir(2, grid=(1,))
  generated_ir = mod.str()
  assert mod.verify()
  assert ref_ir == generated_ir

def test_while():
  ref_ir = """module {
  func @generic_while(%arg0: i32) {
    %c-1_i32 = arith.constant -1 : i32
    %0 = scf.while (%arg1 = %c-1_i32) : (i32) -> i32 {
      %c0_i32 = arith.constant 0 : i32
      %1 = arith.cmpi sle, %arg1, %c0_i32 : i32
      scf.condition(%1) %arg1 : i32
    } do {
    ^bb0(%arg1: i32):
      %c1_i32 = arith.constant 1 : i32
      %1 = arith.addi %arg1, %c1_i32 : i32
      scf.yield %1 : i32
    }
    return
  }
}
"""
  @triton.jit
  def generic_while(x):
    c = -1
    while c <= 0:
      c += 1

  mod, _ = generic_while.compile_to_ttir(2, grid=(1,))
  generated_ir = mod.str()
  assert mod.verify()
  assert ref_ir == generated_ir

def test_nested():
  ref_ir = """module {
  func @nested_cf(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c0_i32 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.index_cast %c1_i32 : i32 to index
    %3 = scf.for %arg4 = %0 to %1 step %2 iter_args(%arg5 = %cst) -> (f32) {
      %5 = arith.cmpi slt, %arg1, %arg2 : i32
      %6 = scf.if %5 -> (f32) {
        %c0_i32_1 = arith.constant 0 : i32
        %c1_i32_2 = arith.constant 1 : i32
        %7 = arith.index_cast %c0_i32_1 : i32 to index
        %8 = arith.index_cast %arg3 : i32 to index
        %9 = arith.index_cast %c1_i32_2 : i32 to index
        %10 = scf.for %arg6 = %7 to %8 step %9 iter_args(%arg7 = %arg5) -> (f32) {
          %cst_3 = arith.constant 2.000000e+00 : f32
          %11 = arith.addf %arg7, %cst_3 : f32
          scf.yield %11 : f32
        }
        scf.yield %10 : f32
      } else {
        %7 = scf.while (%arg6 = %arg5) : (f32) -> f32 {
          %cst_1 = arith.constant 1.200000e+00 : f32
          %8 = arith.cmpf olt, %arg6, %cst_1 : f32
          scf.condition(%8) %arg6 : f32
        } do {
        ^bb0(%arg6: f32):
          %cst_1 = arith.constant 2.000000e+00 : f32
          %8 = arith.mulf %arg6, %cst_1 : f32
          scf.yield %8 : f32
        }
        scf.yield %7 : f32
      }
      scf.yield %6 : f32
    }
    %cst_0 = arith.constant 1.000000e+00 : f32
    %4 = arith.subf %3, %cst_0 : f32
    return
  }
}
"""
  @triton.jit
  def nested_cf(X, lb, ub, Z):
    a = 0.0
    for x in range(0, X):
      if lb < ub:
        for z in range(0, Z):
          a += 2.0
      else:
        while a < 1.2:
          a *= 2.0
    a -= 1.0

  mod, _ = nested_cf.compile_to_ttir(3, 4, 5, 6, grid=(1,))
  generated_ir = mod.str()
  assert mod.verify(), generated_ir
  assert ref_ir == generated_ir

def test_matmul():
  ref_ir = """module {
  func @matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %c64_i32 = arith.constant 64 : i32
    %1 = arith.addi %arg3, %c64_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %2 = arith.subi %1, %c1_i32 : i32
    %c64_i32_0 = arith.constant 64 : i32
    %3 = arith.divsi %2, %c64_i32_0 : i32
    %c64_i32_1 = arith.constant 64 : i32
    %4 = arith.addi %arg4, %c64_i32_1 : i32
    %c1_i32_2 = arith.constant 1 : i32
    %5 = arith.subi %4, %c1_i32_2 : i32
    %c64_i32_3 = arith.constant 64 : i32
    %6 = arith.divsi %5, %c64_i32_3 : i32
    %c8_i32 = arith.constant 8 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.divsi %0, %7 : i32
    %c8_i32_4 = arith.constant 8 : i32
    %9 = arith.muli %8, %c8_i32_4 : i32
    %10 = arith.subi %3, %9 : i32
    %c8_i32_5 = arith.constant 8 : i32
    %11 = arith.cmpi slt, %10, %c8_i32_5 : i32
    %c8_i32_6 = arith.constant 8 : i32
    %12 = select %11, %10, %c8_i32_6 : i32
    %13 = arith.remsi %0, %12 : i32
    %14 = arith.addi %9, %13 : i32
    %15 = arith.remsi %0, %7 : i32
    %16 = arith.divsi %15, %12 : i32
    %c64_i32_7 = arith.constant 64 : i32
    %17 = arith.muli %14, %c64_i32_7 : i32
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %19 = tt.broadcast %17 : (i32) -> tensor<64xi32>
    %20 = arith.addi %19, %18 : tensor<64xi32>
    %c64_i32_8 = arith.constant 64 : i32
    %21 = arith.muli %16, %c64_i32_8 : i32
    %22 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %23 = tt.broadcast %21 : (i32) -> tensor<64xi32>
    %24 = arith.addi %23, %22 : tensor<64xi32>
    %25 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %26 = tt.reshape %20 : (tensor<64xi32>) -> tensor<64x1xi32>
    %27 = tt.broadcast %arg6 : (i32) -> tensor<64x1xi32>
    %28 = arith.muli %26, %27 : tensor<64x1xi32>
    %29 = tt.reshape %25 : (tensor<32xi32>) -> tensor<1x32xi32>
    %c1_i32_9 = arith.constant 1 : i32
    %30 = tt.broadcast %c1_i32_9 : (i32) -> tensor<1x32xi32>
    %31 = arith.muli %29, %30 : tensor<1x32xi32>
    %32 = tt.broadcast %28 : (tensor<64x1xi32>) -> tensor<64x32xi32>
    %33 = tt.broadcast %31 : (tensor<1x32xi32>) -> tensor<64x32xi32>
    %34 = arith.addi %32, %33 : tensor<64x32xi32>
    %35 = tt.broadcast %arg0 : (!tt.ptr<f16>) -> tensor<64x32x!tt.ptr<f16>>
    %36 = tt.getelementptr %35, %34, : tensor<64x32x!tt.ptr<f16>>
    %37 = tt.reshape %25 : (tensor<32xi32>) -> tensor<32x1xi32>
    %38 = tt.broadcast %arg7 : (i32) -> tensor<32x1xi32>
    %39 = arith.muli %37, %38 : tensor<32x1xi32>
    %40 = tt.reshape %24 : (tensor<64xi32>) -> tensor<1x64xi32>
    %c1_i32_10 = arith.constant 1 : i32
    %41 = tt.broadcast %c1_i32_10 : (i32) -> tensor<1x64xi32>
    %42 = arith.muli %40, %41 : tensor<1x64xi32>
    %43 = tt.broadcast %39 : (tensor<32x1xi32>) -> tensor<32x64xi32>
    %44 = tt.broadcast %42 : (tensor<1x64xi32>) -> tensor<32x64xi32>
    %45 = arith.addi %43, %44 : tensor<32x64xi32>
    %46 = tt.broadcast %arg1 : (!tt.ptr<f16>) -> tensor<32x64x!tt.ptr<f16>>
    %47 = tt.getelementptr %46, %45, : tensor<32x64x!tt.ptr<f16>>
    %cst = arith.constant 0.000000e+00 : f32
    %48 = tt.broadcast %cst : (f32) -> tensor<64x64xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %49 = arith.index_cast %c0_i32 : i32 to index
    %50 = arith.index_cast %arg5 : i32 to index
    %51 = arith.index_cast %c32_i32 : i32 to index
    %52:3 = scf.for %arg9 = %49 to %50 step %51 iter_args(%arg10 = %48, %arg11 = %36, %arg12 = %47) -> (tensor<64x64xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x64x!tt.ptr<f16>>) {
      %cst_14 = arith.constant dense<true> : tensor<64x32xi1>
      %cst_15 = arith.constant dense<0.000000e+00> : tensor<64x32xf16>
      %82 = tt.load %arg11, %cst_14, %cst_15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x32xf16>
      %cst_16 = arith.constant dense<true> : tensor<32x64xi1>
      %cst_17 = arith.constant dense<0.000000e+00> : tensor<32x64xf16>
      %83 = tt.load %arg12, %cst_16, %cst_17 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16>
      %cst_18 = arith.constant 0.000000e+00 : f32
      %84 = tt.broadcast %cst_18 : (f32) -> tensor<64x64xf32>
      %85 = tt.dot %82, %83, %84 {allowTF32 = true} : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
      %86 = arith.addf %arg10, %85 : tensor<64x64xf32>
      %c32_i32_19 = arith.constant 32 : i32
      %87 = tt.broadcast %c32_i32_19 : (i32) -> tensor<64x32xi32>
      %88 = tt.getelementptr %arg11, %87, : tensor<64x32x!tt.ptr<f16>>
      %c32_i32_20 = arith.constant 32 : i32
      %89 = arith.muli %arg7, %c32_i32_20 : i32
      %90 = tt.broadcast %89 : (i32) -> tensor<32x64xi32>
      %91 = tt.getelementptr %arg12, %90, : tensor<32x64x!tt.ptr<f16>>
      scf.yield %86, %88, %91 : tensor<64x64xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x64x!tt.ptr<f16>>
    }
    %53 = arith.truncf %52#0 : tensor<64x64xf32> to tensor<64x64xf16>
    %c64_i32_11 = arith.constant 64 : i32
    %54 = arith.muli %14, %c64_i32_11 : i32
    %55 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %56 = tt.broadcast %54 : (i32) -> tensor<64xi32>
    %57 = arith.addi %56, %55 : tensor<64xi32>
    %c64_i32_12 = arith.constant 64 : i32
    %58 = arith.muli %16, %c64_i32_12 : i32
    %59 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %60 = tt.broadcast %58 : (i32) -> tensor<64xi32>
    %61 = arith.addi %60, %59 : tensor<64xi32>
    %62 = tt.reshape %57 : (tensor<64xi32>) -> tensor<64x1xi32>
    %63 = tt.broadcast %arg8 : (i32) -> tensor<64x1xi32>
    %64 = arith.muli %63, %62 : tensor<64x1xi32>
    %65 = tt.broadcast %arg2 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>>
    %66 = tt.getelementptr %65, %64, : tensor<64x1x!tt.ptr<f16>>
    %67 = tt.reshape %61 : (tensor<64xi32>) -> tensor<1x64xi32>
    %c1_i32_13 = arith.constant 1 : i32
    %68 = tt.broadcast %c1_i32_13 : (i32) -> tensor<1x64xi32>
    %69 = arith.muli %67, %68 : tensor<1x64xi32>
    %70 = tt.broadcast %66 : (tensor<64x1x!tt.ptr<f16>>) -> tensor<64x64x!tt.ptr<f16>>
    %71 = tt.broadcast %69 : (tensor<1x64xi32>) -> tensor<64x64xi32>
    %72 = tt.getelementptr %70, %71, : tensor<64x64x!tt.ptr<f16>>
    %73 = tt.reshape %57 : (tensor<64xi32>) -> tensor<64x1xi32>
    %74 = tt.broadcast %arg3 : (i32) -> tensor<64x1xi32>
    %75 = arith.cmpi slt, %73, %74 : tensor<64x1xi32>
    %76 = tt.reshape %61 : (tensor<64xi32>) -> tensor<1x64xi32>
    %77 = tt.broadcast %arg4 : (i32) -> tensor<1x64xi32>
    %78 = arith.cmpi slt, %76, %77 : tensor<1x64xi32>
    %79 = tt.broadcast %75 : (tensor<64x1xi1>) -> tensor<64x64xi1>
    %80 = tt.broadcast %78 : (tensor<1x64xi1>) -> tensor<64x64xi1>
    %81 = arith.andi %79, %80 : tensor<64x64xi1>
    tt.store %72, %53, %81, : tensor<64x64xf16>
    return
  }
}
"""
  @triton.jit
  def matmul_kernel(
      # Pointers to matrices
      a_ptr, b_ptr, c_ptr,
      # Matrix dimensions
      M, N, K,
      # The stride variables represent how much to increase the ptr by when moving by 1
      # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
      # by to get the element one row down (A has M rows)
      stride_am, stride_ak,
      stride_bk, stride_bn,
      stride_cm, stride_cn,
      # Meta-parameters
      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
      GROUP_SIZE_M: tl.constexpr,
  ):
      """Kernel for computing the matmul C = A x B.
      A has shape (M, K), B has shape (K, N) and C has shape (M, N)
      """
      # -----------------------------------------------------------
      # Map program ids `pid` to the block of C it should compute.
      # This is done in a grouped ordering to promote L2 data reuse
      # See above `L2 Cache Optimizations` section for details
      pid = tl.program_id(axis=0)
      num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
      num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
      num_pid_in_group = GROUP_SIZE_M * num_pid_n
      group_id = pid // num_pid_in_group
      first_pid_m = group_id * GROUP_SIZE_M
      group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
      pid_m = first_pid_m + (pid % group_size_m)
      pid_n = (pid % num_pid_in_group) // group_size_m

      # ----------------------------------------------------------
      # Create pointers for the first blocks of A and B.
      # We will advance this pointer as we move in the K direction
      # and accumulate
      # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
      # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_n] pointers
      # see above `Pointer Arithmetics` section for details
      offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
      offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      offs_k = tl.arange(0, BLOCK_SIZE_K)
      a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
      b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

      # -----------------------------------------------------------
      # Iterate to compute a block of the C matrix
      # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
      # of fp32 values for higher accuracy.
      # `accumulator` will be converted back to fp16 after the loop
      accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
      for k in range(0, K, BLOCK_SIZE_K):
          # Note that for simplicity, we don't apply a mask here.
          # This means that if K is not a multiple of BLOCK_SIZE_K,
          # this will access out-of-bounds memory and produce an
          # error or (worse!) incorrect results.
          a = tl.load(a_ptrs)
          b = tl.load(b_ptrs)
          # We accumulate along the K dimension
          accumulator += tl.dot(a, b)
          # Advance the ptrs to the next K block
          a_ptrs += BLOCK_SIZE_K * stride_ak
          b_ptrs += BLOCK_SIZE_K * stride_bk
      c = accumulator.to(tl.float16)

      # -----------------------------------------------------------
      # Write back the block of the output matrix C
      offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
      offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
      c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
      tl.store(c_ptrs, c, mask=c_mask)

  a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
  b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
  c = torch.empty((512, 512), device='cuda', dtype=torch.float16)


  mod, ctx = matmul_kernel.compile_to_ttir(
    a, b, c,
    512, 512, 512,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    64, 64, 32,
    8, grid=(2,)
  )
  verify = mod.verify()
  assert verify
  assert ref_ir == mod.str()
