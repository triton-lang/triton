import triton
import pytest
import torch
import triton.language as tl
import numpy as np
from numpy.random import RandomState


@pytest.mark.parametrize("M, N, K",
                         [(shape)
                          for shape in [[128, 16, 16]]])
def test_slow(M, N, K, device='cuda'):
    ir = f"""
    #blocked = #triton_gpu.blocked<{{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}}>
    #blocked1 = #triton_gpu.blocked<{{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}}>
    #mma = #triton_gpu.mma<{{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1]}}>
    #shared = #triton_gpu.shared<{{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0]}}>
    #shared1 = #triton_gpu.shared<{{vec = 8, perPhase = 2, maxPhase = 2, order = [1, 0]}}>
    """ + """
    module attributes {"triton_gpu.num-warps" = 4 : i32} {
    func.func public @kernel_0d1d2c3d4d5c6d7d8c9d10d11c(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) {
        %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
        %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
        %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi32, #blocked>
        %2 = tt.splat %arg1 : (i32) -> tensor<128x1xi32, #blocked>
        %3 = arith.muli %1, %2 : tensor<128x1xi32, #blocked>
        %4 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>, #blocked>
        %5 = tt.addptr %4, %3 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
        %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %7 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
        %8 = tt.expand_dims %6 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x16xi32, #blocked>
        %9 = tt.expand_dims %7 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x16xi32, #blocked1>
        %10 = tt.broadcast %5 : (tensor<128x1x!tt.ptr<f32>, #blocked>) -> tensor<128x16x!tt.ptr<f32>, #blocked>
        %11 = tt.broadcast %8 : (tensor<1x16xi32, #blocked>) -> tensor<128x16xi32, #blocked>
        %12 = tt.addptr %10, %11 : tensor<128x16x!tt.ptr<f32>, #blocked>, tensor<128x16xi32, #blocked>
        %13 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
        %14 = tt.expand_dims %13 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<16x1xi32, #blocked1>
        %15 = tt.splat %arg3 : (i32) -> tensor<16x1xi32, #blocked1>
        %16 = arith.muli %14, %15 : tensor<16x1xi32, #blocked1>
        %17 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<16x1x!tt.ptr<f32>, #blocked1>
        %18 = tt.addptr %17, %16 : tensor<16x1x!tt.ptr<f32>, #blocked1>, tensor<16x1xi32, #blocked1>
        %19 = tt.broadcast %18 : (tensor<16x1x!tt.ptr<f32>, #blocked1>) -> tensor<16x16x!tt.ptr<f32>, #blocked1>
        %20 = tt.broadcast %9 : (tensor<1x16xi32, #blocked1>) -> tensor<16x16xi32, #blocked1>
        %21 = tt.addptr %19, %20 : tensor<16x16x!tt.ptr<f32>, #blocked1>, tensor<16x16xi32, #blocked1>
        %22 = tt.splat %arg5 : (i32) -> tensor<16x1xi32, #blocked1>
        %23 = arith.muli %14, %22 : tensor<16x1xi32, #blocked1>
        %24 = tt.splat %arg4 : (!tt.ptr<f32>) -> tensor<16x1x!tt.ptr<f32>, #blocked1>
        %25 = tt.addptr %24, %23 : tensor<16x1x!tt.ptr<f32>, #blocked1>, tensor<16x1xi32, #blocked1>
        %26 = tt.broadcast %25 : (tensor<16x1x!tt.ptr<f32>, #blocked1>) -> tensor<16x16x!tt.ptr<f32>, #blocked1>
        %27 = tt.addptr %26, %20 : tensor<16x16x!tt.ptr<f32>, #blocked1>, tensor<16x16xi32, #blocked1>
        %28 = tt.splat %arg7 : (i32) -> tensor<128x1xi32, #blocked>
        %29 = arith.muli %1, %28 : tensor<128x1xi32, #blocked>
        %30 = tt.splat %arg6 : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>, #blocked>
        %31 = tt.addptr %30, %29 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
        %32 = tt.broadcast %31 : (tensor<128x1x!tt.ptr<f32>, #blocked>) -> tensor<128x16x!tt.ptr<f32>, #blocked>
        %33 = tt.addptr %32, %11 : tensor<128x16x!tt.ptr<f32>, #blocked>, tensor<128x16xi32, #blocked>
        %34 = tt.load %12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x16xf32, #blocked>
        %35 = triton_gpu.convert_layout %34 : (tensor<128x16xf32, #blocked>) -> tensor<128x16xf32, #shared>
        %36 = tt.load %21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf32, #blocked1>
        %37 = triton_gpu.convert_layout %36 : (tensor<16x16xf32, #blocked1>) -> tensor<16x16xf32, #shared1>
        %38 = triton_gpu.convert_layout %35 : (tensor<128x16xf32, #shared>) -> tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
        %39 = triton_gpu.convert_layout %37 : (tensor<16x16xf32, #shared1>) -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
        %40 = tt.dot %38, %39, %cst {allowTF32 = true} : tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x16xf32, #mma>
        %41 = tt.reduce %40 {axis = 0 : i32, redOp = 12 : i32} : tensor<128x16xf32, #mma> -> tensor<16xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
        %42 = tt.expand_dims %41 {axis = 0 : i32} : (tensor<16xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>>) -> tensor<1x16xf32, #mma>
        %43 = tt.broadcast %42 : (tensor<1x16xf32, #mma>) -> tensor<128x16xf32, #mma>
        %44 = arith.subf %40, %43 : tensor<128x16xf32, #mma>
        %45 = tt.reduce %44 {axis = 0 : i32, redOp = 11 : i32} : tensor<128x16xf32, #mma> -> tensor<16xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
        %46 = tt.expand_dims %45 {axis = 0 : i32} : (tensor<16xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>>) -> tensor<1x16xf32, #mma>
        %47 = tt.broadcast %46 : (tensor<1x16xf32, #mma>) -> tensor<128x16xf32, #mma>
        %48 = arith.subf %44, %47 : tensor<128x16xf32, #mma>
        %49 = tt.load %27 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf32, #blocked1>
        %50 = triton_gpu.convert_layout %49 : (tensor<16x16xf32, #blocked1>) -> tensor<16x16xf32, #shared1>
        %51 = triton_gpu.convert_layout %48 : (tensor<128x16xf32, #mma>) -> tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
        %52 = triton_gpu.convert_layout %50 : (tensor<16x16xf32, #shared1>) -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
        %53 = tt.dot %51, %52, %cst {allowTF32 = true} : tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x16xf32, #mma>
        %54 = triton_gpu.convert_layout %53 : (tensor<128x16xf32, #mma>) -> tensor<128x16xf32, #blocked>
        tt.store %33, %54 {cache = 1 : i32, evict = 1 : i32} : tensor<128x16xf32, #blocked>
        return
    }
    }
    """

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    inc = [[row*N + col for col in range(K)] for row in range(M)]
    ident = [[1 if i == j else 0 for i in range(N)] for j in range(K)]
    zeros = [[0 for i in range(N)] for j in range(M)]
    x = np.array(inc).astype('float32')
    y = np.array(ident).astype('float32')
    w = np.array(ident).astype('float32')
    z = np.array(zeros).astype('float32')
    x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
    y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
    w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)
    w_tri = torch.tensor(w, device=device)
    z_tri = torch.tensor(z, device=device)

    pgm = kernel[(1, 1, 4)](x_tri, 16,
                         y_tri, 16,
                         w_tri, 16,
                         z_tri, 16)

    z_ref = np.matmul(x, y)
    z_ref = z_ref - np.max(z_ref, axis=0, keepdims=True)
    z_ref = z_ref - np.min(z_ref, axis=0, keepdims=True)
    z_ref = np.matmul(z_ref, w)

    # compare
    # print(z_ref[:,0], z_tri[:,0])

    # XXX: Somehow there's a larger difference when we use float32
    np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)

@pytest.mark.parametrize("M, N, K",
                         [(shape)
                          for shape in [[128, 16, 16]]])
def test_fast(M, N, K, device='cuda'):
    ir = f"""
#blocked = #triton_gpu.blocked<{{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}}>
#blocked1 = #triton_gpu.blocked<{{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}}>
#mma = #triton_gpu.mma<{{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1]}}>
#shared = #triton_gpu.shared<{{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0]}}>
#shared1 = #triton_gpu.shared<{{vec = 8, perPhase = 2, maxPhase = 2, order = [1, 0]}}>
""" + """
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  func.func public @kernel_0d1d2c3d4d5c6d7d8c9d10d11c(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<128x1xi32, #blocked>
    %2 = tt.splat %arg1 : (i32) -> tensor<128x1xi32, #blocked>
    %3 = arith.muli %1, %2 : tensor<128x1xi32, #blocked>
    %4 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
    %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %8 = tt.expand_dims %6 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x16xi32, #blocked>
    %9 = tt.expand_dims %7 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x16xi32, #blocked1>
    %10 = tt.broadcast %5 : (tensor<128x1x!tt.ptr<f32>, #blocked>) -> tensor<128x16x!tt.ptr<f32>, #blocked>
    %11 = tt.broadcast %8 : (tensor<1x16xi32, #blocked>) -> tensor<128x16xi32, #blocked>
    %12 = tt.addptr %10, %11 : tensor<128x16x!tt.ptr<f32>, #blocked>, tensor<128x16xi32, #blocked>
    %13 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<16x1xi32, #blocked1>
    %15 = tt.splat %arg3 : (i32) -> tensor<16x1xi32, #blocked1>
    %16 = arith.muli %14, %15 : tensor<16x1xi32, #blocked1>
    %17 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<16x1x!tt.ptr<f32>, #blocked1>
    %18 = tt.addptr %17, %16 : tensor<16x1x!tt.ptr<f32>, #blocked1>, tensor<16x1xi32, #blocked1>
    %19 = tt.broadcast %18 : (tensor<16x1x!tt.ptr<f32>, #blocked1>) -> tensor<16x16x!tt.ptr<f32>, #blocked1>
    %20 = tt.broadcast %9 : (tensor<1x16xi32, #blocked1>) -> tensor<16x16xi32, #blocked1>
    %21 = tt.addptr %19, %20 : tensor<16x16x!tt.ptr<f32>, #blocked1>, tensor<16x16xi32, #blocked1>
    %22 = tt.splat %arg5 : (i32) -> tensor<16x1xi32, #blocked1>
    %23 = arith.muli %14, %22 : tensor<16x1xi32, #blocked1>
    %24 = tt.splat %arg4 : (!tt.ptr<f32>) -> tensor<16x1x!tt.ptr<f32>, #blocked1>
    %25 = tt.addptr %24, %23 : tensor<16x1x!tt.ptr<f32>, #blocked1>, tensor<16x1xi32, #blocked1>
    %26 = tt.broadcast %25 : (tensor<16x1x!tt.ptr<f32>, #blocked1>) -> tensor<16x16x!tt.ptr<f32>, #blocked1>
    %27 = tt.addptr %26, %20 : tensor<16x16x!tt.ptr<f32>, #blocked1>, tensor<16x16xi32, #blocked1>
    %28 = tt.splat %arg7 : (i32) -> tensor<128x1xi32, #blocked>
    %29 = arith.muli %1, %28 : tensor<128x1xi32, #blocked>
    %30 = tt.splat %arg6 : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %31 = tt.addptr %30, %29 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
    %32 = tt.broadcast %31 : (tensor<128x1x!tt.ptr<f32>, #blocked>) -> tensor<128x16x!tt.ptr<f32>, #blocked>
    %33 = tt.addptr %32, %11 : tensor<128x16x!tt.ptr<f32>, #blocked>, tensor<128x16xi32, #blocked>
    %34 = tt.load %12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x16xf32, #blocked>
    %35 = triton_gpu.convert_layout %34 : (tensor<128x16xf32, #blocked>) -> tensor<128x16xf32, #shared>
    %36 = tt.load %21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf32, #blocked1>
    %37 = triton_gpu.convert_layout %36 : (tensor<16x16xf32, #blocked1>) -> tensor<16x16xf32, #shared1>
    %38 = triton_gpu.convert_layout %35 : (tensor<128x16xf32, #shared>) -> tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
    %39 = triton_gpu.convert_layout %37 : (tensor<16x16xf32, #shared1>) -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
    %40 = tt.dot %38, %39, %cst {allowTF32 = true} : tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x16xf32, #mma>
    %41 = tt.reduce %40 {axis = 1 : i32, redOp = 12 : i32} : tensor<128x16xf32, #mma> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %42 = tt.expand_dims %41 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<128x1xf32, #mma>
    %43 = tt.broadcast %42 : (tensor<128x1xf32, #mma>) -> tensor<128x16xf32, #mma>
    %44 = arith.subf %40, %43 : tensor<128x16xf32, #mma>
    %45 = tt.reduce %44 {axis = 1 : i32, redOp = 11 : i32} : tensor<128x16xf32, #mma> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %46 = tt.expand_dims %45 {axis = 1 : i32} : (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<128x1xf32, #mma>
    %47 = tt.broadcast %46 : (tensor<128x1xf32, #mma>) -> tensor<128x16xf32, #mma>
    %48 = arith.subf %44, %47 : tensor<128x16xf32, #mma>
    %49 = tt.load %27 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf32, #blocked1>
    %50 = triton_gpu.convert_layout %49 : (tensor<16x16xf32, #blocked1>) -> tensor<16x16xf32, #shared1>
    %51 = triton_gpu.convert_layout %48 : (tensor<128x16xf32, #mma>) -> tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
    %52 = triton_gpu.convert_layout %50 : (tensor<16x16xf32, #shared1>) -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
    %53 = tt.dot %51, %52, %cst {allowTF32 = true} : tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x16xf32, #mma>
    %54 = triton_gpu.convert_layout %53 : (tensor<128x16xf32, #mma>) -> tensor<128x16xf32, #blocked>
    tt.store %33, %54 {cache = 1 : i32, evict = 1 : i32} : tensor<128x16xf32, #blocked>
    return
  }
}       
"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    inc = [[row*N + col for col in range(K)] for row in range(M)]
    ident = [[1 if i == j else 0 for i in range(N)] for j in range(K)]
    zeros = [[0 for i in range(N)] for j in range(M)]
    x = np.array(inc).astype('float32')
    y = np.array(ident).astype('float32')
    w = np.array(ident).astype('float32')
    z = np.array(zeros).astype('float32')
    x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
    y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
    w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)
    w_tri = torch.tensor(w, device=device)
    z_tri = torch.tensor(z, device=device)

    pgm = kernel[(1, 1, 4)](x_tri, 16,
                         y_tri, 16,
                         w_tri, 16,
                         z_tri, 16)

    z_ref = np.matmul(x, y)
    z_ref = z_ref - np.max(z_ref, axis=1, keepdims=True)
    z_ref = z_ref - np.min(z_ref, axis=1, keepdims=True)
    z_ref = np.matmul(z_ref, w)
    # import sys
    # np.set_printoptions(edgeitems=10, linewidth=200, threshold=sys.maxsize)
    # torch.set_printoptions(edgeitems=10, linewidth=200, threshold=5000)
    # print(z_ref.astype(int))
    # print(z_tri)
    # breakpoint()
    # XXX: Somehow there's a larger difference when we use float32
    np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)
                                
@pytest.mark.parametrize("M, N, K, num_warps, epilogue, allow_tf32, in_dtype, out_dtype, axis",
                         [(*shape_nw, 'softmax', allow_tf32, in_dtype, out_dtype, axis)
                          for shape_nw in [[128, 16, 16, 4]]
                          for allow_tf32 in [True]
                          for in_dtype, out_dtype in [('float32', 'float32')]
                          for axis in [0, 1]])
def test_dot(M, N, K, num_warps, epilogue, allow_tf32, in_dtype, out_dtype, axis, device='cuda'):
    capability = torch.cuda.get_device_capability()
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xk,
               Y, stride_yk, stride_yn,
               W, stride_wn, stride_wl,
               Z, stride_zm, stride_zn,
               out_dtype: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,
               ALLOW_TF32: tl.constexpr,
               DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,
               AXIS: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_l = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Ws = W + off_n[:, None] * stride_wn + off_l[None, :] * stride_wl
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        x = tl.load(Xs)
        y = tl.load(Ys)
        z = tl.dot(x, y, allow_tf32=ALLOW_TF32, out_dtype=out_dtype)
        max = tl.max(z, AXIS)
        if AXIS == 1:
            z = z - max[:, None]
        else:
            z = z - max[None, :]
        min = tl.min(z, AXIS)
        if AXIS == 1:
            z = z - min[:, None]
        else:
            z = z - min[None, :]
        w = tl.load(Ws)
        z = tl.dot(z.to(w.dtype), w, out_dtype=out_dtype)
        tl.store(Zs, z)
    # input
    rs = RandomState(17)
    inc = [[row*N + col for col in range(K)] for row in range(M)]
    ident = [[1 if i == j else 0 for i in range(N)] for j in range(K)]
    x = np.array(inc).astype(in_dtype)
    y = np.array(ident).astype(in_dtype)
    w = np.array(ident).astype(in_dtype)
    # x = rs.randint(0, 4, (M, K)).astype(in_dtype)
    # y = rs.randint(0, 4, (K, N)).astype(in_dtype)
    # w = np.ones((N, N)).astype(in_dtype)
    if in_dtype == 'float32' and allow_tf32:
        x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
        y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
        w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)
    w_tri = torch.tensor(w, device=device)
    z = 1 + rs.randint(0, 4, (M, N)).astype(in_dtype)

    z_tri = torch.tensor(z, device=device)
    out_dtype = tl.float32

    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1),
                         y_tri, y_tri.stride(0), y_tri.stride(1),
                         w_tri, w_tri.stride(0), w_tri.stride(1),
                         z_tri, z_tri.stride(0), z_tri.stride(1),
                         out_dtype,
                         BLOCK_M=M, BLOCK_K=K, BLOCK_N=N,
                         ADD_MATRIX=epilogue == 'add-matrix',
                         ADD_ROWS=epilogue == 'add-rows',
                         ADD_COLS=epilogue == 'add-cols',
                         DO_SOFTMAX=epilogue == 'softmax',
                         CHAIN_DOT=epilogue == 'chain-dot',
                         AXIS=axis,
                         ALLOW_TF32=allow_tf32,
                         num_warps=num_warps)
    z_ref = np.matmul(x, y)
    z_ref = z_ref - np.max(z_ref, axis=axis, keepdims=True)
    z_ref = z_ref - np.min(z_ref, axis=axis, keepdims=True)
    z_ref = np.matmul(z_ref, w)
    # import sys
    # np.set_printoptions(edgeitems=10, linewidth=200, threshold=sys.maxsize)
    # torch.set_printoptions(edgeitems=10, linewidth=200, threshold=5000)
    # print(z_ref.astype(int))
    # print(z_tri)
    # breakpoint()
    # compare
    # print(z_ref[:,0], z_tri[:,0])
    if in_dtype == 'float32':
        # XXX: Somehow there's a larger difference when we use float32
        np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)
    elif out_dtype == tl.float16:
        np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)
    else:
        np.savetxt('np.out', z_ref)
        np.savetxt('triton.out', z_tri.cpu().numpy())
        np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01)

@pytest.mark.parametrize("M, N, K, num_warps, epilogue, allow_tf32, in_dtype, out_dtype, axis",
                         [(*shape_nw, 'softmax', allow_tf32, in_dtype, out_dtype, axis)
                          for shape_nw in [[128, 16, 16, 4]]
                          for allow_tf32 in [True]
                          for in_dtype, out_dtype in [('float32', 'float32')]
                          for axis in [0]])
def test_reduce(M, N, K, num_warps, epilogue, allow_tf32, in_dtype, out_dtype, axis, device='cuda'):
    capability = torch.cuda.get_device_capability()

    # triton kernel
    @triton.jit
    def reduce_kernel(X, Z, stride_xm, stride_xn, stride_zm, stride_zn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
        Zs = Z + off_n * stride_zn
        x = tl.load(Xs)
        z = tl.max(x, AXIS)
        tl.store(Zs, z)
    # input
    rs = RandomState(17)
    inc = [[row*N + col for col in range(N)] for row in range(M)]
    x = np.array(inc).astype(in_dtype)
    # x = rs.randint(0, 4, (M, N)).astype(in_dtype)
    x_tri = torch.tensor(x, device=device)
    z = 1 + rs.randint(0, 4, (1, N)).astype(in_dtype)
    z_tri = torch.tensor(z, device=device)

    pgm = reduce_kernel[(1, 1)](x_tri, z_tri, x_tri.stride(0), x_tri.stride(1), z_tri.stride(0), z_tri.stride(1), M, N, axis)
    z_ref = x
    z_ref = np.max(z_ref, axis=axis, keepdims=True)

    # compare
    # print(z_ref[:,0], z_tri[:,0])
    # XXX: Somehow there's a larger difference when we use float32
    np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)