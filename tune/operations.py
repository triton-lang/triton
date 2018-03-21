from __future__ import division
import isaac as sc
import numpy as np
from time import time
from tools import benchmark
from isaac._isaac.templates import Conv, GEMM, Pool
import multiprocessing

lock = multiprocessing.Lock()

def tuning_ranges(OpType):
    L3 = [1, 2, 4]
    L4 = [2, 4, 8, 16]
    L5 = [1, 2, 4, 8]
    LR = [1, 2, 4, 8, 16, 32]
    if OpType==Conv:
        return [L3, L4, L4, L5, L5, L4, [1], L5, LR]
    if OpType==GEMM:
        return [L3, L4, L4, L4, L5, [1], L5, L4, L4, L4, L4, [1], L5, LR]
    if OpType==Pool:
        return [L3, [32, 64, 128, 256], L5, L5]

def input_ranges(OpType, device):
    LDT = [4]
    if OpType==Conv:
        LNpix = [4, 8, 16, 32, 128, 512, 1024, 2048, 4096, 8192]
        LC = [8, 16, 32, 64, 128, 256, 512, 1024]
        LNfilt = [9, 25, 49, 81, 200]
        return [LDT,LNpix,LC,LC,LNfilt]
    if OpType==GEMM:
        FP64 = [8] if device.compute_capability[1] == 0 else []
        LDT = LDT + FP64
        LT = [1, 2]
        LK = [16, 32, 64, 256, 4096, 8192, 16382, 65536]
        LMN = [4, 8, 16, 32, 128, 512, 1024, 2048, 4096, 8192]
        return [LDT, LT, LT, LMN, LMN, LK]
    if OpType==Pool:
        LNpix = [2**x for x in np.arange(10,25,.5)]
        LNfilt = [9, 25, 49, 81, 200]
        return [LDT, LNpix, LNfilt]

def valid_shapes(OpType):
    if OpType==Conv:
        return [(dtype, 1024, 1024, 512, 9) for dtype in [4]]
    if OpType==GEMM:
        return [(dtype, AT, BT, 1024, 1024, 1024) for dtype in [4] for AT in [1,2] for BT in [1,2]]
    if OpType==Pool:
        return [(dtype, 1024, 9) for dtype in [4]]

def num_ops(OpType, X):
    if OpType==Conv:
        Npix, K, C, Nfilt = X[:,1], X[:,2], X[:,3], X[:,4]
        return 2.*Npix*K*C*Nfilt
    if OpType==GEMM:
        M, N, K = X[:, 3], X[:, 4], X[:, 5]
        return 2.*M*N*K
    if OpType==Pool:
        Npix, Nfilt = X[:,1], X[:,2]
        return Npix*Nfilt

def keep_valid(OpType, device, X):
    idx = OpType.check_valid(device, X.astype(np.uint32))
    return X[idx, :]

def pool_shapes(device):
    DTs = [4]
    result = []
    for DTYPE in DTs:
        for Nfilt in [9, 27, 125]:
            for Npix in [2**x for x in np.arange(14,25,1)]:
                result += [[DTYPE, Npix, Nfilt]]
    return result

def conv_shapes(device):
    DTs = [4]
    result = []
    #Deep Learning
    for DTYPE in DTs:
         for Npix in [512, 2048, 8192, 65536, 131072]:
            for K in [16, 32, 64, 128, 512]:
                for C in [1, 16, 32, 64, 128, 512]:
                    result += [[DTYPE, Npix, K, C, 25]]
    result = np.array(result)
    flops = num_ops(Conv, result)
    idx = np.logical_and(flops > 1e7, flops < 1e12)
    return result[idx,:]

def gemm_shapes(device):
    sizes = []
    FP64 = [8] if device.compute_capability[1] == 0 else []

    #LinPACK
    for DTYPE in [4] + FP64:
        for AT, BT in [(1, 2)]:
            for N in [256, 512, 1024, 2048, 4096]:
                sizes += [(DTYPE, AT, BT, N, N, N)]

    #Deep Learning
    for DTYPE in [4]:
        #Deep Bench
        for AT, BT in [(1,1), (2,1)]:
            for M in [1760, 2560]:
                for N in [16, 32, 64, 128]:
                    sizes += [(DTYPE, AT, BT, M, N, M)]
        #OpenNMT
        sizes += [(DTYPE, 1, 1, 2000, 128, 500),
                  (DTYPE, 1, 1, 2000, 640, 500),
                  (DTYPE, 1, 1, 2000, 2048, 500),
                  (DTYPE, 1, 1, 2000, 640, 1000),
                  (DTYPE, 1, 1, 500, 640, 1000),
                  (DTYPE, 1, 1, 500, 640, 500),
                  (DTYPE, 1, 1, 50000, 640, 500)]

    #Covariance
    for DTYPE in [4] + FP64:
        for AT, BT in [(1, 2)]:
            for N in [1, 2, 4, 8, 16, 64, 128, 256]:
                for K in [32000, 64000, 128000]:
                    sizes += [(DTYPE, AT, BT, N, N, K)]

    #LaPack
    for DTYPE in [4] + FP64:
        for AT, BT in [(1, 2)]:
            for N in [512, 1024, 2048, 4096]:
                for K in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                    sizes += [(DTYPE, AT, BT, N, N, K)]

    #Remove small problems
    sizes = [(DTYPE, AT, BT, M, N, K) for DTYPE, AT, BT, M, N, K in sizes if 2*M*N*K > 1e7]
    #Remove large problems
    sizes = [(DTYPE, AT, BT, M, N, K) for DTYPE, AT, BT, M, N, K in sizes if 2*M*N*K < 1e12]

    return sizes

def bench_shapes(OpType, device):
    if OpType==Conv:
        return conv_shapes(device)
    if OpType==GEMM:
        return gemm_shapes(device)
    if OpType==Pool:
        return pool_shapes(device)


def isaacConv(ctx, stream, shapes, layouts):
    # Shapes
    dtype, Npix, K, C, Nfilt = shapes
    N, M, P, Q = 1, 1, 1, Npix
    T, R, S = 1, 1, Nfilt
    dtype = sc.float64 if dtype==8 else sc.float32
    pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w = 0, 0, 0, 1, 1, 1, 1, 1, 1
    D = M*stride_d + T - 1 - 2*pad_d - stride_d + 1
    H = P*stride_h + R - 1 - 2*pad_h - stride_h + 1
    W = Q*stride_w + S - 1 - 2*pad_w - stride_w + 1
    # Kernel
    generator = sc.templates.Conv(dtype, dtype, C, D, H, W, N, K, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, upsample_d, upsample_h, upsample_w, sc.templates.LINEAR, 1, sc.templates.NO_RESIDUAL, 0, 1, 1, 1, 1, 1, 1, *layouts)


    src = generator.dump(ctx.device, "conv_fprop")
    module = sc.driver.Module(ctx, src)
    kernel = sc.driver.Kernel(module, "conv_fprop")
    with lock:
        # Buffers
        O = sc.driver.Buffer(ctx, K*M*P*Q*N*sc.size_of(dtype))
        I = sc.driver.Buffer(ctx, C*D*H*W*N*sc.size_of(dtype))
        F = sc.driver.Buffer(ctx, C*T*R*S*K*sc.size_of(dtype))
        alpha, beta = sc.Scalar(1., dtype), sc.Scalar(0., dtype)
        # Result
        time = benchmark(lambda: (generator.enqueue(kernel, stream, I, F, O, None, 1., 1., 1., [1.], 1., None), stream.synchronize()), ctx.device, 1e-2)
    tflops = 2*M*P*Q*N*K*C*T*R*S/time*1e-12
    return tflops

def isaacPool(ctx, stream, shapes, layouts):
    # Shapes
    dtype, Npix, Nfilt = shapes
    N, K, M, P, Q = 1, 1, 1, 1, Npix
    T, R, S = 1, 1, Nfilt
    dtype = sc.float64 if dtype==8 else sc.float32
    pad_d, pad_h, pad_w, stride_d, stride_h, stride_w = 0, 0, 0, 1, 1, 1
    D = M*stride_d + T - 1 - 2*pad_d - stride_d + 1
    H = P*stride_h + R - 1 - 2*pad_h - stride_h + 1
    W = Q*stride_w + S - 1 - 2*pad_w - stride_w + 1
    # Kernel
    generator = sc.templates.Pool(dtype, dtype, sc.templates.MAX_POOL, K, D, H, W, N, M, P, Q, T, R, S, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, *layouts)
    src = generator.dump(ctx.device, "pool_fprop")
    module = sc.driver.Module(ctx, src)
    kernel = sc.driver.Kernel(module, "pool_fprop")
    with lock:
        # BuffeNfilt
        O = sc.driver.Buffer(ctx, K*M*P*Q*N*sc.size_of(dtype))
        I = sc.driver.Buffer(ctx, K*D*H*W*N*sc.size_of(dtype))
        # Result
        time = benchmark(lambda: (generator.enqueue(kernel, stream, I, O, 1., 1.), stream.synchronize()), ctx.device, 1e-2)
    tflops = M*P*Q*N*K*T*R*S/time*1e-12
    return tflops

def isaacGemm(ctx, stream, shapes, layouts):
    # Shapes
    offa, offb, offc = 0, 0, 0
    dtype, AT, BT, M, N, K = shapes
    dtype = sc.float64 if dtype==8 else sc.float32
    AT, BT = sc.templates.op(AT), sc.templates.op(BT)
    ldc = M
    lda = M if AT==sc.templates.OP_N else K
    ldb = K if BT==sc.templates.OP_N else N
    # Kernel
    generator = sc.templates.GEMM(dtype, dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc, *layouts)
    src = generator.dump(ctx.device, "gemm")
    module = sc.driver.Module(ctx, src)
    kernel = sc.driver.Kernel(module, "gemm")
    with lock:
        # BuffeNfilt
        C = sc.driver.Buffer(ctx, M*N*sc.size_of(dtype))
        A = sc.driver.Buffer(ctx, M*K*sc.size_of(dtype))
        B = sc.driver.Buffer(ctx, K*N*sc.size_of(dtype))
        alpha, beta = sc.Scalar(1., dtype), sc.Scalar(0., dtype)
        # Result
        ts = benchmark(lambda: (generator.enqueue(kernel, stream, alpha, A, B, beta, C, 1., 1., 1., None), stream.synchronize()), ctx.device, 1e-2)
    tflops = 2*M*N*K/ts*1e-12
    return tflops

def evaluate(OpType, ctx, stream, params):
    shapes, layouts = params[:OpType.Nshapes], params[OpType.Nshapes:]
    if OpType==Conv:
        return isaacConv(ctx, stream, shapes, layouts)
    if OpType==GEMM:
        return isaacGemm(ctx, stream, shapes, layouts)
    if OpType==Pool:
        return isaacPool(ctx, stream, shapes, layouts)

def cudaConv(ctx, stream, dtype, N, K, P, Q, C, R, S):
    pad_h, pad_w, stride_h, stride_w = 0, 0, 1, 1
    H = P*stride_h + R - 1 - 2*pad_h
    W = Q*stride_w + S - 1 - 2*pad_w
    dtype = sc.dtype(dtype)
    O = sc.driver.Buffer(ctx, K*P*Q*N*sc.size_of(dtype))
    I = sc.driver.Buffer(ctx, C*H*W*N*sc.size_of(dtype))
    F = sc.driver.Buffer(ctx, C*R*S*K*sc.size_of(dtype))
    alpha, beta = sc.Scalar(1., dtype), sc.Scalar(0., dtype)
    time = benchmark(lambda: (sc.driver.cudnnConv(dtype, ctx, stream, H, W, N, K, P, Q, C, R, S, pad_h, pad_w, stride_h, stride_w, alpha, I, F, beta, O), stream.synchronize()), ctx.device, 1e-2)
    tflops = 2*P*Q*K*N*C*R*S/time*1e-12
    return tflops

def cudaGemm(ctx, stream, dtype, AT, BT, M, N, K):
    ldc = M
    lda = M if AT==1 else K
    ldb = K if BT==1 else N
    dtype = sc.dtype(dtype)
    C = sc.driver.Buffer(ctx, M*N*sc.size_of(dtype))
    A = sc.driver.Buffer(ctx, M*K*sc.size_of(dtype))
    B = sc.driver.Buffer(ctx, K*N*sc.size_of(dtype))
    alpha, beta = sc.Scalar(1., dtype), sc.Scalar(0., dtype)
    time = benchmark(lambda: (sc.driver.cublasGemm(dtype, ctx, stream, 'N' if AT==1 else 'T', 'N' if BT==1 else 'T', M, N, K, alpha, A, lda, B,  ldb, beta, C, ldc), stream.synchronize()), ctx.device, 1e-2)
    tflops = 2*M*N*K/time*1e-12
    return tflops
