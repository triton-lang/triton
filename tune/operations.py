import isaac as sc
import numpy as np
import itertools
from time import time

def cartesian_coord(arrays):
    grid = np.meshgrid(*arrays)        
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points
    
def cartesian_iterator(arrays):
    N = len(arrays)
    split = [np.array_split(ary, min(len(ary), 2) if i < 4 else 1) for i, ary in enumerate(arrays)]
    for x in itertools.product(*split):
        yield cartesian_coord(x)
    
def benchmark(fn, device, nsec):
    total, hist = 0, []
    fn()
    while total < nsec:
        norm = device.current_sm_clock/device.max_sm_clock #* device.current_mem_clock/device.max_mem_clock
        start = time()
        fn()
        end = time()
        hist.append(norm*(end - start))
        total += hist[-1]
    return min(hist)
        
class ConvWrapper:
    
    name = 'conv'
    nparams = 23
    nshape_params = 8
    
    
    @staticmethod
    def bench_shapes(device):
        result = []
        #DeepBench
        result +=  [[4, 4, 32, 79, 341, 1, 5, 20],
                    [4, 8, 32, 79, 341, 1, 5, 20],
                    [4, 16, 32, 79, 341, 1, 5, 20],
                    [4, 32, 32, 79, 341, 1, 5, 20],
                    
                    [4, 4, 32, 38, 166, 32, 5, 10],
                    [4, 8, 32, 38, 166, 32, 5, 10],
                    [4, 16, 32, 38, 166, 32, 5, 10],
                    [4, 32, 32, 38, 166, 32, 5, 10],
                    
                    [4, 16, 16, 48, 480, 1, 3, 3],
                    [4, 16, 32, 24, 240, 16, 3, 3],
                    [4, 16, 64, 12, 120, 32, 3, 3],
                    [4, 16, 128, 6, 60, 64, 3, 3],

                    [4, 8, 64, 54, 54, 3, 3, 3],
                    [4, 8, 64, 54, 54, 64, 3, 3],
                    [4, 8, 128, 27, 27, 128, 3, 3],
                    [4, 8, 256, 14, 14, 128, 3, 3],
                    [4, 8, 512, 7, 7, 256, 3, 3],

                    [4, 8, 64, 224, 224, 3, 3, 3],
                    [4, 8, 128, 112, 112, 64, 3, 3],
                    [4, 8, 256, 56, 56, 128, 3, 3],
                    [4, 8, 512, 28, 28, 256, 3, 3],
                    [4, 8, 512, 14, 14, 512, 3, 3],
                    [4, 8, 512, 7, 7, 512, 3, 3],

                    [4, 16, 64, 224, 224, 3, 3, 3],
                    [4, 16, 128, 112, 112, 64, 3, 3],
                    [4, 16, 256, 56, 56, 128, 3, 3],
                    [4, 16, 512, 28, 28, 256, 3, 3],
                    [4, 16, 512, 14, 14, 512, 3, 3],
                    [4, 16, 512, 7, 7, 512, 3, 3],
                    
                    [4, 16, 64, 112, 112, 3, 7, 7],
                    [4, 16, 32, 28, 28, 192, 5, 5],
                    [4, 16, 64, 28, 28, 192, 1, 1],
                    [4, 16, 48, 14, 14, 512, 5, 5],
                    [4, 16, 192, 14, 14, 512, 1, 1],
                    [4, 16, 256, 7, 7, 832, 1, 1],
                    [4, 16, 128, 7, 7, 832, 5, 5],
                    
                    [4, 32, 64, 224, 224, 3, 3, 3],
                    [4, 32, 128, 112, 112, 64, 3, 3],
                    [4, 32, 256, 56, 56, 128, 3, 3],
                    [4, 32, 512, 28, 28, 256, 3, 3],
                    [4, 32, 512, 14, 14, 512, 3, 3],
                    [4, 32, 512, 7, 7, 512, 3, 3],
                    
                    [4, 32, 64, 112, 112, 3, 7, 7],
                    [4, 32, 32, 28, 28, 192, 5, 5],
                    [4, 32, 64, 28, 28, 192, 1, 1],
                    [4, 32, 48, 14, 14, 512, 5, 5],
                    [4, 32, 192, 14, 14, 512, 1, 1],
                    [4, 32, 256, 7, 7, 832, 1, 1],
                    [4, 32, 128, 7, 7, 832, 5, 5]]
        return result

    @staticmethod
    def train_shapes(device, nsamples):
        DTs = [4, 8]
        if device.compute_capability[0] >= 6:
            DTs += [2]
        Ns = [4, 8, 16, 32, 64]
        Ks = [16, 32, 64, 128, 256, 512]
        Ps = [1, 16, 32, 64, 128, 256]
        Qs = [1, 16, 32, 64, 128, 256]
        Cs = [3, 16, 32, 64, 128, 256, 512]
        Rs = [3, 5, 7]
        X = np.array([np.random.choice(x, size=nsamples) for x in [DTs, Ns, Ks, Ps, Qs, Cs, Rs, Rs]]).T
        N, K, P, Q, C, R, S = X[:,1], X[:,2], X[:,3], X[:,4], X[:,5], X[:,6], X[:,7]
        idx = np.logical_and.reduce((K*P*Q*N <= 1e7, C*P*Q*N <= 1e7, C*R*S*K <= 1e7, P*Q>1))
        return X[idx, :]


    @staticmethod
    def exhaust_param_ranges():
        rv, rl, rs = [2, 4], [1,2,4,8], [1,2,4,8]
        return [rv, [1,2,4,8], [1,2,4,8], rl, rl, rl, [1,2,4,8], [1,2,4,8], rs, rs, rl, [1], [1,2,4], [1,2,4], rs]
    
    @staticmethod
    def check_valid(device, X):
        return sc.templates.Conv.check_valid(device, X)
        
    @staticmethod
    def generate_valid(device, probabilities):
        X = np.array([np.random.choice(x, size=10000, p=prob) for x, prob in zip(ConvWrapper.param_ranges(), probabilities)], dtype=np.uint32).T.copy()
        idx = sc.templates.Conv.check_valid(device, X)
        return X[idx, :]


    @staticmethod
    def get_valid(V, shape):
        return V[shape[0]]
        
    @staticmethod
    def all_valid(device):
        nparams, param_ranges = ConvWrapper.nparams, ConvWrapper.exhaust_param_ranges()
        X = np.empty((0, nparams - 8))
        N, K, P, Q, C, R, S = 128, 128, 128, 128, 128, 5, 5
        for dtype in [2, 4, 8]:
            for T in cartesian_iterator(param_ranges):
                Y = np.zeros((T.shape[0], nparams), dtype=np.uint32)
                Y[:, :ConvWrapper.nshape_params] = [dtype, N, K, P, Q, C, R, S]
                Y[:, ConvWrapper.nshape_params:] = T
                X  = np.vstack((X, Y[ConvWrapper.check_valid(device, Y), 8:]))
        return X
        
    @staticmethod
    def param_ranges():
        LDT = [2, 4, 8]
        LRS = [3, 5, 7]
        L0 = [16, 32, 64, 256, 512]
        L1 = [1, 2, 4, 8, 16, 32, 64, 128]
        L2 = [1, 2, 4, 8, 16, 32, 64, 128]
        L3 = [1, 2, 4]
        L4 = [1, 2, 4, 8, 16]
        L5 = [1, 2, 4, 8]
        return [LDT,L1,L1,L2,L2,L0,LRS,LRS] + [L3,L5,L5,L4,L4,L4,L5,L5,L5,L5,L4,[1],L5,L5,L5]

    def __init__(s,  params):
        for x, name in zip(params, ['dtype', 'N','K','P','Q','C','R','S','vec','bp','bq','bn','bk','bf_n','ps','qs','ns','ks','crs_l','crs_s','cs','bc','gridc']):
            setattr(s, name, int(x))
        s.dtype = sc.dtype(s.dtype)
        s.pad_h, s.pad_w, s.stride_h, s.stride_w = 0, 0, 1, 1
        s.H = s.P*s.stride_h + s.R - 1 - 2*s.pad_h
        s.W = s.Q*s.stride_w + s.S - 1 - 2*s.pad_w
        
    def skip(s):
        return s.K*s.P*s.Q*s.N > 1e7 or s.C*s.H*s.W*s.N > 1e7 or s.C*s.R*s.S*s.K > 1e7
    
    def benchmark(s, ctx, stream):
        O = sc.driver.Buffer(ctx, s.K*s.P*s.Q*s.N*sc.size_of(s.dtype))
        I = sc.driver.Buffer(ctx, s.C*s.H*s.W*s.N*sc.size_of(s.dtype))
        F = sc.driver.Buffer(ctx, s.C*s.R*s.S*s.K*sc.size_of(s.dtype))
        alpha, beta = sc.Scalar(1., s.dtype), sc.Scalar(0., s.dtype)
        generator = sc.templates.Conv(s.dtype, s.C, s.H, s.W, s.N, s.K, s.P, s.Q,s.R, s.S,
                                      s.pad_h, s.pad_w, s.stride_h, s.stride_w,
                                      s.vec, s.bp, s.bq, s.bn, s.bk, s.bf_n, s.ps, s.qs, s.ns, s.ks, s.crs_l, s.crs_s, s.cs, s.bc, s.gridc)
        src = generator.dump(ctx.device, "conv_fprop")
        module = sc.driver.Module(ctx, src, True)
        kernel = sc.driver.Kernel(module, "conv_fprop")
        time = benchmark(lambda: (generator.enqueue(kernel, stream, alpha, I, F, beta, O), stream.synchronize()), ctx.device, 1e-2)
        tflops = 2*s.P*s.Q*s.K*s.N*s.C*s.R*s.S/time*1e-12
        return tflops

class GEMMWrapper:
    
    name = 'gemm'
    nparams = 20
    nshape_params = 6
    ntune_params = 14

    #M, N, K
    @staticmethod
    def bench_shapes(device):
        sizes = []
        DTs = [4]
        if device.compute_capability == (6,0):
            DTS += [2, 8]
            
        for DTYPE in DTs:
            for AT, BT in [(1, 2)]:
                for N in [512, 1024, 2048]:
                    sizes += [(DTYPE, AT, BT, N, N, N)]
        #DeepBench
        for DTYPE in DTs:
            for AT, BT in [(1,1), (2,1)]:
                for M in [512, 1760, 2560]:
                    for N in [1, 2, 4, 8, 16, 32, 64, 128]:
                        sizes += [(DTYPE, AT, BT, M, N, M)]
                
        #Covariance
        for DTYPE in DTs:
            for AT, BT in [(1, 2)]:
                for N in [1, 2, 4, 8, 16, 64, 256]:
                    for K in [32000, 64000, 128000]:
                        sizes += [(DTYPE, AT, BT, N, N, K)]
                    
        #LaPack
        for DTYPE in DTs:
            for AT, BT in [(1, 2)]:
                for N in [512, 1024, 2048, 4096]:
                    for K in [1, 2, 4, 8, 16, 32, 64]:
                        sizes += [(DTYPE, AT, BT, N, N, K)]
                    
        return sizes
                    
    @staticmethod
    def exhaust_param_ranges():
        L3 = [2, 4]
        L4 = [1, 2, 4, 8, 16]
        L5 = [1, 2, 4, 8]
        return [L3, L4, L4, L4, L5, [1], L5, L4, L4, L4, L4, [1], [1], L5]
  
    @staticmethod
    def train_shapes(device, nsamples):
        DTs = [4, 8]
        if device.compute_capability[0] >= 6:
            DTS += [2]
        LT = [1, 2]
        LK = [16, 32, 64, 256, 4096, 8192, 16382, 65536]
        LMN = [4, 8, 16, 32, 128, 512, 1024, 2048, 4096]
        X = np.array([np.random.choice(x, size=nsamples) for x in [DTs, LT, LT, LMN, LMN, LK]]).T
        M, N, K = X[:,3], X[:,4], X[:,5]
        idx = np.logical_and.reduce((2*M*N*K<=1e10, 2*M*N*K >= 1e6))
        return X[idx, :]

      
    @staticmethod
    def param_ranges():
        LDT = [2, 4, 8]
        LT = [1, 2]
        LK = [16, 32, 64, 256, 4096, 8192, 16382]
        LMN = [4, 8, 16, 32, 128, 512, 1024, 2048, 4096]
        L2 = [8, 16, 32, 64]
        L3 = [1, 2, 4]
        L4 = [1, 2, 4, 8, 16]
        L5 = [1, 2, 4, 8]
        return [LDT, LT, LT, LMN, LMN, LK, L3, L4, L4, L4, L5, [1], L5, L4, L4, L4, L4, L5, L4, L4]

    @staticmethod
    def get_valid(V, shape):
        return V[(shape[0], shape[1], shape[2])]
        
    @staticmethod
    def all_valid(device):
        nparams, param_ranges = GEMMWrapper.nparams, GEMMWrapper.exhaust_param_ranges()
        X = np.empty((0, nparams - 6))
        M, N, K = 1024, 1024, 1024
        for dtype in [2, 4, 8]:
            for AT in [1, 2]:
                for BT in [1, 2]:
                    for T in cartesian_iterator(param_ranges):
                        Y = np.zeros((T.shape[0], nparams), dtype=np.uint32)
                        Y[:, :GEMMWrapper.nshape_params] = [dtype, AT, BT, M, N, K]
                        Y[:, GEMMWrapper.nshape_params:] = T
                        X  = np.vstack((X, Y[GEMMWrapper.check_valid(device, Y), 6:]))
        return X
    
    @staticmethod
    def check_valid(device, X):
        return sc.templates.GEMM.check_valid(device, X)
        
    @staticmethod
    def generate_valid(device, probabilities):
        X = np.array([np.random.choice(x, size=10000, p=prob) for x, prob in zip(GEMMWrapper.param_ranges(), probabilities)]).astype(np.uint32).T.copy()
        idx = sc.templates.GEMM.check_valid(device, X)
        return X[idx, :]
        
    def __init__(s, *params):
        for x, name in zip(params[0], ['dtype', 'AT','BT','M','N','K','vec','bm','kl','bn','ms','ks','ns','a_bf0','a_bf1','b_bf0','b_bf1','rs','br','gridr']):
            setattr(s, name, x)
        s.dtype = sc.dtype(s.dtype)
        s.AT = sc.templates.op(s.AT)
        s.BT = sc.templates.op(s.BT)
        s.params = params
    
    def skip(s):
        return s.M*s.N > 1e7 or s.M*s.K > 1e7 or s.K*s.N > 1e7
    
    def benchmark(s, ctx, stream):
        
        C = sc.driver.Buffer(ctx, s.M*s.N*sc.size_of(s.dtype))
        A = sc.driver.Buffer(ctx, s.M*s.K*sc.size_of(s.dtype))
        B = sc.driver.Buffer(ctx, s.K*s.N*sc.size_of(s.dtype))
        
        alpha, beta = sc.Scalar(1., s.dtype), sc.Scalar(0., s.dtype)
        ldc = s.M
        lda = s.M if s.AT==sc.templates.OP_N else s.K
        ldb = s.K if s.BT==sc.templates.OP_N else s.N        
  
        generator = sc.templates.GEMM(s.dtype, s.AT, s.BT, s.M, s.N, s.K, 0, lda, 0, ldb, 0, ldc,
                                      s.vec, s.bm, s.kl, s.bn, s.ms, s.ks, s.ns, s.a_bf0, s.a_bf1, s.b_bf0, s.b_bf1, s.rs, s.br, s.gridr)
        src = generator.dump(ctx.device, "gemm")
        #start = time()
        module = sc.driver.Module(ctx, src, True)
        kernel = sc.driver.Kernel(module, "gemm")
        #end = time()
        #print('Compile:', end - start, s.params)
        ts = benchmark(lambda: (generator.enqueue(kernel, stream, alpha, A, B, beta, C), stream.synchronize()), ctx.device, 1e-2)
        tflops = 2*s.M*s.N*s.K/ts*1e-12
        return tflops

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
