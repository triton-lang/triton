import isaac as sc
import numpy as np
from itertools import product as prod
from time import time
    
ctx = sc.driver.default_context()
stream = sc.driver.default_stream()
dtype = sc.float32

M, N, K = 2048, 4, 8192
AT, BT = sc.templates.OP_T, sc.templates.OP_N
#Device buffers
C = sc.driver.Buffer(ctx, M*N*sc.size_of(dtype))
A = sc.driver.Buffer(ctx, M*K*sc.size_of(dtype))
B = sc.driver.Buffer(ctx, K*N*sc.size_of(dtype))
alpha = sc.Scalar(1, dtype)
beta = sc.Scalar(0, dtype)
#Execute

ldc = M
lda = M if AT==sc.templates.OP_N else K
ldb = K if BT==sc.templates.OP_N else N        

generator = sc.templates.GEMM(dtype, AT, BT, M, N, K, 0, lda, 0, ldb, 0, ldc,
                              2, 1, 1, 16, 4, 1, 8, 8, 4, 2, 16, 8, 2, 1)
src = generator.dump(ctx.device, "gemm")
#start = time()
module = sc.driver.Module(ctx, src, True)
kernel = sc.driver.Kernel(module, "gemm")
#end = time()
#print('Compile:', end - start, s.params)
generator.enqueue(kernel, stream, alpha, A, B, beta, C)
