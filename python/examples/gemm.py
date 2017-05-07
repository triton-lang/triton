import isaac as sc
import numpy as np
from itertools import product as prod
from time import time
    
ctx = sc.driver.default_context()
stream = sc.driver.default_stream()
dtype = sc.float32

M, N, K = 16, 4096, 256
AT, BT = sc.templates.OP_N, sc.templates.OP_N
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
                             2, 4, 8, 8, 4, 1, 4, 4, 8, 4, 8, 1, 1, 2)
src = generator.dump(ctx.device, "gemm")
#start = time()
module = sc.driver.Module(ctx, src, True)
kernel = sc.driver.Kernel(module, "gemm")
#end = time()
#print('Compile:', end - start, s.params)
generator.enqueue(kernel, stream, alpha, A, B, beta, C)
