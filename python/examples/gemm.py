import isaac as sc
import numpy as np
from itertools import product as prod
from time import time
    
ctx = sc.driver.default_context()
stream = sc.driver.default_stream()
dtype = sc.float32
alpha = sc.Scalar(1, dtype)
beta = sc.Scalar(0, dtype)

M, N, K = 1024, 1024, 1024
AT, BT = sc.templates.OP_N, sc.templates.OP_T
#Device buffers
C = sc.driver.Buffer(ctx, M*N*sc.size_of(dtype))
A = sc.driver.Buffer(ctx, M*K*sc.size_of(dtype))
B = sc.driver.Buffer(ctx, K*N*sc.size_of(dtype))
#Execute

ldc = M
lda = M if AT==sc.templates.OP_N else K
ldb = K if BT==sc.templates.OP_N else N        

generator = sc.templates.GEMM(dtype, AT, BT, M, N, K, 0, lda, 0, ldb, 0, ldc,
                              1, 2, 8, 1, 8, 1, 2, 1, 8, 1, 8, 2, 4, 1)
src = generator.dump(ctx.device, "gemm")
#start = time()
module = sc.driver.Module(ctx, src, True)
kernel = sc.driver.Kernel(module, "gemm")
#end = time()
#print('Compile:', end - start, s.params)
generator.enqueue(kernel, stream, alpha, A, B, beta, C)


N, K, P, Q, C, R, S = 8, 8, 64, 8, 8, 7, 7
H, W = P + R - 1, Q + S - 1
O = sc.driver.Buffer(ctx, N*K*P*Q*sc.size_of(dtype))
I = sc.driver.Buffer(ctx, N*C*H*W*sc.size_of(dtype))
F = sc.driver.Buffer(ctx, K*C*R*S*sc.size_of(dtype))
generator = sc.templates.Conv(dtype, C, H, W, N, K, P, Q, R, S, 0, 0, 1, 1,
                              1, 2, 1, 2, 4, 1, 4, 2, 8, 4, 1, 1, 1, 2, 1)
src = generator.dump(ctx.device, "conv")
module = sc.driver.Module(ctx, src, True)
kernel = sc.driver.Kernel(module, "conv")
generator.enqueue(kernel, stream, alpha, I, O, beta, F)


#[ 4  8  8 64  8 16  3  3  1  2  1  8  2  2  4  1  1  4  4  1  1  8  1]
#Benchmarks       : [#####                    ]  20%terminate called after throwing an instance of 'isaac::driver::exception::cuda::illegal_address'
#[  2 128  32  64  32  32   7   3   1   2   1   2   4   1   4   2   8   4
#   1   1   1   2   1]
