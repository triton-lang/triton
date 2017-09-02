import isaac as sc
import numpy as np
from itertools import product as prod
from time import time
    
ctx = sc.driver.default_context()

#Dimensions
C, H, W, N = (832, 5, 5, 16)
R, S, K = (1, 1, 128)
pad_h, pad_w = (0, 0)
stride_h, stride_w = (1, 1)
P, Q = ((H - R + 1 + 2*pad_h)/stride_h,  (W - S + 1 + 2*pad_w)/stride_w)

#Device buffers
O = sc.driver.Buffer(ctx, K*P*Q*N*4)
I = sc.driver.Buffer(ctx, C*H*W*N*4)
F = sc.driver.Buffer(ctx, C*R*S*K*4)

#Queue
queue = sc.driver.CommandQueue(ctx)

#Tune
rv, rl, rs = [4], [1,2,4,8,16,32], [1,2,4,8]
for vec, bp, bq, bn, bk, bf_n, ps, qs, ns, ks, crs_l, crs_s, cs, bc, gridc \
  in prod(rv, [1], [1], rl, rl, rl, [1], [1], rs, rs, rl, [2], rs, rs, rs):
    
    #Compile
    conv = sc.templates.Conv(C, H, W, N, K, P, Q, R, S, pad_h, pad_w, stride_h, stride_w,\
                             vec, bp, bq, bn, bk, bf_n, ps, qs, ns, ks, crs_l, crs_s, cs, bc, gridc)
    try:
        src = conv.dump(ctx.device)
    except:
        continue
    program = sc.driver.Program(ctx, src, True)
    kernel = sc.driver.Kernel(program, "conv_fprop")
    
    #Launch
    try:
        perf = benchmark(lambda: (conv.enqueue(kernel, queue, 1., I, F, 0., O), queue.synchronize()), 1e-2)
    except:
        continue
    print '{0:2f}'.format(tflops(P, Q, K, N, C, R, S, perf))
    
        
