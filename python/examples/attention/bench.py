import torch
import numpy as np
import reference
import optimized
from time import time

use_half = False
def cast(x):
    if use_half:
        return x.half()
    else:
        return x

# GPU device
device = torch.device("cuda:0")
# shapes
batch, nhead = 16, 8
dm, dk, dv = 512, 512, 512
lq, lk, lv = 256, 256, 256
# initialize tensors
torch.manual_seed(0)
np.random.seed(0)
query = cast(torch.randn(batch, lq, dm)).cuda()
key   = cast(torch.randn(batch, lk, dm)).cuda()
value = cast(torch.randn(batch, lv, dm)).cuda()
# initialize layers
torch.manual_seed(0)
np.random.seed(0)
rattn = cast(reference.MultiHeadAttention(nhead, dm, dk, dv).to(device))
torch.manual_seed(0)
np.random.seed(0)
tattn = cast(optimized.MultiHeadAttention(nhead, dm, dk, dv).to(device))
# test
routput, _ = rattn(query, key, value)
toutput, _ = tattn(query, key, value)
diff = torch.max(torch.abs(routput - toutput))
assert diff < 1e-2
# benchmark
start = time()
routput, _ = rattn(query, key, value)
end = time()
rtime = end - start
start = time()
toutput, _ = tattn(query, key, value)
end = time()
ttime = end - start
print(rtime, ttime)