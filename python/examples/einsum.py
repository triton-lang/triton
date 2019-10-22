import numpy as np
import torch
import triton

batch_dim  = 16
ctx_dim    = 32
head_dim   = 8
state_dim  = 32
key_dim    = 32
n_keys     = 32
bs         = batch_dim * ctx_dim

# shapes
x_shape  = (bs, state_dim)
qw_shape = (state_dim, head_dim * key_dim)
kw_shape = (head_dim, 2, n_keys,  key_dim // 2)

np.random.seed(0)
x  = np.random.uniform(-1.0, 1.0,  x_shape).astype(np.float32) # layer input
qw = np.random.uniform(-1.0, 1.0, qw_shape).astype(np.float32) # query weights
kw = np.random.uniform(-1.0, 1.0, kw_shape).astype(np.float32) # key   weights
# (bs, head_dim * key_dim) = (bs, state_dim) * (state_dim, head_dim * key_dim)
# (bs, head_dim, 2, key_dim//2) <==  (bs, head_dim * key_dim)
q = np.dot(x, qw).reshape(bs, head_dim, 2, key_dim//2) # normal matmul

# (bs, head_dim, 2, n_keys) = (bs, head_dim, 2, key_dim//2) * (head_dim, 2, n_keys,  key_dim//2)
# outer: bs, n_keys
# inner: key_dim//2
# batch: head_dim, 2 (key_axis)
qk = np.einsum("bhak,hank->bhan", q, kw)

tq = torch.from_numpy(q).contiguous().cuda()
tkw = torch.from_numpy(kw).contiguous().cuda()
tqk = triton.ops.einsum("bhak,hank->bhan", tq, tkw)
diff = np.abs(qk - tqk.cpu().numpy())
print(np.max(diff))
print(np.min(diff))

