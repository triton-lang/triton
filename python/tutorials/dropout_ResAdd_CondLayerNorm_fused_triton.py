

import torch
from torch import autograd
import torch.nn.functional as F

import triton
import triton.language as tl



@triton.jit
def _drcln_fwd_fused_training(
    X_input_ptr,  # pointer to the original input (that will be added -- res add)
    Y_output_ptr,  # pointer to the output
    Z_input_to_dropout_ptr, # pointer to the input to dropout
    YIN_resadd_out_ptr, # pointer to output after dropout and res add
    MASK_out_aftr_dropout, # pointer to output mask after dropout
    p_drop, # dropout prob
    seed, # dropout seed
    W_ptr,  # pointer to the weights
    B_ptr,  # pointer to the biases
    Mean_ptr,  # pointer to the mean
    Rstd_ptr,  # pointer to the 1/std
    input_row_stride,  # how much to increase the pointer when moving by 1 row
    N_cols,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):

	"""
	X --> [some operations] --> Z --> Dropout --> Zout --> ResAdd(Zout + X) --> Yin --> LN/CLN --> Y

	"""
	row_idx = tl.program_id(0)

	Z_input_to_dropout_ptr += row_idx * input_row_stride
	X_input_ptr += row_idx * input_row_stride
	YIN_resadd_out_ptr += row_idx * input_row_stride
	Y_output_ptr += row_idx * input_row_stride
	MASK_out_aftr_dropout += row_idx * input_row_stride

	mean = 0
	_mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

	for off in range(0, N_cols, BLOCK_SIZE):
		all_elem = row_idx * input_row_stride + (off + tl.arange(0, BLOCK_SIZE))
		cols_ptr = off + tl.arange(0, BLOCK_SIZE)
		mask = cols_ptr < N_cols

		# Dropout
		random = tl.rand(seed, all_elem)
		x_keep = random > p_drop
		z_drop_in = tl.load(Z_input_to_dropout_ptr + cols_ptr, mask=mask, other=0.).to(tl.float32)
		z_dropout_output = tl.where(x_keep, z_drop_in / (1 - p_drop), 0.0)

		#ResidualAdd (X + Zout)
		x = tl.load(X_input_ptr + cols_ptr, mask=mask, other=0.).to(tl.float32)
		yin_ln = x + z_dropout_output
		# calculate mean for LayerNorm
		_mean += yin_ln

		tl.store(MASK_out_aftr_dropout + cols_ptr, x_keep, mask=mask)
		tl.store(YIN_resadd_out_ptr + cols_ptr, yin_ln, mask=mask)

	mean = tl.sum(_mean, axis=0) / N_cols

	# compute variance
	_var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
	for off in range(0, N_cols, BLOCK_SIZE):
		cols_ptr = off + tl.arange(0, BLOCK_SIZE)
		yin_ln = tl.load(YIN_resadd_out_ptr + cols_ptr, mask=cols_ptr < N_cols, other=0.).to(tl.float32)
		yin_ln = tl.where(cols_ptr < N_cols, yin_ln - mean, 0.)
		_var += yin_ln * yin_ln

	var = tl.sum(_var, axis=0) / N_cols
	rstd = 1 / tl.sqrt(var + eps)

	tl.store(Mean_ptr + row_idx, mean)
	tl.store(Rstd_ptr + row_idx, rstd)

	# layernorm : normalize and apply linear transformation
	for off in range(0, N_cols, BLOCK_SIZE):
		cols_ptr = off + tl.arange(0, BLOCK_SIZE)
		mask = cols_ptr < N_cols
		w = tl.load(W_ptr + cols_ptr, mask=mask)
		b = tl.load(B_ptr + cols_ptr, mask=mask)
		yin_ln = tl.load(YIN_resadd_out_ptr + cols_ptr, mask=cols_ptr < N_cols, other=0.).to(tl.float32)
		yinhat = (yin_ln - mean) * rstd
		y_out = yinhat * w + b
		tl.store(Y_output_ptr + cols_ptr, y_out, mask=mask)


"""
@triton.jit
def _drcln_fwd_fused_inference ():
	pass
"""

# simple flag based approach to distinguish between regular (conventional) and
# conditional Layernorm runs into problems with Triton compiler
# which checks memory allocation code - for conditional LN, it still checks
# memory is allocated for dW, dB etc.
# need to implement both (conventional, conditional) LM separately.

@triton.jit
def _drcln_bwd_dx_fused(
    DY_ptr, # pointer to output gradient
    DZ_ptr, # pointer to input gradient (input ahead of dropout)
    DX_ptr, # pointer input gradient (input for residual add),
    YIN_ptr, # input to CLN
    # Z, # pointer to input
    W_ptr, # CLN weights,
    Mean_ptr,
    Rstd_ptr,
    DROPOUT_MASK, # pointer to dropout mask
    p_drop, # dropout rate
    input_row_stride,  # how much to increase the pointer when moving by 1 row
    N_cols,  # number of columns in X
    # eps,  # epsilon to avoid division by zero
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):

    """

	 X --> [some operations] --> Z --> Dropout --> Zout --> ResAdd(Zout + X) --> Yin --> LN/CLN --> Y

     X dw = dy * yin
     X db = dy
     dyin =(wdy - (yin_hat * c1 + c2)) * rstd
     dx = dyin
     dz = dyin * mask / (1 - p)
	"""

    # Map the program id to the elements of X, DX, and DY it should compute.
    row_idx = tl.program_id(0)
    cols_ptr = tl.arange(0, BLOCK_SIZE_N)
    mask = cols_ptr < N_cols

    DY_ptr += row_idx * input_row_stride
    YIN_ptr += row_idx * input_row_stride
    W_ptr += row_idx * input_row_stride
    DX_ptr += row_idx * input_row_stride
    DROPOUT_MASK += row_idx * input_row_stride
    DZ_ptr += row_idx * input_row_stride


    # input to CLN
    yin = tl.load(YIN_ptr + cols_ptr, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY_ptr + cols_ptr, mask=mask, other=0).to(tl.float32)
    w = tl.load(W_ptr + cols_ptr, mask=mask).to(tl.float32)

    mean = tl.load(Mean_ptr + row_idx)
    rstd = tl.load(Rstd_ptr + row_idx)

    # Compute dx
    yin_hat = (yin - mean) * rstd
    wdy = w * dy
    yin_hat = tl.where(mask, yin_hat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(yin_hat * wdy, axis=0) / N_cols
    c2 = tl.sum(wdy, axis=0) / N_cols
    dyin = (wdy - (yin_hat * c1 + c2)) * rstd

    # dx is just dyin (res add)
    tl.store(DX_ptr + cols_ptr, dyin, mask=mask)


    # compute dz (dropout then add)
    dropout_mask = tl.load(DROPOUT_MASK + cols_ptr, mask=mask, other=0.).to(tl.float32)
    dz = dyin * dropout_mask / (1 - p_drop)
    tl.store(DZ_ptr + cols_ptr, dz, mask=mask)


@triton.jit
def _drcln_bwd_dwdb_fused(
    DY_ptr, # pointer to output gradient
    DZ_ptr, # pointer to input gradient (input ahead of dropout)
    DX_ptr, # pointer input gradient (input for residual add),
    YIN_ptr, # input to CLN
    # Z, # pointer to input
    W_ptr, # CLN weights,
    DW_ptr,# pointer to output gradient for CLN scale
    DB_ptr, # pointer to output gradient for CLN shift
    Mean_ptr,
    Rstd_ptr,
    DROPOUT_MASK, # pointer to dropout mask
    p_drop, # dropout rate
    Lock, # pointer to the lock
    input_row_stride,  # how much to increase the pointer when moving by 1 row
    N_cols,  # number of columns in X
    # eps,  # epsilon to avoid division by zero
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):

    """

	 X --> [some operations] --> Z --> Dropout --> Zout --> ResAdd(Zout + X) --> Yin --> LN/CLN --> Y

     X dw = dy * yin
     X db = dy
     dyin =(wdy - (yin_hat * c1 + c2)) * rstd
     dx = dyin
     dz = dyin * mask / (1 - p)
	"""

    # Map the program id to the elements of X, DX, and DY it should compute.
    row_idx = tl.program_id(0)
    cols_ptr = tl.arange(0, BLOCK_SIZE_N)
    mask = cols_ptr < N_cols

    DY_ptr += row_idx * input_row_stride
    YIN_ptr += row_idx * input_row_stride
    W_ptr += row_idx * input_row_stride
    DX_ptr += row_idx * input_row_stride
    DROPOUT_MASK += row_idx * input_row_stride
    DZ_ptr += row_idx * input_row_stride

    # conventional layernorm
    #if not conditional_layernorm_flag:
    lock_id = row_idx % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW_ptr = DW_ptr + lock_id * N_cols + cols_ptr
    DB_ptr = DB_ptr + lock_id * N_cols + cols_ptr

    # input to CLN
    yin = tl.load(YIN_ptr + cols_ptr, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY_ptr + cols_ptr, mask=mask, other=0).to(tl.float32)
    w = tl.load(W_ptr + cols_ptr, mask=mask).to(tl.float32)

    mean = tl.load(Mean_ptr + row_idx)
    rstd = tl.load(Rstd_ptr + row_idx)

    # Compute dx
    yin_hat = (yin - mean) * rstd
    wdy = w * dy
    yin_hat = tl.where(mask, yin_hat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(yin_hat * wdy, axis=0) / N_cols
    c2 = tl.sum(wdy, axis=0) / N_cols
    dyin = (wdy - (yin_hat * c1 + c2)) * rstd

    # dx is just dyin (res add)
    tl.store(DX_ptr + cols_ptr, dyin, mask=mask)


    # compute dz (dropout then add)
    dropout_mask = tl.load(DROPOUT_MASK + cols_ptr, mask=mask, other=0.).to(tl.float32)
    dz = dyin * dropout_mask / (1 - p_drop)
    tl.store(DZ_ptr + cols_ptr, dz, mask=mask)

    # needed only if normal conventional (not conditional) Layernorm
    	# Accumulate partial sums for dw/db
    partial_dw = (dy * yin_hat).to(w.dtype)
    partial_db = (dy).to(w.dtype)

    while tl.atomic_cas(Lock, 0, 1) == 1:
      pass
    count = tl.load(Count)
    if count == 0:
    	tl.atomic_xchg(Count, 1)
    else:
      partial_dw += tl.load(DW_ptr, mask=mask)
      partial_db += tl.load(DB_ptr, mask=mask)
    tl.store(DW_ptr, partial_dw, mask=mask)
    tl.store(DB_ptr, partial_db, mask=mask)
    	# Release the lock
    tl.atomic_xchg(Lock, 0)



@triton.jit
def _drcln_bwd_dwdb_fused(
    DW_ptr,  # pointer to the partial sum of weights gradient
    DB_ptr,  # pointer to the partial sum of biases gradient
    FINAL_DW_ptr,  # pointer to the weights gradient
    FINAL_DB_ptr,  # pointer to the biases gradient
    M,  # GROUP_SIZE_M
    N_cols,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N_cols)
        offs = rows[:, None] * N_cols + cols[None, :]
        dw += tl.load(DW_ptr + offs, mask=mask, other=0.)
        db += tl.load(DB_ptr + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW_ptr + cols, sum_dw, mask=cols < N_cols)
    tl.store(FINAL_DB_ptr + cols, sum_db, mask=cols < N_cols)


class DropoutResAddCLN(torch.autograd.Function):

    

    @staticmethod
    def forward(ctx, z, x, p, weight, bias, eps, seed=42):
        # z --> input ahead of dropout
        # x --> original input (residual add)

        # allocate output
        y = torch.empty_like(x)
        yin = torch.empty_like(z)
        mask_out = torch.empty_like(z)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        mean = torch.empty((M, ), dtype=torch.float32, device='cuda')
        rstd = torch.empty((M, ), dtype=torch.float32, device='cuda')
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        
        #TO DO - alter calling parameters based on conventional / conditional layernorm
        _drcln_fwd_fused_training[(M,)](x_arg, y, z, yin, mask_out, p, seed, weight, bias, mean, rstd,
                                    x_arg.stride(0), N, eps,
                                    BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        
        ctx.save_for_backward(mask_out, mean, rstd, weight, yin ) 
        
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.p = p
        return y 

    @staticmethod
    def backward(ctx, dy):
        dropout_mask, mean, rstd, weight, yin = ctx.saved_tensors

        #for TESTING  - change this into a parameter
        COND_LN_FLAG = True

        # heuristics for amount of parallel reduction stream for DW/DB
        M, N = dy.shape #[1]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256

        # needed for learning w, b (conventional layernorm)
        
        if not COND_LN_FLAG:
          locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device='cuda')
          _dw = torch.empty((GROUP_SIZE_M, weight.shape[0]), dtype=yin.dtype, device='cuda')
          _db = torch.empty((GROUP_SIZE_M, weight.shape[0]), dtype=yin.dtype, device='cuda')
        # x_arg = x.reshape(-1, x.shape[-1])
        # M, N = x_arg.shape
        dz = torch.empty_like(dy)
        dx = torch.empty_like(dy)
        dw = torch.empty_like(dy)
        db = torch.empty_like(dy)

        if COND_LN_FLAG:
          # _dw, _db, locks are none, not needed for conditional layernorm
          _drcln_bwd_dx_fused[(M,)](dy, dz, dx, yin, weight, mean, rstd,  dropout_mask, ctx.p,
                                    dy.stride(0), N, #ctx.eps,
                                    GROUP_SIZE_M=GROUP_SIZE_M,
                                    BLOCK_SIZE_N=ctx.BLOCK_SIZE,
                                    num_warps=ctx.num_warps)  
        else:               
        # for normal layernorm _dw, _db are needed.
          _drcln_bwd_dwdb_fused[(M,)](dy, dz, dx, yin, weight, _dw, _db,  mean, rstd,  dropout_mask, ctx.p,
                                    locks, dy.stride(0), N, #ctx.eps,
                                    GROUP_SIZE_M=GROUP_SIZE_M,
                                    BLOCK_SIZE_N=ctx.BLOCK_SIZE,
                                    num_warps=ctx.num_warps)
        
          grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
          _drcln_bwd_dwdb_fused[grid](_dw, _db, dw, db, 
                                    GROUP_SIZE_M, N, 
                                    BLOCK_SIZE_M=32,
                                    BLOCK_SIZE_N=128)

        return dz, dx, None, dw, db, None


### following code used as-is from NA's implementation. Needs to be verified. Results don't match

def vanilla_conditional_drcln(z, x, weight, bias, eps, p=0.5): #, weight, bias, eps=1e-5):
    # vanilla CLN --> different scale (weight) and shift (bias) params per element in batch
    M, N = x.size()
    p = 0.5
    z_out = F.dropout(z, p=p)
    yin =  x + z_out
    # return y
    return vanilla_conditional_layer_norm(yin, weight, bias, eps=1e-5)
    x_keep = (torch.rand(size=(10,)) > p).to(torch.int32).cuda()


    #assert weight.size() == x.size()
    #assert bias.size() == x.size()
    mean = torch.mean(x, -1)[..., None]
    var = torch.var(x, -1)[..., None]
    normalized_x = (x - mean) / torch.sqrt(var + eps)

    out = weight * normalized_x + bias
    return out

def vanilla_conditional_layer_norm(x, weight, bias, eps=1e-5):
    # vanilla CLN --> different scale (weight) and shift (bias) params per element in batch
    M, N = x.size()
    #assert weight.size() == x.size()
    #assert bias.size() == x.size()
    mean = torch.mean(x, -1)[..., None]
    # print('mean', mean)
    var = torch.var(x, -1)[..., None]
    normalized_x = (x - mean) / torch.sqrt(var + eps)

    out = weight * normalized_x + bias
    return out

# @pytest.fixture
def test_drcln(M, N, dtype, eps=1e-5, device='cuda'):

    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )

    # for reg layernorm 
    #weight = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    #bias = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    z = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    z.requires_grad_(True)

    p = 0.5
    # forward pass

    # for conditional layer norm, weights and biases are (M, N)
    weight = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=False)
    bias = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=False)
    
    z_out_tri = dracln(z, x, p, weight, bias, eps) #, w_shape) #, p=0.5, seed=123)
    print(z_out_tri[0], z_out_tri[-1])
    # print(mask_out_tri[0], mask_out_tri[1], mask_out_tri[-1])
    # y_tri = layer_norm(x, w_shape, weight, bias, eps)
    
    z_out_ref = vanilla_conditional_drcln(z, x, weight, bias, eps).to(dtype) #torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    print ("***")
    print(z_out_ref[0], z_out_ref[-1])



    # # # backward pass (triton)
    z_out_tri.backward(dy, retain_graph=True)
    # dw_tri, db_tri
    #dz_tri, dx_tri, dw_tri, db_tri  = [_.grad.clone() for _ in [z, x, weight, bias, ]] #, weight, bias]]
    dz_tri, dx_tri   = [_.grad.clone() for _ in [z, x]] #, weight, bias]]
    # print(dz_tri.size())
    print ("*** backward pass")
    print(dz_tri[0], dz_tri[1], dz_tri[-1])
    x.grad = None #, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    z_out_ref.backward(dy, retain_graph=True)
    #dz_ref, dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [z, x, weight, bias,]] #, weight, bias]]
    dz_ref, dx_ref = [_.grad.clone() for _ in [z, x ]] #, weight, bias]]
    # # # compare
    print ("***")
    print(dz_ref[0], dz_ref[1], dz_ref[-1])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='dracln-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'}
    )
)
def bench_drcln(M, N, dtype, provider, mode='backward', eps=1e-5, device='cuda'):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=False)
    bias = torch.rand(x_shape, dtype=dtype, device='cuda', requires_grad=False)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    z = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    z.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]
    p = 0.5
    # utility functions
    if provider == 'triton':
        #  z, x, p, weight, bias, eps,
        def y_fwd(): return dracln(z, x, p, weight, bias, eps) #dracln(x, w_shape, weight, bias, eps)  # noqa: F811, E704
    if provider == 'torch':
        def y_fwd(): return vanilla_conditional_drcln(z, x, weight, bias, eps, p=p)  # noqa: F811, E704
    """    
    if provider == 'apex':
        apex_drcln = apex.normalization.FusedDropoutResAddCLN(
            w_shape).to(x.device).to(x.dtype)
        def y_fwd(): return apex_drcln(x)  # noqa: F811, E704
    """
    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms = triton.testing.do_bench_cudagraph(y_fwd)
        # ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        def gbps(ms): return 3 * x.numel() * x.element_size() / ms * 1e-6  # noqa: F811, E704
        y = y_fwd()
        ms = triton.testing.do_bench_cudagraph(lambda: y.backward(dy, retain_graph=True))
        # ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True),
                                                    #  quantiles=quantiles, grad_to_none=[x], rep=500)
    return gbps(ms) #, gbps(max_ms), gbps(min_ms)



if __name__ == '__main__':
	HAS_APEX = False
	dracln = DropoutResAddCLN.apply
	test_drcln(1151, 10, torch.float16)
	
	bench_drcln.run(save_path='.', print_data=True) 



