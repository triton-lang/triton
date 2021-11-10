import argparse
import itertools
import torch
import triton


def profile_matmul(args):
  Ms = list(map(int, args.M.split(',')))
  Ns = list(map(int, args.N.split(',')))
  Ks = list(map(int, args.K.split(',')))
  device = args.device

  at = True if args.trans_a else False
  bt = True if args.trans_b else False
  dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
  for M, N, K in itertools.product(Ms, Ns, Ks):
    a_shape = (M, K) if not at else (K, M)
    b_shape = (K, N) if not bt else (N, K)
    a = torch.randn(a_shape, device=device, dtype=dtype)
    b = torch.randn(b_shape, device=device, dtype=dtype)
    a = a.t() if at else a
    b = b.t() if bt else b
    if args.config: # override config if necessary
      BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages = list(map(int, args.config.split(',')))
      kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K, 'SPLIT_K': SPLIT_K}
      pre_hook = None if SPLIT_K == 1 else lambda nargs: nargs['C'].zero_()
      configs = [triton.Config(kwargs=kwargs, num_warps=num_warps, num_stages=num_stages, pre_hook=pre_hook)]
      kernel = triton.ops._matmul.kernel
      decorators = kernel.kernel_decorators
      kernel.kernel_decorators = []
      triton.autotune(configs, [])(kernel)
      kernel.kernel_decorators += decorators[1:]
    try:
      ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton.ops.matmul(a, b))
    except triton.code_gen.OutOfResources as e:
      pass
    tflops = 2*M*N*K*1e-9/ms

    if not at and not bt:
      geo = 'tt'
    elif at and not bt:
      geo = 'nt'
    elif not at and bt:
      geo = 'tn'
    elif at and bt:
      geo = 'nn'
    print(f'matmul-{geo}: {M}x{N}x{K}: {ms:.4f} ms, {tflops:.2f} TFLOPS')

    

def profile_blocksparse(args):
  raise NotImplementedError("profile_blocksparse not implemented")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Triton profiler')
  parser.add_argument('--operation', required=True, choices=['matmul', 'blocksparse'])
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--varify-result', action='store_true')
  parser.add_argument('--random-seed', type=int, default=0)
  # matmul args
  parser.add_argument('--M')
  parser.add_argument('--N')
  parser.add_argument('--K')
  parser.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32'])
  parser.add_argument('--trans_a', action='store_true')
  parser.add_argument('--trans_b', action='store_true')
  # override configs
  parser.add_argument('--config', help='Config, format: BLOCK_M,BLOCK_N,BLOCK_K,SPLIT_K,num_warps,num_stages')

  args = parser.parse_args()

  torch.manual_seed(args.random_seed)
  if args.operation == 'matmul':
    profile_matmul(args)
  elif args.operation == 'blocksparse':
    profile_blocksparse(args)