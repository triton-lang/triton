import argparse
import itertools
import re
import importlib.util
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

def _parse_value(dtype, shape, value):
  if not shape: # scalar value
    if value.isdigit():
      return int(value)
    try:
      res = float(value)
      return res
    except ValueError as e:
      raise ValueError(f"Don't know what to do with {value}")
  else: # Tensor
    return torch.ones(shape, dtype=dtype, device='cuda')


def _parse_params(args_str):
  _str_to_dtype = {'float16': torch.float16, 'float32': torch.float32}
  res = []
  for arg_str in args_str:
    # name:dtype:shape:value(optional)
    re_result = re.match('(?P<name>\w+)(:(?P<dtype>float16|float32):(?P<shape>\d+(,\d+)*))?(:(?P<value>[^:]*))?', arg_str)
    if not re_result:
      raise ValueError(f'Cannot parse {arg_str}')
    name = re_result.group('name')
    dtype = re_result.group('dtype')
    shape = re_result.group('shape')
    value = re_result.group('value')
    if dtype and shape:
      dtype = _str_to_dtype[dtype]
      shape = list(map(int, shape.split(',')))
    value = _parse_value(dtype, shape, value)
    res.append(value)
  return res
    

def profile_custom_kernel(args):
  '''
  python -m triton.tools.profiler --kernel /path/to/kernel.py::kernel_name --arg [name]:dtype:value
  '''
  # print(args.args)
  try:
    path, kname = re.match('([^:]+)::(.+)', args.kernel).groups()
  except Exception as e:
    print('invalid kernel, abort')
    exit()
  kernel_args = _parse_params(args.args)
  # load the module
  spec = importlib.util.spec_from_file_location('', path)
  pymod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(pymod)
  jit_fn = getattr(pymod, kname)
  grid = list(map(int, args.grid.split(',')))
  ms, min_ms, max_ms = triton.testing.do_bench(lambda: jit_fn[grid](*kernel_args))

  print(f'{ms:.3f} ms')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Triton profiler')
  parser.add_argument('--kernel', required=True)
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

  # custom kernels
  parser.add_argument('--args', '-a', action='append')
  parser.add_argument('--grid', required=True)

  args = parser.parse_args()

  torch.manual_seed(args.random_seed)
  if args.kernel == 'matmul':
    profile_matmul(args)
  elif args.kernel == 'blocksparse':
    profile_blocksparse(args)
  else:
    profile_custom_kernel(args)