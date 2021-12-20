import torch
import triton
import triton._C.libtriton.triton as _triton
import heapq

def get_dram_bw(backend=None, device=None):
  ''' return DRAM bandwidth in GB/s '''
  # assert backend == CUDA
  if not backend:
    backend = _triton.runtime.backend.CUDA
  if not device:
    device = torch.cuda.current_device()
  mem_clock_khz = _triton.runtime.memory_clock_rate(backend, device)
  bus_width = _triton.runtime.global_memory_bus_width(backend, device)
  bw = mem_clock_khz * bus_width * 2 // 1024 // 1024 // 8 # In GB/s
  return bw

def get_matmul_tput(backend, device, num_ctas, num_warps):
  ''' return compute throughput in TOPS '''
  num_subcores = _triton.runtime.num_sm(backend, device) * 4 # on recent GPUs
  total_warps = num_ctas * min(num_warps, 4)
  clock_rate = _triton.runtime.clock_rate(backend, device) # in kHz
  # # Fix clock rate for now
  # clock_rate = 1410000 # kHz
  # assume fp32 += fp16*fp16
  tput = min(num_subcores, total_warps) * clock_rate *2*4*4*4*2 # 2 4x4x4 Tensor Cores
  tput /= 1024*1024*1024
  return tput

def estimate_matmul_time(
  # backend, device, 
  num_warps, num_stages,
  M, N, K, 
  BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K,
  debug=False, **kwargs
):
  ''' return estimated running time in ms 
        = max(compute, loading) + store '''
  backend = _triton.runtime.backend.CUDA
  device = torch.cuda.current_device()

  num_cta_m = triton.cdiv(M, BLOCK_M)
  num_cta_n = triton.cdiv(N, BLOCK_N)
  num_cta_k = SPLIT_K
  num_ctas = num_cta_m * num_cta_n * num_cta_k

  # If the input is smaller than the block size
  M, N = max(M, BLOCK_M), max(N, BLOCK_N)

  # time to compute
  total_ops = 2*M*N*K / (1024*1024*1024) # GOPS
  tput = get_matmul_tput(backend, device, num_ctas, num_warps)
  compute_ms = total_ops / tput

  # time to load data
  num_sm  = _triton.runtime.num_sm(backend, device)
  active_cta_ratio = min(1, num_ctas/num_sm)
  active_cta_ratio_bw1 = min(1, num_ctas/32) # 32 active ctas are enough to saturate 
  active_cta_ratio_bw2 = max(min(1, (num_ctas-32)/(108-32)), 0) # 32-108, remaining 5%
  dram_bw = get_dram_bw(backend, device) * (active_cta_ratio_bw1*0.95 + active_cta_ratio_bw2*0.05)  # in GB/s
  l2_bw   = dram_bw * 4 # rough estimation (should be 4.7 for A100?)
  # assume 80% of (following) loads are in L2 cache
  load_a_dram = M*K*2*(1+0.2*(num_cta_n-1)) # assume dtype=float16 (size==2)
  load_a_l2   = M*K*2*0.8*(num_cta_n-1)
  load_b_dram = N*K*2*(1+0.2*(num_cta_m-1))
  load_b_l2   = N*K*2*0.8*(num_cta_m-1)
  # total
  total_dram = (load_a_dram + load_b_dram) / (1024*1024) # MB
  total_l2   = (load_a_l2   + load_b_l2)   / (1024*1024)     
  # loading time in ms
  load_ms = total_dram/dram_bw + total_l2/l2_bw

  # estimate storing time
  store_bw = dram_bw * 0.6 # :o
  store_c_dram = M*N*2*SPLIT_K / (1024*1024) # MB
  if SPLIT_K == 1:
    store_ms = store_c_dram /store_bw
  else:
    reduce_bw = store_bw
    store_ms = store_c_dram/reduce_bw
    # c.zero_()
    zero_ms = M*N*2/(1024*1024)/store_bw
    store_ms += zero_ms

  total_time_ms = max(compute_ms, load_ms) + store_ms
  if debug:
    print(f'Total time: {total_time_ms}ms, compute time: {compute_ms}ms, '
          f'loading time: {load_ms}ms, store time: {store_ms}ms, '
          f'Activate CTAs: {active_cta_ratio*100}%')
  return total_time_ms

def prune_num_stages(configs):
  # TODO: should cache this
  backend = _triton.runtime.backend.CUDA
  device = torch.cuda.current_device()
  cc = _triton.runtime.cc(backend, device)
  # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages

  # group configs by (BLOCK_M,_N,_K, SPLIT_K, num_warps)
  configs_map = {}
  for config in configs:
    kw = config.kwargs
    BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages = \
      kw['BLOCK_M'], kw['BLOCK_N'], kw['BLOCK_K'], kw['SPLIT_K'], config.num_warps, config.num_stages
    
    key = (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps)
    if key in configs_map:
      configs_map[key].append((config, num_stages))
    else:
      configs_map[key] = [(config, num_stages)]

  pruned_configs = []
  for k, v in configs_map.items():
    BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps = k
    if cc >= 80:
      # compute cycles (only works for ampere GPUs)
      mmas = BLOCK_M * BLOCK_N * BLOCK_K / (16*8*16)
      mma_cycles = mmas/min(4, num_warps) * 8 

      ldgsts_latency = 300 # Does this matter?
      optimal_num_stages = ldgsts_latency/mma_cycles

      # nearest stages, prefer large #stages
      nearest = heapq.nsmallest(2, v, key=lambda x: 10 + abs(x[1] - optimal_num_stages) \
                                            if (x[1] - optimal_num_stages) < 0 else x[1] - optimal_num_stages)

      for n in nearest:
        pruned_configs.append(n[0])
    else: # Volta & Turing only supports num_stages <= 2
      random_config = v[0][0]
      random_config.num_stages = 2
      pruned_configs.append(random_config)
  return pruned_configs