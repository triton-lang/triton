### Install Latest rocprofv3 from sources

Compile that latest rocprofv3  from sources (use `amd-staging`  branch)

```bash
cd ~
mkdir -p ~/usr/rocprofv3
INSTALL_DIR=$(realpath ~/usr/rocprofv3)
git clone https://github.com/rocm/rocprofiler-sdk
cd rocprofiler-sdk
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DROCPROFILER_BUILD_TESTS=ON -DROCPROFILER_BUILD_SAMPLES=ON
make -j
make install
```

Set the corresponding env. variables

```bash
$ cat ~/load.rocprofv3.sh
#!/bin/bash

INSTALL_DIR=$(realpath ~/usr/rocprofv3)

export PATH=${INSTALL_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${INSTALL_DIR}/lib:${LIBRARY_PATH}

$ source ~/load.rocprofv3.sh
```

### Adjust Triton Source Code

The `flash-attention.py` kernel comes with auto-tuning. In this example, we want to measure performance of the best performing FA configuration. Run the kernel with the enabled auto-tuner.


```bash
$ TRITON_PRINT_AUTOTUNING=1 python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -causal -layout thd
Autotuning kernel attn_fwd with config BLOCK_M: 128, BLOCK_N: 128, waves_per_eu: 2, PRE_LOAD_V: False, GRID_CU_MULTIP: 2, schedule_hint: none, num_warps: 4, num_ctas: 1, num_stages: 1, maxnreg: None
Autotuning kernel attn_fwd with config BLOCK_M: 128, BLOCK_N: 64, waves_per_eu: 2, PRE_LOAD_V: False, GRID_CU_MULTIP: 2, schedule_hint: none, num_warps: 4, num_ctas: 1, num_stages: 1, maxnreg: None
Autotuning kernel attn_fwd with config BLOCK_M: 128, BLOCK_N: 64, waves_per_eu: 3, PRE_LOAD_V: False, GRID_CU_MULTIP: 2, schedule_hint: none, num_warps: 4, num_ctas: 1, num_stages: 1, maxnreg: None
Autotuning kernel attn_fwd with config BLOCK_M: 128, BLOCK_N: 64, waves_per_eu: 1, PRE_LOAD_V: False, GRID_CU_MULTIP: 2, schedule_hint: none, num_warps: 4, num_ctas: 1, num_stages: 1, maxnreg: None
Autotuning kernel attn_fwd with config BLOCK_M: 128, BLOCK_N: 32, waves_per_eu: 2, PRE_LOAD_V: False, GRID_CU_MULTIP: 2, schedule_hint: none, num_warps: 4, num_ctas: 1, num_stages: 1, maxnreg: None
Triton autotuning for function attn_fwd finished after 15.06s; best config selected: BLOCK_M: 128, BLOCK_N: 64, waves_per_eu: 2, PRE_LOAD_V: False, GRID_CU_MULTIP: 2, schedule_hint: none, num_warps: 4, num_ctas: 1, num_stages: 1, maxnreg: None;
fused-attention-fwd-d128-layoutthd:
   BATCH    HQ    HK  N_CTX_Q  N_CTX_K      triton      torch
0    2.0  16.0  16.0   8192.0   8192.0  221.869662  17.140226
```


Open the script and find the function which sets tuning parameters (i.e., `get_cdna_autotune_configs`). You can see that the function returns a list of suggested configs to the tuner. Comment everything except the winning config that we found in the previous step. For example,


```python
def get_cdna_autotune_configs():
    return [
        #triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
        #              num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
                      num_stages=1, num_warps=4),
        #triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
        #              num_stages=1, num_warps=4),
        #triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
        #              num_stages=1, num_warps=4),
        #triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'waves_per_eu': 2, 'PRE_LOAD_V': False, 'GRID_CU_MULTIP': 2},
        #              num_stages=1, num_warps=4),
    ], ['IS_CAUSAL', 'dropout_p', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'VARLEN', 'HQ', 'HK']

```


### Collect Performance Data

Make a softlink to `./rocm-triton-prof.py` in the directory where you perform the test. For example,

```bash
ln -s <rocm-triton-dir>/python/perf-kernels/tools/rocm-triton-prof/rocm-triton-prof.py rocm-triton-prof.py
```

Run the tool as follows:

```bash
$ python3 ./rocm-triton-prof.py --kernel attn_fwd --cmd python3 ./flash-attention.py -b 2 -hq 16 -hk 16 -sq 8192 -sk 8192 -d 128 -causal -layout thd
Timing info in `nsec`:
count       269.000000
mean     326119.100372
std        7120.765559
min      304946.000000
25%      322147.000000
50%      327960.000000
75%      331047.000000
max      352857.000000
dtype: float64

NON-FLOP related data:
   Counter Name        Max        Min          Mean     Median
0    GRBM_COUNT  8955952.0  4043501.0  4.284156e+06  4261916.0
1   TCC_HIT_sum  5347185.0  4074880.0  4.112117e+06  4107955.0
2  TCC_MISS_sum  5932281.0  3526537.0  3.572396e+06  3556786.5

FLOP related data:
                    Counter Name     Raw Data          FLOP  Relative FLOP, %
0          SQ_INSTS_VALU_ADD_F16          0.0  0.000000e+00          0.000000
1          SQ_INSTS_VALU_MUL_F16          0.0  0.000000e+00          0.000000
2          SQ_INSTS_VALU_FMA_F16     192512.0  2.464154e+07          0.030844
3        SQ_INSTS_VALU_TRANS_F16          0.0  0.000000e+00          0.000000
4          SQ_INSTS_VALU_ADD_F32    4898176.0  3.134833e+08          0.392393
5          SQ_INSTS_VALU_MUL_F32    2411456.0  1.543332e+08          0.193182
6          SQ_INSTS_VALU_FMA_F32    2486720.0  3.183002e+08          0.398422
7        SQ_INSTS_VALU_TRANS_F32    2489728.0  1.593426e+08          0.199452
8          SQ_INSTS_VALU_ADD_F64          0.0  0.000000e+00          0.000000
9          SQ_INSTS_VALU_MUL_F64          0.0  0.000000e+00          0.000000
10         SQ_INSTS_VALU_FMA_F64          0.0  0.000000e+00          0.000000
11       SQ_INSTS_VALU_TRANS_F64          0.0  0.000000e+00          0.000000
12   SQ_INSTS_VALU_MFMA_MOPS_F16  154140672.0  7.892002e+10         98.785706
13  SQ_INSTS_VALU_MFMA_MOPS_BF16          0.0  0.000000e+00          0.000000
14   SQ_INSTS_VALU_MFMA_MOPS_F32          0.0  0.000000e+00          0.000000
15   SQ_INSTS_VALU_MFMA_MOPS_F64          0.0  0.000000e+00          0.000000

Performance info in TFLOP/s:
count    269.000000
mean     245.090089
std        5.420713
min      226.409352
25%      241.325627
50%      243.597161
75%      247.992764
max      261.981219
dtype: float64
```

### Known limits

The tool currently supports only FP64, FP32 and FP16 operations.
Note, it can be extended to supoprt other data types.
