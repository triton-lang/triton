# GEMM tuning script v2

This is the v2 version of the gemm tuning script, which is based on @scxiao's v1 (https://github.com/ROCmSoftwarePlatform/triton/pull/309) and @alefimov-amd's thread pool https://github.com/ROCmSoftwarePlatform/triton/pull/310

### Main features
- `rocprof` is used to measure the time for kernels in the full tuning space
- Each kernel is executed 10 times and the execution time of the last instance is used
- All kernels are compiled in parallel
- Two modes for correctness checking
    - During tuning, check correctness with the best perf_config for the current gemm size
    - Without tuning, check correctness based on the tuning results, which includes best perf_config for each gemm size
- The process takes about 30 - 40 minutes for the full tuning space with ~15000 configs
- Limitations
   - For now, only support fp16 as inputs. It should be trivial to extend to other types, but may require some work for mixed inputs

### Usage
Go to the script dir
```bash
cd triton/scripts/amd/gemm/
```

1. Tune gemm sizes given in a yaml file and check correctness on the way
```bash
python tune_gemm.py --gemm_size_file input_gemm_sizes.yaml --compare
```

2. Tune a single gemm size
```bash
python tune_gemm.py -m 16 -n 16 -k 16
```

3. Choose the file to store tuning results
```bash
python tune_gemm.py --gemm_size_file input_gemm_sizes.yaml --o output_tuning.yaml
```

4. Only check correctness given the tuning results
```bash
python tune_gemm.py --gemm_size_file output_tuning.yaml --compare_wo_tuning
```
Note that the tuning results file are provided as the `gemm_size_file` in this scenario.

### Overview of implementations

Workflow of the tuning process
1. Generate the full tuning space. For now the `range`s for each tuning parameter are hard-coded
2. Prune the tuning space according to the current GEMM size and some rules
    - BLOCK_SIZE must be equal or larger than the mfma instruction size.
    - SPLIT_K * BLOCK_SIZE_K must divide K. Therefore, we do not need EVEN_K in the kernel. 
    - When split-k is not needed, i.e. both M and N are large, it must be 1
    - GROUP_M * BLOCK_SIZE_M must be smaller than M. Otherwise, GROUP_M must be 1
    - When BLOCK_SIZE_K = 128, neither BLOCK_SIZE_M or BLOCK_SIZE_N can be 128. Otherwise too much LDS will be required. **Needs further investigation**
    - Skip BLOCK_SIZE_M or BLOCK_SIZE_N if they are over 2 times larger than M or N.
3. Open a file `generated_kernel{M}-{N}-{K}-{gpuid}.py` and write the following into the file
    1. For each config in the pruned space, generate a kernel with name `matmul_kernel_{configStr}`, where `configStr` contains the gemm size and the tuning parameters.
    2. Generate `matmul` function for each config in a similar way
    3. Generate `try_config` functions for each `matmul` function.
    4. Generate `test_gemm`, which does
        1. Add all `try_config` functions in the thread_pool by `thread_pool.apply_async(try_config)`. This is used to compile all kernels in parallel.  
        2. Call each `matmul` function in a for loop of 10 iterations
    5. Generate `main` function
4. Run the generated script with 16 workers. This will compile all kernels in parallel.
5. Invoke `rocprof` on the generated script
6. Post process `results.csv` by extract the execution time of the last instance of each kernel. Pick the best one, write to file, and return.

# GEMM Tuning Script v3

### API changes

- Input and output data types can be provided as `-dtype_a`, `-dtype_b`, and `-dtype_c`.
The provided types must be one of ['fp32', 'fp16', 'bf16', 'fp8', 'bf8', 'int8'].
- Row/col major-ness of operand a and b can be provided as `-col_a` and `-col_b`.
If set, it means the corresponding operand is column major.
The major-ness is considered as problem input. 
So they should be included in the input yaml file. However, in the yaml file, user should
set `rowMajowA` and `rowMajorB` as shown in the example below.
- `--benchmark` is used to control if the perf config in the input yaml file is used as the tuning space.
- `--jobs` is used to control the number of .py files for generated kernels.
Note that this can be different from `ngpus`. This usually means multiple kernel files
will be profiled on each GPU.
This is necessary to keep each file "small" in terms of execution time.

### Implementation changes
- `gen_input` is used to generate matmul inputs.
- Time measurement
    - In benchmark mode, the kernel is executed 1000 times.
    - In tuning mode, each kernel is executed 200 times. We cannot afford to larger runs since rocprof hangs if the session takes too long.
    - In both tuning and benchmark mode, kernel time is measured as the average execution time of the last 100 instances.
- Added error recovery. This helps when rocprof crashes in multi-processing mode. 


### Example Usage

Let's say we have an input yaml file, named `gemm_input.yaml`, that contains the following configs
```yaml
- {'M': 4864, 'N': 4096, 'K': 8192, 'rowMajorA': 'T', 'rowMajorB': 'N'}
- {'M': 8192, 'N': 8192, 'K': 8192, 'rowMajorA': 'T', 'rowMajorB': 'N'}
```
1. Tuning with bf8 input types with gpu 4,5,6,7, and save output to `output.yaml`
```bash
python ./tune_gemm.py --gemm_size_file gemm_input.yaml -dtype_a bf8 -dtype_b bf8 --gpu_ids 4,5,6,7 --o output.yaml
```

2. Check the correctness of the tuned configs
```bash
python ./tune_gemm.py --gemm_size_file output.yaml -dtype_a bf8 -dtype_b bf8 --compare_wo_tuning
```

3. Run benchmark of the tuned configs
```bash
python ./tune_gemm.py --gemm_size_file output.yaml -dtype_a bf8 -dtype_b bf8 --benchmark
```

A sample output from `benchmark` looks like
```bash
Benchmarking gemm with bf8 inputs (peak tflops: 1298)
trans    M     N     K    TFLOPS  Efficiency
NT    4864  4096  8192    841.22         65%
NT    8192  8192  8192    745.31         57%
```

# GEMM Tuning Script v3.1

### API changes

- Added `matrix_instr_nonkdim` into the tuning space. Now we can tune mfma instruction size.


# One config running script

`one_config.py` is a script that runs one given matmul config.
It is an interface to `tune_gemm.py` functionality and could be used for triton debugging.

### Usage

This script supports two methods to specify configuration parameters.

Variant 1: Separate command line attributes.

```bash
python one_config.py -m 256 -n 256 -k 256 --block_m 64 --block_n 64 --block_k 64 --group_m 1 --split_k 2 --num_warps 2 --num_stages 0 --waves_per_eu 0
```

Variant 2: one-line config description.
This is how configs are printed by `tune_gemm.py` script

```bash
python one_config.py --config_str M16_N8_K128_BM64_BN64_BK64_GM1_SK2_nW2_nS0_EU0
```

