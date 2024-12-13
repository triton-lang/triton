# streamk gemm script v0.3

### features added:

- new persistent gemm kernel
- gemm benchmark tool using nearest neighbour approach.

### benchmark commandline

```
 python gemm_benchmark.py
```

# streamk gemm script v0.2

### features added:

- new streamk tuning script to reduce compiling and profiling time

- use load/store cache modifier to reimplement spinning lock

- add CI test for streamk-kernel

### potential issues:

- there may be hanging issue when use random grid sizes
- large register spills when using tile size 256x256x64

### tuning command

```
python tune_streamk.py --gemm_size_file input_nn_size.yaml --ngpus 8 --jobs 24
```

### calculate occ

```
../tools/occ.sh "python tune_streamk.py --gemm_size_file single_item.yaml --compare_wo_tuning"
```

# streamk gemm script v0.1

The plan is to use this version as the base version for the future triton streamk gemm development.

### Main features
- comparable performance with tune gemm

- use the persistent loop so that a WG may work on multiple output tiles, and also allowing workgroups to do part of the work for an output tile.

- use atomics for spinning lock to replace atomic_add for the final output.

- pid renumbering based on chiplet structure of MI300X

- dynamic grid setting

- tuning script adapt from tune_gemm

### Usage

Go to the script dir
```bash
cd triton/python/perf_kernels/streamk
```

1. Tune gemm sizes given in a yaml file and check correctness on the way
```bash
python tune_streamk.py --gemm_size_file input_gemm_sizes.yaml --compare
```

2. Tune a single gemm size
```bash
python tune_streamk.py -m 16 -n 16 -k 16
```

3. Choose the file to store tuning results
```bash
python tune_streamk.py --gemm_size_file input_gemm_sizes.yaml --o output_tuning.yaml
```

4. Only check correctness given the tuning results
```bash
python tune_streamk.py --gemm_size_file output_tuning.yaml --compare_wo_tuning
```
