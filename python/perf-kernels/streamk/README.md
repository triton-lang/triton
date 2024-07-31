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
