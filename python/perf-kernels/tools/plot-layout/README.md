# Plot script for triton layouts

This script is used to draw Triton layouts in the context of matmul.
Here is the help info from the script.

```bash
>$ python3 plot_layout.py -h
usage: Draw triton layouts [-h] [--output OUTPUT] [--keep] [--force] PLOT_TYPE ...

options:
  -h, --help       show this help message and exit
  --output OUTPUT  output pdf file name (without surfix)
  --keep           If set, keep the generated .tex file
  --force          If set, overwrite the pdf file with the same name

subcommands:
  Choose to plot blocked, lds, dot or wmma

  PLOT_TYPE        Choose one of the four plot mode
    blocked        plot blocked layout for global memory access
    dot            plot dot layout for MFMA
    lds            plot LDS (shared memory) layout
    wmma           plot dot layout for wmma
```

## Installation
This script does not require torch or triton to be installed. The only package
it depends on is latex. On Ubuntu, do
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra

```

## Draw blocked layout (`python plot_layout.py blocked`)
```bash
>$ python plot_layout.py blocked --help
usage: Draw triton layouts blocked [-h] [-r ROW] [-c COL] [-B] [-s s0 s1] [-t t0 t1] [-w w0 w1] [-o minor major] [-b b0 b1]

options:
  -h, --help                           show this help message and exit
  -r ROW, --rowName ROW                tensor dim0 name (default: M)
  -c COL, --colName COL                tensor dim1 name (default: K)
  -B, --matrixB                        shortcut to plot operand B with dimension name of (K, N) (default: False)
  -s s0 s1, --sizePerThread s0 s1      how many elements each thread holds in the 2D block per CTA (default: (1, 4))
  -t t0 t1, --threadsPerWarp t0 t1     how thread is partitioned into a 2D grid in a warp with 64 threads (default: (16, 4))
  -w w0 w1, --warpsPerCTA w0 w1        how warps tile a CTA (default: (1, 4))
  -o minor major, --order minor major  order from most minor to most major (default: (1, 0))
  -b b0 b1, --blockShape b0 b1         block size (dim0, dim1) of the tile. If not specified it presumably equals to the shape of CTA
```

Examples:
```bash
python3 plot_layout.py blocked --sizePerThread 1 8 --threadsPerWarp 8 8 --warpsPerCTA 4 1
python3 plot_layout.py blocked --blockShape 16 64 --sizePerThread 1 8 --threadsPerWarp 16 4 --warpsPerCTA 1 2
python3 plot_layout.py blocked --blockShape 32 64 --sizePerThread 8 1 --threadsPerWarp 4 16 --warpsPerCTA 1 2 --order 0 1
```

Blocked layouts are used during global load. It is used to describe the layout of the tensor
for pointers and results.
We can provide blocked layout parameters (
`--sizePerThread x y`, `--threadsPerWarp x y`, and `--warpsPerCTA x y`).
We can also provide the order of the tensor as `--order x y` to control which dim
is the fastest changing dimension.

Note that the parameters above forms a Cooperative Thread Array (CTA) and the block may contain multiple of it.
Please specifiy block shape (`--blockShape b0 b1`) to explicitly set the block size, otherwise it will assumes block size equals to CTA size.

Notes
- The script does not support the case when threads are loading elements that are
  out of the boundary of the tensor dimensions. This means
  - For dim0: sizePerThread[0] * threadsPerWarps[0] * warpsPerCTA[0] <= dim0
  - For dim1: sizePerThread[1] * threadsPerWarps[1] * warpsPerCTA[1] <= dim1


## Draw mfma operand and result layouts (`python plot_layout.py dot`)
```bash
>$ python plot_layout.py dot --help
usage: Draw triton layouts dot [-h] [--dotShape M N K] [--warpsPerCTA w0 w1] [--nonKDim {16,32}] [--kWidth {4,8,16,32}] [--kGroup {1,2}]
                               [--dtypeA {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}] [--dtypeB {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}] [--mfmaTrans] [--scale]

options:
  -h, --help                                  show this help message and exit
  --dotShape M N K                            Dot op shape in the form of M, N, K (default: (32, 128, 64))
  --warpsPerCTA w0 w1                         how warps tile the dot result matrix (default: (1, 4))
  --nonKDim {16,32}                           mfma instruction dimension of M/N (default: 16)
  --kWidth {4,8,16,32}                        number of contiguous elements each thread owns during MFMA (default: 4)
  --kGroup {1,2}                              total number of elements / kWidth per mfma instruction (default: 1)
  --dtypeA {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}  element type of operand A (default: fp16)
  --dtypeB {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}  element type of operand B (default: fp16)
  --mfmaTrans                                 If set, then use mfma.trans layout (default: False)
  --scale                                     If set, plot the scale tensor for mfma_f8f6f4 instructions (default: False)
```

Examples:
```bash
## i8 inputs
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 8 --dtype-a i8 --dtype-b i8
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 16 --dtype-a i8 --dtype-b i8
## fp16/bf16 inputs
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 4 --dtype-a fp16 --dtype-b fp16
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 8 --dtype-a fp16 --dtype-b fp16
## fp8/bf8 inputs
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 8 --dtype-a fp8 --dtype-b bf8
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 16 --dtype-a fp8 --dtype-b bf8
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 16 --kGroup 2 --dtype-a fp8 --dtype-b bf8
## f4 and fp6/bf6 inputs
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 32 --kGroup 1 --dtype-a f4 --dtype-b bf6
## fp8/bf8 and fp6/bf6/f4 inputs
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 16 --kGroup 2 --dtype-a fp6 --dtype-b bf8
## mixed precision with scaling
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 16 --kGroup 2 --dtype-a fp6 --dtype-b bf8 --scale
```

One can add `--nonKDim [16,32]` and `--mfmaTrans` to all of the above examples.

This mode draws two graphs:
1. The layout of the dot operation, i.e. tile C = tile A x tile B
2. The layout of a single mfma block, operands and results of one or more mfma
   instructions that share the same accumulating VGPRs.

Knobs
- `--kWidth [4,8,16,32]`: the number of elements that will be loaded into one thread at once
- `--kGroup [1,2]`: total number of elements / kWidth for on mfma instruction.
   This is 1 for all mfma instructions except for mfma_f32_16x16x128_f8f6f4 and mfma_f32_32x32x64_f8f6f4
   with fp8 input types (CBSZ=0 or 1 and/or BLGP=0 or 1)
- `--nonKDim [16,32]`: mfma instruction size. The default is set to 16.
- `--mfmaTrans`: if set, the transposed mfma layout will be plotted.
- `--dtype-a` and `-dtype-b`: element types of operand A and B. The default value is fp16.
- `--scale`: plot scale tensors for A and B. This is only supported with f4/f6 and f8 with `kGroup=2`.
  If `--scale` is set but not supported, it's ignored.

Notes
- The layout shows the mapping from the threads/wave to the elements in the
  original tensor. It does not matter if LDS is used.
- The script does not allow settings for k dim of the mfma instruction.
  This can be controled by the `--kWidth` and `--kGroup`.

## Draw LDS access (`python plot_layout.py lds`)
```bash
>$ python plot_layout.py lds --help
usage: Draw triton layouts lds [-h] [--tensorShape TENSORSHAPE TENSORSHAPE] [--kWidth {4,8,16,32}] [--dtype {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}] [--nonKDim {16,32}]
                               [--banks {32,64}] [--layout {swizzle,padding,none}] [--access {read,write,none}] [--mnContig] [--mfma-trans-load]
                               [--swizzleVec {4,8,16,32}] [--padInterval PADINTERVAL] [--padAmount PADAMOUNT]

options:
  -h, --help                                 show this help message and exit
  --tensorShape TENSORSHAPE TENSORSHAPE      2D block shape in the form of (dim0, dim1) (default: (128, 64))
  --kWidth {4,8,16,32}                       number of contiguous elements per thread (default: 4)
  --dtype {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}  element type of tensor to be stored in LDS (default: fp16)
  --nonKDim {16,32}                          mfma instruction dim (default: 16)
  --banks {32,64}                            choose the number of banks in LDS (default: 32)
  --layout {swizzle,padding,none}            choose the LDS data layout (default: none)
  --access {read,write,none}                 choose LDS access mode (default: none)
  --mnContig                                 If set, the tensor is K x N and n-contig (default: False)
  --mfma-trans-load                          If set, use MFMA transpose load instructions (default: False)
  --swizzleVec {4,8,16,32}                   number of contiguous elements in a vector to swizzle (default: 4)
  --padInterval PADINTERVAL                  Add padding for every padInterval bytes (default: 1)
  --padAmount PADAMOUNT                      Pad padAmount bytes for every padInterval bytes (default: 0)
```
Examples:
```bash
python3 plot_layout.py lds --lds-layout none --lds-access none --tensorShape 128 128 --kWidth 8
python3 plot_layout.py lds --lds-layout none --lds-access none --tensorShape 128 128 --kWidth 32 --dtype f4
python3 plot_layout.py lds --lds-layout none --lds-access none --tensorShape 128 128 --kWidth 16 --dtype fp8 --banks 64
python3 plot_layout.py lds --lds-layout swizzle --lds-access none --tensorShape 128 128 --kWidth 16 --dtype fp8 --banks 64
python3 plot_layout.py lds --lds-layout swizzle --lds-access read --tensorShape 128 128 --kWidth 16 --dtype bf8 --banks 64
python3 plot_layout.py lds --lds-layout swizzle --lds-access write --tensorShape 128 128 --kWidth 16 --dtype f4 --banks 32
python3 plot_layout.py lds --lds-layout none --lds-access read --tensorShape 128 32 --kWidth 4 --dtype fp16 --banks 64 --mnContig
python3 plot_layout.py lds --lds-layout swizzle --lds-access read --tensorShape 128 32 --kWidth 16 --dtype fp8 --banks 64 --mnContig --mfma_trans_load
python3 plot_layout.py lds --lds-layout padding --lds-access none --tensorShape 128 32 --kWidth 8 --dtype fp16 --banks 32 --padInterval 128 --padAmount 16
```

Knobs
- `kWidth`: the vector size (in unit of elements) when accessing LDS
- `banks`: the number of banks in LDS. (64 for gfx950, 32 for pre-gfx950)
- `dtype_a`: element data type
- Three options for `--lds-layout`:
  - `none`: no swizzling, no padding
  - `swizzle`: apply the swizzling pattern, which is derived from tensor shape and kWidth.
  - `padding`: pad `padAmount` bytes for every `padInterval` bytes of data
    - `padAmount`: default is 0
    - `padInterval`: default is 1
- Three options for `--lds-access`:
  - `none`: do not plot access pattern
  - `read`: plot accessed elements at the first cycle of ds_read
  - `write`: plot accessed elements during ds_write. For global load access, we assume
    a fully coalesced dwordx4 access pattern along the K dim.
- `mnContig`: If set, the tile is stored in mn-contig layout. In this layout, elements along
  the M/N dim are contiguous in both global memory and LDS.
- `mfma_trans_load`: This flag only works when `mnContig` is set. When set, `ds_read_b64_tr_bx`
  instructions are used to read from LDS. Note that current triton LDS layout mechanism will
  lead to bank conflicts.

## Draw WMMA access (`python plot_layout.py lds`)
WMMA layout drawing is intended for Radeon consumer GPU usage. Currently it has very limited support.
