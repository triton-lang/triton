# Plot script for triton layouts

This script is used to draw triton layouts in the context of matmul.
Here is the help info from the script.

```bash
>$ python3 plot_layout.py -h
usage: Draw triton layouts [-h] [-tensorShape TENSORSHAPE TENSORSHAPE] [-dotShape DOTSHAPE DOTSHAPE DOTSHAPE] [-plot {blocked,dot,wmma,lds}] [-dim0 DIM0] [-dim1 DIM1] [-sizePerThread SIZEPERTHREAD SIZEPERTHREAD]
                           [-threadsPerWarp THREADSPERWARP THREADSPERWARP] [-warpsPerCTA WARPSPERCTA WARPSPERCTA] [-order ORDER ORDER] [-nonKDim {16,32}] [-kWidth {4,8,16,32}] [-kGroup {1,2}]
                           [-dtype_a {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}] [-dtype_b {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}] [-mfmaTrans] [-scale] [-banks {32,64}] [-lds_layout {swizzle,padding,none}] [-lds_access {read,write,none}]
                           [-mnContig] [-mfma_trans_load] [-swizzleVec {4,8,16,32}] [-padInterval PADINTERVAL] [-padAmount PADAMOUNT] [-wave_size {32,64}] [-o O] [-keep]

options:
  -h, --help            show this help message and exit
  -tensorShape TENSORSHAPE TENSORSHAPE
                        2D tensor shape in the form of dim0,dim1
  -dotShape DOTSHAPE DOTSHAPE DOTSHAPE
                        Dot op shape in the form of M,N,K
  -plot {blocked,dot,wmma,lds}
                        choose plot mode
  -dim0 DIM0            tensor dim0 name
  -dim1 DIM1            tensor dim1 name
  -sizePerThread SIZEPERTHREAD SIZEPERTHREAD
  -threadsPerWarp THREADSPERWARP THREADSPERWARP
  -warpsPerCTA WARPSPERCTA WARPSPERCTA
  -order ORDER ORDER
  -nonKDim {16,32}      mfma instruction dim
  -kWidth {4,8,16,32}   number of contiguous elements per thread
  -kGroup {1,2}         total number of elements / kWidth per mfma instruction
  -dtype_a {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}
                        element type of operand A
  -dtype_b {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}
                        element type of operand B
  -mfmaTrans            If set, then use mfma.trans layout
  -scale                If set, plot the scale tensor for mfma_f8f6f4 instructions
  -banks {32,64}        choose the number of banks in LDS
  -lds_layout {swizzle,padding,none}
                        choose the LDS data layout
  -lds_access {read,write,none}
                        choose LDS access mode
  -mnContig             If set, the tensor is K x N and n-contig
  -mfma_trans_load      If set, use MFMA transpose load instructions
  -swizzleVec {4,8,16,32}
                        number of contiguous elements in a vector to swizzle
  -padInterval PADINTERVAL
                        Add padding for every padInterval bytes
  -padAmount PADAMOUNT  Pad padAmount bytes for every padInterval bytes
  -wave_size {32,64}    choose the wmma instruction mode
  -o O                  output pdf file name (without surfix)
  -keep                 If set, keep the generated .tex file
```

## Installation
This script does not require torch or triton to be installed. The only package
it depends on is latex. On Ubuntu, do
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra

```

## Draw blocked layout (`-plot blocked`)

Examples:
```bash
python3 plot_layout.py -plot blocked -tensorShape 128 64 -sizePerThread 1 8 -threadsPerWarp 8 8 -warpsPerCTA 4 1
python3 plot_layout.py -plot blocked -tensorShape 16 64 -sizePerThread 1 8 -threadsPerWarp 16 4 -warpsPerCTA 1 2
python3 plot_layout.py -plot blocked -tensorShape 32 64 -sizePerThread 8 1 -threadsPerWarp 4 16 -warpsPerCTA 1 2 -order 0 1
```

Blocked layouts are used during global load. It is used to describe the layout of the tensor
for pointers and results.
We can provide tensor shape (`-tensorShape dim0 dim1`) and blocked layout parameters (
`-sizePerThread x y`, `-threadsPerWarp x y`, and `-warpsPerCTA x y`).
We can also provide the order of the tensor as `-order x y` to control which dim
is the fastest changing dimension.

Notes
- The script does not support the case when threads are loading elements that are
  out of the boundary of the tensor dimensions. This means
  - For dim0: sizePerThread[0] * threadsPerWarps[0] * warpsPerCTA[0] <= dim0
  - For dim1: sizePerThread[1] * threadsPerWarps[1] * warpsPerCTA[1] <= dim1


## Draw mfma operand and result layouts (`-plot dot`)

Examples:
```bash
## i8 inputs
python3 plot_layout.py -plot dot -dotShape 128 128 128 -warpsPerCTA 2 4 -kWidth 8 -dtype_a i8 -dtype_b i8
python3 plot_layout.py -plot dot -dotShape 128 128 128 -warpsPerCTA 2 4 -kWidth 16 -dtype_a i8 -dtype_b i8
## fp16/bf16 inputs
python3 plot_layout.py -plot dot -dotShape 128 128 128 -warpsPerCTA 2 4 -kWidth 4 -dtype_a fp16 -dtype_b fp16
python3 plot_layout.py -plot dot -dotShape 128 128 128 -warpsPerCTA 2 4 -kWidth 8 -dtype_a fp16 -dtype_b fp16
## fp8/bf8 inputs
python3 plot_layout.py -plot dot -dotShape 128 128 128 -warpsPerCTA 2 4 -kWidth 8 -dtype_a fp8 -dtype_b bf8
python3 plot_layout.py -plot dot -dotShape 128 128 128 -warpsPerCTA 2 4 -kWidth 16 -dtype_a fp8 -dtype_b bf8
python3 plot_layout.py -plot dot -dotShape 128 128 128 -warpsPerCTA 2 4 -kWidth 16 -kGroup 2 -dtype_a fp8 -dtype_b bf8
## f4 and fp6/bf6 inputs
python3 plot_layout.py -plot dot -dotShape 128 128 128 -warpsPerCTA 2 4 -kWidth 32 -kGroup 1 -dtype_a f4 -dtype_b bf6
## fp8/bf8 and fp6/bf6/f4 inputs
python3 plot_layout.py -plot dot -dotShape 128 128 128 -warpsPerCTA 2 4 -kWidth 16 -kGroup 2 -dtype_a fp6 -dtype_b bf8
## mixed precision with scaling
python3 plot_layout.py -plot dot -dotShape 128 128 128 -warpsPerCTA 2 4 -kWidth 16 -kGroup 2 -dtype_a fp6 -dtype_b bf8 -scale
```

One can add `-nonKDim [16,32]` and `-mfmaTrans` to all of the above examples.

This mode draws two graphs:
1. The layout of the dot operation, i.e. tile C = tile A x tile B
2. The layout of a single mfma block, operands and results of one or more mfma
   instructions that share the same accumulating VGPRs.

Knobs
- `-kWidth [4,8,16,32]`: the number of elements that will be loaded into one thread at once
- `-kGroup [1,2]`: total number of elements / kWidth for on mfma instruction.
   This is 1 for all mfma instructions except for mfma_f32_16x16x128_f8f6f4 and mfma_f32_32x32x64_f8f6f4
   with fp8 input types (CBSZ=0 or 1 and/or BLGP=0 or 1)
- `-nonKDim [16,32]`: mfma instruction size. The default is set to 16.
- `-mfmaTrans`: if set, the transposed mfma layout will be plotted.
- `-dtype_a` and `-dtype_b`: element types of operand A and B. The default value is fp16.
- `-scale`: plot scale tensors for A and B. This is only supported with f4/f6 and f8 with `kGroup=2`.
  If `-scale` is set but not supported, it's ignored.

Notes
- The layout shows the mapping from the threads/wave to the elements in the
  original tensor. It does not matter if LDS is used.
- The script does not allow settings for k dim of the mfma instruction.
  This can be controled by the `-kWidth` and `-kGroup`.

## Draw LDS access (`-plot lds`)

Examples:
```bash
python3 plot_layout.py -plot lds -lds_layout none -lds_access none -tensorShape 128 128 -kWidth 8
python3 plot_layout.py -plot lds -lds_layout none -lds_access none -tensorShape 128 128 -kWidth 32 -dtype_a f4
python3 plot_layout.py -plot lds -lds_layout none -lds_access none -tensorShape 128 128 -kWidth 16 -dtype_a fp8 -banks 64
python3 plot_layout.py -plot lds -lds_layout swizzle -lds_access none -tensorShape 128 128 -kWidth 16 -dtype_a fp8 -banks 64
python3 plot_layout.py -plot lds -lds_layout swizzle -lds_access read -tensorShape 128 128 -kWidth 16 -dtype_a bf8 -banks 64
python3 plot_layout.py -plot lds -lds_layout swizzle -lds_access write -tensorShape 128 128 -kWidth 16 -dtype_a f4 -banks 32
python3 plot_layout.py -plot lds -lds_layout none -lds_access read -tensorShape 128 32 -kWidth 4 -dtype_a fp16 -banks 64 -mnContig
python3 plot_layout.py -plot lds -lds_layout swizzle -lds_access read -tensorShape 128 32 -kWidth 16 -dtype_a fp8 -banks 64 -mnContig -mfma_trans_load
python3 plot_layout.py -plot lds -lds_layout padding -lds_access none -tensorShape 128 32 -kWidth 8 -dtype_a fp16 -banks 32 -padInterval 128 -padAmount 16
```

Knobs
- `kWidth`: the vector size (in unit of elements) when accessing LDS
- `banks`: the number of banks in LDS. (64 for gfx950, 32 for pre-gfx950)
- `dtype_a`: element data type
- Three options for `-lds_layout`:
  - `none`: no swizzling, no padding
  - `swizzle`: apply the swizzling pattern, which is derived from tensor shape and kWidth.
  - `padding`: pad `padAmount` bytes for every `padInterval` bytes of data
    - `padAmount`: default is 0
    - `padInterval`: default is 1
- Three options for `-lds_access`:
  - `none`: do not plot access pattern
  - `read`: plot accessed elements at the first cycle of ds_read
  - `write`: plot accessed elements during ds_write. For global load access, we assume
    a fully coalesced dwordx4 access pattern along the K dim.
- `mnContig`: If set, the tile is stored in mn-contig layout. In this layout, elements along
  the M/N dim are contiguous in both global memory and LDS.
- `mfma_trans_load`: This flag only works when `mnContig` is set. When set, `ds_read_b64_tr_bx`
  instructions are used to read from LDS. Note that current triton LDS layout mechanism will
  lead to bank conflicts.
