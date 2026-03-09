# Triton Apple MPS Backend

WIP. Make-or-break foundation laid.

## What's done

### 1. Empirical layout verification (`verify_simdgroup.metal` + `.swift`)
Confirmed Apple `simdgroup_float8x8` thread→element mapping:
```
lane T, reg R → row = (T >> 3) + R*4,  col = T & 7
```
Layout grid:
```
     col: 0  1  2  3  4  5  6  7
row 0:    0  1  2  3  4  5  6  7   ← lanes 0-7,  reg 0
row 1:    8  9 10 11 12 13 14 15   ← lanes 8-15, reg 0
row 2:   16 17 18 19 20 21 22 23
row 3:   24 25 26 27 28 29 30 31
row 4:    0  1  2  3  4  5  6  7   ← lanes 0-7,  reg 1
row 5:    8  9 10 11 12 13 14 15
row 6:   16 17 18 19 20 21 22 23
row 7:   24 25 26 27 28 29 30 31   ← lanes 24-31,reg 1
```

### 2. LinearLayout basis vectors (`AppleMmaGroup.cpp`)
XOR basis vectors for the Triton LinearLayout system:
```cpp
layout *= LinearLayout::identity1D(8, "lane",     dimCol);  // 8 cols
layout *= LinearLayout::identity1D(4, "lane",     dimRow);  // rows 0-3
layout *= LinearLayout::identity1D(2, "register", dimRow);  // rows 4-7
```

### 3. toLinearLayout() (`AppleMmaLayoutConversions.cpp`)
Full `AppleMmaEncodingAttr::toLinearLayout()` implementation including:
- Single 8x8 tile basis
- warpsPerCTA tiling across M and N
- combineCtaCgaWithShape for arbitrary tensor shapes

### 4. Attribute definition (`AppleGPUAttrDefs.td`)
MLIR tablegen for `AppleMmaEncodingAttr` mirroring `NvidiaMmaEncodingAttr`.

### 5. Backend skeleton (`backend/compiler.py`)
Full pipeline stub: ttir → ttgir → llir → metalir → metallib

## What's next (in order)

1. **CMakeLists.txt** — wire up the new dialect + passes into Triton build
2. **AccelerateAppleMatmul pass** — MLIR rewrite: `tt.dot` → `AppleMmaEncoding`
   - Analog of `AccelerateAMDMatmul.cpp` (~1800 lines)
   - This is the second hardest piece after LinearLayout
3. **DotOpToLLVM** — emit `simdgroup_multiply_accumulate` Metal intrinsic calls
4. **Metal IR lowering** — connect to MetalASM for in-process compilation
5. **driver.py** — MTLDevice dispatch, buffer management
6. **Software pipelining** — reuse existing `SoftwarePipeliner.cpp` passes

## Key insight
The LinearLayout is the keystone — once correct, Triton's existing:
- Layout propagation passes work automatically
- Shared memory access analysis works automatically
- Layout conversion insertion works automatically
- Software pipelining works automatically

Only the dot op lowering (step 3) needs Apple-specific code after the layout.
