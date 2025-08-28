// RUN: triton-opt %s -split-input-file -verify-diagnostics

// expected-error@+2 {{ttg.dot_op opIdx parameter can be 0 or 1, got: 2}}
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#dot_op = #ttg.dot_op<{opIdx = 2, parent = #blocked, kWidth = 2}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter is not supported when the parent is a blocked layout}}
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 8}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter can only be non-zero for Ampere or Hopper MMA parent}}
#mma = #ttg.nvidia_mma<{versionMajor = 1, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter is mandatory for Ampere or Hopper MMA parent}}
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_op = #ttg.dot_op<{opIdx = 0, parent = #mma}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter is mandatory for Ampere or Hopper MMA parent}}
#mma = #ttg.nvidia_mma<{versionMajor = 3, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_op = #ttg.dot_op<{opIdx = 0, parent = #mma}>

// -----

// expected-error@+2 {{ttg.dot_op opIdx parameter must be 0 for Hopper MMA parent, since Hopper WGMMA only allows first operand to be in registers}}
#mma = #ttg.nvidia_mma<{versionMajor = 3, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter is mandatory for MFMA parent}}
#mfma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1, 1], instrShape = [32, 32], isTransposed = false}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #mfma}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter must be 8/16 for gfx11 and 4/8/16 for gfx12 (including packed cases for `scaled_dot`)}}
#wmma = #ttg.amd_wmma<{version = 1, warpsPerCTA = [1, 4]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #wmma}>

// -----

// expected-error@+2 {{ttg.dot_op kWidth parameter must be 8/16 for gfx11 and 4/8/16 for gfx12 (including packed cases for `scaled_dot`)}}
#wmma = #ttg.amd_wmma<{version = 2, warpsPerCTA = [1, 4]}>
#dot_op = #ttg.dot_op<{opIdx = 1, parent = #wmma, kWidth = 32}>

// -----

// expected-error@+1 {{version must be in the [0, 4] range}}
#mfma = #ttg.amd_mfma<{version = 10, warpsPerCTA = [1, 1, 1], instrShape = [32, 32], isTransposed = false}>

// -----

// expected-error@+1 {{invalid (mDim, nDim) combination}}
#mfma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1, 1], instrShape = [16, 8], isTransposed = false}>

// -----

// expected-error@+1 {{element type must be f64, f32, i32, or none}}
#mfma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1, 1], instrShape = [16, 16], isTransposed = false, elementType = f16}>

// -----

// expected-error@+1 {{interval values must all be power of two}}
#shared = #ttg.padded_shared<[3:+2]>

// -----

// expected-error@+1 {{interval values must all be power of two}}
#shared = #ttg.padded_shared<[0:+2]>

// -----

// expected-error@+1 {{padding values must all be power of two}}
#shared = #ttg.padded_shared<[2:+3]>

// -----

// expected-error@+1 {{padding values must all be power of two}}
#shared = #ttg.padded_shared<[2:+0]>

// -----

// expected-error@+1 {{interval values cannot have duplicates}}
#shared = #ttg.padded_shared<[2:+1, 2:+4]>

// -----

// expected-error@+1 {{order cannot be empty}}
#shared = #ttg.padded_shared<[2:+1, 4:+2]>

// -----

// expected-error@+1 {{unexpected key: unknown}}
#shared = #ttg.padded_shared<[2:+1, 4:+2] {order = [1, 0], unknown = 5}>

// -----

// expected-error@+1 {{order size (3) must match CTALayout rank (2)}}
#shared = #ttg.padded_shared<[2:+1, 4:+2] {order = [2, 1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
