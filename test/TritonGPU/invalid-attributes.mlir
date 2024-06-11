// RUN: triton-opt %s -split-input-file -verify-diagnostics

// expected-error@+2 {{triton_gpu.dot_op opIdx paramenter can be 0 or 1, got: 2}}
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#dot_op = #triton_gpu.dot_op<{opIdx = 2, parent = #blocked}>

// -----

// expected-error@+2 {{triton_gpu.dot_op kWidth parameter is not supported when the parent is a blocked layout}}
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#dot_op = #triton_gpu.dot_op<{opIdx = 1, parent = #blocked, kWidth = 8}>

// -----

// expected-error@+2 {{triton_gpu.dot_op kWidth parameter can only be non-zero for Ampere MMA parent}}
#mma = #triton_gpu.nvidia_mma<{versionMajor = 1, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_op = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// -----

// expected-error@+2 {{triton_gpu.dot_op kWidth parameter is mandatory for Ampere MMA parent}}
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_op = #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>

// -----

// expected-error@+2 {{triton_gpu.dot_op kWidth parameter can only be non-zero for Ampere MMA parent}}
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_op = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>

// -----

// expected-error@+2 {{triton_gpu.dot_op kWidth parameter is mandatory for MFMA parent}}
#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1, 1], instrShape = [32, 32], isTransposed = false}>
#dot_op = #triton_gpu.dot_op<{opIdx = 1, parent = #mfma}>

// -----

// expected-error@+2 {{triton_gpu.dot_op kWidth parameter supports only 16 for WMMA parent}}
#wmma = #triton_gpu.amd_wmma<{warpsPerCTA = [1, 4]}>
#dot_op = #triton_gpu.dot_op<{opIdx = 1, parent = #wmma}>

// -----

// expected-error@+2 {{triton_gpu.dot_op kWidth parameter supports only 16 for WMMA parent}}
#wmma = #triton_gpu.amd_wmma<{warpsPerCTA = [1, 4]}>
#dot_op = #triton_gpu.dot_op<{opIdx = 1, parent = #wmma, kWidth = 8}>

// -----

// expected-error@+1 {{major version must be in the [0, 3] range}}
#mfma = #triton_gpu.amd_mfma<{versionMajor = 10, versionMinor = 0, warpsPerCTA = [1, 1, 1], instrShape = [32, 32], isTransposed = false}>

// -----

// expected-error@+1 {{minor version must be 0}}
#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 5, warpsPerCTA = [1, 1, 1], instrShape = [32, 32], isTransposed = false}>

// -----

// expected-error@+1 {{(M, N) cases other than (32, 32) or (16, 16) unimplemented}}
#mfma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1, 1], instrShape = [16, 8], isTransposed = false}>
