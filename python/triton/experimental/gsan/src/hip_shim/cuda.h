// hip_shim/cuda.h — dummy cuda.h for building GSanAllocator.cc on ROCm.
// All actual CUDA types/functions are remapped by GSanAllocatorHIP.cc
// before this file is included; we just need the include to succeed.
