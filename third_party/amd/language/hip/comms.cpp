//
// Created by mlevental on 3/20/25.
//

#include <cstdint>

// -S -emit-llvm --offload-arch=gfx942 -mno-tgsplit
// https://godbolt.org/z/Y35Tf9nfo

__attribute__((used)) __device__ uint32_t
__triton_hip_atom_add_acqrel_agent(int *atomic_address, int *value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_ACQ_REL,
                                __HIP_MEMORY_SCOPE_AGENT);
}
__attribute__((used)) __device__ uint32_t
__triton_hip_atom_add_acqrel_system(int *atomic_address, int *value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_ACQ_REL,
                                __HIP_MEMORY_SCOPE_SYSTEM);
}
__attribute__((used)) __device__ uint32_t
__triton_hip_atom_add_acquire_agent(int *atomic_address, int *value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_ACQUIRE,
                                __HIP_MEMORY_SCOPE_AGENT);
}
__attribute__((used)) __device__ uint32_t
__triton_hip_atom_add_acquire_system(int *atomic_address, int *value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_ACQUIRE,
                                __HIP_MEMORY_SCOPE_SYSTEM);
}
__attribute__((used)) __device__ uint32_t
__triton_hip_atom_add_relaxed_agent(int *atomic_address, int *value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT);
}
__attribute__((used)) __device__ uint32_t
__triton_hip_atom_add_relaxed_system(int *atomic_address, int *value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_SYSTEM);
}

__attribute__((used)) __device__ bool
__triton_hip_atom_cas_acqrel_relaxed_agent(int *atomic_address, int *compare,
                                           int *value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_AGENT);
}
__attribute__((used)) __device__ bool
__triton_hip_atom_cas_acqrel_relaxed_system(int *atomic_address, int *compare,
                                            int *value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_SYSTEM);
}
__attribute__((used)) __device__ bool
__triton_hip_atom_cas_acquire_relaxed_agent(int *atomic_address, int *compare,
                                            int *value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_AGENT);
}
__attribute__((used)) __device__ bool
__triton_hip_atom_cas_acquire_relaxed_system(int *atomic_address, int *compare,
                                             int *value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_SYSTEM);
}
__attribute__((used)) __device__ bool
__triton_hip_atom_cas_relaxed_relaxed_agent(int *atomic_address, int *compare,
                                            int *value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_AGENT);
}
__attribute__((used)) __device__ bool
__triton_hip_atom_cas_relaxed_relaxed_system(int *atomic_address, int *compare,
                                             int *value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_SYSTEM);
}
__attribute__((used)) __device__ bool
__triton_hip_atom_cas_release_relaxed_agent(int *atomic_address, int *compare,
                                            int *value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_RELEASE, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_AGENT);
}
__attribute__((used)) __device__ bool
__triton_hip_atom_cas_release_relaxed_system(int *atomic_address, int *compare,
                                             int *value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_RELEASE, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_SYSTEM);
}

__attribute__((used)) __device__ uint64_t
__triton_hip_load_acquire_agent(uint64_t *input) {
  return __hip_atomic_load(input, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
__attribute__((used)) __device__ uint64_t
__triton_hip_load_acquire_system(uint64_t *input) {
  return __hip_atomic_load(input, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__attribute__((used)) __device__ uint64_t
__triton_hip_load_acquire_workgroup(uint64_t *input) {
  return __hip_atomic_load(input, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP);
}
__attribute__((used)) __device__ uint64_t
__triton_hip_load_relaxed_agent(uint64_t *input) {
  return __hip_atomic_load(input, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__attribute__((used)) __device__ uint64_t
__triton_hip_load_relaxed_system(uint64_t *input) {
  return __hip_atomic_load(input, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__attribute__((used)) __device__ uint64_t
__triton_hip_load_relaxed_workgroup(uint64_t *input) {
  return __hip_atomic_load(input, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_WORKGROUP);
}

__global__ void __triton_hip_store_relaxed_agent(uint64_t *input) {
  uint64_t value{1};
  __hip_atomic_store(input, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__global__ void __triton_hip_store_relaxed_system(uint64_t *input) {
  uint64_t value{1};
  __hip_atomic_store(input, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void __triton_hip_store_relaxed_workgroup(uint64_t *input) {
  uint64_t value{1};
  __hip_atomic_store(input, value, __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_WORKGROUP);
}
__global__ void __triton_hip_store_release_agent(uint64_t *input) {
  uint64_t value{1};
  __hip_atomic_store(input, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
__global__ void __triton_hip_store_release_system(uint64_t *input) {
  uint64_t value{1};
  __hip_atomic_store(input, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void __triton_hip_store_release_workgroup(uint64_t *input) {
  uint64_t value{1};
  __hip_atomic_store(input, value, __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_WORKGROUP);
}

__global__ void __triton_hip_syncthreads() { __syncthreads(); }

__attribute__((used)) __device__ uint32_t
__triton_hip_red_add_release_agent(int *atomic_address, int *value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_RELEASE,
                                __HIP_MEMORY_SCOPE_AGENT);
}
__attribute__((used)) __device__ uint32_t
__triton_hip_red_add_release_system(int *atomic_address, int *value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_RELEASE,
                                __HIP_MEMORY_SCOPE_SYSTEM);
}
