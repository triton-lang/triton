/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "device_launch_parameters.h"
#include <builtin_types.h>
#include <cuda_device_runtime_api.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __NVCC__
#define __DEVICE__ __device__ inline
#else
#define __DEVICE__
#endif

#ifndef BARRIER_RANDOM_DELAY
#define BARRIER_RANDOM_DELAY 0
#endif

#if BARRIER_RANDOM_DELAY
__DEVICE__ uint64_t random_avalanche(uint64_t r) {
  r ^= r >> 33;
  r *= 0xff51afd7ed558ccd;
  r ^= r >> 33;
  r *= 0xc4ceb9fe1a85ec53;
  r ^= r >> 33;
  return r;
}

__DEVICE__ uint64_t random_stateless_uint64() {
  uint64_t r = blockIdx.z << 27 ^ blockIdx.y << 22 ^ blockIdx.x;
  r = r << 32 ^ threadIdx.z << 20 ^ threadIdx.y << 10 ^ threadIdx.x;

  uint64_t timer;
  asm volatile("mov.u64 %0, %%globaltimer;\n\t" : "=l"(timer) : : "memory");

  r ^= timer;
  return random_avalanche(r);
}
#endif

__DEVICE__ void random_stateless_delay() {
#if BARRIER_RANDOM_DELAY
  uint64_t r = random_stateless_uint64();

  // About 2 milliseconds.
  uint32_t ns = r >> (64 - 21);
  asm volatile("nanosleep.u32 %0;\n\t" : : "r"(ns));
#endif
}

__DEVICE__ __attribute__((__always_inline__)) uint64_t
__nv_get_wgmma_desc(uint32_t smem_nvvm_pointer, uint32_t mode,
                    uint32_t height) {
  uint64_t desc = 0;
  uint64_t swizzling = (mode == 1 ? 128 : mode == 2 ? 64 : 32);
  uint64_t smem_address_bit = smem_nvvm_pointer;

  uint64_t stride_dimension = swizzling << 3 >> 4;
  uint64_t leading_dimension = height * swizzling >> 4;
  // [benzh] from cutlass
  uint64_t base_offset = 0; //(smem_address_bit >> 7) % (swizzling >> 4);
  uint64_t start_addr = (smem_address_bit << 46) >> 50;

  desc |= ((uint64_t)mode) << 62;
  desc |= stride_dimension << 32;
  desc |= leading_dimension << 16;
  desc |= base_offset << 49;
  desc |= start_addr;

  return desc;
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n");
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n");
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n");
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_mbarrier_init(uint32_t bar, uint32_t expected, uint32_t pred) {
  if (pred) {
    asm volatile("{\n\t"
                 "mbarrier.init.shared.b64 [%0], %1;\n\t"
                 "}"
                 :
                 : "r"(bar), "r"(expected));
  }
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_mbarrier_wait(uint32_t bar, uint32_t phase) {
  uint32_t large_val = 0x989680;
  asm volatile("{\n\t"
               ".reg .pred                P1; \n\t"
               "LAB_WAIT: \n\t"
               "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1, %2; \n\t"
               "@P1                       bra.uni DONE; \n\t"
               "bra.uni                   LAB_WAIT; \n\t"
               "DONE: \n\t"
               "}"
               :
               : "r"(bar), "r"(phase), "r"(large_val));
  random_stateless_delay();
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_mbarrier_arrive_normal(uint32_t bar, uint32_t pred) {
  random_stateless_delay();
  if (pred) {
    asm volatile("{\n\t"
                 "mbarrier.arrive.shared.b64 _, [%0];\n\t"
                 "}"
                 :
                 : "r"(bar));
  }
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_mbarrier_arrive_cp_async(uint32_t bar, uint32_t pred) {
  random_stateless_delay();
  if (pred) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];"
                 :
                 : "r"(bar));
  }
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_mbarrier_arrive_expect_tx(uint32_t bar, uint32_t txCount, uint32_t pred) {
  random_stateless_delay();
  if (pred) {
    asm volatile("{\n\t"
                 "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n\t"
                 "}"
                 :
                 : "r"(bar), "r"(txCount));
  }
}

// for warp special empty barrier.
__DEVICE__ __attribute__((__always_inline__)) void
__nv_mbarrier_arrive_remote(uint32_t bar, uint32_t ctaId, uint32_t pred) {
  random_stateless_delay();
  if (pred) {
    asm volatile("{\n\t"
                 ".reg .b32 remAddr32;\n\t"
                 "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
                 "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
                 "}"
                 :
                 : "r"(bar), "r"(ctaId));
  }
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_fence_mbarrier_init() {
  asm volatile("{\n\t"
               "fence.mbarrier_init.release.cluster; \n"
               "}" ::
                   : "memory");
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_bar_arrive(uint32_t bar, uint32_t numThreads) {
  random_stateless_delay();
  asm volatile("bar.arrive %0, %1;\n" ::"r"(bar), "r"(numThreads));
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_bar_wait(uint32_t bar, uint32_t numThreads) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(bar), "r"(numThreads));
  random_stateless_delay();
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_cga_barrier_sync() {
  asm volatile("barrier.cluster.sync.aligned;\n" ::);
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_cga_barrier_arrive() {
  asm volatile("barrier.cluster.arrive;\n" ::);
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_cga_barrier_wait() {
  asm volatile("barrier.cluster.wait;\n" ::);
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_cluster_arrive() {
  asm volatile("barrier.cluster.arrive.aligned;\n" : :);
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;\n" : :);
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_sts64(uint32_t ptr, uint32_t d0, uint32_t d1) {
  asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n"
               :
               : "r"(ptr), "r"(d0), "r"(d1));
}

__DEVICE__ __attribute__((__always_inline__)) uint32_t
__nv_offset_of_sts64(uint32_t threadIdx, uint32_t rowOfWarp, int32_t elemIdx,
                     uint32_t rowStride) {
  uint32_t perPhase = 0;
  uint32_t maxPhase = 0;
  if (rowStride == 128) {
    perPhase = 1;
    maxPhase = 8;
  } else if (rowStride == 64) {
    perPhase = 2;
    maxPhase = 4;
  } else if (rowStride == 32) {
    perPhase = 4;
    maxPhase = 2;
  }

  uint32_t laneId = threadIdx & 0x1f;

  uint32_t myRow = ((elemIdx >> 1) & 0x1) * 8 + laneId / 4;
  uint32_t myCol = (elemIdx / 4) * 8 + (laneId % 4) * 2;
  myRow += rowOfWarp;

  uint32_t phase = (myRow / perPhase) % maxPhase;

  uint32_t lineOffset = (myRow % perPhase) * rowStride + myCol * 4;
  uint32_t colOffset = ((lineOffset / 16) ^ phase) * 16 + lineOffset % 16;
  uint32_t offset = (myRow / perPhase) * 128 + colOffset;

  return offset;
}
