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

// GMMA expects data to be in TN format. if A is column major, transa should be
// set GMMA expects data to be in TN format. if B is row major, transb should be
// set
__DEVICE__ __attribute__((__always_inline__)) float32
__nv_wgmma_m64n64k16_f32_f16_f16_row_col(const uint64_t desc_a,
                                         const uint64_t desc_b, float32 acc) {
  const uint32_t scale_d = 1;
  asm volatile("{\n"
               ".reg .pred p;\n\t"
               "setp.eq.u32 p, %34, 1;\n\t"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
               "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
               " %8, %9, %10, %11, %12, %13, %14, %15,\n"
               " %16, %17, %18, %19, %20, %21, %22, %23,\n"
               " %24, %25, %26, %27, %28, %29, %30, %31},\n"
               "%32, \n"
               "%33, \n"
               "p, 1, 1, 0, 0;\n"
               "}"
               : "+f"(acc.d0), "+f"(acc.d1), "+f"(acc.d2), "+f"(acc.d3),
                 "+f"(acc.d4), "+f"(acc.d5), "+f"(acc.d6), "+f"(acc.d7),
                 "+f"(acc.d8), "+f"(acc.d9), "+f"(acc.d10), "+f"(acc.d11),
                 "+f"(acc.d12), "+f"(acc.d13), "+f"(acc.d14), "+f"(acc.d15),
                 "+f"(acc.d16), "+f"(acc.d17), "+f"(acc.d18), "+f"(acc.d19),
                 "+f"(acc.d20), "+f"(acc.d21), "+f"(acc.d22), "+f"(acc.d23),
                 "+f"(acc.d24), "+f"(acc.d25), "+f"(acc.d26), "+f"(acc.d27),
                 "+f"(acc.d28), "+f"(acc.d29), "+f"(acc.d30), "+f"(acc.d31)
               : "l"(desc_a), "l"(desc_b), "r"(scale_d));

  return acc;
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_bar_cta_sync(uint32_t bar, uint32_t numThreads) {
  asm volatile("bar.sync %0, %1;" : : "r"(bar), "r"(numThreads));
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_bar_cta_sync_all(uint32_t bar) {
  asm volatile("bar.sync %0;" : : "r"(bar));
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
}

__DEVICE__ __attribute__((__always_inline__)) int
__nv_mbarrier_peek(uint32_t bar, uint32_t phase) {
  int ready = 0;
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "mbarrier.try_wait.shared.b64 p, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, p;\n\t"
               "}"
               : "=r"(ready)
               : "r"(bar), "l"((unsigned long long)phase)
               : "memory");
  return ready;
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_mbarrier_arrive_normal(uint32_t bar, uint32_t pred) {
  if (pred) {
    asm volatile("{\n\t"
                 "mbarrier.arrive.shared.b64 _, [%0];\n\t"
                 "}"
                 :
                 : "r"(bar));
  }
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_mbarrier_arrive_cp_async(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" : : "r"(bar));
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_mbarrier_arrive_expect_tx(uint32_t bar, uint32_t txCount, uint32_t pred) {
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
__nv_fence_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n");
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_fence_async_shared_cluster() {
  asm volatile("fence.proxy.async.shared::cluster;\n");
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_cp_async_bulk(char *gmem_ptr, unsigned smem_ptr, unsigned barrier,
                   int bytes, uint32_t pred) {
  if (pred) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
                 "bytes [%0], [%1], %2, [%3];\n"
                 :
                 : "r"(smem_ptr), "l"(gmem_ptr), "r"(bytes), "r"(barrier)
                 : "memory");
  }
}
__DEVICE__ __attribute__((__always_inline__)) void
__nv_tma_load_tiled_2d(const uint64_t p_tma_desc, uint32_t dst_smem,
                       uint32_t barrier, int32_t c0, int32_t c1,
                       unsigned long long mem_desc, uint32_t pred) {
  if (pred) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx"
        "::bytes.L2::cache_hint [%0], [%1, {%2, %3}], "
        "[%4], %5;\n"
        :
        : "r"(dst_smem), "l"(p_tma_desc), "r"(c0), "r"(c1), "r"(barrier),
          "l"(mem_desc)
        : "memory");
  }
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_tma_load_tiled_mcast_2d(
    const uint64_t p_tma_desc, uint32_t dst_smem, uint32_t barrier, int32_t c0,
    int32_t c1, unsigned long long mem_desc, uint16_t mcast, uint32_t pred) {
  if (pred) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes.multicast::cluster.L2::cache_hint"
                 " [%0], [%1, {%2, %3}], [%4], %5, %6;"
                 :
                 : "r"(dst_smem), "l"(p_tma_desc), "r"(c0), "r"(c1),
                   "r"(barrier), "h"(mcast), "l"(mem_desc)
                 : "memory");
  }
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_tma_load_tiled_4d(const uint64_t p_tma_desc, uint32_t dst_smem,
                       uint32_t barrier, int32_t c0, int32_t c1, int32_t c2,
                       int32_t c3, unsigned long long mem_desc, uint32_t pred) {
  if (pred) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx"
        "::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5}], "
        "[%6], %7;\n"
        :
        : "r"(dst_smem), "l"(p_tma_desc), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
          "r"(barrier), "l"(mem_desc)
        : "memory");
  }
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_stmatrix_x1(uint32_t ptr, const uint32_t d0) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n" ::"r"(ptr),
      "r"(d0));
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_stmatrix_x2(uint32_t ptr, const uint32_t d0, const uint32_t d1) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n" ::"r"(ptr),
      "r"(d0), "r"(d1));
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_stmatrix_x4(uint32_t ptr, const uint32_t d0, const uint32_t d1,
                 const uint32_t d2, const uint32_t d3) {
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::
          "r"(ptr),
      "r"(d0), "r"(d1), "r"(d2), "r"(d3));
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_async_group_commit() {
  asm volatile("cp.async.bulk.commit_group;");
}

__DEVICE__ __attribute__((__always_inline__)) void __nv_async_group_wait0() {
  asm volatile("cp.async.bulk.wait_group %0;" : : "n"(0) : "memory");
}

__DEVICE__ __attribute__((__always_inline__)) uint32_t
__nv_dsmem_addr(uint32_t buffer_ptr, uint32_t ctaid) {
  uint32_t buffer_ptr_;
  asm volatile("{\n\t"
               "mapa.shared::cluster.u32 %0, %1, %2;\n\t"
               "}"
               : "=r"(buffer_ptr_)
               : "r"(buffer_ptr), "r"(ctaid));
  return buffer_ptr_;
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_bar_arrive(uint32_t bar, uint32_t numThreads) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(bar), "r"(numThreads));
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_bar_wait(uint32_t bar, uint32_t numThreads) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(bar), "r"(numThreads));
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
__nv_tma_store_tiled_2d(const uint64_t p_tma_desc, int32_t src_smem, int32_t c0,
                        int32_t c1, uint32_t pred) {
  if (pred) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
                 "[%0, {%2, %3}], [%1];\n"
                 :
                 : "l"(p_tma_desc), "r"(src_smem), "r"(c0), "r"(c1)
                 : "memory");
  }
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_tma_store_tiled_3d(const uint64_t p_tma_desc, uint32_t src_smem,
                        int32_t c0, int32_t c1, int32_t c2, uint32_t pred) {
  if (pred) {
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group"
                 "[%0, {%2, %3, %4}], [%1];\n"
                 :
                 : "l"(p_tma_desc), "r"(src_smem), "r"(c0), "r"(c1), "r"(c2)
                 : "memory");
  }
}

__DEVICE__ __attribute__((__always_inline__)) void
__nv_tma_store_tiled_4d(const uint64_t p_tma_desc, uint32_t src_smem,
                        int32_t c0, int32_t c1, int32_t c2, int32_t c3,
                        uint32_t pred) {
  if (pred) {
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
                 "[%0, {%2, %3, %4, %5}], [%1];\n"
                 :
                 : "l"(p_tma_desc), "r"(src_smem), "r"(c0), "r"(c1), "r"(c2),
                   "r"(c3)
                 : "memory");
  }
}
__DEVICE__ __attribute__((__always_inline__)) uint32_t
__nv_offset_of_stmatrix_v4(uint32_t threadIdx, uint32_t rowOfWarp,
                           uint32_t elemIdx, uint32_t leadingDimOffset,
                           uint32_t rowStride) {
  uint32_t perPhase = 0;
  uint32_t maxPhase = 0;
  if (rowStride == 64) {
    perPhase = 1;
    maxPhase = 8;
  } else if (rowStride == 32) {
    perPhase = 2;
    maxPhase = 4;
  } else if (rowStride == 16) {
    perPhase = 4;
    maxPhase = 2;
  }

  uint32_t iterOfCol = elemIdx / 8;

  uint32_t myRow = rowOfWarp + (threadIdx & 0xf);
  uint32_t myCol = ((threadIdx >> 4) & 0x1) * 8;
  myCol = myCol + iterOfCol * 16;

  uint32_t offset0 = (myCol / rowStride) * leadingDimOffset;
  myCol = myCol % rowStride;

  uint32_t phase = (myRow / perPhase) % maxPhase;

  uint32_t lineOffset = (myRow % perPhase) * rowStride + myCol;
  uint32_t colOffset = ((lineOffset / 8) ^ phase) * 8 + lineOffset % 8;
  uint32_t offset1 = (myRow / perPhase) * 64 + colOffset;

  return offset1 + offset0;
}

__DEVICE__ __attribute__((__always_inline__)) uint32_t
__nv_offset_of_stmatrix_v4_no_swizzle(uint32_t threadIdx, uint32_t rowOfWarp,
                                      uint32_t elemIdx, uint32_t rowStride) {
  uint32_t iterOfCol = elemIdx / 4;
  uint32_t myRow = rowOfWarp + (threadIdx & 0xf);
  uint32_t myCol = ((threadIdx >> 4) & 0x1) * 8;

  myCol = myCol + iterOfCol * 16;
  uint32_t offset = myRow * rowStride + myCol * 2;
  return offset;
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

__DEVICE__ __attribute__((__always_inline__)) uint32_t
__nv_cvt_pack(uint16_t d0, uint16_t d1) {
  uint32_t ret;
  asm volatile("cvt.pack.sat.u16.s32 %0, %1, %2;\n"
               : "=r"(ret)
               : "r"(d0), "r"(d1));
  return ret;
}
