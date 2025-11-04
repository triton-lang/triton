// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 -cse | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @tc_gen5_mma
  // CHECK: %[[WID:.+]] = nvgpu.warp_id
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[P0:.+]] = llvm.icmp "eq" %[[WID]], %[[C0]] : i32
  // CHECK: %[[P1:.+]] = llvm.and %{{.*}}, %[[P0]]  : i1
  // CHECK: llvm.cond_br %[[P1]]
  // CHECK: %[[E:.+]] = nvvm.elect.sync -> i1
  // CHECK-COUNT-8: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[E]]
  // CHECK: %[[PRED:.+]] = llvm.and %arg6, %[[E]]
  // CHECK: @$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];", "b,l" %[[PRED]]
  tt.func @tc_gen5_mma(%a: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async} :
       !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_multi_m_n
  // CHECK: %[[TMEM_BASE:.+]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<3> to i32
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[TMEM_BASE]]
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 64 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[TMEM_BASE]]
  // 1048576 = row << 16 + col = 16 << 16 + 0
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 1048576 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[TMEM_BASE]]
  // 1048640 = row << 16 + col = 16 << 16 + 64
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 1048640 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[TMEM_BASE]]

  tt.func @tc_gen5_mma_multi_m_n(%a: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async} :
       !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 2], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16, CTAsPerCGA = [1, 2], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16, CTAsPerCGA = [1, 2], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [2], CTASplitNum = [1], CTAOrder = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, unpacked = true, CTASplitN = 2>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_multi_ctas
  // CHECK: %[[TMEM_BASE:.+]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<3> to i32
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 0 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[TMEM_BASE]]
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 32 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[TMEM_BASE]]
  // 1048576 = row << 16 + col = 16 << 16 + 0
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 1048576 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[TMEM_BASE]]
  // 1048640 = row << 16 + col = 16 << 16 + 32
  // CHECK: @$5 tcgen05.mma.cta_group::1.kind::f16 [ $0 + 1048608 ], $1, $2, $3, $4;", "r,l,l,r,b,b" %[[TMEM_BASE]]

  tt.func @tc_gen5_mma_multi_ctas(%a: !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async} :
       !ttg.memdesc<128x16xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<16x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld
  // CHECK: nvgpu.tensor_memory_base
  // CHECK: tcgen05.st.sync.aligned.32x32b.x128.b32
  // CHECK: tcgen05.wait::st.sync.aligned
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x128.b32
  // CHECK: tcgen05.wait::ld.sync.aligned
  tt.func public @tensor_memory_ld(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    tt.return
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [8, 0], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[32, 0], [64, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_16x256
  // CHECK: tcgen05.st.sync.aligned.16x256b.x16.b32
  // CHECK: tcgen05.st.sync.aligned.16x256b.x16.b32
  // CHECK: tcgen05.ld.sync.aligned.16x256b.x16.b32
  // CHECK: tcgen05.ld.sync.aligned.16x256b.x16.b32
  tt.func public @tensor_memory_ld_16x256(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #linear>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_allocation
  // CHECK: llvm.mlir.constant(4194306 : i32) : i32
  tt.func public @tensor_memory_allocation() {
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 2 : i32, tensor_memory_row_offset = 64 : i32} : () -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_m64
  // CHECK: nvgpu.tensor_memory_base
  // CHECK: tcgen05.st.sync.aligned.16x32bx2.x64.b32
  // CHECK: tcgen05.st.sync.aligned.16x32bx2.x64.b32
  // CHECK: tcgen05.wait::st.sync.aligned
  // CHECK: tcgen05.ld.sync.aligned.16x32bx2.x64.b32
  // CHECK: tcgen05.ld.sync.aligned.16x32bx2.x64.b32
  // CHECK: tcgen05.wait::ld.sync.aligned
  tt.func public @tensor_memory_ld_m64(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_unpack_f16
  // CHECK: nvgpu.tensor_memory_base
  // CHECK: tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32
  // CHECK: tcgen05.wait::st.sync.aligned
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32
  // CHECK: tcgen05.wait::ld.sync.aligned
  tt.func public @tensor_memory_unpack_f16() {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked1>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf16, #blocked1>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale
  // CHECK: %[[TMEM_BASE:.+]] = llvm.ptrtoint %arg2 : !llvm.ptr<3> to i32
  // CHECK: %[[WID:.+]] = nvgpu.warp_id
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[P0:.+]] = llvm.icmp "eq" %[[WID]], %[[C0]] : i32
  // CHECK: %[[P1:.+]] = llvm.and %{{.*}}, %[[P0]]  : i1
  // CHECK: llvm.cond_br %[[P1]]
  // CHECK: %[[DESC0:.+]] = llvm.mlir.constant(144708608 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[TMEM_BASE]], %{{.+}}, %{{.+}}, %[[DESC0]], %{{.+}}, %{{.+}}, %arg5
  // CHECK: %[[TRUE:.+]] = llvm.mlir.constant(true) : i1
  // CHECK: %[[DESC1:.+]] = llvm.mlir.constant(681579536 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[TMEM_BASE]], %{{.+}}, %{{.+}}, %[[DESC1]], %{{.+}}, %{{.+}}, %[[TRUE]]
  tt.func @tc_gen5_mma_block_scale(%a: !ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<32x128xi8, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e4m3 rhs = e2m1, %barrier[%barrierPred] {is_async} :
    !ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
    !ttg.memdesc<32x128xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale_fp4_a
  // CHECK: %[[DESC0:.+]] = llvm.mlir.constant(144769664 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %{{.+}}, %{{.+}}, %{{.+}}, %[[DESC0]]
  // CHECK: %[[DESC1:.+]] = llvm.mlir.constant(681640592 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %{{.+}}, %{{.+}}, %{{.+}}, %[[DESC1]]
  // CHECK: %[[DESC2:.+]] = llvm.mlir.constant(1218511520 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %{{.+}}, %{{.+}}, %{{.+}}, %[[DESC2]]
  // CHECK: %[[DESC3:.+]] = llvm.mlir.constant(1755382448 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %{{.+}}, %{{.+}}, %{{.+}}, %[[DESC3]]
  tt.func @tc_gen5_mma_block_scale_fp4_a(%a: !ttg.memdesc<128x64xi8, #shared1, #ttg.shared_memory>,
                       %b: !ttg.memdesc<128x128xi8, #shared, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e4m3, %barrier[%barrierPred] {is_async} :
    !ttg.memdesc<128x64xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<128x128xi8, #shared, #ttg.shared_memory>,
    !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<128x2xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CTAsPerCGA = [2, 1], CTASplitNum = [2, 1], CTAOrder = [1, 0]}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16, CTAsPerCGA = [1, 2], CTASplitNum = [1, 2], CTAOrder = [1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [2], CTASplitNum = [1], CTAOrder = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true, CTASplitM = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_2ctas
  tt.func @tc_gen5_mma_2ctas(%a: !ttg.memdesc<256x32xf16, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    // CHECK: tcgen05.mma.cta_group::2.kind::f16
    // CHECK: tcgen05.mma.cta_group::2.kind::f16
    // CHECK: tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64
    ttng.tc_gen5_mma %a, %b, %c, %useAcc, %pred, %barrier[%barrierPred] {is_async, two_ctas} :
       !ttg.memdesc<256x32xf16, #shared, #ttg.shared_memory>,
       !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>,
       !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread=[1, 4], threadsPerWarp=[32, 1], warpsPerCTA=[4, 1], order=[0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {

tt.func public @tmem_copy_2d(%src: !ttg.memdesc<256x16xi8, #shared, #ttg.shared_memory>,
                             %dst: !ttg.memdesc<128x32xi8, #tmem_scales, #ttng.tensor_memory, mutable>,
		                         %barrier: !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory>) {
  // CHECK-COUNT-8: tcgen05.cp.cta_group::1.warpx4.32x128b
  // CHECK: tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64
  ttng.tmem_copy %src, %dst, %barrier : !ttg.memdesc<256x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<128x32xi8, #tmem_scales, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #ttg.shared_memory>
  tt.return
}

tt.func public @tmem_copy_2d_slice(%src: !ttg.memdesc<256x16xi8, #shared, #ttg.shared_memory, 1024x16>,
                                   %dst: !ttg.memdesc<128x32xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
  // CHECK: [[OFF0:%.*]] = llvm.extractvalue %arg0[1]
  // CHECK: [[OFF1:%.*]] = llvm.extractvalue %arg0[2]
  // CHECK-COUNT-8: tcgen05.cp.cta_group::1.warpx4.32x128b
  ttng.tmem_copy %src, %dst : !ttg.memdesc<256x16xi8, #shared, #ttg.shared_memory, 1024x16>, !ttg.memdesc<128x32xi8, #tmem_scales, #ttng.tensor_memory, mutable>
  tt.return
}

}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale_nvfp4
  // CHECK: %[[TMEM_BASE:.+]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<3> to i32
  // CHECK: %[[DESC0:.+]] = llvm.mlir.constant(138413184 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[TMEM_BASE]], %{{.+}}, %{{.+}}, %[[DESC0]]
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[TMEM_BASE]], %{{.+}}, %{{.+}}, %[[DESC0]]
  tt.func @tc_gen5_mma_block_scale_nvfp4(%a: !ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<64x256xi8, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<256x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e2m1, %barrier[%barrierPred] {is_async} :
    !ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
    !ttg.memdesc<64x256xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<256x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tc_gen5_mma_block_scale_mxfp4
  // CHECK-DAG: %[[TMEM_BASE:.+]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<3> to i32
  // CHECK: %[[DESC0:.+]] = llvm.mlir.constant(146801792 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[TMEM_BASE]], %{{.+}}, %{{.+}}, %[[DESC0]]
  // CHECK: %[[DESC1:.+]] = llvm.mlir.constant(1220543648 : i32) : i32
  // CHECK: @$7 tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [ $0 + 0 ], $1, $2, $3, [ $4 + 0 ], [ $5 + 0 ], $6;", "r,l,l,r,r,r,b,b" %[[TMEM_BASE]], %{{.+}}, %{{.+}}, %[[DESC1]]
  tt.func @tc_gen5_mma_block_scale_mxfp4(%a: !ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
                       %b: !ttg.memdesc<64x256xi8, #shared1, #ttg.shared_memory>,
                       %c: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
                       %scale_a: !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
                       %scale_b: !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>,
                       %useAcc: i1,
                       %pred: i1,
                       %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>,
                       %barrierPred: i1) {
    ttng.tc_gen5_mma_scaled %a, %b, %c, %scale_a, %scale_b, %useAcc, %pred lhs = e2m1 rhs = e2m1, %barrier[%barrierPred] {is_async} :
    !ttg.memdesc<128x64xi8, #shared, #ttg.shared_memory>,
    !ttg.memdesc<64x256xi8, #shared1, #ttg.shared_memory>,
    !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
    !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>,
    !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_128x256
  // CHECK-COUNT-4: tcgen05.st.sync.aligned.32x32b.x64.b32
  // CHECK-NOT: tcgen05.st
  // CHECK: tcgen05.wait::st.sync.aligned
  // CHECK-COUNT-4: tcgen05.ld.sync.aligned.32x32b.x64.b32
  // CHECK-NOT: tcgen05.ld
  // CHECK: tcgen05.wait::ld.sync.aligned
  tt.func public @tensor_memory_ld_128x256(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x256xf32, #blocked>) -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttng.tmem_load %0 : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_128x256_8_warps
  // CHECK: tcgen05.st.sync.aligned.32x32b.x128.b32
  // CHECK: tcgen05.wait::st.sync.aligned
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x128.b32
  // CHECK: tcgen05.wait::ld.sync.aligned
  tt.func public @tensor_memory_ld_128x256_8_warps(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %0 = ttng.tmem_alloc %cst_0 {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x256xf32, #blocked>) -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttng.tmem_load %0 : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_256x64_8_warps_blocked
  tt.func public @tensor_memory_ld_256x64_8_warps_blocked(%tmem: !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    // CHECK-COUNT-1: tcgen05.ld.sync.aligned.32x32b.x64.b32
    // CHECK-NOT: tcgen05.ld
    %result = ttng.tmem_load %tmem : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    tt.return
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[32, 0], [64, 0], [16, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_256x64_8_warps_splitM
  tt.func public @tensor_memory_ld_256x64_8_warps_splitM(%tmem: !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    // CHECK-COUNT-2: tcgen05.ld.sync.aligned.16x32bx2.x32.b32
    // CHECK-NOT: tcgen05.ld
    %result = ttng.tmem_load %tmem : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #linear>
    tt.return
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]], warp = [[32, 0], [64, 0], [16, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_128x128_8_warps_splitM
  tt.func public @tensor_memory_ld_128x128_8_warps_splitM(%tmem: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
    // CHECK-COUNT-1: tcgen05.ld.sync.aligned.16x32bx2.x64.b32
    // CHECK-NOT: tcgen05.ld
    %result = ttng.tmem_load %tmem : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    tt.return
  }
}

// -----

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[32, 0], [64, 0], [16, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @tensor_memory_ld_128x64_8_warps_splitM
  tt.func public @tensor_memory_ld_128x64_8_warps_splitM(%tmem: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>) {
    // CHECK-COUNT-1: tcgen05.ld.sync.aligned.16x32bx2.x32.b32
    // CHECK-NOT: tcgen05.ld
    %result = ttng.tmem_load %tmem : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 80 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32} {

// CHECK-LABEL: @tmem_message_maxnreg_80
tt.func public @tmem_message_maxnreg_80(%desc: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>) {
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x32.b32 {{.*}} [$32 + 0]
  // CHECK: tcgen05.ld.sync.aligned.32x32b.x32.b32 {{.*}} [$32 + 32]
  // CHECK-NOT: tcgen05.ld
  ttng.tmem_load %desc : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory> -> tensor<128x64xf32, #blocked>
  tt.return
}

// CHECK-LABEL: @module_constraint_supercedes_local
tt.func public @module_constraint_supercedes_local(%desc: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>) {
  ttg.warp_specialize(%desc) attributes {actualRegisters = array<i32: 256, 256>}
  default {
    // CHECK-COUNT-2: tcgen05.ld.sync.aligned.32x32b.x32.b32
    // CHECK-NOT: tcgen05.ld
    // CHECK: ttg.warp_yield
    ttng.tmem_load %desc : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory> -> tensor<128x64xf32, #blocked>
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>) num_warps(4) {
    // CHECK-COUNT-2: tcgen05.ld.sync.aligned.32x32b.x32.b32
    // CHECK-NOT: tcgen05.ld
    // CHECK: ttg.warp_return
    ttng.tmem_load %arg0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory> -> tensor<128x64xf32, #blocked>
    ttg.warp_return
  } : (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>) -> ()
  tt.return
}

}

module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 256 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32} {

// CHECK-LABEL: @tmem_message_local_constraint
tt.func public @tmem_message_local_constraint(%desc: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>) {
  ttg.warp_specialize(%desc) attributes {actualRegisters = array<i32: 80, 48>}
  default {
    // CHECK: tcgen05.ld.sync.aligned.32x32b.x32.b32 {{.*}} [$32 + 0]
    // CHECK: tcgen05.ld.sync.aligned.32x32b.x32.b32 {{.*}} [$32 + 32]
    // CHECK-NOT: tcgen05.ld
    // CHECK: ttg.warp_yield
    ttng.tmem_load %desc : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory> -> tensor<128x64xf32, #blocked>
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>) num_warps(4) {
    // CHECK: tcgen05.ld.sync.aligned.32x32b.x16.b32 {{.*}} [$16 + 0]
    // CHECK: tcgen05.ld.sync.aligned.32x32b.x16.b32 {{.*}} [$16 + 16]
    // CHECK: tcgen05.ld.sync.aligned.32x32b.x16.b32 {{.*}} [$16 + 32]
    // CHECK: tcgen05.ld.sync.aligned.32x32b.x16.b32 {{.*}} [$16 + 48]
    // CHECK-NOT: tcgen05.ld
    // CHECK: ttg.warp_return
    ttng.tmem_load %arg0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory> -> tensor<128x64xf32, #blocked>
    ttg.warp_return
  } : (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory>) -> ()
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#packed_b16 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 4 : i32, ttg.maxnreg = 128 : i32} {
// CHECK-LABEL: @store_packedb16_2x64xf16
tt.func @store_packedb16_2x64xf16(%arg0: !ttg.memdesc<128x128xf16, #packed_b16, #ttng.tensor_memory, mutable, 1x128x128>, %arg1: tensor<128x128xf16, #blocked>) {
  %true = arith.constant true
  // CHECK: tcgen05.st.sync.aligned.32x32b.x64.b32
  // CHECK-NOT: tcgen05.st
  ttng.tmem_store %arg1, %arg0, %true : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #packed_b16, #ttng.tensor_memory, mutable, 1x128x128>
  tt.return
}
}

module attributes {ttg.target = "cuda:100", "ttg.num-warps" = 4 : i32, ttg.maxnreg = 80 : i32} {
// CHECK-LABEL: @store_packedb16_4x32xf16
tt.func @store_packedb16_4x32xf16(%arg0: !ttg.memdesc<128x128xf16, #packed_b16, #ttng.tensor_memory, mutable, 1x128x128>, %arg1: tensor<128x128xf16, #blocked>) {
  %true = arith.constant true
  // CHECK: tcgen05.st.sync.aligned.32x32b.x32.b32 [$1 + 0]
  // CHECK: tcgen05.st.sync.aligned.32x32b.x32.b32 [$1 + 32]
  // CHECK-NOT: tcgen05.st
  ttng.tmem_store %arg1, %arg0, %true : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #packed_b16, #ttng.tensor_memory, mutable, 1x128x128>
  tt.return
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 32, unpacked = false>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @tc_gen5_mma_lhs_tmem(%arg0: !ttg.memdesc<128x32xf16, #tmem, #ttng.tensor_memory>, %arg1: !ttg.memdesc<32x128xf16, #shared, #smem>, %arg2: !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, %arg3: i1, %arg4: i1, %arg5: !ttg.memdesc<1xi64, #shared1, #smem>, %barrierPred: i1) {
    // CHECK-LABEL: tc_gen5_mma_lhs_tmem
    //       CHECK: tcgen05.mma.cta_group::1.kind::f16
    ttng.tc_gen5_mma %arg0, %arg1, %arg2, %arg3, %arg4, %arg5[%barrierPred] {is_async} :
      !ttg.memdesc<128x32xf16, #tmem, #ttng.tensor_memory>,
      !ttg.memdesc<32x128xf16, #shared, #smem>,
      !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<1xi64, #shared1, #smem>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_commit
tt.func @tc_gen5_commit(%arg0: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %pred: i1) {
  // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: [[IS_WARP_0:%.*]] = llvm.icmp "eq" [[ZERO]], [[ZERO]]
  // CHECK: [[ELECT:%.*]] = nvvm.elect.sync
  // CHECK: [[WARP_PRED:%.*]] = llvm.and [[IS_WARP_0]], [[ELECT]]
  // CHECK: [[PRED:%.*]] = llvm.and %arg1, [[WARP_PRED]]
  // CHECK: @$0 tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$1];", "b,l" [[PRED]]
  ttng.tc_gen5_commit %arg0, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable>
  tt.return
}
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 16, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @reinterpret
tt.func private @reinterpret(%arg0: !ttg.memdesc<128x32xf32, #tmem, #ttng.tensor_memory>) -> !ttg.memdesc<256x32xf16, #tmem, #ttng.tensor_memory> {
  %0 = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<128x32xf32, #tmem, #ttng.tensor_memory> -> !ttg.memdesc<256x32xf16, #tmem, #ttng.tensor_memory>
  // CHECK-NEXT: return %arg0
  tt.return %0 : !ttg.memdesc<256x32xf16, #tmem, #ttng.tensor_memory>
}

}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>
#tmem_unpacked = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_x1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 2, unpacked = false>
#tmem_x1_unpacked = #ttng.tensor_memory_encoding<blockM = 128, blockN = 2, unpacked = true>

#blocked_x1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @subslice_unpacked
tt.func private @subslice_unpacked(%arg0: !ttg.memdesc<128x128xf16, #tmem_unpacked, #ttng.tensor_memory>) -> !ttg.memdesc<128x64xf16, #tmem_unpacked, #ttng.tensor_memory, 128x128> {
  // CHECK: [[OFFSET:%.*]] = llvm.mlir.constant(64 : i32)
  // CHECK: [[PTR:%.*]] = llvm.ptrtoint
  // CHECK: llvm.add [[PTR]], [[OFFSET]]
  %0 = ttng.tmem_subslice %arg0 {N = 64 : i32} : !ttg.memdesc<128x128xf16, #tmem_unpacked, #ttng.tensor_memory> -> !ttg.memdesc<128x64xf16, #tmem_unpacked, #ttng.tensor_memory, 128x128>
  tt.return %0 : !ttg.memdesc<128x64xf16, #tmem_unpacked, #ttng.tensor_memory, 128x128>
}


// CHECK-LABEL: @subslice_packed
tt.func private @subslice_packed(%arg0: !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>) -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, 128x128> {
  // CHECK: [[OFFSET:%.*]] = llvm.mlir.constant(32 : i32)
  // CHECK: [[PTR:%.*]] = llvm.ptrtoint
  // CHECK: llvm.add [[PTR]], [[OFFSET]]
  %0 = ttng.tmem_subslice %arg0 {N = 64 : i32} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory> -> !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, 128x128>
  tt.return %0 : !ttg.memdesc<128x64xf16, #tmem, #ttng.tensor_memory, 128x128>
}

// CHECK-LABEL: @load_store_x1
tt.func @load_store_x1(%arg0: !ttg.memdesc<128x2xf16, #tmem_x1, #ttng.tensor_memory, mutable>) {
  %true = arith.constant true
  // CHECK: [[V:%.*]] = llvm.inline_asm {{.*}}tcgen05.ld.sync{{.*}} (i32) -> i32
  // CHECK: [[V1:%.*]] = llvm.bitcast [[V]] : i32 to i32
  // CHECK: [[F:%.*]] = llvm.bitcast [[V1]] : i32 to vector<2xf16>
  // CHECK: [[E0:%.*]] = llvm.extractelement [[F]]{{.*}} : vector<2xf16>
  // CHECK: [[E1:%.*]] = llvm.extractelement [[F]]{{.*}} : vector<2xf16>
  // CHECK: [[U:%.*]] = llvm.mlir.undef : !llvm.struct<(f16, f16)>
  // CHECK: [[I0:%.*]] = llvm.insertvalue [[E0]], [[U]][0] : !llvm.struct<(f16, f16)>
  // CHECK: [[I1:%.*]] = llvm.insertvalue [[E1]], [[I0]][1] : !llvm.struct<(f16, f16)>
  %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x2xf16, #tmem_x1, #ttng.tensor_memory, mutable> -> tensor<128x2xf16, #blocked_x1>
  ttng.tmem_store %0, %arg0, %true : tensor<128x2xf16, #blocked_x1> -> !ttg.memdesc<128x2xf16, #tmem_x1, #ttng.tensor_memory, mutable>
  tt.return
}

// CHECK-LABEL: @load_store_x1_unpacked
tt.func @load_store_x1_unpacked(%arg0: !ttg.memdesc<128x2xf16, #tmem_x1_unpacked, #ttng.tensor_memory, mutable>) {
  %true = arith.constant true
  // CHECK: [[V:%.*]] = llvm.inline_asm {{.*}}tcgen05.ld.sync{{.*}} (i32) -> i32
  // CHECK: [[V1:%.*]] = llvm.bitcast [[V]] : i32 to i32
  // CHECK: [[F:%.*]] = llvm.bitcast [[V1]] : i32 to vector<2xf16>
  // CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: extractelement [[F]][[[C0]] : i32]
  // CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : i32)
  // CHECK: extractelement [[F]][[[C1]] : i32]
  %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x2xf16, #tmem_x1_unpacked, #ttng.tensor_memory, mutable> -> tensor<128x2xf16, #blocked_x1>
  ttng.tmem_store %0, %arg0, %true : tensor<128x2xf16, #blocked_x1> -> !ttg.memdesc<128x2xf16, #tmem_x1_unpacked, #ttng.tensor_memory, mutable>
  tt.return
}

}

// -----

// CHECK-LABEL: max_reduction
//       CHECK:  %[[M:.+]] = llvm.mlir.constant(-1 : i32) : i32
//       CHECK:   nvvm.redux.sync  fmax %{{.*}}, %[[M]] {nan = true} : f32 -> f32
//       CHECK:   nvvm.barrier0
//       CHECK:   nvvm.shfl.sync bfly
//       CHECK:   nvvm.shfl.sync bfly
//       CHECK:   nvvm.barrier0
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @max_reduction(%arg0: tensor<1x1024xf32, #blocked>) {
    %11 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %15 = arith.maximumf %arg2, %arg3 : f32
      tt.reduce.return %15 : f32
    }) {allocation.offset = 0 : i32} : (tensor<1x1024xf32, #blocked>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: maxnum_reduction
//       CHECK:  %[[M:.+]] = llvm.mlir.constant(-1 : i32) : i32
//       CHECK:   nvvm.redux.sync  fmax %{{.*}}, %[[M]] : f32 -> f32
//       CHECK:   nvvm.barrier0
//       CHECK:   nvvm.shfl.sync bfly
//       CHECK:   nvvm.shfl.sync bfly
//       CHECK:   nvvm.barrier0
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @maxnum_reduction(%arg0: tensor<1x1024xf32, #blocked>) {
    %11 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %15 = arith.maxnumf %arg2, %arg3 : f32
      tt.reduce.return %15 : f32
    }) {allocation.offset = 0 : i32} : (tensor<1x1024xf32, #blocked>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return
  }
}

// -----

#bm64_bn128 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
#bm64_bn64 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, unpacked = true>

#bm64_bn128_packed = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = false>
#bm64_bn64_packed = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, unpacked = false>

#bm64_bn32 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 32, unpacked = true>
#bm64_bn16 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 16, unpacked = true>

#tmem = #ttng.tensor_memory

module attributes {"ttg.target" = "cuda:100", "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @subslice_16x32bx2
tt.func private @subslice_16x32bx2(%arg0: !ttg.memdesc<64x128xf32, #bm64_bn128, #tmem>) -> !ttg.memdesc<64x64xf32, #bm64_bn64, #tmem> {
  // CHECK: [[OFFSET:%.*]] = llvm.mlir.constant(64 : i32)
  // CHECK: [[PTR:%.*]] = llvm.ptrtoint
  // CHECK: llvm.add [[PTR]], [[OFFSET]]
  %0 = ttng.tmem_subslice %arg0 {N = 64 : i32} : !ttg.memdesc<64x128xf32, #bm64_bn128, #tmem> -> !ttg.memdesc<64x64xf32, #bm64_bn64, #tmem>
  tt.return %0 : !ttg.memdesc<64x64xf32, #bm64_bn64, #tmem>
}

// CHECK-LABEL: @subslice_16x32bx2_packed
tt.func private @subslice_16x32bx2_packed(%arg0: !ttg.memdesc<64x128xf16, #bm64_bn128_packed, #tmem>) -> !ttg.memdesc<64x64xf16, #bm64_bn64_packed, #tmem> {
  // CHECK: [[OFFSET:%.*]] = llvm.mlir.constant(32 : i32)
  // CHECK: [[PTR:%.*]] = llvm.ptrtoint
  // CHECK: llvm.add [[PTR]], [[OFFSET]]
  %0 = ttng.tmem_subslice %arg0 {N = 64 : i32} : !ttg.memdesc<64x128xf16, #bm64_bn128_packed, #tmem> -> !ttg.memdesc<64x64xf16, #bm64_bn64_packed, #tmem>
  tt.return %0 : !ttg.memdesc<64x64xf16, #bm64_bn64_packed, #tmem>
}

// CHECK-LABEL: @subslice_16x32bx2_interleaved_block1
tt.func private @subslice_16x32bx2_interleaved_block1(%arg0: !ttg.memdesc<64x128xf32, #bm64_bn32, #tmem>) -> !ttg.memdesc<64x32xf32, #bm64_bn32, #tmem, 64x128> {
  // 16 << 16 => 1048576
  // CHECK: [[OFFSET:%.*]] = llvm.mlir.constant(1048576 : i32)
  // CHECK: [[PTR:%.*]] = llvm.ptrtoint
  // CHECK: llvm.add [[PTR]], [[OFFSET]]
  %0 = ttng.tmem_subslice %arg0 {N = 32 : i32} : !ttg.memdesc<64x128xf32, #bm64_bn32, #tmem> -> !ttg.memdesc<64x32xf32, #bm64_bn32, #tmem, 64x128>
  tt.return %0 : !ttg.memdesc<64x32xf32, #bm64_bn32, #tmem, 64x128>
}

// CHECK-LABEL: @subslice_16x32bx2_interleaved_block0
tt.func private @subslice_16x32bx2_interleaved_block0(%arg0: !ttg.memdesc<64x128xf32, #bm64_bn32, #tmem>) -> !ttg.memdesc<64x16xf32, #bm64_bn16, #tmem, 64x128> {
  // CHECK: [[OFFSET:%.*]] = llvm.mlir.constant(16 : i32)
  // CHECK: [[PTR:%.*]] = llvm.ptrtoint
  // CHECK: llvm.add [[PTR]], [[OFFSET]]
  %0 = ttng.tmem_subslice %arg0 {N = 16 : i32} : !ttg.memdesc<64x128xf32, #bm64_bn32, #tmem> -> !ttg.memdesc<64x16xf32, #bm64_bn16, #tmem, 64x128>
  tt.return %0 : !ttg.memdesc<64x16xf32, #bm64_bn16, #tmem, 64x128>
}

// CHECK-LABEL: @subslice_16x32bx2_interleaved_block0_offset
tt.func private @subslice_16x32bx2_interleaved_block0_offset(%arg0: !ttg.memdesc<64x128xf32, #bm64_bn32, #tmem>) -> !ttg.memdesc<64x16xf32, #bm64_bn16, #tmem, 64x128> {
  // (16 << 16) | 16 => 1048592
  // CHECK: [[OFFSET:%.*]] = llvm.mlir.constant(1048592 : i32)
  // CHECK: [[PTR:%.*]] = llvm.ptrtoint
  // CHECK: llvm.add [[PTR]], [[OFFSET]]
  %0 = ttng.tmem_subslice %arg0 {N = 48 : i32} : !ttg.memdesc<64x128xf32, #bm64_bn32, #tmem> -> !ttg.memdesc<64x16xf32, #bm64_bn16, #tmem, 64x128>
  tt.return %0 : !ttg.memdesc<64x16xf32, #bm64_bn16, #tmem, 64x128>
}

// CHECK-LABEL: @subslice_16x32bx2_interleaved_block4_offset
tt.func private @subslice_16x32bx2_interleaved_block4_offset(%arg0: !ttg.memdesc<64x128xf32, #bm64_bn32, #tmem>) -> !ttg.memdesc<64x16xf32, #bm64_bn16, #tmem, 64x128> {
  // CHECK: [[OFFSET:%.*]] = llvm.mlir.constant(80 : i32)
  // CHECK: [[PTR:%.*]] = llvm.ptrtoint
  // CHECK: llvm.add [[PTR]], [[OFFSET]]
  %0 = ttng.tmem_subslice %arg0 {N = 144 : i32} : !ttg.memdesc<64x128xf32, #bm64_bn32, #tmem> -> !ttg.memdesc<64x16xf32, #bm64_bn16, #tmem, 64x128>
  tt.return %0 : !ttg.memdesc<64x16xf32, #bm64_bn16, #tmem, 64x128>
}

}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 1, unpacked = true>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @load_store_16x32bx1_broadcast
tt.func private @load_store_16x32bx1_broadcast(%arg0: !ttg.memdesc<64x1xf32, #tmem, #ttng.tensor_memory, mutable>) {
  %true = arith.constant true
  // CHECK: tcgen05.ld.sync.aligned.16x32bx2.x1.b32 {$0}, [$1 + 0], 0
  %0 = ttng.tmem_load %arg0 : !ttg.memdesc<64x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x1xf32, #blocked>
  // CHECK: @$0 tcgen05.st.sync.aligned.16x32bx2.x1.b32 [$1 + 0], 0, {$2}
  ttng.tmem_store %0, %arg0, %true : tensor<64x1xf32, #blocked> -> !ttg.memdesc<64x1xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}

}
// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_st
  // CHECK: nvgpu.tensor_memory_base
  // CHECK: tcgen05.st.sync.aligned.32x32b.x128.b32
  // CHECK: tcgen05.wait::st.sync.aligned
  tt.func public @tensor_memory_st(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %true = arith.constant true
    ttng.tmem_store %cst_0, %0, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
