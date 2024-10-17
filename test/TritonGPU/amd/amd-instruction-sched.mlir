// RUN: triton-opt %s -split-input-file -triton-amdgpu-insert-instruction-sched-hints -allocate-shared-memory -convert-scf-to-cf -convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s -check-prefix=INSTR_INSERTION
// RUN: triton-opt %s -split-input-file -triton-amdgpu-insert-instruction-sched-hints -allocate-shared-memory -convert-scf-to-cf -convert-triton-amdgpu-to-llvm=arch=gfx942 -triton-amdgpu-lower-insert-instruction-sched-hints=variant="iglp0" | FileCheck %s -check-prefix=LOWER_IGLP0

#shared0_ex0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#mma0_ex0 = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = false}>

#blocked0_ex1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1_ex1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2_ex1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared0_ex1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
#shared1_ex1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#mma0_ex1 = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = false}>
#dot0_ex1 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma0_ex1, kWidth = 8}>
#dot1_ex1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma0_ex1, kWidth = 8}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // LOWER_IGLP0-LABEL: test_instruction_hints_lowering
  tt.func @test_instruction_hints_lowering(
    %arg0: tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma0_ex0, kWidth = 16}>>,
    %arg1: tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma0_ex0, kWidth = 16}>>,
    %arg2: tensor<32x32xf16, #mma0_ex0>) {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 1 : i32

    scf.for %arg11 = %c0_i32 to %c64_i32 step %c1_i32 iter_args() -> () : i32 {
      // LOWER_IGLP0: llvm.add
      // LOWER_IGLP0-NEXT: %[[OPT_LEVEL:.*]] = llvm.mlir.constant(0 : i32) : i32
      // LOWER_IGLP0-NEXT: llvm.call_intrinsic "llvm.amdgcn.iglp.opt"(%[[OPT_LEVEL]]) : (i32) -> ()
      %0 = tt.dot %arg0, %arg1, %arg2, inputPrecision = ieee : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma0_ex0, kWidth = 16}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma0_ex0, kWidth = 16}>> -> tensor<32x32xf16, #mma0_ex0>
      scf.yield
    }
    tt.return
  }

  // INSTR_INSERTION-LABEL: @test_llvm_instruction_count
  tt.func public @test_llvm_instruction_count(
    %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}
    ) attributes {noinline = false} {

    %cst = arith.constant dense<64> : tensor<256x64xi32, #blocked0_ex1>
    %cst_0 = arith.constant dense<64> : tensor<64x128xi32, #blocked1_ex1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32

    %19 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked0_ex1}>>
    %20 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2_ex1}>>
    %21 = tt.splat %c256_i32 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked0_ex1}>>
    %22 = tt.splat %c256_i32 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2_ex1}>>
    %23 = arith.addi %21, %19 : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked0_ex1}>>
    %24 = arith.addi %22, %20 : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2_ex1}>>

    %26 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1_ex1}>>
    %27 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2_ex1}>>
    %28 = tt.splat %c128_i32 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1_ex1}>>
    %29 = tt.splat %c128_i32 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2_ex1}>>
    %30 = arith.addi %28, %26 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1_ex1}>>
    %31 = arith.addi %29, %27 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2_ex1}>>
    %32 = tt.expand_dims %23 {axis = 1 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked0_ex1}>> -> tensor<256x1xi32, #blocked0_ex1>
    %33 = tt.expand_dims %24 {axis = 1 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2_ex1}>> -> tensor<256x1xi32, #blocked2_ex1>
    %34 = tt.splat %c64_i32 : i32 -> tensor<256x1xi32, #blocked0_ex1>
    %35 = arith.muli %32, %34 : tensor<256x1xi32, #blocked0_ex1>
    %36 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked0_ex1>
    %37 = tt.addptr %36, %35 : tensor<256x1x!tt.ptr<f16>, #blocked0_ex1>, tensor<256x1xi32, #blocked0_ex1>
    %38 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked0_ex1}>>
    %39 = tt.expand_dims %38 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked0_ex1}>> -> tensor<1x64xi32, #blocked0_ex1>
    %40 = tt.broadcast %37 : tensor<256x1x!tt.ptr<f16>, #blocked0_ex1> -> tensor<256x64x!tt.ptr<f16>, #blocked0_ex1>
    %41 = tt.broadcast %39 : tensor<1x64xi32, #blocked0_ex1> -> tensor<256x64xi32, #blocked0_ex1>
    %42 = tt.addptr %40, %41 : tensor<256x64x!tt.ptr<f16>, #blocked0_ex1>, tensor<256x64xi32, #blocked0_ex1>

    %43 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1_ex1}>>
    %44 = tt.expand_dims %43 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1_ex1}>> -> tensor<64x1xi32, #blocked1_ex1>
    %45 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked1_ex1>
    %46 = tt.addptr %45, %44 : tensor<64x1x!tt.ptr<f16>, #blocked1_ex1>, tensor<64x1xi32, #blocked1_ex1>
    %47 = tt.expand_dims %30 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1_ex1}>> -> tensor<1x128xi32, #blocked1_ex1>
    %48 = tt.expand_dims %31 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2_ex1}>> -> tensor<1x128xi32, #blocked2_ex1>
    %49 = tt.splat %c64_i32 : i32 -> tensor<1x128xi32, #blocked1_ex1>
    %50 = arith.muli %47, %49 : tensor<1x128xi32, #blocked1_ex1>
    %51 = tt.broadcast %46 : tensor<64x1x!tt.ptr<f16>, #blocked1_ex1> -> tensor<64x128x!tt.ptr<f16>, #blocked1_ex1>
    %52 = tt.broadcast %50 : tensor<1x128xi32, #blocked1_ex1> -> tensor<64x128xi32, #blocked1_ex1>
    %53 = tt.addptr %51, %52 : tensor<64x128x!tt.ptr<f16>, #blocked1_ex1>, tensor<64x128xi32, #blocked1_ex1>

    %56 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x256x64xf16, #shared0_ex1, #triton_gpu.shared_memory, mutable>
    %57 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x64x128xf16, #shared1_ex1, #triton_gpu.shared_memory, mutable>

    %cst_1 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma0_ex1>

    %cc0_i1 = arith.constant 1 : i1
    %59 = tt.splat %cc0_i1 : i1 -> tensor<256x64xi1, #blocked0_ex1>
    %60 = tt.load %42, %59 : tensor<256x64x!tt.ptr<f16>, #blocked0_ex1>
    %61 = tt.splat %cc0_i1 : i1 -> tensor<64x128xi1, #blocked1_ex1>
    %62 = tt.load %53, %61 : tensor<64x128x!tt.ptr<f16>, #blocked1_ex1>

    %63 = triton_gpu.memdesc_subview %56[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x256x64xf16, #shared0_ex1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<256x64xf16, #shared0_ex1, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %60, %63 : tensor<256x64xf16, #blocked0_ex1> -> !tt.memdesc<256x64xf16, #shared0_ex1, #triton_gpu.shared_memory, mutable>
    %64 = triton_gpu.memdesc_subview %57[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x64x128xf16, #shared1_ex1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x128xf16, #shared1_ex1, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %62, %64 : tensor<64x128xf16, #blocked1_ex1> -> !tt.memdesc<64x128xf16, #shared1_ex1, #triton_gpu.shared_memory, mutable>

    %66:5 = scf.for %arg11 = %c0_i32 to %c63_i32 step %c1_i32 iter_args(
      %arg12 = %cst_1,
      %arg13 = %42,
      %arg14 = %53,
      %arg16 = %63,
      %arg17 = %64) -> (
        tensor<256x128xf32, #mma0_ex1>,
        tensor<256x64x!tt.ptr<f16>, #blocked0_ex1>,
        tensor<64x128x!tt.ptr<f16>, #blocked1_ex1>,
        !tt.memdesc<256x64xf16, #shared0_ex1, #triton_gpu.shared_memory, mutable>,
        !tt.memdesc<64x128xf16, #shared1_ex1, #triton_gpu.shared_memory, mutable>)  : i32 {

      %82 = triton_gpu.local_load %arg16 : !tt.memdesc<256x64xf16, #shared0_ex1, #triton_gpu.shared_memory, mutable> -> tensor<256x64xf16, #dot0_ex1>
      %83 = triton_gpu.local_load %arg17 : !tt.memdesc<64x128xf16, #shared1_ex1, #triton_gpu.shared_memory, mutable> -> tensor<64x128xf16, #dot1_ex1>

      // INSTR_INSERTION: amdgpu.instruction_sched_hint
      // INSTR_INSERTION-SAME: numDsReadsA = #amdgpu.InstCounter<16, vector<8xf16>>
      // INSTR_INSERTION-SAME: numDsReadsB = #amdgpu.InstCounter<8, vector<8xf16>>
      // INSTR_INSERTION-SAME: numDsWritesA = #amdgpu.InstCounter<8, vector<8xf16>>
      // INSTR_INSERTION-SAME: numDsWritesB = #amdgpu.InstCounter<4, vector<8xf16>>
      // INSTR_INSERTION-SAME: numGlobalLoadsA = #amdgpu.InstCounter<8, vector<8xf16>>
      // INSTR_INSERTION-SAME: numGlobalLoadsB = #amdgpu.InstCounter<4, vector<8xf16>>
      // INSTR_INSERTION-SAME: numMMAs = #amdgpu.InstCounter<64, tensor<32x32x8xf16>>

      %84 = tt.dot %82, %83, %arg12 : tensor<256x64xf16, #dot0_ex1> * tensor<64x128xf16, #dot1_ex1> -> tensor<256x128xf32, #mma0_ex1>
      %85 = tt.addptr %arg13, %cst : tensor<256x64x!tt.ptr<f16>, #blocked0_ex1>, tensor<256x64xi32, #blocked0_ex1>
      %86 = tt.addptr %arg14, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked1_ex1>, tensor<64x128xi32, #blocked1_ex1>
      %87 = tt.load %85 : tensor<256x64x!tt.ptr<f16>, #blocked0_ex1>
      %88 = tt.load %86 : tensor<64x128x!tt.ptr<f16>, #blocked1_ex1>
      %89 = triton_gpu.memdesc_subview %56[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x256x64xf16, #shared0_ex1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<256x64xf16, #shared0_ex1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %87, %89 : tensor<256x64xf16, #blocked0_ex1> -> !tt.memdesc<256x64xf16, #shared0_ex1, #triton_gpu.shared_memory, mutable>
      %90 = triton_gpu.memdesc_subview %57[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x64x128xf16, #shared1_ex1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x128xf16, #shared1_ex1, #triton_gpu.shared_memory, mutable>
      triton_gpu.local_store %88, %90 : tensor<64x128xf16, #blocked1_ex1> -> !tt.memdesc<64x128xf16, #shared1_ex1, #triton_gpu.shared_memory, mutable>

      scf.yield %84, %85, %86, %89, %90 :
        tensor<256x128xf32, #mma0_ex1>,
        tensor<256x64x!tt.ptr<f16>, #blocked0_ex1>,
        tensor<64x128x!tt.ptr<f16>, #blocked1_ex1>,
        !tt.memdesc<256x64xf16, #shared0_ex1, #triton_gpu.shared_memory, mutable>,
        !tt.memdesc<64x128xf16, #shared1_ex1, #triton_gpu.shared_memory, mutable>
    }
    tt.return
  }
}
