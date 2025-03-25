// RUN: triton-opt --split-input-file %s --verify-diagnostics

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @aref_get_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>
    // expected-error @below {{Number of Batch axes on the aref type does not match the number of indexes}}
    %1 = nvws.aref.get %0 as (%b0 : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
      nvws.aref.return %b0 : !ttg.memdesc<1x64x16xf16, #shared0, #smem>
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>) -> !ttg.memdesc<1x64x16xf16, #shared0, #smem>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @aref_put_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    // expected-error @below {{Aref has different number of arguments than region}}
    nvws.aref.put %0 as (%b0 : !ttg.memdesc<1x64x16xf16, #shared0, #smem>) {
      nvws.aref.return
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>) -> ()
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @aref_get_batch(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    // expected-error @below {{Number of Batch axes on the aref type does not match the number of indexes}}
    %1 = nvws.aref.get %0[%c0_i32] as (%b0 : !ttg.memdesc<64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<16x32xf16, #shared0, #smem>) {
      nvws.aref.return %b0 : !ttg.memdesc<64x16xf16, #shared0, #smem>
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>, i32) -> !ttg.memdesc<64x16xf16, #shared0, #smem>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @aref_put_batch(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>
    // expected-error @below {{Dimensions don't match}}
    nvws.aref.put %0[%c0_i32] as (%b0 : !ttg.memdesc<64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<32x32xf16, #shared0, #smem>) {
      nvws.aref.return
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>, i32) -> ()
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @aref_put_batch(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>
    // expected-error @below {{MLIR Types don't match}}
    nvws.aref.put %0[%c0_i32] as (%b0 : tensor<64x16xf16>, %b1 : tensor<16x32xf16>) {
      nvws.aref.return
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>, i32) -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_put_tensor_wrong_num_outputs
  // CHECK: nvws.aref.put
  tt.func @aref_put_tensor_wrong_num_outputs(%d : tensor<1x64x16xf16>, %e : tensor<1x16x32xf16>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[tensor<1x64x16xf16>, tensor<1x16x32xf16>], 1>
    nvws.aref.put %0[%c0_i32] as (%b0 : tensor<64x16xf16>, %b1 : tensor<16x32xf16>) {
      %1 = math.exp %b0 : tensor<64x16xf16>
      %2 = math.cos %b1 : tensor<16x32xf16>
      // expected-error @below {{Mismatching number of returns}}
      nvws.aref.return %1 : tensor<64x16xf16>
    } : (!nvws.aref<[tensor<1x64x16xf16>, tensor<1x16x32xf16>], 1>, i32) -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_put_tensor_wrong_type
  // CHECK: nvws.aref.put
  tt.func @aref_put_tensor_wrong_type(%d : tensor<1x64x16xf16>, %e : tensor<1x16x32xf16>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[tensor<1x64x16xf16>, tensor<1x16x32xf16>], 1>
    nvws.aref.put %0[%c0_i32] as (%b0 : tensor<64x16xf16>, %b1 : tensor<16x32xf16>) {
      %1 = math.exp %b0 : tensor<64x16xf16>
      %2 = tt.fp_to_fp %b1 : tensor<16x32xf16> -> tensor<16x32xf32>
      // expected-error @below {{Return sources and Block Arguments have different types}}
      nvws.aref.return %1 , %2: tensor<64x16xf16>, tensor<16x32xf32>
    } : (!nvws.aref<[tensor<1x64x16xf16>, tensor<1x16x32xf16>], 1>, i32) -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_put_tensor_wrong_num_outputs
  // CHECK: nvws.aref.put
  tt.func @aref_put_tensor_wrong_num_outputs(%d : tensor<1x64x16xf16>, %e : tensor<1x16x32xf16>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[tensor<1x64x16xf16>, tensor<1x16x32xf16>], 1>
    nvws.aref.put %0[%c0_i32] as (%b0 : tensor<64x16xf16>, %b1 : tensor<16x32xf16>) {
      %1 = math.exp %b0 : tensor<64x16xf16>
      %2 = math.cos %b1 : tensor<16x32xf16>
      // expected-error @below {{Mismatching number of returns}}
      nvws.aref.return %1 : tensor<64x16xf16>
    } : (!nvws.aref<[tensor<1x64x16xf16>, tensor<1x16x32xf16>], 1>, i32) -> ()
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_get_dot
  // CHECK: nvws.aref.get
  tt.func @aref_get_dot(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>
    %1 = nvws.aref.get %0[%c0_i32] as (%b0 : !ttg.memdesc<64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<16x32xf16, #shared0, #smem>) {
      %1 = ttg.local_load %b0 : !ttg.memdesc<64x16xf16, #shared0, #smem> -> tensor<64x16xf16>
      %2 = ttg.local_load %b1 : !ttg.memdesc<16x32xf16, #shared0, #smem> -> tensor<16x32xf16>
      %cst_0 = arith.constant dense<0.0> : tensor<64x32xf16>
      %3 = tt.dot %1, %2, %cst_0 : tensor<64x16xf16> * tensor<16x32xf16> -> tensor<64x32xf16>
      // expected-error @below {{Return sources and Parent Op Results have different types}}
      nvws.aref.return %3 : tensor<64x32xf16>
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>, i32) -> !ttg.memdesc<64x16xf16, #shared0, #smem>
    tt.return
  }
}
