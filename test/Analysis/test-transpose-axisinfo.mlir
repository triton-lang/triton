// RUN: triton-opt %s -test-print-alignment -split-input-file -verify-diagnostics=only-expected -o /dev/null
//
// -----// IR Dump Before TritonRewriteTensorPointer (triton-rewrite-tensor-pointer) ('builtin.module' operation) //----- //
#loc = loc("/tmp/transpose.py":8:0)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#loc13 = loc("X_ptr"(#loc))
#loc14 = loc("stride_xa"(#loc))
module {
  tt.func public @transpose_read_kernel(%X_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("X_ptr"(#loc)), %stride_xa: i32 {tt.divisibility = 16 : i32} loc("stride_xa"(#loc))) attributes {noinline = false} {
    // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
    %buffer = arith.constant 0 : i32
    %buffers = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %buffer_0 = ttg.memdesc_index %buffers[%buffer] : !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>

    // expected-remark @below {{contiguity = [64], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
    %offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    // expected-remark @below {{contiguity = [64, 1], divisibility = [1073741824, 1], constancy = [1, 1], constant_value = <none>}}
    %offsets_1 = tt.expand_dims %offsets {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    // expected-remark @below {{contiguity = [64], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
    %offsets_2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    // expected-remark @below {{contiguity = [1, 64], divisibility = [1, 1073741824], constancy = [1, 1], constant_value = <none>}}
    %offsets_3 = tt.expand_dims %offsets_2 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 64], constant_value = <none>}}
    %offsets_4 = tt.splat %stride_xa : i32 -> tensor<1x64xi32>
    // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>}}
    %offsets_5 = arith.muli %offsets_3, %offsets_4 : tensor<1x64xi32>

    // expected-remark @below {{contiguity = [64, 1], divisibility = [1073741824, 1], constancy = [1, 64], constant_value = <none>}}
    %offsets_6 = tt.broadcast %offsets_1 : tensor<64x1xi32> -> tensor<64x64xi32>
    // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [64, 1], constant_value = <none>}}
    %offsets_7 = tt.broadcast %offsets_5 : tensor<1x64xi32> -> tensor<64x64xi32>
    // expected-remark @below {{contiguity = [64, 1], divisibility = [16, 1], constancy = [1, 1], constant_value = <none>}}
    %offsets_8 = arith.addi %offsets_6, %offsets_7 : tensor<64x64xi32>

    // expected-remark @below {{contiguity = [1, 64], divisibility = [1, 16], constancy = [1, 1], constant_value = <none>}}
    %offsets_9 = tt.trans %offsets_8 {order = array<i32: 1, 0>} : tensor<64x64xi32> -> tensor<64x64xi32>

    // expected-remark @below {{contiguity = [1, 1], divisibility = [16, 16], constancy = [64, 64], constant_value = <none>}}
    %0 = tt.splat %X_ptr : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
    // expected-remark @below {{contiguity = [1, 64], divisibility = [2, 16], constancy = [1, 1], constant_value = <none>}}
    %1 = tt.addptr %0, %offsets_9 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>

    %2 = ttg.async_copy_global_to_local %1, %buffer_0 : tensor<64x64x!tt.ptr<f16>> -> <64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
