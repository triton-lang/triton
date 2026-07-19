// RUN: triton-opt %s --tritonamdgpu-convert-tensor-descriptor=arch-generation-name=gfx950 |  FileCheck %s --check-prefixes=CHECK-GFX950
// RUN: triton-opt %s --tritonamdgpu-convert-tensor-descriptor=arch-generation-name=gfx1250 |  FileCheck %s --check-prefixes=CHECK-GFX1250

// Test for descriptor_reduce and descriptor_load use different descriptors
module {
// CHECK-GFX1250: tt.atomic_rmw
// CHECK-GFX1250-NEXT: tt.make_tensor_descriptor
// CHECK-GFX1250-NEXT: tt.descriptor_load
// CHECK-GFX950: tt.atomic_rmw
// CHECK-GFX950-NOT: tt.make_tensor_descriptor
// CHECK-GFX950-NOT: tt.descriptor_load
  tt.func public @kernel(%out_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %out_ptr_for_tensor: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %a_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %desc: i64, %load_desc: i64, %val_56: tensor<8x128xbf16>, %1: tensor<8x128x!tt.ptr<bf16>>, %moffset_12: i32, %noffset_19: i32, %M: i32 {tt.divisibility = 16 : i32} , %N: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc_57 = tt.make_tensor_descriptor %out_ptr, [%M, %N], [%desc, %c1_i64] : <bf16>, <8x128xbf16>
    tt.descriptor_reduce add, %desc_57[%moffset_12, %noffset_19], %val_56 : !tt.tensordesc<8x128xbf16>, tensor<8x128xbf16>
    %load_desc_76 = tt.make_tensor_descriptor %out_ptr, [%M, %N], [%load_desc, %c1_i64] : <bf16>, <8x128xbf16>
    %block = tt.descriptor_load %load_desc_76[%c0_i32, %c0_i32] : !tt.tensordesc<8x128xbf16> -> tensor<8x128xbf16>
    tt.store %1, %block : tensor<8x128x!tt.ptr<bf16>>
    tt.return
  }
}

// ------

// Test for descriptor_reduce and descriptor_load use the same descriptor
module {
// CHECK-LABEL: kernel
// CHECK-GFX1250: {{.*}} = tt.atomic_rmw
// CHECK-GFX1250: [[LD:%.*]] = tt.load
// CHECK-GFX1250: tt.store {{.*}}, [[LD]]
// CHECK-GFX950: {{.*}} = tt.atomic_rmw
// CHECK-GFX950: [[LD:%.*]] = tt.load
// CHECK-GFX950: tt.store {{.*}}, [[LD]]
  tt.func public @kernel(%out_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %out_ptr_for_tensor: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %a_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %desc: i64, %load_desc: i64, %val_56: tensor<8x128xbf16>, %1: tensor<8x128x!tt.ptr<bf16>>, %moffset_12: i32, %noffset_19: i32, %M: i32 {tt.divisibility = 16 : i32} , %N: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc_57 = tt.make_tensor_descriptor %out_ptr, [%M, %N], [%desc, %c1_i64] : <bf16>, <8x128xbf16>
    tt.descriptor_reduce add, %desc_57[%moffset_12, %noffset_19], %val_56 : !tt.tensordesc<8x128xbf16>, tensor<8x128xbf16>
    %block = tt.descriptor_load %desc_57[%c0_i32, %c0_i32] : !tt.tensordesc<8x128xbf16> -> tensor<8x128xbf16>
    tt.store %1, %block : tensor<8x128x!tt.ptr<bf16>>
    tt.return
  }
}


// -----

// Test for the descriptor created on the host side used by the reduce op
module {
// CEHCK-LABEL:
// CHECK-GFX1250: tt.func public @kernel(%arg0: !tt.ptr<bf16>
// CHECK-GFX1250: tt.atomic_rmw
// CHECK-GFX1250-NEXT: tt.make_tensor_descriptor
// CHECK-GFX1250-NEXT: tt.descriptor_load
// CHECK-GFX950: tt.func public @kernel(%arg0: !tt.ptr<bf16>
// CHECK-GFX950: tt.atomic_rmw
// CHECK-GFX950-NOT: tt.make_tensor_descriptor
// CHECK-GFX950-NOT: tt.descriptor_load
  tt.func public @kernel(%desc_57: !tt.tensordesc<8x128xbf16>, %reduce_desc.shape.0: i32, %reduce_desc.shape.1: i32, %reduce_desc.stride.0: i64, %reduce_desc.stride.1: i64, %out_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %out_ptr_for_tensor: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %a_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %desc: i64, %load_desc: i64, %val_56: tensor<8x128xbf16>, %1: tensor<8x128x!tt.ptr<bf16>>, %moffset_12: i32, %noffset_19: i32, %M: i32 {tt.divisibility = 16 : i32} , %N: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    tt.descriptor_reduce add, %desc_57[%moffset_12, %noffset_19], %val_56 : !tt.tensordesc<8x128xbf16>, tensor<8x128xbf16>
    %load_desc_76 = tt.make_tensor_descriptor %out_ptr, [%M, %N], [%load_desc, %c1_i64] : <bf16>, <8x128xbf16>
    %block = tt.descriptor_load %load_desc_76[%c0_i32, %c0_i32] : !tt.tensordesc<8x128xbf16> -> tensor<8x128xbf16>
    tt.store %1, %block : tensor<8x128x!tt.ptr<bf16>>
    tt.return
  }
}


// -----

// Test for the descriptor created on the host side used by the non-reduce op
module {
// CEHCK-LABEL:
// CHECK-GFX1250: tt.func public @kernel(%arg0: !tt.tensordesc<8x128xbf16>
// CHECK-GFX1250: tt.atomic_rmw
// CHECK-GFX1250-NEXT: tt.descriptor_load
// CHECK-GFX950: tt.func public @kernel(%arg0: !tt.ptr<bf16>
// CHECK-GFX950: tt.atomic_rmw
// CHECK-GFX950-NOT: tt.make_tensor_descriptor
// CHECK-GFX950-NOT: tt.descriptor_load
  tt.func public @kernel(%load_desc_76: !tt.tensordesc<8x128xbf16>, %reduce_desc.shape.0: i32, %reduce_desc.shape.1: i32, %reduce_desc.stride.0: i64, %reduce_desc.stride.1: i64, %out_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %out_ptr_for_tensor: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %a_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %desc: i64, %load_desc: i64, %val_56: tensor<8x128xbf16>, %1: tensor<8x128x!tt.ptr<bf16>>, %moffset_12: i32, %noffset_19: i32, %M: i32 {tt.divisibility = 16 : i32} , %N: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %desc_57 = tt.make_tensor_descriptor %out_ptr, [%M, %N], [%desc, %c1_i64] : <bf16>, <8x128xbf16>
    tt.descriptor_reduce add, %desc_57[%moffset_12, %noffset_19], %val_56 : !tt.tensordesc<8x128xbf16>, tensor<8x128xbf16>
    %block = tt.descriptor_load %load_desc_76[%c0_i32, %c0_i32] : !tt.tensordesc<8x128xbf16> -> tensor<8x128xbf16>
    tt.store %1, %block : tensor<8x128x!tt.ptr<bf16>>
    tt.return
  }
}
