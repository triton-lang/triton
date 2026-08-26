// RUN: triton-opt %s --convert-nv-gpu-to-llvm -allow-unregistered-dialect -split-input-file | FileCheck %s
// RUN: triton-opt %s --nvidia-optimize-conditional-xor -allow-unregistered-dialect -split-input-file | FileCheck %s --check-prefix=LOP3

// CHECK-LABEL: @cluster_id
llvm.func @cluster_id() -> i32 {
  // CHECK: nvvm.read.ptx.sreg.cluster.ctarank
  // CHECK-NOT: nvvm.read.ptx.sreg.cluster.ctaid.x
  // CHECK-NOT: nvvm.read.ptx.sreg.cluster.ctaid.y
  // CHECK-NOT: nvvm.read.ptx.sreg.cluster.ctaid.z
  // CHECK-NOT: nvvm.read.ptx.sreg.cluster.nctaid.x
  // CHECK-NOT: nvvm.read.ptx.sreg.cluster.nctaid.y
  %id = nvg.cluster_id
  llvm.return %id : i32
}

// -----

!struct_128xf32 = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
)>

!struct_64xf32 = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
)>

// CHECK-LABEL: @wgmma
llvm.func @wgmma(%desc: i64, %in: !struct_64xf32) {
// CHECK: wgmma.mma_async.sync.aligned.m64n256k32.f32.e5m2.e5m2
%false = llvm.mlir.constant(false) : i1
%acc0 = nvg.wgmma %desc, %desc, %false {
  eltTypeA = 3 : i32,
  eltTypeB = 3 : i32,
  eltTypeC = 7 : i32,
  layoutA = 0 : i32,
  layoutB = 1 : i32,
  m = 64 : i32,
  n = 256 : i32,
  k = 32 : i32
} : (i64, i64, i1) -> !struct_128xf32

  // CHECK: // wait for regs: $0,$1,$2,{{.*}},$127
  // CHECK: wgmma.wait_group.sync.aligned 0;
  %out = nvg.wgmma_wait_group %in {pendings = 0 : i32} : !struct_64xf32
  llvm.return
}

// -----

!struct = !llvm.struct<(f32, f32, i32, i32, f16, f16)>

// CHECK-LABEL: @wgmma_wait
llvm.func @wgmma_wait(%in: !struct) {
  // CHECK: // wait for regs: $0,$1,$2,$3,$4,$5
  // CHECK: wgmma.wait_group.sync.aligned 0;
  // CHECK: "=f,=f,=r,=r,=h,=h,0,1,2,3,4,5"
  %out = nvg.wgmma_wait_group %in {pendings = 0 : i32} : !struct
  llvm.return
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_base_lowering
  //      CHECK:    %[[TID:.+]] = nvvm.read.ptx.sreg.tid.x : i32
  //      CHECK:    %[[C32:.+]] = llvm.mlir.constant(32 : i32) : i32
  //      CHECK:    %[[PRED:.+]] = llvm.icmp "ult" %[[TID]], %[[C32]] : i32
  //      CHECK:    %[[SHMEM:.+]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
  //      CHECK:    %[[A:.+]] = llvm.inline_asm has_side_effects
  // CHECK-SAME:    "@$0 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 128;", "b,r" %[[PRED]], %[[SHMEM]] : (i1, !llvm.ptr<3>) -> !llvm.void
  //      CHECK:    %[[AR:.+]] = llvm.load %[[SHMEM]] : !llvm.ptr<3> -> i32
  //      CHECK:    nvvm.barrier
  //      CHECK:    "@$0 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "b" %[[PRED]]  : (i1) -> !llvm.void
  //      CHECK:    nvvm.barrier
  //      CHECK:    llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.dealloc.cta_group::1.sync.aligned.b32 $1, 128;", "b,r" %[[PRED]], %{{.+}} : (i1, !llvm.ptr<6>) -> !llvm.void
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @tensor_memory_base_lowering() -> i32 attributes {nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
    %263 = nvg.tensor_memory_base
    %264 = llvm.ptrtoint %263 : !llvm.ptr<6> to i32
    llvm.return %264 : i32
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @tensor_memory_base_warpgroup
llvm.func @tensor_memory_base_warpgroup() attributes {nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
  // CHECK: [[PTR:%.*]] = llvm.inttoptr %{{.*}} : i32 to !llvm.ptr<6>
  // CHECK: ttg.warp_specialize([[PTR]])
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0() num_warps(1) {
    %0 = nvg.tensor_memory_base
    // CHECK-NEXT: "use"(%arg0)
    "use"(%0) : (!llvm.ptr<6>) -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 576 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_exclusive_rubin
  //      CHECK:    %[[TID:.+]] = nvvm.read.ptx.sreg.tid.x : i32
  //      CHECK:    %[[C32:.+]] = llvm.mlir.constant(32 : i32) : i32
  //      CHECK:    %[[PRED:.+]] = llvm.icmp "ult" %[[TID]], %[[C32]] : i32
  //      CHECK:    %[[SHMEM:.+]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
  //      CHECK:    %[[A:.+]] = llvm.inline_asm has_side_effects
  // CHECK-SAME:    "@$0 tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [$1], 576;", "b,r" %[[PRED]], %[[SHMEM]] : (i1, !llvm.ptr<3>) -> !llvm.void
  //      CHECK:    %[[AR:.+]] = llvm.load %[[SHMEM]] : !llvm.ptr<3> -> i32
  //      CHECK:    nvvm.barrier
  //      CHECK:    "@$0 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "b" %[[PRED]]  : (i1) -> !llvm.void
  //      CHECK:    nvvm.barrier
  //      CHECK:    llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 $1, 576;", "b,r" %[[PRED]], %{{.+}} : (i1, !llvm.ptr<6>) -> !llvm.void
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @tensor_memory_exclusive_rubin() -> i32 attributes {nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
    %263 = nvg.tensor_memory_base
    %264 = llvm.ptrtoint %263 : !llvm.ptr<6> to i32
    llvm.return %264 : i32
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 512 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_no_exclusive_rubin
  //      CHECK:    %[[TID:.+]] = nvvm.read.ptx.sreg.tid.x : i32
  //      CHECK:    %[[C32:.+]] = llvm.mlir.constant(32 : i32) : i32
  //      CHECK:    %[[PRED:.+]] = llvm.icmp "ult" %[[TID]], %[[C32]] : i32
  //      CHECK:    %[[SHMEM:.+]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
  //      CHECK:    %[[A:.+]] = llvm.inline_asm has_side_effects
  // CHECK-SAME:    "@$0 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 512;", "b,r" %[[PRED]], %[[SHMEM]] : (i1, !llvm.ptr<3>) -> !llvm.void
  //      CHECK-NOT:  .exclusive
  //      CHECK:    %[[AR:.+]] = llvm.load %[[SHMEM]] : !llvm.ptr<3> -> i32
  //      CHECK:    nvvm.barrier
  //      CHECK:    "@$0 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "b" %[[PRED]]  : (i1) -> !llvm.void
  //      CHECK:    nvvm.barrier
  //      CHECK:    llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.dealloc.cta_group::1.sync.aligned.b32 $1, 512;", "b,r" %[[PRED]], %{{.+}} : (i1, !llvm.ptr<6>) -> !llvm.void
  //      CHECK-NOT:  .exclusive
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @tensor_memory_no_exclusive_rubin() -> i32 attributes {nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
    %263 = nvg.tensor_memory_base
    %264 = llvm.ptrtoint %263 : !llvm.ptr<6> to i32
    llvm.return %264 : i32
  }
}

// -----

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @one_warp
tt.func @one_warp() -> i32 {
  // CHECK-NEXT: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
  %0 = ttg.warp_id
  // CHECK-NEXT: return [[C0]]
  tt.return %0 : i32
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @one_contextual_warp
tt.func @one_contextual_warp() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0() num_warps(1) {
    // CHECK-NEXT: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
    %0 = ttg.warp_id
    // CHECK-NEXT: "use"([[C0]])
    "use"(%0) : (i32) -> ()
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

module attributes {ttg.target = "cuda:90"} {
  // Only the value eagerly consumed by LOP3 is frozen. A separate select
  // must still use the original, potentially poison, shuffle result.
  // LOP3-LABEL: llvm.func @conditional_xor_masks_poison(
  // LOP3-SAME: [[ACC:%.*]]: i32, [[SOURCE:%.*]]: i32, [[COND:%.*]]: i1, [[OTHER_COND:%.*]]: i1, [[OUT:%.*]]: !llvm.ptr
  // LOP3: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // LOP3: [[FULL:%.*]] = llvm.mlir.constant(-1 : i32) : i32
  // LOP3: [[CLAMP:%.*]] = llvm.mlir.constant(31 : i32) : i32
  // LOP3-NEXT: [[BASIS:%.*]] = nvvm.shfl.sync idx [[FULL]], [[SOURCE]], [[ZERO]], [[CLAMP]] : i32 -> i32
  // LOP3-NEXT: [[FROZEN:%.*]] = llvm.freeze [[BASIS]] : i32
  // LOP3-NEXT: [[MASK:%.*]] = llvm.sext [[COND]] : i1 to i32
  // LOP3-NEXT: [[FUSED:%.*]] = llvm.inline_asm asm_dialect = att {{.*}}"lop3.b32 $0, $1, $2, $3, 0x78;", "=r,r,r,r" [[ACC]], [[FROZEN]], [[MASK]] : (i32, i32, i32) -> i32
  // LOP3-NEXT: [[OTHER:%.*]] = llvm.select [[OTHER_COND]], [[BASIS]], [[ZERO]] : i1, i32
  // LOP3-NEXT: llvm.store [[OTHER]], [[OUT]] : i32, !llvm.ptr
  // LOP3-NEXT: llvm.return [[FUSED]] : i32
  llvm.func @conditional_xor_masks_poison(%acc: i32, %source: i32, %cond: i1, %other_cond: i1, %out: !llvm.ptr) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %full = llvm.mlir.constant(-1 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %basis = nvvm.shfl.sync idx %full, %source, %zero, %clamp {ttg.one_hot_xor_reduction} : i32 -> i32
    %contribution = llvm.select %cond, %basis, %zero : i1, i32
    %result = llvm.xor %acc, %contribution : i32
    %other = llvm.select %other_cond, %basis, %zero : i1, i32
    llvm.store %other, %out : i32, !llvm.ptr
    llvm.return %result : i32
  }

  // The conditional contribution can be either XOR operand.
  // LOP3-LABEL: llvm.func @conditional_xor_commuted(
  // LOP3-SAME: [[ACC:%.*]]: i32, [[SOURCE:%.*]]: i32, [[COND:%.*]]: i1
  // LOP3: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // LOP3: [[FULL:%.*]] = llvm.mlir.constant(-1 : i32) : i32
  // LOP3: [[CLAMP:%.*]] = llvm.mlir.constant(31 : i32) : i32
  // LOP3-NEXT: [[BASIS:%.*]] = nvvm.shfl.sync idx [[FULL]], [[SOURCE]], [[ZERO]], [[CLAMP]] : i32 -> i32
  // LOP3-NEXT: [[FROZEN:%.*]] = llvm.freeze [[BASIS]] : i32
  // LOP3-NEXT: [[MASK:%.*]] = llvm.sext [[COND]] : i1 to i32
  // LOP3-NEXT: [[FUSED:%.*]] = llvm.inline_asm asm_dialect = att {{.*}}"lop3.b32 $0, $1, $2, $3, 0x78;", "=r,r,r,r" [[ACC]], [[FROZEN]], [[MASK]] : (i32, i32, i32) -> i32
  // LOP3-NEXT: llvm.return [[FUSED]] : i32
  llvm.func @conditional_xor_commuted(%acc: i32, %source: i32, %cond: i1) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %full = llvm.mlir.constant(-1 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %basis = nvvm.shfl.sync idx %full, %source, %zero, %clamp {ttg.one_hot_xor_reduction} : i32 -> i32
    %contribution = llvm.select %cond, %basis, %zero : i1, i32
    %result = llvm.xor %contribution, %acc : i32
    llvm.return %result : i32
  }

  // Gather lowering may still select the owning register after its shuffle.
  // Freeze the marked final result, not just the shuffle operand.
  // LOP3-LABEL: llvm.func @conditional_xor_gather_register_select(
  // LOP3-SAME: [[ACC:%.*]]: i32, [[SOURCE:%.*]]: i32, [[COND:%.*]]: i1
  // LOP3: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // LOP3: [[FULL:%.*]] = llvm.mlir.constant(-1 : i32) : i32
  // LOP3: [[CLAMP:%.*]] = llvm.mlir.constant(31 : i32) : i32
  // LOP3-NEXT: [[UNDEF:%.*]] = llvm.mlir.undef : i32
  // LOP3-NEXT: [[REG:%.*]] = llvm.xor [[ZERO]], [[ZERO]] : i32
  // LOP3-NEXT: [[BASIS:%.*]] = nvvm.shfl.sync idx [[FULL]], [[SOURCE]], [[ZERO]], [[CLAMP]] : i32 -> i32
  // LOP3-NEXT: [[OWNS:%.*]] = llvm.icmp "eq" [[REG]], [[ZERO]] : i32
  // LOP3-NEXT: [[GATHERED:%.*]] = llvm.select [[OWNS]], [[BASIS]], [[UNDEF]] : i1, i32
  // LOP3-NEXT: [[FROZEN:%.*]] = llvm.freeze [[GATHERED]] : i32
  // LOP3-NEXT: [[MASK:%.*]] = llvm.sext [[COND]] : i1 to i32
  // LOP3-NEXT: [[FUSED:%.*]] = llvm.inline_asm asm_dialect = att {{.*}}"lop3.b32 $0, $1, $2, $3, 0x78;", "=r,r,r,r" [[ACC]], [[FROZEN]], [[MASK]] : (i32, i32, i32) -> i32
  // LOP3-NEXT: llvm.return [[FUSED]] : i32
  llvm.func @conditional_xor_gather_register_select(%acc: i32, %source: i32, %cond: i1) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %full = llvm.mlir.constant(-1 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %undef = llvm.mlir.undef : i32
    %reg = llvm.xor %zero, %zero : i32
    %basis = nvvm.shfl.sync idx %full, %source, %zero, %clamp {ttg.one_hot_xor_reduction} : i32 -> i32
    %owns = llvm.icmp "eq" %reg, %zero : i32
    %gathered = llvm.select %owns, %basis, %undef {ttg.one_hot_xor_reduction} : i1, i32
    %contribution = llvm.select %cond, %gathered, %zero : i1, i32
    %result = llvm.xor %acc, %contribution : i32
    llvm.return %result : i32
  }

  // A pre-existing shuffle is not evidence of a rewritten one-hot reduction.
  // LOP3-LABEL: llvm.func @conditional_xor_untagged_shuffle(
  // LOP3-SAME: [[ACC:%.*]]: i32, [[SOURCE:%.*]]: i32, [[COND:%.*]]: i1
  // LOP3: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // LOP3: [[BASIS:%.*]] = nvvm.shfl.sync idx {{.*}} : i32 -> i32
  // LOP3-NEXT: [[SELECT:%.*]] = llvm.select [[COND]], [[BASIS]], [[ZERO]] : i1, i32
  // LOP3-NEXT: [[RESULT:%.*]] = llvm.xor [[ACC]], [[SELECT]] : i32
  // LOP3-NEXT: llvm.return [[RESULT]] : i32
  llvm.func @conditional_xor_untagged_shuffle(%acc: i32, %source: i32, %cond: i1) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %full = llvm.mlir.constant(-1 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %basis = nvvm.shfl.sync idx %full, %source, %zero, %clamp : i32 -> i32
    %contribution = llvm.select %cond, %basis, %zero : i1, i32
    %result = llvm.xor %acc, %contribution : i32
    llvm.return %result : i32
  }

  // A reused select is left intact, but its shuffle's temporary marker is
  // still removed from the NVIDIA LLVM module.
  // LOP3-LABEL: llvm.func @conditional_xor_shared_select(
  // LOP3-SAME: [[ACC:%.*]]: i32, [[SOURCE:%.*]]: i32, [[COND:%.*]]: i1
  // LOP3: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // LOP3: [[FULL:%.*]] = llvm.mlir.constant(-1 : i32) : i32
  // LOP3: [[CLAMP:%.*]] = llvm.mlir.constant(31 : i32) : i32
  // LOP3-NEXT: [[BASIS:%.*]] = nvvm.shfl.sync idx [[FULL]], [[SOURCE]], [[ZERO]], [[CLAMP]] : i32 -> i32
  // LOP3-NEXT: [[SELECT:%.*]] = llvm.select [[COND]], [[BASIS]], [[ZERO]] : i1, i32
  // LOP3-NEXT: [[XOR:%.*]] = llvm.xor [[ACC]], [[SELECT]] : i32
  // LOP3-NEXT: [[RESULT:%.*]] = llvm.add [[XOR]], [[SELECT]] : i32
  // LOP3-NEXT: llvm.return [[RESULT]] : i32
  llvm.func @conditional_xor_shared_select(%acc: i32, %source: i32, %cond: i1) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %full = llvm.mlir.constant(-1 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %basis = nvvm.shfl.sync idx %full, %source, %zero, %clamp {ttg.one_hot_xor_reduction} : i32 -> i32
    %contribution = llvm.select %cond, %basis, %zero : i1, i32
    %xor = llvm.xor %acc, %contribution : i32
    %result = llvm.add %xor, %contribution : i32
    llvm.return %result : i32
  }

  // A nonzero inactive contribution is not the supported mask/XOR identity.
  // LOP3-LABEL: llvm.func @conditional_xor_nonzero_inactive(
  // LOP3-SAME: [[ACC:%.*]]: i32, [[SOURCE:%.*]]: i32, [[COND:%.*]]: i1
  // LOP3: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // LOP3: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // LOP3: [[FULL:%.*]] = llvm.mlir.constant(-1 : i32) : i32
  // LOP3: [[CLAMP:%.*]] = llvm.mlir.constant(31 : i32) : i32
  // LOP3-NEXT: [[BASIS:%.*]] = nvvm.shfl.sync idx [[FULL]], [[SOURCE]], [[ZERO]], [[CLAMP]] : i32 -> i32
  // LOP3-NEXT: [[SELECT:%.*]] = llvm.select [[COND]], [[BASIS]], [[ONE]] : i1, i32
  // LOP3-NEXT: [[RESULT:%.*]] = llvm.xor [[ACC]], [[SELECT]] : i32
  // LOP3-NEXT: llvm.return [[RESULT]] : i32
  llvm.func @conditional_xor_nonzero_inactive(%acc: i32, %source: i32, %cond: i1) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %full = llvm.mlir.constant(-1 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %basis = nvvm.shfl.sync idx %full, %source, %zero, %clamp {ttg.one_hot_xor_reduction} : i32 -> i32
    %contribution = llvm.select %cond, %basis, %one : i1, i32
    %result = llvm.xor %acc, %contribution : i32
    llvm.return %result : i32
  }

  // Wider arithmetic is not fused through a cast of a marked shuffle.
  // LOP3-LABEL: llvm.func @conditional_xor_i64(
  // LOP3-SAME: [[ACC:%.*]]: i64, [[SOURCE:%.*]]: i32, [[COND:%.*]]: i1
  // LOP3: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // LOP3: [[ZERO64:%.*]] = llvm.mlir.constant(0 : i64) : i64
  // LOP3: [[FULL:%.*]] = llvm.mlir.constant(-1 : i32) : i32
  // LOP3: [[CLAMP:%.*]] = llvm.mlir.constant(31 : i32) : i32
  // LOP3-NEXT: [[BASIS:%.*]] = nvvm.shfl.sync idx [[FULL]], [[SOURCE]], [[ZERO]], [[CLAMP]] : i32 -> i32
  // LOP3-NEXT: [[WIDE:%.*]] = llvm.zext [[BASIS]] : i32 to i64
  // LOP3-NEXT: [[SELECT:%.*]] = llvm.select [[COND]], [[WIDE]], [[ZERO64]] : i1, i64
  // LOP3-NEXT: [[RESULT:%.*]] = llvm.xor [[ACC]], [[SELECT]] : i64
  // LOP3-NEXT: llvm.return [[RESULT]] : i64
  llvm.func @conditional_xor_i64(%acc: i64, %source: i32, %cond: i1) -> i64 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %zero64 = llvm.mlir.constant(0 : i64) : i64
    %full = llvm.mlir.constant(-1 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %basis = nvvm.shfl.sync idx %full, %source, %zero, %clamp {ttg.one_hot_xor_reduction} : i32 -> i32
    %wide = llvm.zext %basis : i32 to i64
    %contribution = llvm.select %cond, %wide, %zero64 : i1, i64
    %result = llvm.xor %acc, %contribution : i64
    llvm.return %result : i64
  }
}

// -----

module attributes {ttg.target = "hip:gfx950"} {
  // A non-CUDA module is untouched, including its attributes.
  // LOP3-LABEL: llvm.func @conditional_xor_non_cuda(
  // LOP3-SAME: [[ACC:%.*]]: i32, [[SOURCE:%.*]]: i32, [[COND:%.*]]: i1
  // LOP3: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // LOP3: [[BASIS:%.*]] = nvvm.shfl.sync idx {{.*}} {ttg.one_hot_xor_reduction} : i32 -> i32
  // LOP3-NEXT: [[SELECT:%.*]] = llvm.select [[COND]], [[BASIS]], [[ZERO]] : i1, i32
  // LOP3-NEXT: [[RESULT:%.*]] = llvm.xor [[ACC]], [[SELECT]] : i32
  // LOP3-NEXT: llvm.return [[RESULT]] : i32
  llvm.func @conditional_xor_non_cuda(%acc: i32, %source: i32, %cond: i1) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %full = llvm.mlir.constant(-1 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %basis = nvvm.shfl.sync idx %full, %source, %zero, %clamp {ttg.one_hot_xor_reduction} : i32 -> i32
    %contribution = llvm.select %cond, %basis, %zero : i1, i32
    %result = llvm.xor %acc, %contribution : i32
    llvm.return %result : i32
  }
}

// -----

module {
  // An explicit CUDA target is required before emitting NVIDIA assembly.
  // LOP3-LABEL: llvm.func @conditional_xor_missing_target(
  // LOP3-SAME: [[ACC:%.*]]: i32, [[SOURCE:%.*]]: i32, [[COND:%.*]]: i1
  // LOP3: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // LOP3: [[BASIS:%.*]] = nvvm.shfl.sync idx {{.*}} {ttg.one_hot_xor_reduction} : i32 -> i32
  // LOP3-NEXT: [[SELECT:%.*]] = llvm.select [[COND]], [[BASIS]], [[ZERO]] : i1, i32
  // LOP3-NEXT: [[RESULT:%.*]] = llvm.xor [[ACC]], [[SELECT]] : i32
  // LOP3-NEXT: llvm.return [[RESULT]] : i32
  llvm.func @conditional_xor_missing_target(%acc: i32, %source: i32, %cond: i1) -> i32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %full = llvm.mlir.constant(-1 : i32) : i32
    %clamp = llvm.mlir.constant(31 : i32) : i32
    %basis = nvvm.shfl.sync idx %full, %source, %zero, %clamp {ttg.one_hot_xor_reduction} : i32 -> i32
    %contribution = llvm.select %cond, %basis, %zero : i1, i32
    %result = llvm.xor %acc, %contribution : i32
    llvm.return %result : i32
  }
}
