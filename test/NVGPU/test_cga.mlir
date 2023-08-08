// RUN: triton-translate %s | FileCheck %s
#SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 2 : i32} {
  tt.func @test_mbarrier() {
    %addr = arith.constant 32 : i32
    %data = arith.constant 123 : i32
    %pred = arith.constant 1 : i1
    %id0 = arith.constant 0 : i32
    %id1 = arith.constant 1 : i32
    // CHECK: call void @__nv_cga_barrier_sync()
    // CHECK: call void @__nv_cga_barrier_arrive()
    // CHECK: call void @__nv_cga_barrier_wait()
    nvgpu.cga_barrier_sync
    nvgpu.cga_barrier_arrive
    nvgpu.cga_barrier_wait

    %ptr = llvm.mlir.null : !llvm.ptr<i32, 3>

    // CHECK: %[[X:.+]] = tail call i32 asm "mov.u32 $0, %cluster_ctaid.x;", "=r"()
    // CHECK: %[[Y:.+]] = tail call i32 asm "mov.u32 $0, %cluster_ctaid.y;", "=r"()
    // CHECK: %[[Z:.+]] = tail call i32 asm "mov.u32 $0, %cluster_ctaid.z;", "=r"()
    // CHECK: %[[NX:.+]] = tail call i32 asm "mov.u32 $0, %cluster_nctaid.x;", "=r"()
    // CHECK: %[[NY:.+]] = tail call i32 asm "mov.u32 $0, %cluster_nctaid.y;", "=r"()
    // CHECK: %[[A0:.+]] = mul i32 %[[NY]], %[[Z]]
    // CHECK: %[[A1:.+]] = add i32 %[[A0]], %[[Y]]
    // CHECK: %[[A2:.+]] = mul i32 %[[A1]], %[[NX]]
    // CHECK: add i32 %[[A2]], %[[X]]
    %v = nvgpu.cluster_id
    llvm.store %v, %ptr : !llvm.ptr<i32, 3>

    tt.return
  }
} // end module
