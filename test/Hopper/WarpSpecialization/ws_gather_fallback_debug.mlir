// REQUIRES: asserts
// RUN: triton-opt %S/ws_gather_fallback.mlir --nvgpu-warp-specialization=num-stages=2 --debug-only=nvgpu-ws-data-partition 2>&1 | FileCheck %s

// CHECK: dim <unassigned>
// CHECK: @gather_after_loop_falls_back
