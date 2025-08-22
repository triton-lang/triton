// RUN: triton-opt %s -canonicalize | FileCheck %s

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0], [64, 0]], block = []}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {

// CHECK-LABEL: @test_dce_tmem_alloc
tt.func @test_dce_tmem_alloc(%arg: tensor<128x4xi8, #linear>) {
  // CHECK-NOT: ttng.tmem_alloc
  %a = ttng.tmem_alloc %arg : (tensor<128x4xi8, #linear>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
  // CHECK-NEXT: tt.return
  tt.return
}

// CHECK-LABEL: @reinterpret_fold
tt.func @reinterpret_fold(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> {
  %0 = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
  // CHECK-NEXT: return %arg0
  tt.return %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
}

}  // end module
