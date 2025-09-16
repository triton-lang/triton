module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", ttg.warp_specialize.tag = 0 : i32} {
  tt.func @optimize_broadcast(%arg0: i32) {
    %c0_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 0 : i32
    %c1_i32 = arith.constant {ttg.partition = array<i32: 0, 1>} 1 : i32
    scf.for %arg1 = %c0_i32 to %arg0 step %c1_i32  : i32 {
      %0 = "producer"() {data, ttg.partition = array<i32: 0>} : () -> tensor<128xf32>
      %1 = tt.expand_dims %0 {axis = 0 : i32, data, ttg.partition = array<i32: 0>} : tensor<128xf32> -> tensor<1x128xf32>
      %2 = tt.broadcast %1 {data, ttg.partition = array<i32: 1>} : tensor<1x128xf32> -> tensor<128x128xf32>
      "use"(%2) {data, ttg.partition = array<i32: 0>} : (tensor<128x128xf32>) -> ()
      "use"(%2) {data, ttg.partition = array<i32: 1>} : (tensor<128x128xf32>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

