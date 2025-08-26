module attributes {nvws.group.g = {num_warps = 4 : i32, start_warp = 0 : i32}, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @optimize_broadcast(%arg0: i32) {
    %c0_i32 = arith.constant {groups = [@nvws.group.g]} 0 : i32
    %c1_i32 = arith.constant {groups = [@nvws.group.g]} 1 : i32
    scf.for %arg1 = %c0_i32 to %arg0 step %c1_i32  : i32 {
      %0 = "producer"() {data, groups = [@nvws.group.g], ttg.partition = 0 : i32} : () -> tensor<128xf32>
      %1 = tt.expand_dims %0 {axis = 0 : i32, data, groups = [@nvws.group.g]} : tensor<128xf32> -> tensor<1x128xf32>
      %2 = tt.broadcast %1 {data, groups = [@nvws.group.g]} : tensor<1x128xf32> -> tensor<128x128xf32>
      "use"(%2) {data, groups = [@nvws.group.g], ttg.partition = 0 : i32} : (tensor<128x128xf32>) -> ()
      "use"(%2) {data, groups = [@nvws.group.g], ttg.partition = 1 : i32} : (tensor<128x128xf32>) -> ()
    } {groups = [@nvws.group.g], tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

