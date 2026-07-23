// RUN: triton-opt %s -allow-unregistered-dialect -test-print-allocation="partition-size=65536" -verify-diagnostics -o /dev/null

// Test allocation with shared memory partitioning enabled (64KB partition size like for AMD GFX1250)
// With partition-size=65536, partition buffers should be placed in different 64KB physical partitions

#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#PADDED_SHARED = #ttg.padded_shared<[256:+8] {order = [1, 0], shape = [16, 32]}>

// 2 partitions, 2 groups each
// Each piece is 1052 bytes, so each partition buffer is 1052 * 2 = 2104 bytes
#PARTITIONED_2P_2G = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #PADDED_SHARED}>

// 4 partitions, 1 group each
// Each piece is 1052 bytes, each partition buffer is 1052 * 1 = 1052 bytes
#PARTITIONED_4P_1G = #ttg.partitioned_shared<{numPartitions = 4, numGroups = 1, partitionDim = 0, partitionLayout = #PADDED_SHARED}>

// 2 partitions, 4 groups each (using swizzled layout)
// 64x32xf16 = 4096 bytes total, 8 pieces = 512 bytes each, partition buffer = 512 * 4 = 2048 bytes
#PARTITIONED_2P_4G = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 4, partitionDim = 0, partitionLayout = #A_SHARED}>

// 4 partitions, 2 groups each (using swizzled layout)
// 64x32xf16 = 4096 bytes total, 8 pieces = 512 bytes each, partition buffer = 512 * 2 = 1024 bytes
#PARTITIONED_4P_2G = #ttg.partitioned_shared<{numPartitions = 4, numGroups = 2, partitionDim = 0, partitionLayout = #A_SHARED}>

// Swizzled (unpadded) 2/4 partitions, 1 group. Piece size depends purely on the
// tensor shape, which the tests below use to land on / straddle 64KB boundaries.
#PARTITIONED_2P_1G_SW = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 1, partitionDim = 0, partitionLayout = #A_SHARED}>
#PARTITIONED_4P_1G_SW = #ttg.partitioned_shared<{numPartitions = 4, numGroups = 1, partitionDim = 0, partitionLayout = #A_SHARED}>

#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {

// Test basic 2 partitions, 2 groups allocation
// expected-remark @below {{partitioned_2p_2g}}
// expected-remark @below {{size = 67640}}
tt.func @partitioned_2p_2g() {
  // 2 partition buffers: one for partition 0 (contains groups 0,1) and one for partition 1 (contains groups 0,1)
  // Each partition buffer is 2104 bytes (1052 bytes per piece * 2 groups)
  // With 64KB partition size, the two partition buffers are placed in different 64KB physical partitions
  // expected-remark @below {{offset = 0, size = 2104}}
  // expected-remark @below {{offset = 65536, size = 2104}}
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_2P_2G, #ttg.shared_memory, mutable>
  tt.return
}

// Test 4 partitions, 1 group each - each partition is a neighbor to all others
// expected-remark @below {{partitioned_4p_1g}}
// expected-remark @below {{size = 197660}}
tt.func @partitioned_4p_1g() {
  // 4 partition buffers, each containing 1 group = 1052 bytes each
  // All 4 partitions are neighbors, so they must be in different 64KB physical partitions
  // expected-remark @below {{offset = 0, size = 1052}}
  // expected-remark @below {{offset = 65536, size = 1052}}
  // expected-remark @below {{offset = 131072, size = 1052}}
  // expected-remark @below {{offset = 196608, size = 1052}}
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_4P_1G, #ttg.shared_memory, mutable>
  tt.return
}

// Test 2 partitions, 4 groups each with swizzled layout
// expected-remark @below {{partitioned_2p_4g}}
// expected-remark @below {{size = 67584}}
tt.func @partitioned_2p_4g() {
  // 2 partition buffers, each containing 4 groups concatenated = 2048 bytes each
  // With 64KB partition size, the two partition buffers are placed in different 64KB physical partitions
  // expected-remark @below {{offset = 0, size = 2048}}
  // expected-remark @below {{offset = 65536, size = 2048}}
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_2P_4G, #ttg.shared_memory, mutable>
  tt.return
}

// Test 4 partitions, 2 groups each - four partitions need four different physical partitions
// expected-remark @below {{partitioned_4p_2g}}
// expected-remark @below {{size = 197632}}
tt.func @partitioned_4p_2g() {
  // 4 partition buffers, each containing 2 groups concatenated = 1024 bytes each
  // All 4 partitions are neighbors, so they must be in different 64KB physical partitions
  // expected-remark @below {{offset = 0, size = 1024}}
  // expected-remark @below {{offset = 65536, size = 1024}}
  // expected-remark @below {{offset = 131072, size = 1024}}
  // expected-remark @below {{offset = 196608, size = 1024}}
  %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_4P_2G, #ttg.shared_memory, mutable>
  tt.return
}

// Test partitioned allocation alongside non-partitioned allocation
// expected-remark @below {{partitioned_with_regular}}
// expected-remark @below {{size = 67640}}
tt.func @partitioned_with_regular() {
  // Non-partitioned allocation: 1024 bytes, packed into partition 0 right after
  // the partitioned buffer's first piece.
  // expected-remark @below {{offset = 2112, size = 1024}}
  %regular = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // Partitioned allocation: 2 partition buffers of 2104 bytes each
  // expected-remark @below {{offset = 0, size = 2104}}
  // expected-remark @below {{offset = 65536, size = 2104}}
  %partitioned = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_2P_2G, #ttg.shared_memory, mutable>
  // Use both allocations so they overlap in liveness
  "use"(%regular) : (!ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable>) -> ()
  "use"(%partitioned) : (!ttg.memdesc<64x32xf16, #PARTITIONED_2P_2G, #ttg.shared_memory, mutable>) -> ()
  tt.return
}

// Test multiple partitioned allocations (both live at the same time)
// expected-remark @below {{multiple_partitioned}}
// expected-remark @below {{size = 69696}}
tt.func @multiple_partitioned() {
  // First partitioned allocation: 2 partition buffers of 2104 bytes each
  // expected-remark @below {{offset = 0, size = 2104}}
  // expected-remark @below {{offset = 65536, size = 2104}}
  %alloc1 = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_2P_2G, #ttg.shared_memory, mutable>
  // Second partitioned allocation: 2 partition buffers of 2048 bytes each.
  // Co-locates alloc2's pieces in the same partitions as alloc1's pieces
  // (alloc2 piece 0 packed right after alloc1 piece 0 in partition 0, etc).
  // expected-remark @below {{offset = 2112, size = 2048}}
  // expected-remark @below {{offset = 67648, size = 2048}}
  %alloc2 = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_2P_4G, #ttg.shared_memory, mutable>
  // Use both allocations so they overlap in liveness
  "use"(%alloc1) : (!ttg.memdesc<64x32xf16, #PARTITIONED_2P_2G, #ttg.shared_memory, mutable>) -> ()
  "use"(%alloc2) : (!ttg.memdesc<64x32xf16, #PARTITIONED_2P_4G, #ttg.shared_memory, mutable>) -> ()
  tt.return
}

// Test liveness/reuse of partitioned buffers
// expected-remark @below {{partitioned_reuse}}
// expected-remark @below {{size = 67640}}
tt.func @partitioned_reuse() {
  // First allocation: 2 partition buffers of 2104 bytes each
  // expected-remark @below {{offset = 0, size = 2104}}
  // expected-remark @below {{offset = 65536, size = 2104}}
  %alloc1 = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_2P_2G, #ttg.shared_memory, mutable>
  ttg.local_dealloc %alloc1 : !ttg.memdesc<64x32xf16, #PARTITIONED_2P_2G, #ttg.shared_memory, mutable>
  // Second allocation after dealloc: should reuse the same memory
  // expected-remark @below {{offset = 0, size = 2048}}
  // expected-remark @below {{offset = 65536, size = 2048}}
  %alloc2 = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_2P_4G, #ttg.shared_memory, mutable>
  tt.return
}

// Degeneration: with partition-size set but no partitioned tensors the analysis
// must behave exactly like the classic first-fit allocator (no partition holes).
// expected-remark @below {{regular_only}}
// expected-remark @below {{size = 2048}}
tt.func @regular_only() {
  // Two live 1024-byte buffers packed back-to-back.
  // expected-remark @below {{offset = 0, size = 1024}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{offset = 1024, size = 1024}}
  %b = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  "use"(%a) : (!ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable>) -> ()
  "use"(%b) : (!ttg.memdesc<32x16xf16, #A_SHARED, #ttg.shared_memory, mutable>) -> ()
  tt.return
}

// Three 2-partition tensors (2048-byte pieces), all live.
// The partitioned first stage packs every tensor's piece 0 into partition 0 and
// every piece 1 into partition 1, so the whole thing fits in 2 physical
// (71680 bytes).
// expected-remark @below {{three_tensor_colocation}}
// expected-remark @below {{size = 71680}}
tt.func @three_tensor_colocation() {
  // partition 0: pieces at 0, 2048, 4096 ; partition 1: at 65536, 67584, 69632.
  // expected-remark @below {{offset = 0, size = 2048}}
  // expected-remark @below {{offset = 65536, size = 2048}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>
  // expected-remark @below {{offset = 2048, size = 2048}}
  // expected-remark @below {{offset = 67584, size = 2048}}
  %b = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>
  // expected-remark @below {{offset = 4096, size = 2048}}
  // expected-remark @below {{offset = 69632, size = 2048}}
  %c = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>
  "use"(%a) : (!ttg.memdesc<64x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>) -> ()
  "use"(%b) : (!ttg.memdesc<64x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>) -> ()
  "use"(%c) : (!ttg.memdesc<64x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>) -> ()
  tt.return
}

// Piece size exactly one physical partition (2048x32xf16 = 131072 bytes total,
// 65536 per partition buffer): the two neighbors land on partition boundaries
// 0 and 65536 with no straddle and no wasted space.
// expected-remark @below {{partition_exact_boundary}}
// expected-remark @below {{size = 131072}}
tt.func @partition_exact_boundary() {
  // expected-remark @below {{offset = 0, size = 65536}}
  // expected-remark @below {{offset = 65536, size = 65536}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<2048x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>
  "use"(%a) : (!ttg.memdesc<2048x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>) -> ()
  tt.return
}

// Partition pieces that straddle a physical-partition boundary must
// still be separated correctly and packed tightly. %a's two pieces (40000 bytes)
// go to partitions 0 and 1; %b's four pieces fill the gaps, with piece 0
// straddling partitions 0/1 (allowed: %b is not a neighbor of %a).
// expected-remark @below {{partitioned_straddle}}
// expected-remark @below {{size = 302144}}
tt.func @partitioned_straddle() {
  // expected-remark @below {{offset = 0, size = 40000}}
  // expected-remark @below {{offset = 80000, size = 40000}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<1250x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>
  // expected-remark @below {{offset = 40000, size = 40000}}
  // expected-remark @below {{offset = 131072, size = 40000}}
  // expected-remark @below {{offset = 196608, size = 40000}}
  // expected-remark @below {{offset = 262144, size = 40000}}
  %b = ttg.local_alloc : () -> !ttg.memdesc<2500x32xf16, #PARTITIONED_4P_1G_SW, #ttg.shared_memory, mutable>
  "use"(%a) : (!ttg.memdesc<1250x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>) -> ()
  "use"(%b) : (!ttg.memdesc<2500x32xf16, #PARTITIONED_4P_1G_SW, #ttg.shared_memory, mutable>) -> ()
  tt.return
}

// Two neighbors each larger than one physical partition (96000 bytes
// -> each spans ~1.5 partitions). Comparing only start partitions would leave
// them sharing a partition; the footprint-based conflict check separates them
// (partitions 0-1 and 2-3).
// expected-remark @below {{two_big_neighbors}}
// expected-remark @below {{size = 227072}}
tt.func @two_big_neighbors() {
  // expected-remark @below {{offset = 0, size = 96000}}
  // expected-remark @below {{offset = 131072, size = 96000}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<3000x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>
  "use"(%a) : (!ttg.memdesc<3000x32xf16, #PARTITIONED_2P_1G_SW, #ttg.shared_memory, mutable>) -> ()
  tt.return
}
}
