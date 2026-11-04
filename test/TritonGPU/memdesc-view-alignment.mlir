// RUN: triton-opt %s -split-input-file -verify-diagnostics

#weak = #ttg.shared_linear<{offset = [[1], [2], [4], [8]], block = []}, alignment = 2>
#strong = #ttg.shared_linear<{offset = [[1], [2], [4], [8]], block = []}, alignment = 16>
#smem = #ttg.shared_memory
tt.func @reinterpret_weak_base(%src: !ttg.memdesc<16xi32, #weak, #smem>) {
  // expected-error @+1 {{result view requires 16-byte origin alignment, but the source guarantees 2}}
  %dst = ttg.memdesc_reinterpret %src : !ttg.memdesc<16xi32, #weak, #smem> -> !ttg.memdesc<16xi32, #strong, #smem>
  tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
tt.func @reinterpret_shifted_stages(%src: !ttg.memdesc<4x1xf16, #shared, #smem>) {
  %stages = ttg.memdesc_subslice %src [1, 0] : !ttg.memdesc<4x1xf16, #shared, #smem> -> !ttg.memdesc<2x1xf16, #shared, #smem, 4x1>
  // The two stages own four contiguous bytes starting at byte two.
  // expected-error @+1 {{result view requires 4-byte origin alignment, but the source guarantees 2}}
  %dst = ttg.memdesc_reinterpret %stages : !ttg.memdesc<2x1xf16, #shared, #smem, 4x1> -> !ttg.memdesc<1xi32, #shared, #smem>
  tt.return
}

// -----

#page = #ttg.padded_shared<[64:+1] {order = [0], shape = [128]}>
#ring = #ttg.padded_shared<[64:+1] {order = [0], shape = [64]}>
#smem = #ttg.shared_memory
tt.func @reinterpret_hidden_padded_stage(%src: !ttg.memdesc<128xf16, #page, #smem>) {
  // An indexed source stage can start at (128 + 2) * 2 bytes.
  // expected-error @+1 {{result view requires 16-byte origin alignment, but the source guarantees 4}}
  %dst = ttg.memdesc_reinterpret %src : !ttg.memdesc<128xf16, #page, #smem> -> !ttg.memdesc<2x64xf16, #ring, #smem>
  tt.return
}

// -----

#full = #ttg.shared_linear<{offset = [[1], [2], [4], [8]], block = []}, alignment = 16>
#weak = #ttg.shared_linear<{offset = [[1], [2], [4], [8]], block = []}, alignment = 4>
#one = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
tt.func @weaken_and_widen(%src: !ttg.memdesc<16xf64, #full, #smem>, %bytes: !ttg.memdesc<16xi8, #full, #smem>) {
  %weak = ttg.memdesc_reinterpret %src : !ttg.memdesc<16xf64, #full, #smem> -> !ttg.memdesc<16xf64, #weak, #smem>
  %slice = ttg.memdesc_subslice %bytes [2] : !ttg.memdesc<16xi8, #full, #smem> -> !ttg.memdesc<2xi8, #full, #smem, 16>
  %word = ttg.memdesc_reinterpret %slice : !ttg.memdesc<2xi8, #full, #smem, 16> -> !ttg.memdesc<1xi16, #one, #smem>
  tt.return
}
