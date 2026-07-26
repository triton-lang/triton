#ifndef TRITONNVIDIAGPU_ATTRDEFS
#define TRITONNVIDIAGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.td"
include "mlir/IR/EnumAttr.td"

def TTG_TensorMemorySpace : AttrDef<TritonNvidiaGPU_Dialect, "TensorMemorySpace"> {
  let mnemonic = "tensor_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to tensor memory.
    The memory is laid out in blocks of size blockM x blockN. Each block is distributed
    across TMEM 128 rows.

    Blocks are distributed along M dimension first and then N dimension. This is an arbitrary
    convention that needs to be followed by operations reading/writing to TMEM.

    a tensor <128x128xf32> with blockM = 64 and blockN = 32 will be distributed as follows:

        \ col    0        1            31         32            64            96           127
    rows: 0  ( 0,  0) ( 0,  1) ... ( 0,  31)  ( 0,  32) ... ( 0,  64) ... ( 0,  96) ... ( 0,  127)
          1
         ...
          15 (15,  0) (15,  1) ... (15,  31)  (15,  32) ... (15,  64) ... (15,  96) ... (15,  127)
          16 (64,  0) (64,  1) ... (64,  31)  (64,  32) ... (64,  64) ... (64,  96) ... (64,  127)
         ...
          31 (79,  0) (79,  1) ... (79,  31)  (79,  32) ... (79,  64) ... (79,  96) ... (79,  127)
          32 (16,  0) (16,  1) ... (16,  31)  (16,  32) ... (16,  64) ... (16,  96) ... (16,  127)
         ..
         127 (127, 0) (127, 1) ... (127, 31) (127, 32) ... (127, 64) ... (127, 96) ... (127, 127)
  }];
}

def TTNG_TMEMLoadReduceModifierAttr : I32EnumAttr<
    "TMEMLoadReduceModifier", "",
    [
        I32EnumAttrCase<"MIN", 1, "min">,
        I32EnumAttrCase<"MAX", 2, "max">,
    ]> {
    let cppNamespace = "::mlir::triton::nvidia_gpu";
    let genSpecializedAttr = 0;
}
def TTNG_TMEMLoadReduceModifierEnum : EnumAttr<TritonNvidiaGPU_Dialect, TTNG_TMEMLoadReduceModifierAttr, "redOp"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNG_TensorMemoryScalesBlockRepOrderAttr : I32EnumAttr<
    "TensorMemoryScalesBlockRepOrder", "",
    [
        I32EnumAttrCase<"MN_THEN_K", 0, "mnThenK">,
        I32EnumAttrCase<"K_THEN_MN", 1, "kThenMn">,
    ]> {
    let cppNamespace = "::mlir::triton::nvidia_gpu";
    let genSpecializedAttr = 0;
}
def TTNG_TensorMemoryScalesBlockRepOrderEnum
    : EnumAttr<TritonNvidiaGPU_Dialect,
               TTNG_TensorMemoryScalesBlockRepOrderAttr, "blockRepOrder"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTNG_CGALayoutRankTwoParam
    : AttrOrTypeParameter<"::mlir::triton::gpu::CGAEncodingAttr",
                          "rank-2 CGA layout"> {
  let parser = [{
    [&]() -> ::mlir::FailureOr<::mlir::triton::gpu::CGAEncodingAttr> {
      return ::mlir::triton::nvidia_gpu::parseCGALayoutRankTwo($_parser);
    }()
  }];
  let printer = [{
    ::mlir::triton::nvidia_gpu::printCGALayoutRankTwo($_printer, $_self);
  }];
  let defaultValue =
      "::mlir::triton::gpu::CGAEncodingAttr::get1CTALayout($_ctxt, 2)";
}

def TTG_TensorMemoryEncodingAttr
    : AttrDef<TritonNvidiaGPU_Dialect, "TensorMemoryEncoding",
              [LayoutEncodingTrait]> {
  let mnemonic = "tensor_memory_encoding";
  let attrName = "triton.gpu.tensor_memory_encoding";
  let description = [{
    An encoding to represent the different way the tensor memory is laid out.
    `colStride` describes the stride in elements along the column dimension,
    that is, the stride between two elements in the same row.
    When colStride is 1 the tensor memory is packed. When colStride > 1, the
    tensor memory between elements is undefined.
    `twoCTAs` indicates that the tensor memory is laid out for twoCTA mode,
    i.e., `cta_group::2`.
  }];
  let parameters = (
    ins
    "unsigned":$blockM,
    "unsigned":$blockN,
    "unsigned":$colStride,
    TTNG_CGALayoutRankTwoParam:$CGALayout,
    DefaultValuedParameter<"bool", "false">:$twoCTAs,
    DefaultValuedParameter<"bool", "false">:$fp4Padded
  );
  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins "unsigned":$blockM,
                     "unsigned":$blockN,
                     "unsigned":$colStride,
                     "::mlir::triton::gpu::CGAEncodingAttr":$CGALayout,
                     CArg<"bool", "false">:$twoCTAs,
                     CArg<"bool", "false">:$fp4Padded), [{
      return $_get($_ctxt, blockM, blockN, colStride, CGALayout, twoCTAs,
                   fp4Padded);
    }]>
  ];
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` struct(params) `>`";
}

def TTG_TensorMemoryScalesEncodingAttr
    : AttrDef<TritonNvidiaGPU_Dialect, "TensorMemoryScalesEncoding",
              [LayoutEncodingTrait]> {
  let mnemonic = "tensor_memory_scales_encoding";
  let attrName = "triton.gpu.tensor_memory_scales_encoding";
  let description = [{
    An encoding to represent the layout of tensor memory scales.
    As described in the PTX doc, blocked scales in TMEM must be in a special layout. They are organized
    as a multiple copies of a block, each of which having the size 32x4x4B. Moreover, such blocks are duplicated
    over 4 warps to fill entire 128 rows of TMEM. This encoding indicates that a tensor in TMEM is in such a special
    layout.
    The parameter blockRepOrder determines the ordering of repeated scale blocks along TMEM columns. Here `mn`
    denotes the MMA instruction's non-K matrix dimension: `M` for A scales and `N` for B scales. `mnThenK`
    advances along that dimension before advancing K, while `kThenMn` advances along K before advancing the
    non-K dimension. When one MMA instruction consumes only one scale block along K, the value of this parameter
    does not matter. Otherwise, it must match the hardware instruction layout. For example, Rubin NVFP4 A scales
    with `BLOCK_M = 256` require `kThenMn`, while B scales keep the default `mnThenK`.
  }];
  let parameters = (
    ins
    TTNG_CGALayoutRankTwoParam:$CGALayout,
    DefaultValuedParameter<
        "::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder",
        "::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder::MN_THEN_K">:$blockRepOrder
  );
  let builders = [
    AttrBuilder<(ins "::mlir::triton::gpu::CGAEncodingAttr":$CGALayout), [{
      return get(context, CGALayout,
                 ::mlir::triton::nvidia_gpu::TensorMemoryScalesBlockRepOrder::MN_THEN_K);
    }]>
  ];
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif
