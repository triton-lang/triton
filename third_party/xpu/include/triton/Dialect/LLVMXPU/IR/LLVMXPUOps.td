//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef LLVMXPUIR_OPS
#define LLVMXPUIR_OPS

include "mlir/IR/OpBase.td" // Trait
include "mlir/IR/DialectBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td" // LLVM_OpBase
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType


//===----------------------------------------------------------------------===//
// LLVMXPU dialect definitions
//===----------------------------------------------------------------------===//

def LLVMXPU_Dialect : Dialect {
  let name = "llvm_xpu";

  let cppNamespace = "::mlir::LLVM::XPU";

  let hasOperationAttrVerify = 1;

  let dependentDialects = ["LLVM::LLVMDialect"];

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "xpu.kernel"; }
  }];

  // let useDefaultAttributePrinterParser = 1;
}



//===----------------------------------------------------------------------===//
// LLVMXPU op definitions
//===----------------------------------------------------------------------===//

class XPU_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<LLVMXPU_Dialect, mnemonic, traits> {
}


//===----------------------------------------------------------------------===//
// LLVMXPU intrinsic operations
//===----------------------------------------------------------------------===//

class XPU_IntrOp<string mnem, list<Trait> traits = [],
                  int numResults>
  : LLVM_IntrOpBase<LLVMXPU_Dialect, mnem, "xpu_" # !subst(".", "_", mnem),
                    /*list<int> overloadedResults=*/[],
                    /*list<int> overloadedOperands=*/[],
                    traits, numResults>;


def XPU_GM2LMOp : XPU_IntrOp<"gm2lm", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src, // TODO[dyq]: Check i8?
                         LLVM_AnyPointer:$dst,
                         I32:$offset,
                         I32:$size);
    string llvmBuilder = [{
        // XPU2, offset only has 7 bits, so we let LLVM fold the value
        auto addr  = builder.CreatePtrToInt($src, $offset->getType());
        auto faddr = builder.CreateAdd(addr, $offset);
        auto srcptr  = builder.CreateIntToPtr(faddr, $src->getType());
        auto zero = builder.getInt32(0);
        createIntrinsicCall(builder, llvm::Intrinsic::xpu_gm2lm, {$dst, srcptr, zero, $size});
    }];
}

def XPU_LM2GMOp : XPU_IntrOp<"lm2gm", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src,
                         LLVM_AnyPointer:$dst,
                         I32:$offset,
                         I32:$size);
    string llvmBuilder = [{
        // XPU2, offset only has 7 bits, so we let LLVM fold the value
        auto addr  = builder.CreatePtrToInt($dst, $offset->getType());
        auto faddr = builder.CreateAdd(addr, $offset);
        auto dstptr  = builder.CreateIntToPtr(faddr, $dst->getType());
        auto zero = builder.getInt32(0);
        createIntrinsicCall(builder, llvm::Intrinsic::xpu_lm2gm, {dstptr, zero, $src, $size});
    }];
}

def XPU_SM2GMOp : XPU_IntrOp<"sm2gm", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src,
                         LLVM_AnyPointer:$dst,
                         I32:$offset,
                         I32:$size);
    string llvmBuilder = [{
        // XPU2, offset only has 7 bits, so we let LLVM fold the value
        auto addr  = builder.CreatePtrToInt($dst, $offset->getType());
        auto faddr = builder.CreateAdd(addr, $offset);
        auto dstptr  = builder.CreateIntToPtr(faddr, $dst->getType());
        auto zero = builder.getInt32(0);
        createIntrinsicCall(builder, llvm::Intrinsic::xpu_sm2gm, {dstptr, zero, $src, $size});
    }];
}


def XPU_GM2LMOp_v3 : XPU_IntrOp<"gm2lm_v3", [], 0> {
  let arguments = (ins LLVM_AnyPointer:$src,
                       LLVM_AnyPointer:$dst,
                       I32:$offset,
                       I32:$size);
  string llvmBuilder = [{
    auto isrc = builder.CreatePtrToInt($src, builder.getInt64Ty());
    auto ioft = builder.CreateZExtOrTrunc($offset, builder.getInt64Ty());
    auto asrc = builder.CreateAdd(isrc, ioft);
    auto fsrc = builder.CreateIntToPtr(asrc, $src->getType());
    auto zero = builder.getInt32(0);
    createIntrinsicCall(builder, llvm::Intrinsic::xpu_gm2lm_v3, {$dst, fsrc, zero, $size});
  }];
}

def XPU_LM2GMOp_v3 : XPU_IntrOp<"lm2gm_v3", [], 0> {
  let arguments = (ins LLVM_AnyPointer:$src,
                       LLVM_AnyPointer:$dst,
                       I32:$offset,
                       I32:$size);
  string llvmBuilder = [{
    auto idst = builder.CreatePtrToInt($dst, builder.getInt64Ty());
    auto ioft = builder.CreateZExtOrTrunc($offset, builder.getInt64Ty());
    auto adst = builder.CreateAdd(idst, ioft);
    auto fdst = builder.CreateIntToPtr(adst, $dst->getType());
    auto zero = builder.getInt32(0);
    createIntrinsicCall(builder, llvm::Intrinsic::xpu_lm2gm_v3, {fdst, $src, zero, $size});
  }];
}

def XPU_SM2GMOp_v3 : XPU_IntrOp<"sm2gm_v3", [], 0> {
  let arguments = (ins LLVM_AnyPointer:$src,
                       LLVM_AnyPointer:$dst,
                       I32:$offset,
                       I32:$size);
  string llvmBuilder = [{
    auto idst = builder.CreatePtrToInt($dst, builder.getInt64Ty());
    auto ioft = builder.CreateZExtOrTrunc($offset, builder.getInt64Ty());
    auto adst = builder.CreateAdd(idst, ioft);
    auto fdst = builder.CreateIntToPtr(adst, $dst->getType());
    auto zero = builder.getInt32(0);
    createIntrinsicCall(builder, llvm::Intrinsic::xpu_sm2gm_v3, {fdst, $src, zero, $size});
  }];
}

//===----------------------------------------------------------------------===//
// XPU special register op definitions
//===----------------------------------------------------------------------===//

class XPU_SpecialRegisterOp<string mnemonic, list<Trait> traits = []> :
  XPU_IntrOp<mnemonic, !listconcat(traits, [Pure]), 1> {
  let arguments = (ins);
  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Physiscal cluster index and range (0-7)
//===----------------------------------------------------------------------===//

def XPU_ClusterIdOp : XPU_SpecialRegisterOp<"cluster_id">;

//===----------------------------------------------------------------------===//
// Core index and range
//===----------------------------------------------------------------------===//

def XPU_CoreIdOp : XPU_SpecialRegisterOp<"core_id">;

//===----------------------------------------------------------------------===//
// XPU load parameters op definitions
//===----------------------------------------------------------------------===//

def XPU_LoadParamOp : XPU_Op<"load_param"> {
  let arguments = (ins I32:$num);
  let results = (outs I32:$res);
  string llvmBuilder = [{
    $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu_load_param, {$num});
  }];
  let assemblyFormat = "$num attr-dict";
}

//===----------------------------------------------------------------------===//
// XPU mfence v2 op definitions
//===----------------------------------------------------------------------===//

def XPU_MfenceOp : XPU_Op<"mfence"> {
  let arguments = (ins I32:$num);
  string llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::xpu2_mfence, {$num});
  }];
  let assemblyFormat = "$num attr-dict";
}

//===----------------------------------------------------------------------===//
// XPU barrier for inter-cluster
// mfence + sync_cluster
//===----------------------------------------------------------------------===//
def XPU_BarrierOp : XPU_Op<"barrier"> {
  let arguments = (ins);
  string llvmBuilder = [{
    auto five = builder.getInt32(7);
    createIntrinsicCall(builder, llvm::Intrinsic::xpu2_mfence, {five});
    auto mask = builder.getInt32(65535);
    createIntrinsicCall(builder, llvm::Intrinsic::xpu_csr_set_sync_group, {mask});
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// XPU set haddr op definitions
//===----------------------------------------------------------------------===//

def XPU_SetHaddrOp : XPU_Op<"set_haddr"> {
  let arguments = (ins I32:$num);
  string llvmBuilder = [{
    createIntrinsicCall(builder, llvm::Intrinsic::xpu_set_haddr, {$num});
  }];
  let assemblyFormat = "$num attr-dict";
}

//===----------------------------------------------------------------------===//
// XPU log definitions
//===----------------------------------------------------------------------===//

def XPU_LogOp : XPU_Op<"log", [Elementwise, SameOperandsAndResultType, SameOperandsAndResultShape]> {
  let arguments = (ins F32:$operand);
  let results = (outs F32:$result);
  string llvmBuilder = [{
    $result = createIntrinsicCall(builder, llvm::Intrinsic::xpu_log2f, {$operand});
  }];
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// XPU min definitions
//===----------------------------------------------------------------------===//
def XPU_MinOp : XPU_IntrOp<"min", [], 0> {
    let arguments = (ins I32:$lhs,
                         I32:$rhs);
    let results = (outs I32:$res);
    string llvmBuilder = [{
        // XTDK only supports llvm.xpu.min
        $res = createIntrinsicCall(builder, llvm::Intrinsic::smin, {$lhs, $rhs});
    }];
}

//===----------------------------------------------------------------------===//
// XPU VGatherF definitions
//===----------------------------------------------------------------------===//
def XPU_VGatherFOp : XPU_IntrOp<"vgather_mask16_mr1", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src,
                         LLVM_AnyVector:$mask);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu2_vgather_mask16_mr1, {$src, $mask});
    }];
}

def XPU_VGatherHFOp : XPU_IntrOp<"vgather_mr1", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src,
                         LLVM_AnyVector:$mask);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu2_vgather_mr1, {$src, $mask});
    }];
}

//===----------------------------------------------------------------------===//
// XPU vvor_f_mh_rn definitions
//===----------------------------------------------------------------------===//
def XPU_VVOR_F_MHOp : XPU_IntrOp<"vvor_f_mh_rn", [], 0> {
    let arguments = (ins LLVM_AnyVector:$mask,
                         LLVM_AnyVector:$a,
                         LLVM_AnyVector:$b,
                         LLVM_AnyVector:$c);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu2_vvor_f_mh_rn, {$mask, $a, $b, $c});
    }];
}

def XPU_VVOR_HF_MHOp : XPU_IntrOp<"vvor_hf_mh_rn", [], 0> {
    let arguments = (ins LLVM_AnyVector:$mask,
                         LLVM_AnyVector:$a,
                         LLVM_AnyVector:$b,
                         LLVM_AnyVector:$c);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu3_vvor_hf_mh_rn, {$mask, $a, $b, $c});
    }];
}

def XPU_VVOR_S_MHOp : XPU_IntrOp<"vvor_s_mh", [], 0> {
    let arguments = (ins LLVM_AnyVector:$mask,
                         LLVM_AnyVector:$a,
                         LLVM_AnyVector:$b,
                         LLVM_AnyVector:$c);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu2_vvor_s_mh, {$mask, $a, $b, $c});
    }];
}

//===----------------------------------------------------------------------===//
// XPU vload/vstore definitions
//===----------------------------------------------------------------------===//
def XPU_VLOAD_MZOp : XPU_IntrOp<"vload_mz", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src_ptr,
                         LLVM_AnyVector:$mask);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto zero = builder.getInt32(0);
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu2_vload_mz, {$mask, $src_ptr, zero});
    }];
}

def XPU_VLOAD_MHOp : XPU_IntrOp<"vload_mh", [], 0> {
    let arguments = (ins LLVM_AnyPointer:$src_ptr,
                         LLVM_AnyVector:$dst_data,
                         LLVM_AnyVector:$mask);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        auto zero = builder.getInt32(0);
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu2_vload_mh, {$mask, $src_ptr, $dst_data, zero});
    }];
}

def XPU_VSTORE_MHOp : XPU_IntrOp<"vstore_mh", [], 0> {
    let arguments = (ins LLVM_AnyVector:$src,
                         LLVM_AnyPointer:$dst_Ptr,
                         LLVM_AnyVector:$mask);
    string llvmBuilder = [{
        auto zero = builder.getInt32(0);
        createIntrinsicCall(builder, llvm::Intrinsic::xpu2_vstore_mh, {$src, $mask, $dst_Ptr, zero});
    }];
}

//===----------------------------------------------------------------------===//
// XPU svsllp/svsrlp definitions
//===----------------------------------------------------------------------===//
def XPU_SVSLLPOp : XPU_IntrOp<"svsllp", [], 0> {
    let arguments = (ins I32:$offset,
                         LLVM_AnyVector:$src);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu2_svsllp_s, {$offset, $src});
    }];
}

def XPU_SVSRLPOp : XPU_IntrOp<"svsrlp", [], 0> {
    let arguments = (ins I32:$offset,
                         LLVM_AnyVector:$src);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu2_svsrlp_s, {$offset, $src});
    }];
}

//===----------------------------------------------------------------------===//
// XPU vmerge definitions
//===----------------------------------------------------------------------===//
def XPU_VMERGE_L_HFOp : XPU_IntrOp<"vmerge_l_hf", [], 0> {
    let arguments = (ins LLVM_AnyVector:$src1,
                         LLVM_AnyVector:$src2);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu3_vmerge_l_hf, {$src1, $src2});
    }];
}

def XPU_VMERGE_H_HFOp : XPU_IntrOp<"vmerge_h_hf", [], 0> {
    let arguments = (ins LLVM_AnyVector:$src1,
                         LLVM_AnyVector:$src2);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu3_vmerge_h_hf, {$src1, $src2});
    }];
}

//===----------------------------------------------------------------------===//
// XPU vscatter definitions
//===----------------------------------------------------------------------===//
def XPU_SCATTER_MHOp : XPU_IntrOp<"vscatter_mh", [], 0> {
    let arguments = (ins LLVM_AnyVector:$value,
                         LLVM_AnyVector:$mask,
                         LLVM_AnyPointer:$dst_ptr,
                         LLVM_AnyVector:$offset);
    string llvmBuilder = [{
        createIntrinsicCall(builder, llvm::Intrinsic::xpu2_vscatter_mh, {$value, $mask, $dst_ptr, $offset});
    }];
}

//===----------------------------------------------------------------------===//
// XPU vshuffle definitions
//===----------------------------------------------------------------------===//
def XPU_VSHUFFLE2Op : XPU_IntrOp<"vshuffle2", [], 0> {
    let arguments = (ins LLVM_AnyVector:$src);
    let results = (outs LLVM_AnyVector:$res);
    string llvmBuilder = [{
        $res = createIntrinsicCall(builder, llvm::Intrinsic::xpu3_vshuffle2_hf, {$src});
    }];
}

#endif // LLVMXPUIR_OPS
