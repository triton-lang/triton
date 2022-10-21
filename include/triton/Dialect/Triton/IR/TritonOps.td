#ifndef TRITON_OPS
#define TRITON_OPS

include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // NoSideEffect
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td" // NoSideEffect

def TensorSizeTrait : NativeOpTrait<"TensorSizeTrait">;
def SameOperandsAndResultEncoding : NativeOpTrait<"SameOperandsAndResultEncoding">;
def SameOperandsEncoding : NativeOpTrait<"SameOperandsEncoding">;

//
// Op Base
//
class TT_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Dialect, mnemonic, !listconcat(traits, [TensorSizeTrait])> {
}

//
// CastOps
//
// Use cast ops in arith:
//   bitcast
//   fptoui, fptosi, uitofp, sitofp,
//   extf, tructf,
//   extui, extsi, tructi
def TT_IntToPtrOp : TT_Op<"int_to_ptr", [SameOperandsAndResultShape, 
                                         SameOperandsAndResultEncoding, 
                                         NoSideEffect,
                                         /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast int64 to pointer";

    let arguments = (ins TT_I64Like:$from);

    let results = (outs TT_PtrLike:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";
}

def TT_PtrToIntOp : TT_Op<"ptr_to_int", [SameOperandsAndResultShape, 
                                         SameOperandsAndResultEncoding,
                                         NoSideEffect,
                                         /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast pointer to int64";

    let arguments = (ins TT_PtrLike:$from);

    let results = (outs TT_I64Like:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";
}

def TT_FpToFp : TT_Op<"fp_to_fp", [SameOperandsAndResultShape, 
                                   SameOperandsAndResultEncoding,
                                   NoSideEffect,
                                   /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Floating point casting for custom types";

    let description = [{
        Floating point casting for custom types (F8, BF8).

        F8 <-> BF8, FP16, FP32
        BF8 <-> F8, FP16, FP32
    }];

    let arguments = (ins TT_FloatLike:$from);

    let results = (outs TT_FloatLike:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";

    // TODO: We need a verifier here.
}

//
// Pointer Arith Ops
//

def TT_AddPtrOp : TT_Op<"addptr",
                     [NoSideEffect, 
                     SameOperandsAndResultShape,
                     SameOperandsAndResultEncoding,
                      TypesMatchWith<"result type matches ptr type",
                                     "result", "ptr", "$_self">,
                      TypesMatchWith<"result shape matches offset shape",
                                     "result", "offset",
                                     "getI32SameShape($_self)">]> {
    let arguments = (ins TT_PtrLike:$ptr, TT_I32Like:$offset);

    let results = (outs TT_PtrLike:$result);

    let assemblyFormat = "$ptr `,` $offset attr-dict `:` type($result)";
}


//
// Load/Store Ops
//
def TT_LoadOp : TT_Op<"load",
                      [SameOperandsAndResultShape,
                       SameOperandsAndResultEncoding,
                       AttrSizedOperandSegments,
                       MemoryEffects<[MemRead]>,
                       TypesMatchWith<"infer ptr type from result type",
                                      "result", "ptr", "getPointerTypeSameShape($_self)">,
                       TypesMatchWith<"infer mask type from result type or none",
                                      "result", "mask", "getI1SameShape($_self)",
                                      "($_op.getOperands().size() <= 1) || std::equal_to<>()">,
                       TypesMatchWith<"infer other type from result type or none",
                                      "result", "other", "$_self",
                                      "($_op.getOperands().size() <= 2) || std::equal_to<>()">]> {
    let summary = "load";

    let arguments = (ins TT_PtrLike:$ptr, Optional<TT_BoolLike>:$mask, Optional<TT_Type>:$other,
                         TT_CacheModifierAttr:$cache, TT_EvictionPolicyAttr:$evict,
                         BoolAttr:$isVolatile);

    let results = (outs TT_Type:$result);

    let builders = [
        OpBuilder<(ins "Value":$ptr, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        OpBuilder<(ins "Value":$ptr, "Value":$mask, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
    ];

    // let assemblyFormat = "operands attr-dict `:` type($result)";
    let parser = [{ return mlir::triton::parseLoadOp(parser, result); }];

    let printer = [{ return mlir::triton::printLoadOp(p, *this); }];

    let hasCanonicalizer = 1;
}

def TT_StoreOp : TT_Op<"store",
                       [SameOperandsShape,
                        SameOperandsEncoding,
                        MemoryEffects<[MemWrite]>,
                        TypesMatchWith<"infer ptr type from value type",
                                       "value", "ptr",
                                       "getPointerTypeSameShape($_self)">,
                        TypesMatchWith<"infer mask type from value type",
                                       "value", "mask", "getI1SameShape($_self)",
                                       "($_op.getOperands().size() <= 2) || std::equal_to<>()">]> {
    let summary = "store";

    let arguments = (ins TT_PtrLike:$ptr, TT_Type:$value, Optional<TT_BoolLike>:$mask);

    let builders = [
        OpBuilder<(ins "Value":$ptr, "Value":$value)>,
    ];

    // let assemblyFormat = "operands attr-dict `:` type($value)";
    let parser = [{ return mlir::triton::parseStoreOp(parser, result); }];

    let printer = [{ return mlir::triton::printStoreOp(p, *this); }];

    let hasCanonicalizer = 1;
}

//
// Atomic Op
//
def TT_AtomicRMWOp : TT_Op<"atomic_rmw", [SameOperandsAndResultShape,
                                          SameOperandsAndResultEncoding]> {
    let summary = "atomic rmw";

    let description = [{
        load data at $ptr, do $rmw_op with $val, and store result to $ptr.

        return old value at $ptr
    }];

    let arguments = (ins TT_AtomicRMWAttr:$atomic_rmw_op, TT_PtrTensor:$ptr,
                         TT_Type:$val, I1Tensor:$mask);

    let results = (outs TT_Type:$result);
}

def TT_AtomicCASOp : TT_Op<"atomic_cas", [SameOperandsAndResultShape,
                                          SameOperandsAndResultEncoding]> {
    let summary = "atomic cas";

    let description = [{
        compare $cmp with data $old at location $ptr,

        if $old == $cmp, store $val to $ptr,

        else store $old to $ptr,

        return $old
    }];

    let arguments = (ins TT_Ptr:$ptr, TT_Type:$cmp, TT_Type:$val);

    let results = (outs TT_Type:$result);
}


//
// Shape Manipulation Ops
//
def TT_SplatOp : TT_Op<"splat", [NoSideEffect, 
                                 SameOperandsAndResultElementType]> {
    let summary = "splat";

    let arguments = (ins TT_Type:$src);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    let hasFolder = 1;
}

def TT_ExpandDimsOp : TT_Op<"expand_dims", [NoSideEffect, 
                                            DeclareOpInterfaceMethods<InferTypeOpInterface>, 
                                            SameOperandsAndResultElementType]> {
    let summary = "expand_dims";

    let arguments = (ins TT_Tensor:$src, I32Attr:$axis);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

def TT_ViewOp : TT_Op<"view", [NoSideEffect, 
                               SameOperandsAndResultElementType]> {
    let summary = "view";

    let arguments = (ins TT_Tensor:$src);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

}

def TT_BroadcastOp : TT_Op<"broadcast", [NoSideEffect, 
                                         SameOperandsAndResultElementType]> {
    let summary = "broadcast. No left-padding as of now.";

    let arguments = (ins TT_Type:$src);

    let results = (outs TT_Type:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    let hasFolder = 1;
}

def TT_CatOp : TT_Op<"cat", [NoSideEffect, 
                             SameOperandsAndResultElementType]> {
    let summary = "concatenate 2 tensors";

    let arguments = (ins TT_Tensor:$lhs, TT_Tensor:$rhs);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

//
// SPMD Ops
//
def TT_GetProgramIdOp : TT_Op<"get_program_id", [NoSideEffect]> {
    let arguments = (ins I32Attr:$axis);

    let results = (outs I32:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}

def TT_GetNumProgramsOp : TT_Op<"get_num_programs", [NoSideEffect]> {
    let arguments = (ins I32Attr:$axis);

    let results = (outs I32:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}

//
// Dot Op
//
def TT_DotOp : TT_Op<"dot", [NoSideEffect, 
                             DeclareOpInterfaceMethods<InferTypeOpInterface>,
                             TypesMatchWith<"result's type matches accumulator's type",
                                            "d", "c", "$_self">]> {
    let summary = "dot";

    let description = [{
        $d = matrix_multiply($a, $b) + $c
    }];

    let arguments = (ins TT_FpIntTensor:$a, TT_FpIntTensor:$b, TT_FpIntTensor:$c, BoolAttr:$allowTF32, BoolAttr:$transA, BoolAttr:$transB);

    let results = (outs TT_FpIntTensor:$d);

    let assemblyFormat = "$a`,` $b`,` $c attr-dict `:` type($a) `*` type($b) `->` type($d)";
}

//
// Reduce Op
//
def TT_ReduceOp : TT_Op<"reduce", [NoSideEffect,
                                   DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "reduce";

    let arguments = (ins TT_RedOpAttr:$redOp, TT_Tensor:$operand, I32Attr:$axis);

    let results = (outs TT_Type:$result);

    let builders = [
        OpBuilder<(ins "triton::RedOp":$redOp, "Value":$operand, "int":$axis)>,
    ];

    let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";

}

//
// External elementwise op
//
def TT_ExtElemwiseOp : TT_Op<"ext_elemwise", [NoSideEffect, Elementwise, SameOperandsAndResultShape,
                                              SameOperandsAndResultEncoding,
                                              SameVariadicOperandSize]> {
    let summary = "ext_elemwise";

    let description = [{
        call an external function $symbol implemented in $libpath/$libname with $args

        return $libpath/$libname:$symbol($args...)
    }];

    let arguments = (ins Variadic<TT_Tensor>:$args, StrAttr:$libname, StrAttr:$libpath, StrAttr:$symbol);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "operands attr-dict `:` type(operands) `->` type($result)";

}

//
// Make Range Op
//
// TODO: should have ConstantLike as Trait
def TT_MakeRangeOp : TT_Op<"make_range", [NoSideEffect]> {
    let summary = "make range";

    let description = [{
        Returns an 1D int32 tensor.

        Values span from $start to $end (exclusive), with step = 1
    }];

    let arguments = (ins I32Attr:$start, I32Attr:$end);

    let results = (outs TT_IntTensor:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}

#endif // Triton_OPS
