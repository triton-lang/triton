#ifndef TRITON_OPS
#define TRITON_OPS

include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/CastInterfaces.td" // CastOpInterface

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
                                         Pure,
                                         /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast int64 to pointer";

    let arguments = (ins TT_I64Like:$from);

    let results = (outs TT_PtrLike:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";
}

def TT_PtrToIntOp : TT_Op<"ptr_to_int", [SameOperandsAndResultShape,
                                         SameOperandsAndResultEncoding,
                                         Pure,
                                         /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast pointer to int64";

    let arguments = (ins TT_PtrLike:$from);

    let results = (outs TT_I64Like:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";
}

// arith.bitcast doesn't support pointers
def TT_BitcastOp : TT_Op<"bitcast", [SameOperandsAndResultShape,
                                     SameOperandsAndResultEncoding,
                                     Pure,
                                     /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast between types of the same bitwidth";

    let arguments = (ins TT_Type:$from);

    let results = (outs TT_Type:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";

    // TODO: Add verifier
}

def TT_FpToFpOp : TT_Op<"fp_to_fp", [SameOperandsAndResultShape,
                                     SameOperandsAndResultEncoding,
                                     Pure,
                                     /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Floating point casting for custom types";

    let description = [{
        Floating point casting for custom types (F8).

        F8 <-> FP16, BF16, FP32, FP64
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
                     [Pure,
                      SameOperandsAndResultShape,
                      SameOperandsAndResultEncoding,
                      TypesMatchWith<"result type matches ptr type",
                                     "result", "ptr", "$_self">]> {
    let arguments = (ins TT_PtrLike:$ptr, TT_IntLike:$offset);

    let results = (outs TT_PtrLike:$result);

    let assemblyFormat = "$ptr `,` $offset attr-dict `:` type($result) `,` type($offset)";
}

def TT_AdvanceOp : TT_Op<"advance",
                         [Pure,
                          TypesMatchWith<"result type matches ptr type",
                                         "result", "ptr", "$_self">]> {
    let summary = "Advance a tensor pointer by offsets";

    let arguments = (ins TT_TensorPtr:$ptr, Variadic<I32>:$offsets);

    let results = (outs TT_TensorPtr:$result);

    let assemblyFormat = "$ptr `,` `[` $offsets `]` attr-dict `:` type($result)";
}

//
// Load/Store Ops
//
def TT_LoadOp : TT_Op<"load",
                      [SameLoadStoreOperandsAndResultShape,
                       SameLoadStoreOperandsAndResultEncoding,
                       AttrSizedOperandSegments,
                       MemoryEffects<[MemRead]>,
                       TypesMatchWith<"infer ptr type from result type",
                                      "result", "ptr", "$_self",
                                      "mlir::OpTrait::impl::verifyLoadStorePointerAndValueType">,
                       TypesMatchWith<"infer mask type from result type or none",
                                      "result", "mask", "getI1SameShape($_self)",
                                      "($_op.getOperands().size() <= 1) || std::equal_to<>()">,
                       TypesMatchWith<"infer other type from result type or none",
                                      "result", "other", "$_self",
                                      "($_op.getOperands().size() <= 2) || std::equal_to<>()">]> {
    let summary = "Load from a tensor of pointers or from a tensor pointer";

    let arguments = (ins AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr, Optional<TT_BoolLike>:$mask,
                         Optional<TT_Type>:$other, OptionalAttr<DenseI32ArrayAttr>:$boundaryCheck,
                         OptionalAttr<TT_PaddingOptionAttr>:$padding, TT_CacheModifierAttr:$cache,
                         TT_EvictionPolicyAttr:$evict, BoolAttr:$isVolatile);

    let results = (outs TT_Type:$result);

    let builders = [
        // A tensor of pointers or a pointer to a scalar
        OpBuilder<(ins "Value":$ptr, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A tensor pointer with boundary check and padding
        OpBuilder<(ins "Value":$ptr, "ArrayRef<int32_t>":$boundaryCheck,
                       "Optional<triton::PaddingOption>":$padding, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A tensor of pointers or a pointer to a scalar with mask
        OpBuilder<(ins "Value":$ptr, "Value":$mask, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A tensor of pointers or a pointer to a scalar with mask and other
        OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A utility function to build the operation with all attributes
        OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other, "Optional<ArrayRef<int32_t>>":$boundaryCheck,
                       "Optional<triton::PaddingOption>":$padding, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>
    ];

    // Format: `tt.load operands attrs : optional(type(ptr)) -> type(result)`
    // We need an extra `optional(type(ptr))` for inferring the tensor pointer type with back compatibility
    let hasCustomAssemblyFormat = 1;

    let hasCanonicalizer = 1;
}

def TT_StoreOp : TT_Op<"store",
                       [SameLoadStoreOperandsShape,
                        SameLoadStoreOperandsEncoding,
                        MemoryEffects<[MemWrite]>,
                        TypesMatchWith<"infer ptr type from value type",
                                       "value", "ptr", "$_self",
                                       "mlir::OpTrait::impl::verifyLoadStorePointerAndValueType">,
                        TypesMatchWith<"infer mask type from value type",
                                       "value", "mask", "getI1SameShape($_self)",
                                       "($_op.getOperands().size() <= 2) || std::equal_to<>()">]> {
    let summary = "Store by a tensor of pointers or by a tensor pointer";

    let arguments = (ins AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr, TT_Type:$value, Optional<TT_BoolLike>:$mask,
                         OptionalAttr<DenseI32ArrayAttr>:$boundaryCheck,
                         DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
                         DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict);

    let builders = [
        // A tensor of pointers or a pointer to a scalar
        OpBuilder<(ins "Value":$ptr, "Value":$value, "triton::CacheModifier":$cache, "triton::EvictionPolicy":$evict)>,
        // A tensor of pointers or a pointer to a scalar with mask
        OpBuilder<(ins "Value":$ptr, "Value":$value, "Value":$mask, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict)>,
        // A tensor pointer with boundary check
        OpBuilder<(ins "Value":$ptr, "Value":$value, "ArrayRef<int32_t>":$boundaryCheck, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict)>
    ];

    // Format: `tt.store operands attrs : optional(type(ptr)), type(val)
    // We need an extra `optional(type(ptr))` for inferring the tensor pointer type with back compatibility
    let hasCustomAssemblyFormat = 1;

    let hasCanonicalizer = 1;
}

//
// Atomic Op
//
def TT_AtomicRMWOp : TT_Op<"atomic_rmw", [SameOperandsAndResultShape,
                                          SameOperandsAndResultEncoding,
                                          MemoryEffects<[MemRead]>,
                                          MemoryEffects<[MemWrite]>,
                                          TypesMatchWith<"infer ptr type from value type",
                                                         "val", "ptr",
                                                         "getPointerTypeSameShape($_self)">,
                                          TypesMatchWith<"infer mask type from value type",
                                                         "val", "mask", "getI1SameShape($_self)",
                                                       "($_op.getOperands().size() <= 2) || std::equal_to<>()">]> {
    let summary = "atomic rmw";

    let description = [{
        load data at $ptr, do $rmw_op with $val, and store result to $ptr.

        return old value at $ptr
    }];

    let arguments = (ins TT_AtomicRMWAttr:$atomic_rmw_op, TT_PtrLike:$ptr,
                         TT_Type:$val, Optional<TT_BoolLike>:$mask);

    let results = (outs TT_Type:$result);
}

def TT_AtomicCASOp : TT_Op<"atomic_cas", [MemoryEffects<[MemRead]>,
                                          MemoryEffects<[MemWrite]>,
                                          SameOperandsAndResultShape,
                                          SameOperandsAndResultEncoding]> {
    let summary = "atomic cas";

    let description = [{
        compare $cmp with data $old at location $ptr,

        if $old == $cmp, store $val to $ptr,

        else store $old to $ptr,

        return $old
    }];

    let arguments = (ins TT_PtrLike:$ptr, TT_Type:$cmp, TT_Type:$val);

    let results = (outs TT_Type:$result);
}


//
// Shape Manipulation Ops
//
def TT_SplatOp : TT_Op<"splat", [Pure,
                                 SameOperandsAndResultElementType,
                                 SameOperandsAndResultEncoding]> {
    let summary = "splat";

    let arguments = (ins TT_Type:$src);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    let hasFolder = 1;
}

def TT_ExpandDimsOp : TT_Op<"expand_dims", [Pure,
                                            DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                            SameOperandsAndResultElementType]> {
    let summary = "expand_dims";

    let arguments = (ins TT_Tensor:$src, I32Attr:$axis);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

// view is not `pure` because it may reorder elements
def TT_ViewOp : TT_Op<"view", [NoMemoryEffect,
                               SameOperandsAndResultElementType]> {
    let summary = "view";

    let arguments = (ins TT_Tensor:$src);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

def TT_BroadcastOp : TT_Op<"broadcast", [Pure,
                                         SameOperandsAndResultElementType,
                                         SameOperandsAndResultEncoding]> {
    let summary = "broadcast. No left-padding as of now.";

    let arguments = (ins TT_Type:$src);

    let results = (outs TT_Type:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

// cat is not `pure` because it may reorder elements
def TT_CatOp : TT_Op<"cat", [NoMemoryEffect,
                             SameOperandsAndResultElementType]> {
    let summary = "concatenate 2 tensors";

    let arguments = (ins TT_Tensor:$lhs, TT_Tensor:$rhs);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def TT_TransOp : TT_Op<"trans", [Pure,
                                 DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                 SameOperandsAndResultElementType]> {

    let summary = "transpose a tensor";

    let arguments = (ins TT_Tensor:$src);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

//
// SPMD Ops
//
def TT_GetProgramIdOp : TT_Op<"get_program_id", [Pure]> {
    let arguments = (ins I32Attr:$axis);

    let results = (outs I32:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}

def TT_GetNumProgramsOp : TT_Op<"get_num_programs", [Pure]> {
    let arguments = (ins I32Attr:$axis);

    let results = (outs I32:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}

//
// Dot Op
//
def TT_DotOp : TT_Op<"dot", [Pure,
                             DeclareOpInterfaceMethods<InferTypeOpInterface>,
                             TypesMatchWith<"result's type matches accumulator's type",
                                            "d", "c", "$_self">]> {
    let summary = "dot";

    let description = [{
        $d = matrix_multiply($a, $b) + $c
    }];

    let arguments = (ins TT_FpIntTensor:$a, TT_FpIntTensor:$b, TT_FpIntTensor:$c, BoolAttr:$allowTF32);

    let results = (outs TT_FpIntTensor:$d);

    let assemblyFormat = "$a`,` $b`,` $c attr-dict `:` type($a) `*` type($b) `->` type($d)";
}

//
// Reduce Op
//
def TT_ReduceOp : TT_Op<"reduce", [Pure,
                                   DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "reduce";

    let arguments = (ins TT_RedOpAttr:$redOp, TT_Tensor:$operand, I32Attr:$axis);

    let results = (outs TT_Type:$result);

    let builders = [
        OpBuilder<(ins "triton::RedOp":$redOp, "Value":$operand, "int":$axis)>,
    ];

    let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";

    let extraClassDeclaration = [{
        // This member function is marked static because we need to call it before the ReduceOp
        // is constructed, see the implementation of create_reduce in triton.cc.
        static bool withIndex(mlir::triton::RedOp redOp);
    }];
}

//
// External elementwise op
//
def TT_ExtElemwiseOp : TT_Op<"ext_elemwise", [Pure, Elementwise, SameOperandsAndResultShape,
                                              SameOperandsAndResultEncoding,
                                              SameVariadicOperandSize]> {
    let summary = "ext_elemwise";

    let description = [{
        call an external function $symbol implemented in $libpath/$libname with $args

        return $libpath/$libname:$symbol($args...)
    }];

    let arguments = (ins Variadic<TT_Type>:$args, StrAttr:$libname, StrAttr:$libpath, StrAttr:$symbol);

    let results = (outs TT_Type:$result);

    let assemblyFormat = "operands attr-dict `:` type(operands) `->` type($result)";
}

//
// Make Range Op
//
// TODO: should have ConstantLike as Trait
def TT_MakeRangeOp : TT_Op<"make_range", [Pure]> {
    let summary = "make range";

    let description = [{
        Returns an 1D int32 tensor.

        Values span from $start to $end (exclusive), with step = 1
    }];

    let arguments = (ins I32Attr:$start, I32Attr:$end);

    let results = (outs TT_IntTensor:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}

//
// Make PrintOp
//
def TT_PrintOp : TT_Op<"print", [MemoryEffects<[MemWrite]>]>,
  Arguments<(ins StrAttr:$prefix, Variadic<AnyTypeOf<[TT_Type]>>:$args)> {
  let summary = "Device-side print, as in CUDA for debugging";
  let description = [{
    `tt.print` takes a literal string prefix and an arbitrary number of scalar or tensor arguments that should be printed.
    format are generated automatically from the arguments.
  }];
  let assemblyFormat = [{
    $prefix attr-dict `:` ($args^ `:` type($args))?
  }];
}

//
// Make AssertOp
//
def TT_AssertOp : TT_Op<"assert", [MemoryEffects<[MemWrite]>]> {
  let summary = "Device-side assert, as in CUDA for correctness checking";
  let description = [{
    `tt.assert` takes a condition tensor, a message string, a file string, a function string, and a line number.
    If the condition is false, the message is printed, and the program is aborted.
  }];
  let arguments = (ins TT_Tensor:$condition, StrAttr:$message, StrAttr:$file, StrAttr:$func, I32Attr:$line);
  let assemblyFormat = "$condition `,` $message `,` $file `,` $func `,` $line attr-dict `:` type($condition)";
}

//
// Make a Tensor Pointer
//
def TT_MakeTensorPtrOp : TT_Op<"make_tensor_ptr",
                               [Pure,
                                SameVariadicOperandSize,
                                TypesMatchWith<"infer pointer type from the result type",
                                               "result", "base",
                                               "getPointerType(getElementTypeOfTensorPointerType($_self))">]> {
  let summary = "Make a tensor pointer type with meta information of the parent tensor and the block specified";

  let description = [{
      `tt.make_tensor_ptr` takes both meta information of the parent tensor and the block tensor, then it returns a
      pointer to the block tensor, e.g. returns a type of `tt.ptr<tensor<8x8xf16>>`.
  }];

  // TODO(Chenggang): unify the integer types. Currently we cannot do that due to hardware constraints.
  let arguments = (ins
    TT_Ptr:$base,
    Variadic<I64>:$shape,
    Variadic<I64>:$strides,
    Variadic<I32>:$offsets,
    DenseI32ArrayAttr:$order
  );

  let results = (outs TT_TensorPtr:$result);

  // Add additional `[]` to increase readability and split variadic lists
  let assemblyFormat = "$base `,` `[` $shape `]` `,` `[` $strides `]` `,` `[` $offsets `]` attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins
        "Value":$base,
        "ValueRange":$shape,
        "ValueRange":$strides,
        "ValueRange":$offsets,
        "ArrayRef<int32_t>":$tensorShape,
        "ArrayRef<int32_t>":$order
    )>
  ];
}

#endif // Triton_OPS
