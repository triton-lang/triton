#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_DOT_OP_HELPERS_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_DOT_OP_HELPERS_H

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PTXAsmFormat.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

#include "Utility.h"

class TritonGPUToLLVMTypeConverter;

namespace mlir {
namespace LLVM {
using namespace mlir::triton;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Helper for conversion of FMA DotOp.
struct DotOpFMAConversionHelper {
  Attribute layout;
  MLIRContext *ctx{};

  using ValueTable = std::map<std::pair<int, int>, Value>;

  explicit DotOpFMAConversionHelper(Attribute layout)
      : layout(layout), ctx(layout.getContext()) {}

  SmallVector<Value>
  getThreadIds(Value threadId, ArrayRef<unsigned> shapePerCTA,
               ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order,
               ConversionPatternRewriter &rewriter, Location loc) const;

  Value loadA(Value A, Value llA, BlockedEncodingAttr dLayout, Value thread,
              Location loc, TritonGPUToLLVMTypeConverter *typeConverter,
              ConversionPatternRewriter &rewriter) const;

  Value loadB(Value B, Value llB, BlockedEncodingAttr dLayout, Value thread,
              Location loc, TritonGPUToLLVMTypeConverter *typeConverter,
              ConversionPatternRewriter &rewriter) const;

  ValueTable getValueTableFromStruct(
      Value val, int K, int n0, int shapePerCTA, int sizePerThread,
      ConversionPatternRewriter &rewriter, Location loc,
      TritonGPUToLLVMTypeConverter *typeConverter, Type type) const;

  Value getStructFromValueTable(ArrayRef<Value> vals,
                                ConversionPatternRewriter &rewriter,
                                Location loc,
                                TritonGPUToLLVMTypeConverter *typeConverter,
                                Type elemTy) const;

  // Get shapePerCTA for M or N axis.
  static int getShapePerCTAForMN(BlockedEncodingAttr layout, bool isM) {
    auto order = layout.getOrder();
    auto shapePerCTA = getShapePerCTA(layout);

    int mShapePerCTA =
        order[0] == 1 ? shapePerCTA[order[1]] : shapePerCTA[order[0]];
    int nShapePerCTA =
        order[0] == 0 ? shapePerCTA[order[1]] : shapePerCTA[order[0]];
    return isM ? mShapePerCTA : nShapePerCTA;
  }

  // Get sizePerThread for M or N axis.
  static int getSizePerThreadForMN(BlockedEncodingAttr layout, bool isM) {
    auto order = layout.getOrder();
    auto sizePerThread = getSizePerThread(layout);

    int mSizePerThread =
        order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
    int nSizePerThread =
        order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];
    return isM ? mSizePerThread : nSizePerThread;
  }
};

} // namespace LLVM
} // namespace mlir

#endif
