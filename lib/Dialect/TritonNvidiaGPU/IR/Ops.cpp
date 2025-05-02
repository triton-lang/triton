/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOpInterfaces.cpp.inc"

using namespace mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

// -- WarpGroupDotOp --
mlir::LogicalResult WarpGroupDotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  // verify encodings
  auto aEnc =
      cast<triton::gpu::TensorOrMemDesc>(operands[0].getType()).getEncoding();
  auto bEnc =
      cast<triton::gpu::TensorOrMemDesc>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc);
    Dialect &dialect = aEnc.getDialect();
    auto interface = cast<DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return mlir::failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return mlir::failure();
  }
  return mlir::success();
}

void WarpGroupDotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto &a = getAMutable();
  auto &b = getBMutable();
  if (isa<MemDescType>(a.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &a, SharedMemory::get());
  if (isa<MemDescType>(b.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &b, SharedMemory::get());
}

bool WarpGroupDotOp::needsPartialAccumulator() {
  const auto &a = getA();
  const auto &d = getD();
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto aElTy = cast<triton::gpu::TensorOrMemDesc>(a.getType()).getElementType();
  bool isFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                         Float8E4M3FNUZType>(aElTy);
  bool accFP32 =
      cast<triton::gpu::TensorOrMemDesc>(d.getType()).getElementType().isF32();
  uint32_t maxNumImpreciseAcc = getMaxNumImpreciseAcc();
  return isFP8 && accFP32 && maxNumImpreciseAcc <= aTensorTy.getShape()[1];
}

bool WarpGroupDotOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

// -- WarpGroupDotWaitOp --
LogicalResult WarpGroupDotWaitOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  for (Value operand : operands)
    inferredReturnTypes.push_back(operand.getType());
  return mlir::success();
}

static LogicalResult
verifyBarrierType(Operation *op, mlir::triton::gpu::MemDescType barrierType) {
  if (!barrierType.getElementType().isInteger(64) ||
      barrierType.getShape() != ArrayRef<int64_t>({1}))
    return op->emitOpError(
        "barrier allocation must be a descriptor of 1xi64 type");
  return success();
}

// -- InitBarrierOp --
LogicalResult InitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- InvalBarrierOp --
LogicalResult InvalBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- BarrierExpectOp --
LogicalResult BarrierExpectOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- WaitBarrierOp --
LogicalResult WaitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- ArriveBarrierOp --
LogicalResult ArriveBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  if (getCount() < 1)
    return emitOpError("count must be greater than or equal to 1");
  return success();
}

// -- TensorDescToTMAPtrOp --
LogicalResult TensorDescToTMAPtrOp::canonicalize(TensorDescToTMAPtrOp op,
                                                 PatternRewriter &rewriter) {
  // tensor_desc_to_tma_ptr(reinterpret_tensor_desc(ptr)) -> ptr
  if (auto reinterpret =
          op.getDesc().getDefiningOp<triton::ReinterpretTensorDescOp>()) {
    rewriter.replaceOp(op, reinterpret.getRawDesc());
    return success();
  }
  return failure();
}

// -- AsyncTMACopyGlobalToLocalOp --
LogicalResult AsyncTMACopyGlobalToLocalOp::verify() {
  if (failed(verifyBarrierType(*this, getBarrier().getType())))
    return failure();
  if (getCoord().size() < 1 || getCoord().size() > 5)
    return emitOpError("TMA copies must have between 1 and 5 coordinates");
  if (!getResult().getType().getMutableMemory())
    return emitOpError("Cannot store into immutable memory");
  return success();
}

// -- AsyncTMAGatherOp --
LogicalResult AsyncTMAGatherOp::verify() {
  if (failed(verifyBarrierType(*this, getBarrier().getType())))
    return failure();

  triton::gpu::MemDescType resultType = getResult().getType();
  if (!resultType.getMutableMemory())
    return emitOpError("cannot store into immutable memory");
  return DescriptorGatherOp::verifyResultType(*this, resultType,
                                              getXOffsets().getType());
}

// -- AsyncTMAScatter --
LogicalResult AsyncTMAScatterOp::verify() {
  return DescriptorGatherOp::verifyResultType(*this, getSrc().getType(),
                                              getXOffsets().getType());
}

// -- TCGen5MMAOp --

// barrier-and-pred := `,` ssa-value `[` ssa-value `]`
// barriers-and-preds := (barrier-and-pred)*
static ParseResult
parseBarriersAndPreds(OpAsmParser &p,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &barriers,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &preds) {
  while (succeeded(p.parseOptionalComma())) {
    if (p.parseOperand(barriers.emplace_back()) || p.parseLSquare() ||
        p.parseOperand(preds.emplace_back()) || p.parseRSquare())
      return failure();
  }
  return success();
}
static void printBarriersAndPreds(OpAsmPrinter &p, Operation *op,
                                  OperandRange barriers, OperandRange preds) {
  assert(barriers.size() == preds.size());
  for (auto [barrier, pred] : llvm::zip(barriers, preds)) {
    p << ", " << barrier << '[' << pred << ']';
  }
}

ParseResult TCGen5MMAOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand a, b, d, useD, pred;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> barriers, barrierPreds;

  // Parse the main operands
  if (parser.parseOperand(a) || parser.parseComma() || parser.parseOperand(b) ||
      parser.parseComma() || parser.parseOperand(d) || parser.parseComma() ||
      parser.parseOperand(useD) || parser.parseComma() ||
      parser.parseOperand(pred))
    return failure();

  // Parse optional comma-separated barriers
  if (succeeded(parser.parseOptionalComma())) {
    // Parse barrier
    OpAsmParser::UnresolvedOperand barrier;
    if (parser.parseOperand(barrier))
      return failure();
    barriers.push_back(barrier);

    // Check if there's a predicate in square brackets
    if (!succeeded(parser.parseOptionalLSquare()))
      return failure();
    OpAsmParser::UnresolvedOperand barrierPred;
    if (parser.parseOperand(barrierPred) || parser.parseRSquare())
      return failure();
    barrierPreds.push_back(barrierPred);

    // Parse any additional barriers
    while (succeeded(parser.parseOptionalComma())) {
      if (parser.parseOperand(barrier))
        return failure();
      barriers.push_back(barrier);

      if (!succeeded(parser.parseOptionalLSquare()))
        return failure();
      OpAsmParser::UnresolvedOperand barrierPred;
      if (parser.parseOperand(barrierPred) || parser.parseRSquare())
        return failure();
      barrierPreds.push_back(barrierPred);
    }
  }

  // Parse the operation attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse types for a, b, d
  Type aType, bType, dType;
  if (parser.parseColon() || parser.parseType(aType) || parser.parseComma() ||
      parser.parseType(bType) || parser.parseComma() || parser.parseType(dType))
    return failure();

  // Parse optional types for barriers
  SmallVector<Type, 4> barrierTypes, barrierPredTypes;

  // If we have barriers, we need to parse their types
  if (!barriers.empty()) {
    if (parser.parseComma())
      return failure();

    // Parse barrier types
    for (unsigned i = 0; i < barriers.size(); ++i) {
      if (i > 0 && parser.parseComma())
        return failure();

      Type barrierType;
      if (parser.parseType(barrierType))
        return failure();

      barrierTypes.push_back(barrierType);
    }

    // Check if there's a type for the predicates
    if (succeeded(parser.parseOptionalComma())) {
      Type predType;
      if (parser.parseType(predType))
        return failure();

      // Use the same predicate type for all barriers
      for (unsigned i = 0; i < barrierPreds.size(); ++i) {
        barrierPredTypes.push_back(predType);
      }
    } else {
      // Default to i1 if no type is specified
      for (unsigned i = 0; i < barrierPreds.size(); ++i) {
        barrierPredTypes.push_back(parser.getBuilder().getI1Type());
      }
    }
  }

  // Resolve the main operands
  if (parser.resolveOperand(a, aType, result.operands) ||
      parser.resolveOperand(b, bType, result.operands) ||
      parser.resolveOperand(d, dType, result.operands) ||
      parser.resolveOperand(useD, parser.getBuilder().getI1Type(),
                            result.operands) ||
      parser.resolveOperand(pred, parser.getBuilder().getI1Type(),
                            result.operands))
    return failure();

  // Resolve the barriers
  for (unsigned i = 0; i < barriers.size(); ++i) {
    if (parser.resolveOperand(barriers[i], barrierTypes[i], result.operands))
      return failure();
  }

  // Resolve the barrier predicates
  for (unsigned i = 0; i < barrierPreds.size(); ++i) {
    if (parser.resolveOperand(barrierPreds[i], barrierPredTypes[i],
                              result.operands))
      return failure();
  }

  // Add result types
  result.addTypes({});

  return success();
}

void TCGen5MMAOp::print(OpAsmPrinter &p) {

  Operation *op = *this;
  auto mmaOp = cast<TCGen5MMAOp>(op);

  // Print the main operands
  p << ' ' << mmaOp.getA() << ", " << mmaOp.getB() << ", " << mmaOp.getD()
    << ", " << mmaOp.getUseD() << ", " << mmaOp.getPred();

  // Print barriers and their predicates if any
  auto barriers = mmaOp.getBarriers();
  auto barrierPreds = mmaOp.getBarrierPreds();
  assert(barriers.size() == barrierPreds.size());

  if (!barriers.empty()) {
    for (unsigned i = 0; i < barriers.size(); ++i) {
      // Print each barrier and its predicate
      p << ", " << barriers[i];
      // Print predicate if available
      p << "[" << barrierPreds[i] << "]";
    }
  }

  // Print attributes
  p.printOptionalAttrDict(op->getAttrs());

  // Print types
  p << " : " << mmaOp.getA().getType() << ", " << mmaOp.getB().getType() << ", "
    << mmaOp.getD().getType();

  if (!barriers.empty()) {
    // Print barrier types
    for (auto barrier : barriers) {
      p << ", " << barrier.getType();
    }

    // Print barrier predicate types
    auto predType = barrierPreds[0].getType();
    // Verify that all barrier predicates have the same type
    for (auto pred : barrierPreds) {
      assert(pred.getType() == predType);
    }
    auto intType = cast<IntegerType>(predType);
    // print the type of the predicate if it is not I1
    if (intType.getWidth() != 1) {
      p << ", " << predType;
    }
  }
}

void TCGen5MMAOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDMutable(),
                       TensorMemory::get());
  if (isa<SharedMemorySpaceAttr>(getA().getType().getMemorySpace())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         SharedMemory::get());

  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
                       SharedMemory::get());
}

bool TCGen5MMAOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

Value TCGen5MMAOp::useAccumulator() { return getUseD(); }

void TCGen5MMAOp::setUseAccumulator(Value flag) {
  getUseDMutable().assign(flag);
}

void TCGen5MMAOp::addCompletionBarrier(Value barrier, Value pred) {
  getBarrierPredsMutable().append(pred);
  getBarriersMutable().append(barrier);
}

TypedValue<MemDescType> TCGen5MMAOp::getAccumulator() { return getD(); }

void TCGen5MMAOp::setAccumulator(Value accum) { getDMutable().assign(accum); }

Value TCGen5MMAOp::getPredicate() { return getPred(); }

void TCGen5MMAOp::setPredicate(Value pred) { getPredMutable().assign(pred); }

void TCGen5MMAOp::build(OpBuilder &builder, OperationState &state, Value a,
                        Value b, Value d, Value useD, Value pred,
                        bool useTwoCTAs, ValueRange barriers,
                        ValueRange barrierPreds) {
  build(builder, state, a, b, d, useD, pred, barriers, barrierPreds,
        useTwoCTAs ? builder.getUnitAttr() : UnitAttr());
}

// -- TCGen5MMAScaledOp --
ParseResult TCGen5MMAScaledOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::UnresolvedOperand a, b, d, a_scale, b_scale, useD, pred;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> barriers, barrierPreds;

  // Parse the main operands
  if (parser.parseOperand(a) || parser.parseComma() || parser.parseOperand(b) ||
      parser.parseComma() || parser.parseOperand(d) || parser.parseComma() ||
      parser.parseOperand(a_scale) || parser.parseComma() ||
      parser.parseOperand(b_scale) || parser.parseComma() ||
      parser.parseOperand(useD) || parser.parseComma() ||
      parser.parseOperand(pred))
    return failure();

  // Parse lhs and rhs scaling types
  StringRef keyword;
  triton::ScaleDotElemType a_type, b_type;

  // Parse "lhs = a_type" format
  if (parser.parseKeyword("lhs") || parser.parseEqual() ||
      parser.parseKeyword(&keyword))
    return failure();

  // Set a_type based on the keyword
  a_type = *triton::symbolizeScaleDotElemType(keyword);
  result.addAttribute("a_type",
                      ScaleDotElemTypeAttr::get(parser.getContext(), a_type));

  // Parse "rhs = b_type" format
  if (parser.parseKeyword("rhs") || parser.parseEqual() ||
      parser.parseKeyword(&keyword))
    return failure();

  // Set b_type based on the keyword
  b_type = *triton::symbolizeScaleDotElemType(keyword);
  result.addAttribute("b_type",
                      ScaleDotElemTypeAttr::get(parser.getContext(), b_type));

  // Parse optional comma-separated barriers
  if (succeeded(parser.parseOptionalComma())) {
    // Parse barrier
    OpAsmParser::UnresolvedOperand barrier;
    if (parser.parseOperand(barrier))
      return failure();
    barriers.push_back(barrier);

    // Check if there's a predicate in square brackets
    if (!succeeded(parser.parseOptionalLSquare()))
      return failure();
    OpAsmParser::UnresolvedOperand barrierPred;
    if (parser.parseOperand(barrierPred) || parser.parseRSquare())
      return failure();
    barrierPreds.push_back(barrierPred);

    // Parse any additional barriers
    while (succeeded(parser.parseOptionalComma())) {
      if (parser.parseOperand(barrier))
        return failure();
      barriers.push_back(barrier);

      if (!succeeded(parser.parseOptionalLSquare()))
        return failure();
      OpAsmParser::UnresolvedOperand barrierPred;
      if (parser.parseOperand(barrierPred) || parser.parseRSquare())
        return failure();
      barrierPreds.push_back(barrierPred);
    }
  }

  // Parse the operation attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse types for a, b, d, a_scale, b_scale
  Type aType, bType, dType, aScaleType, bScaleType;
  if (parser.parseColon() || parser.parseType(aType) || parser.parseComma() ||
      parser.parseType(bType) || parser.parseComma() ||
      parser.parseType(dType) || parser.parseComma() ||
      parser.parseType(aScaleType) || parser.parseComma() ||
      parser.parseType(bScaleType))
    return failure();

  // Parse optional types for barriers
  SmallVector<Type, 4> barrierTypes, barrierPredTypes;

  // If we have barriers, we need to parse their types
  if (!barriers.empty()) {
    if (parser.parseComma())
      return failure();

    // Parse barrier types
    for (unsigned i = 0; i < barriers.size(); ++i) {
      if (i > 0 && parser.parseComma())
        return failure();

      Type barrierType;
      if (parser.parseType(barrierType))
        return failure();

      barrierTypes.push_back(barrierType);
    }

    // Check if there's a type for the predicates
    if (succeeded(parser.parseOptionalComma())) {
      Type predType;
      if (parser.parseType(predType))
        return failure();

      // Use the same predicate type for all barriers
      for (unsigned i = 0; i < barrierPreds.size(); ++i) {
        barrierPredTypes.push_back(predType);
      }
    } else {
      // Default to i1 if no type is specified
      for (unsigned i = 0; i < barrierPreds.size(); ++i) {
        barrierPredTypes.push_back(parser.getBuilder().getI1Type());
      }
    }
  }

  // Resolve the main operands
  if (parser.resolveOperand(a, aType, result.operands) ||
      parser.resolveOperand(b, bType, result.operands) ||
      parser.resolveOperand(d, dType, result.operands) ||
      parser.resolveOperand(a_scale, aScaleType, result.operands) ||
      parser.resolveOperand(b_scale, bScaleType, result.operands) ||
      parser.resolveOperand(useD, parser.getBuilder().getI1Type(),
                            result.operands) ||
      parser.resolveOperand(pred, parser.getBuilder().getI1Type(),
                            result.operands))
    return failure();

  // Resolve the barriers
  for (unsigned i = 0; i < barriers.size(); ++i) {
    if (parser.resolveOperand(barriers[i], barrierTypes[i], result.operands))
      return failure();
  }

  // Resolve the barrier predicates
  for (unsigned i = 0; i < barrierPreds.size(); ++i) {
    if (parser.resolveOperand(barrierPreds[i], barrierPredTypes[i],
                              result.operands))
      return failure();
  }

  // Add result types
  result.addTypes({});

  return success();
}

void TCGen5MMAScaledOp::print(OpAsmPrinter &p) {
  // Print the main operands
  p << ' ' << getA() << ", " << getB() << ", " << getD() << ", " << getAScale()
    << ", " << getBScale() << ", " << getUseD() << ", " << getPred();

  // Print the scaling type attributes
  p << " lhs = " << stringifyScaleDotElemType(getAType())
    << " rhs = " << stringifyScaleDotElemType(getBType());

  // Print barriers and their predicates if any
  auto barriers = getBarriers();
  auto barrierPreds = getBarrierPreds();
  assert(barriers.size() == barrierPreds.size());

  if (!barriers.empty()) {
    for (unsigned i = 0; i < barriers.size(); ++i) {
      // Print each barrier and its predicate
      p << ", " << barriers[i];
      // Print predicate if available
      p << "[" << barrierPreds[i] << "]";
    }
  }

  // Print attributes (excluding a_type and b_type which are printed separately)
  p.printOptionalAttrDict(getOperation()->getAttrs(), {"a_type", "b_type"});

  // Print types
  p << " : " << getA().getType() << ", " << getB().getType() << ", "
    << getD().getType() << ", " << getAScale().getType() << ", "
    << getBScale().getType();

  if (!barriers.empty()) {
    // Print barrier types
    for (auto barrier : barriers) {
      p << ", " << barrier.getType();
    }

    // Print barrier predicate types
    auto predType = barrierPreds[0].getType();
    // Verify that all barrier predicates have the same type
    for (auto pred : barrierPreds) {
      assert(pred.getType() == predType);
    }
    auto intType = cast<IntegerType>(predType);
    // print the type of the predicate if it is not I1
    if (intType.getWidth() != 1) {
      p << ", " << predType;
    }
  }
}

void TCGen5MMAScaledOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDMutable(),
                       TensorMemory::get());
  if (isa<SharedMemorySpaceAttr>(getA().getType().getMemorySpace())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         SharedMemory::get());

  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
                       SharedMemory::get());
}

bool TCGen5MMAScaledOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  auto aKdim = aShape[aShape.size() - 1];
  auto bKdim = bShape[aShape.size() - 2];
  if (this->getAType() == ScaleDotElemType::E2M1 && !transA)
    aKdim *= 2;
  if (this->getBType() == ScaleDotElemType::E2M1 && !transB)
    bKdim *= 2;

  return aKdim == bKdim;
}

bool TCGen5MMAScaledOp::verifyOutputDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();
  auto cShape = this->getD().getType().getShape();
  auto oMdim = cShape[cShape.size() - 2];
  auto oNdim = cShape[cShape.size() - 1];

  int aMdim = aShape[aShape.size() - 2];
  int bNdim = bShape[bShape.size() - 1];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && transA)
    aMdim *= 2;
  if (this->getBType() == ScaleDotElemType::E2M1 && transB)
    bNdim *= 2;

  if (aMdim != oMdim || bNdim != oNdim)
    return false;
  return true;
}

Value TCGen5MMAScaledOp::useAccumulator() { return getUseD(); }

void TCGen5MMAScaledOp::setUseAccumulator(Value flag) {
  getUseDMutable().assign(flag);
}

void TCGen5MMAScaledOp::addCompletionBarrier(Value barrier, Value pred) {
  getBarrierPredsMutable().append(pred);
  getBarriersMutable().append(barrier);
}

TypedValue<MemDescType> TCGen5MMAScaledOp::getAccumulator() { return getD(); }

void TCGen5MMAScaledOp::setAccumulator(Value accum) {
  getDMutable().assign(accum);
}

Value TCGen5MMAScaledOp::getPredicate() { return getPred(); }

void TCGen5MMAScaledOp::setPredicate(Value pred) {
  getPredMutable().assign(pred);
}

int64_t TCGen5MMAScaledOp::getBlockM() {
  ArrayRef<int64_t> shape = getA().getType().getShape();
  int64_t blockM = shape[shape.size() - 2];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && transA)
    blockM *= 2;
  return blockM;
}

int64_t TCGen5MMAScaledOp::getBlockN() {
  ArrayRef<int64_t> shape = getB().getType().getShape();
  int64_t blockN = shape[shape.size() - 1];
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  if (this->getBType() == ScaleDotElemType::E2M1 && transB)
    blockN *= 2;
  return blockN;
}

int64_t TCGen5MMAScaledOp::getBlockK() {
  ArrayRef<int64_t> shape = getA().getType().getShape();
  int64_t blockK = shape[shape.size() - 1];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && !transA)
    blockK *= 2;
  return blockK;
}

void TCGen5MMAScaledOp::build(OpBuilder &builder, OperationState &state,
                              Value a, Value b, Value d, Value aScale,
                              Value bScale, ScaleDotElemType aType,
                              ScaleDotElemType bType, Value useD, Value pred,
                              ValueRange barriers, ValueRange barrierPreds) {
  MLIRContext *ctx = builder.getContext();
  build(builder, state, a, b, d, aScale, bScale,
        ScaleDotElemTypeAttr::get(ctx, aType),
        ScaleDotElemTypeAttr::get(ctx, bType), useD, pred, barriers,
        barrierPreds);
}

// -- TMEMStoreOp --
LogicalResult TMEMStoreOp::verify() {
  if (!isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
          getDst().getType().getMemorySpace()))
    return emitOpError("destination must be a tensor memory buffer.");
  if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr,
           TensorMemoryScalesEncodingAttr>(getDst().getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");
  if (!getDst().getType().getMutableMemory()) {
    return emitOpError("Cannot store into an immutable alloc");
  }
  return success();
}

// -- TMEMLoadOp --
LogicalResult TMEMLoadOp::verify() {
  if (!isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
          getSrc().getType().getMemorySpace()))
    return emitOpError("source must be a tensor memory buffer.");
  if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
          getSrc().getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");
  return success();
}

// -- TMEMAllocOp --
LogicalResult TMEMAllocOp::verify() {
  if (!isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
          getType().getMemorySpace()))
    return emitOpError("should create a buffer of tensor memory.");
  if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr,
           TensorMemoryScalesEncodingAttr>(getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");
  if (!getSrc()) {
    if (!getType().getMutableMemory())
      return emitError("uninitialized alloc must have a mutable memdesc type");
  }
  return success();
}

void TMEMAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  Operation *op = getOperation();
  // If allocation is immutable, mark it as no side effect allow things like
  // CSE, DCE to work in early compiler passes.
  // After the memory offset is computed, we attach the true side effect to the
  // op.
  if (!getType().getMutableMemory() && !op->hasAttr("tensor_memory_col_offset"))
    return;
  effects.emplace_back(MemoryEffects::Allocate::get(), TensorMemory::get());
  if (getSrc())
    effects.emplace_back(MemoryEffects::Write::get(),
                         getOperation()->getOpResult(0), TensorMemory::get());
}

// -- TMEMCopyOp --
LogicalResult TMEMCopyOp::verify() {
  if (!isa<triton::gpu::SharedMemorySpaceAttr>(
          getSrc().getType().getMemorySpace()))
    return emitOpError("The source must be a shared memory buffer");
  if (!isa<TensorMemoryEncodingAttr, TensorMemoryScalesEncodingAttr>(
          getDst().getType().getEncoding()))
    return emitOpError("The destination must be a tensor memory buffer.");

  if (getBarrier() && !isa<triton::gpu::SharedMemorySpaceAttr>(
                          getBarrier().getType().getMemorySpace())) {
    return emitOpError("The optional barrier should be a shared memory buffer");
  }
  if (!getDst().getType().getMutableMemory()) {
    return emitOpError("Cannot copy into an immutable alloc");
  }

  auto srcTy = cast<triton::gpu::MemDescType>(getSrc().getType());
  auto sharedEnc =
      cast<triton::gpu::SwizzledSharedEncodingAttr>(srcTy.getEncoding());

  if (sharedEnc.getMaxPhase() != 1 || sharedEnc.getPerPhase() != 1 ||
      sharedEnc.getVec() != 1)
    return emitOpError("The source should not have swizzling applied for now");

  if (!triton::gpu::isInnermostContiguous(srcTy, 512)) {
    return emitOpError("The source must be in a row-major order.");
  }

  // Given that we want to support flexible input SMEM shapes, kinds of shape
  // checking we can do here are limited. For simplicity, shape checking is
  // omitted.
  return success();
}

// -- TMEMSubSliceOp --
LogicalResult TMEMSubSliceOp::verify() {
  auto srcTy = cast<triton::gpu::MemDescType>(getSrc().getType());
  auto encoding = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
      srcTy.getEncoding());
  if (!encoding)
    return emitOpError("The source must be a tensor memory buffer.");
  if (encoding.getBlockM() != 128)
    return emitOpError("The source must be a 128xN layout.");
  auto dstTy = cast<triton::gpu::MemDescType>(getResult().getType());
  auto dstEncoding = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
      dstTy.getEncoding());
  if (!dstEncoding)
    return emitOpError("The destination must be a tensor memory buffer.");
  if (dstEncoding.getBlockM() != encoding.getBlockM() ||
      dstEncoding.getCTASplitM() != encoding.getCTASplitM() ||
      dstEncoding.getCTASplitN() != encoding.getCTASplitN() ||
      dstEncoding.getUnpacked() != encoding.getUnpacked())
    return emitOpError("The destination must have the same block size and "
                       "CTASplit size as the source.");
  return mlir::success();
}

void TMEMSubSliceOp::build(OpBuilder &builder, OperationState &state,
                           Value alloc, int offset, int size) {
  auto allocTy = cast<triton::gpu::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape(allocTy.getShape());
  shape.back() = size;
  auto encoding =
      cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(allocTy.getEncoding());
  unsigned newBlockN = std::min<unsigned>(encoding.getBlockN(), size);
  auto newEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
      builder.getContext(), encoding.getBlockM(), newBlockN,
      encoding.getUnpacked(), encoding.getCTASplitM(), encoding.getCTASplitN());
  auto subsliceType = gpu::MemDescType::get(
      shape, allocTy.getElementType(), newEncoding, allocTy.getMemorySpace(),
      allocTy.getMutableMemory());
  build(builder, state, subsliceType, alloc, offset);
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#define GET_OP_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.cpp.inc"
