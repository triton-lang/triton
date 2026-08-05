#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeRange.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/NVWS/IR/NVWSAttrEnums.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/NVWS/IR/NVWSOpInterfaces.cpp.inc"
#include "Dialect/NVWS/IR/Ops.cpp.inc"

namespace mlir::triton::nvws {

LogicalResult ArefCreateOp::verify() {
  SmallVector<int> dims;
  for (auto operand : getOperands()) {
    SmallVector<Operation *> users(operand.user_begin(), operand.user_end());
    if (!llvm::all_of(users, [](Operation *op) {
          return isa<ArefCreateOp, gpu::LocalDeallocOp>(op);
        }))
      return emitError("Aref buffer is used elsewhere, Aref cannot guarantee "
                       "async safety");
    auto type = operand.getType();
    if (auto mType = dyn_cast<gpu::MemDescType>(type)) {
      dims.push_back(mType.getShape()[0]);
    } else if (auto rType = dyn_cast<RankedTensorType>(type)) {
      dims.push_back(rType.getShape()[0]);
    } else {
      return emitError("Aref is sliced, but input type isn't supported.");
    }
  }
  if (!llvm::all_equal(dims))
    return emitError("Leading dims of sliced aref inputs don't match.");

  return success();
}

static FailureOr<SetVector<int>> getArefPhiOperandPartitions(Value value) {
  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *def = result.getDefiningOp();
    if (!gpu::hasPartition(def))
      return failure();
    if (!def->hasAttr(gpu::kPartitionOutputsAttrName))
      return gpu::getPartitionIds(def);
    auto outputs = gpu::getPartitionOutputs(def);
    if (result.getResultNumber() >= outputs.size())
      return failure();
    return outputs[result.getResultNumber()];
  }

  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg || arg.getArgNumber() == 0)
    return failure();
  auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
  if (!forOp || !gpu::hasPartition(forOp) ||
      !forOp->hasAttr(gpu::kPartitionOutputsAttrName))
    return failure();
  auto outputs = gpu::getPartitionOutputs(forOp);
  unsigned resultNumber = arg.getArgNumber() - 1;
  if (resultNumber >= outputs.size())
    return failure();
  return outputs[resultNumber];
}

LogicalResult ArefPhiOp::verify() {
  if (!gpu::hasPartition(getOperation()))
    return emitError("requires a non-empty ttg.partition annotation");

  FailureOr<SetVector<int>> local = getArefPhiOperandPartitions(getLocal());
  FailureOr<SetVector<int>> remote = getArefPhiOperandPartitions(getRemote());
  if (failed(local) || local->empty())
    return emitError("local operand must have partition ownership");
  if (failed(remote) || remote->empty())
    return emitError("remote operand must have partition ownership");

  for (int partition : *local)
    if (remote->contains(partition))
      return emitError("partition ")
             << partition << " is owned by both phi operands";

  SetVector<int> covered(*local);
  covered.insert(remote->begin(), remote->end());
  SetVector<int> expected = gpu::getPartitionIds(getOperation());
  if (covered.size() != expected.size() ||
      llvm::any_of(covered, [&](int partition) {
        return !expected.contains(partition);
      }))
    return emitError(
        "operand partition union must exactly match the phi partition set");
  return success();
}

template <typename T>
static std::optional<Twine> verifySlice(T &origType, T &newType) {
  if (!origType || !newType)
    return "MLIR Types don't match";
  if (isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
          origType.getEncoding())) {
    if (origType.getElementType() != newType.getElementType() ||
        origType.getRank() != newType.getRank()) {
      return "Ranks don't match for TensorMemoryScalesEncodingAttr";
    }
    for (size_t i = 0, e = newType.getShape().size(); i < e; i++) {
      if (origType.getShape()[i] != newType.getShape()[i])
        return "Dimensions don't match for TensorMemoryScalesEncodingAttr";
    }
  } else {
    if (origType.getElementType() != newType.getElementType() ||
        origType.getRank() - 1 != newType.getRank()) {
      return "Ranks don't match";
    }
    for (size_t i = 0, e = newType.getShape().size(); i < e; i++) {
      if (origType.getShape()[i + 1] != newType.getShape()[i])
        return "Dimensions don't match";
    }
  }
  return std::nullopt;
}

std::optional<Twine> static arefEnterVerify(
    ArefType aref, mlir::ValueTypeRange<ResultRange> resultTypes) {
  auto typeArray = aref.getBaseType();
  if (typeArray.size() != resultTypes.size())
    return "Aref has different number of arguments than enter";
  // This should probably rely on the memdescSubsliceOp verifier?
  for (auto [orig, arg] : llvm::zip(typeArray, resultTypes)) {
    if (auto origT = dyn_cast<RankedTensorType>(orig)) {
      auto argT = dyn_cast<RankedTensorType>(arg);
      if (auto result = verifySlice(origT, argT))
        return result;
    } else if (auto origT = dyn_cast<triton::gpu::MemDescType>(orig)) {
      auto argT = dyn_cast<triton::gpu::MemDescType>(arg);
      if (auto result = verifySlice(origT, argT))
        return result;
    } else {
      return "Slicing not Implemented for this type";
    }
  }
  return std::nullopt;
}

LogicalResult ArefPutEnterOp::verify() {
  if (auto result =
          arefEnterVerify(getAref().getType(), getBuffers().getType()))
    return emitError(*result);
  return success();
}

LogicalResult ArefGetEnterOp::verify() {
  if (auto result =
          arefEnterVerify(getAref().getType(), getBuffers().getType()))
    return emitError(*result);
  return success();
}

LogicalResult WarpGroupOp::verify() {
  auto numWarps = getNumWarps();
  auto regions = getRegions();
  if (numWarps.size() != regions.size())
    return emitError("Must supply numWarps for each Warp Group.");
  if (getResults().size() > 0) {
    if (regions.size() == 0) {
      return emitError("Must have at least one region when there are results.");
    }
    if (!isa<nvws::WarpGroupYieldOp>(
            regions.front()->front().getTerminator())) {
      return emitError("When nvws.warp_group op has results, the first region "
                       "should be terminated by nvws.warp_group.yield op.");
    }
    auto yieldOp =
        cast<nvws::WarpGroupYieldOp>(regions.front()->front().getTerminator());
    if (getResults().size() != yieldOp.getNumOperands()) {
      return emitError(
          "Mismatch in the number of results returned by nvws.warp_group op "
          "and the number of the operands of the corresponding "
          "nvws.warp_group.yield op in the first region.");
    }
  }
  return success();
}

ParseResult WarpGroupOp::parse(OpAsmParser &p, OperationState &result) {
  if (p.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  SmallVector<int32_t> partitionNumWarps;
  while (succeeded(p.parseOptionalKeyword(
      ("partition" + Twine(partitionNumWarps.size()).str())))) {
    if (p.parseKeyword("num_warps") || p.parseLParen() ||
        p.parseInteger(partitionNumWarps.emplace_back()) || p.parseRParen() ||
        p.parseRegion(*result.addRegion()))
      return failure();
  }

  result.addAttribute(getNumWarpsAttrName(result.name),
                      p.getBuilder().getDenseI32ArrayAttr(partitionNumWarps));

  return success();
}

void WarpGroupOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     {getNumWarpsAttrName()});

  for (auto [i, region, numWarps] :
       llvm::enumerate(getPartitionRegions(), getNumWarps())) {
    p.printNewline();
    p << "partition" << i;
    p << " num_warps(" << numWarps << ") ";
    p.printRegion(region, /*printEntryBlockArgs=*/false);
  }
}

void CreateTokenOp::build(::mlir::OpBuilder &builder,
                          ::mlir::OperationState &state, uint32_t num,
                          TokenLoadType loadType) {
  auto tokenType = TokenType::get(builder.getContext());
  auto resultType = RankedTensorType::get({num}, tokenType);
  build(builder, state, resultType, num, loadType);
}

void ArefPutEnterOp::setStage(Value stage) { getStageMutable().assign(stage); }
void ArefPutExitOp::setStage(Value stage) { getStageMutable().assign(stage); }
void ArefGetExitOp::setStage(Value stage) { getStageMutable().assign(stage); }
void ArefGetEnterOp::setStage(Value stage) { getStageMutable().assign(stage); }
void ArefBufferOp::setStage(Value stage) { getStageMutable().assign(stage); }

} // namespace mlir::triton::nvws
