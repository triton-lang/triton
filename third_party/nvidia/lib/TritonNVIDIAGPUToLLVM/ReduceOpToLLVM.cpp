#include "PatternTritonGPUOpToLLVM.h"
#include "lib/Conversion/TritonGPUToLLVM/ReduceScanCommon.h"

#include <cstdlib>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>

#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::triton;

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

static bool isConstantTruePred(Value pred) {
  if (auto constOp = pred.getDefiningOp<LLVM::ConstantOp>())
    return cast<IntegerAttr>(constOp.getValue()).getInt() == -1;
  return false;
}

static bool isClusterReduceBulkDisabled() {
  if (const char *env = std::getenv("TRITON_DISABLE_CLUSTER_REDUCE_BULK"))
    return env[0] != '\0' && env[0] != '0';
  return false;
}

static Value mapaSharedCluster(RewriterBase &rewriter, Location loc, Value ptr,
                               Value ctaid, Value pred) {
  auto *ctx = rewriter.getContext();
  auto clusterPtrTy = ptr_ty(ctx, /*addrspace=*/7);
  if (isConstantTruePred(pred))
    return NVVM::MapaOp::create(rewriter, loc, clusterPtrTy, ptr, ctaid);

  PTXBuilder builder;
  auto &mapaInstr = *builder.create("mapa");
  mapaInstr.o("shared::cluster.u32");
  auto *dstOpr = builder.newOperand("=r");
  auto *ptrOpr = builder.newOperand(ptr, "r");
  auto *ctaidOpr = builder.newOperand(ctaid, "r");
  mapaInstr(dstOpr, ptrOpr, ctaidOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, clusterPtrTy, /*hasSideEffect=*/false);
}

struct ReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
  ReduceOpConversion(LLVMTypeConverter &typeConverter,
                     const NVIDIA::TargetInfo &targetInfo,
                     PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(typeConverter,
                                                                  benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (targetInfo.getComputeCapability() < 90 || isClusterReduceBulkDisabled())
      return failure();

    ReduceOpHelper helper(op);
    Location loc = op.getLoc();
    auto accs = unpackInputs(loc, op, adaptor, rewriter);
    unsigned axis = op.getAxis();
    auto *ctx = op.getContext();

    LinearLayout regLl = triton::gpu::toLinearLayout(helper.getSrcTy());
    auto removeBroadcast = actionRemoveBroadcastedRegs(regLl);
    if (!removeBroadcast.isIdentity()) {
      regLl = removeBroadcast.apply(regLl);
      for (auto &vals : accs)
        vals = removeBroadcast.apply(vals);
    }

    std::tie(regLl, accs) =
        reduceWithinThreads(op, std::move(regLl), std::move(accs), rewriter);
    std::tie(regLl, accs) =
        reduceWithinWarps(op, std::move(regLl), std::move(accs), rewriter);

    assert(regLl ==
           ReduceOpHelper::reducedRegLaneLayout(helper.getSrcTy(), axis));

    auto kAxis = *(regLl.getOutDimNames().begin() + axis);
    auto kBlock = StringAttr::get(ctx, "block");
    bool lastCvtCrossesCTAs = false;
    while (regLl.getOutDimSize(kAxis) != 1) {
      if (hasAxisBlockBasesOnly(regLl, axis)) {
        if (!llvm::isPowerOf2_32(ttg::lookupNumCTAs(op)))
          return failure();
        std::tie(regLl, accs) =
            reduceAcrossCTAs(op, std::move(regLl), std::move(accs), rewriter);
        lastCvtCrossesCTAs = true;
        continue;
      }

      LinearLayout tmpLl = ReduceOpHelper::getInterLayout(regLl, axis);
      if (!mlir::isCvtDimSync(regLl, tmpLl, kBlock))
        return failure();
      accs = convertLayoutValues(loc, rewriter, op, regLl, tmpLl, accs);
      std::tie(regLl, accs) =
          reduceWithinWarps(op, std::move(tmpLl), std::move(accs), rewriter);
    }

    regLl = removeStandardDim(regLl, axis);
    if (auto resultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto outputLayout = triton::gpu::toLinearLayout(resultTy);
      if (regLl != outputLayout) {
        sync(rewriter, loc, lastCvtCrossesCTAs);
        accs =
            convertLayoutValues(loc, rewriter, op, regLl, outputLayout, accs);
      }
    }

    packResults(op, accs, rewriter);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;

  struct SegmentInfo {
    Type elemTy;
    unsigned bitWidth;
    unsigned regsPerThread;
    unsigned bytesPerThread;
    unsigned offsetBytes;
  };

  static bool hasAxisBases(const LinearLayout &layout, StringAttr dim,
                           unsigned axis) {
    auto it = layout.getBases().find(dim);
    if (it == layout.getBases().end())
      return false;
    return llvm::any_of(it->second,
                        [axis](const auto &basis) { return basis[axis] != 0; });
  }

  static bool hasAxisBlockBasesOnly(const LinearLayout &layout, unsigned axis) {
    auto *ctx = layout.getOutDimNames().begin()->getContext();
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    return hasAxisBases(layout, kBlock, axis) &&
           !hasAxisBases(layout, kWarp, axis);
  }

  static Type getReduceMemElemTy(Type elemTy, MLIRContext *ctx) {
    if (elemTy.isIntOrFloat() && elemTy.getIntOrFloatBitWidth() < 8)
      return IntegerType::get(ctx, 8);
    return elemTy;
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    SmallVector<SmallVector<Value>> srcValues(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i)
      srcValues[i] = unpackLLElements(loc, adaptor.getOperands()[i], rewriter);
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            bool crossCTA) const {
    if (crossCTA)
      targetInfo.clusterBarrier(loc, rewriter);
    else
      targetInfo.barrier(loc, rewriter, ttg::AddrSpace::Local);
  }

  SmallVector<Value> treeReduce(Location loc,
                                ConversionPatternRewriter &rewriter,
                                Region &combineOp,
                                SmallVector<SmallVector<Value>> values,
                                unsigned arity) const {
    while (values.size() > 1) {
      SmallVector<SmallVector<Value>> next;
      for (size_t i = 0; i < values.size(); i += arity) {
        size_t remaining = values.size() - i;
        size_t groupSize = std::min(static_cast<size_t>(arity), remaining);
        if (groupSize == 1) {
          next.push_back(std::move(values[i]));
        } else {
          SmallVector<Value> acc = std::move(values[i]);
          for (size_t j = 1; j < groupSize; ++j)
            accumulate(loc, rewriter, combineOp, acc, values[i + j]);
          next.push_back(std::move(acc));
        }
      }
      values = std::move(next);
    }
    return values.front();
  }

  void accumulate(Location loc, ConversionPatternRewriter &rewriter,
                  Region &combineOp, SmallVector<Value> &acc, ValueRange cur,
                  Value pred = {}) const {
    auto results = applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
    if (acc.size() < results.size())
      acc.resize(results.size());
    for (unsigned i = 0; i < acc.size(); ++i)
      acc[i] = results[i];
  }

  void packVectorized(SmallVector<SmallVector<Value>> &accs,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = accs.front().front().getLoc();
    for (auto &acc : accs) {
      SmallVector<Value> packedAcc;
      for (unsigned reg = 0; reg < acc.size(); reg += 2)
        packedAcc.emplace_back(
            packLLVector(loc, {acc[reg], acc[reg + 1]}, rewriter));
      acc = std::move(packedAcc);
    }
  }

  std::unique_ptr<Region> createVectorCombineRegion(
      Location loc, Type elemTy,
      ReduceOpHelper::InThreadVectorizeOpKind vectorizeKind,
      ConversionPatternRewriter &rewriter) const {
    if (vectorizeKind == ReduceOpHelper::InThreadVectorizeOpKind::None)
      return nullptr;
    auto vecTy = vec_ty(elemTy, 2);
    auto storage = std::make_unique<Region>();
    auto *block = new Block();
    storage->push_back(block);
    block->addArgument(vecTy, loc);
    block->addArgument(vecTy, loc);
    OpBuilder builder(rewriter.getContext());
    builder.setInsertionPointToStart(block);
    Value result = ReduceOpHelper::createInThreadVectorizedCombineOp(
        builder, loc, vectorizeKind, block->getArgument(0),
        block->getArgument(1));
    triton::ReduceReturnOp::create(builder, loc, ValueRange{result});
    return storage;
  }

  void unpackVectorized(Location loc, SmallVector<SmallVector<Value>> &accs,
                        ConversionPatternRewriter &rewriter,
                        Region *reduction) const {
    for (auto &acc : accs) {
      SmallVector<Value> unpacked;
      for (Value val : acc) {
        auto elems = unpackLLVector(loc, val, rewriter);
        if (reduction) {
          SmallVector<Value> cur = {elems[0]};
          accumulate(loc, rewriter, *reduction, cur, {elems[1]});
          unpacked.emplace_back(cur[0]);
        } else {
          unpacked.emplace_back(elems[0]);
          unpacked.emplace_back(elems[1]);
        }
      }
      acc = std::move(unpacked);
    }
  }

  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinThreads(triton::ReduceOp op, LinearLayout layout,
                      SmallVector<SmallVector<Value>> accs,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    unsigned axis = op.getAxis();
    auto kReg = str_attr("register");
    auto linearAttr = triton::gpu::LinearEncodingAttr::get(ctx, layout);
    auto basesPerDim = linearAttr.basesPerDim(kReg, /*skipBroadcast=*/true);
    unsigned axisPack = basesPerDim[axis];
    if (axisPack == 1)
      return {std::move(layout), std::move(accs)};

    ReduceOpHelper helper(op);
    auto vectorizeKind = helper.getInThreadVectorizeOpKind(
        axisPack, targetInfo.supportBitwidth16Elementwise(),
        targetInfo.supportBitwidth32Elementwise());
    bool vectorize =
        vectorizeKind != ReduceOpHelper::InThreadVectorizeOpKind::None;
    auto perm = ReduceOpHelper::moveAxisBasesToFront(layout, axis, vectorize);
    if (!perm.isIdentity()) {
      layout = perm.apply(layout);
      for (auto &vals : accs)
        vals = perm.apply(vals);
    }
    if (vectorize)
      packVectorized(accs, rewriter);

    const auto &regBases = layout.getBases().lookup(kReg);
    bool packAlongAxis = vectorize && regBases.front()[axis] != 0;
    if (packAlongAxis)
      axisPack /= 2;

    auto elemTy =
        cast<RankedTensorType>(op.getOperandTypes().front()).getElementType();
    std::unique_ptr<Region> vectorCombineRegion =
        createVectorCombineRegion(loc, elemTy, vectorizeKind, rewriter);
    Region &combineRegion =
        vectorCombineRegion ? *vectorCombineRegion : op.getCombineOp();
    Operation &combinerOp = combineRegion.front().front();
    unsigned arity = targetInfo.getReductionTreeArity(&combinerOp);

    unsigned numOperands = accs.size();
    SmallVector<SmallVector<Value>> reduced(numOperands);
    unsigned regs = accs.front().size();
    for (unsigned regBase = 0; regBase < regs; regBase += axisPack) {
      SmallVector<SmallVector<Value>> vals;
      for (unsigned i = 0; i < axisPack; ++i) {
        SmallVector<Value> cur(numOperands);
        for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx)
          cur[opIdx] = accs[opIdx][regBase + i];
        vals.push_back(std::move(cur));
      }
      auto acc =
          treeReduce(loc, rewriter, combineRegion, std::move(vals), arity);
      for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx)
        reduced[opIdx].push_back(acc[opIdx]);
    }
    accs = std::move(reduced);

    if (vectorize) {
      Region *reduceAfterUnpacking =
          packAlongAxis ? &op.getCombineOp() : nullptr;
      unpackVectorized(loc, accs, rewriter, reduceAfterUnpacking);
    }

    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, axis, kReg);
    layout = actionRemoveBroadcastedRegs(layout).apply(layout);
    return {std::move(layout), std::move(accs)};
  }

  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinWarps(triton::ReduceOp op, LinearLayout layout,
                    SmallVector<SmallVector<Value>> accs,
                    ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto kLane = str_attr("lane");
    const auto &laneBases = layout.getBases().lookup(kLane);
    unsigned reduceLaneIdMask = 0;
    for (unsigned bit = 0; bit < laneBases.size(); ++bit) {
      if (laneBases[bit][op.getAxis()] != 0)
        reduceLaneIdMask |= 1u << bit;
    }
    if (reduceLaneIdMask == 0)
      return {std::move(layout), std::move(accs)};

    unsigned regs = accs.front().size();
    for (unsigned reg = 0; reg < regs; ++reg) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i)
        acc[i] = accs[i][reg];
      warpReduce(op, reduceLaneIdMask, acc, rewriter);
      for (unsigned i = 0; i < op.getNumOperands(); ++i)
        accs[i][reg] = acc[i];
    }

    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, op.getAxis(),
                                                         kLane);
    return {std::move(layout), std::move(accs)};
  }

  void warpReduce(triton::ReduceOp op, unsigned reduceLaneIdMask,
                  SmallVector<Value> &acc,
                  ConversionPatternRewriter &rewriter) const {
    if (reduceLaneIdMask == 0)
      return;
    auto moduleOp = op->getParentOfType<ModuleOp>();
    unsigned warpSize = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);
    if (targetInfo.warpReduce(rewriter, op.getLoc(), acc, op, reduceLaneIdMask))
      return;
    for (int bit = llvm::Log2_32(warpSize) - 1; bit >= 0; --bit) {
      unsigned mask = 1u << bit;
      if ((reduceLaneIdMask & mask) == 0)
        continue;
      SmallVector<Value> shfl(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i)
        shfl[i] = targetInfo.shuffleXor(rewriter, op.getLoc(), acc[i], mask);
      accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl);
    }
  }

  void packResults(triton::ReduceOp op, SmallVector<SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType()))
        results[i] = packLLElements(loc, getTypeConverter(), accs[i], rewriter,
                                    resultTy);
      else
        results[i] = accs[i].front();
    }
    rewriter.replaceOp(op, results);
  }

  SmallVector<int64_t> getSmemBaseOffsets(triton::ReduceOp op,
                                          const LinearLayout &srcLayout,
                                          const LinearLayout &dstLayout) const {
    std::vector<unsigned> indices(op.getNumOperands());
    std::iota(indices.begin(), indices.end(), 0);
    auto *ctx = op.getContext();
    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      auto lhsTy = getReduceMemElemTy(op.getElementTypes()[i], ctx);
      auto rhsTy = getReduceMemElemTy(op.getElementTypes()[j], ctx);
      return getIntOrFloatOrPtrBitWidth(lhsTy) >
             getIntOrFloatOrPtrBitWidth(rhsTy);
    });
    SmallVector<int64_t> offsets(op.getNumOperands());
    int64_t offset = 0;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      unsigned idx = indices[i];
      offsets[idx] = offset;
      auto inputTy = op.getInputTypes()[idx];
      auto bytes = getNumScratchElemsSwizzledCvt(srcLayout, dstLayout,
                                                 getBitwidth(inputTy)) *
                   (getBitwidth(inputTy) / 8);
      offset += bytes;
    }
    return offsets;
  }

  SmallVector<SmallVector<Value>>
  convertLayoutValues(Location loc, ConversionPatternRewriter &rewriter,
                      triton::ReduceOp op, const LinearLayout &srcLayout,
                      const LinearLayout &dstLayout,
                      const SmallVector<SmallVector<Value>> &inVals) const {
    SmallVector<SmallVector<Value>> outVals(op.getNumOperands());
    auto *ctx = rewriter.getContext();
    SmallVector<int64_t> shape;
    for (auto dim : srcLayout.getOutDimNames())
      shape.push_back(srcLayout.getOutDimSize(dim));
    auto srcEnc = triton::gpu::LinearEncodingAttr::get(ctx, srcLayout);
    auto dstEnc = triton::gpu::LinearEncodingAttr::get(ctx, dstLayout);
    auto baseOffsetAttr = op->getAttrOfType<IntegerAttr>("allocation.offset");
    int64_t baseOffset = baseOffsetAttr.getValue().getZExtValue();
    auto smemBaseOffsets = getSmemBaseOffsets(op, srcLayout, dstLayout);
    auto offsetTy = IntegerType::get(ctx, 32);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = op.getElementTypes()[i];
      auto srcTy = RankedTensorType::get(shape, elemTy, srcEnc);
      auto dstTy = RankedTensorType::get(shape, elemTy, dstEnc);
      Value packed =
          packLLElements(loc, getTypeConverter(), inVals[i], rewriter, srcTy);
      auto srcTensor =
          UnrealizedConversionCastOp::create(rewriter, loc, srcTy, packed)
              .getResult(0);
      auto cvt = ttg::ConvertLayoutOp::create(rewriter, loc, dstTy, srcTensor);
      cvt->setAttr("allocation.offset",
                   IntegerAttr::get(offsetTy, baseOffset + smemBaseOffsets[i]));
      Type packedDstTy = getTypeConverter()->convertType(dstTy);
      auto packedDst = UnrealizedConversionCastOp::create(
                           rewriter, loc, packedDstTy, cvt.getResult())
                           .getResult(0);
      outVals[i] = unpackLLElements(loc, packedDst, rewriter);
    }
    return outVals;
  }

  SmallVector<SegmentInfo>
  getSegmentInfos(triton::ReduceOp op,
                  const SmallVector<SmallVector<Value>> &accs,
                  unsigned numThreads) const {
    auto *ctx = op.getContext();
    SmallVector<SegmentInfo> infos;
    unsigned offset = 0;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Type elemTy = getReduceMemElemTy(op.getElementTypes()[i], ctx);
      unsigned bitWidth = getIntOrFloatOrPtrBitWidth(elemTy);
      unsigned regsPerThread = accs[i].size();
      unsigned bytesPerThread = regsPerThread * bitWidth / 8;
      infos.push_back(
          {elemTy, bitWidth, regsPerThread, bytesPerThread, offset});
      offset += numThreads * bytesPerThread;
    }
    return infos;
  }

  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceAcrossCTAs(triton::ReduceOp op, LinearLayout layout,
                   SmallVector<SmallVector<Value>> accs,
                   ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    unsigned numThreads =
        ttg::lookupNumWarps(op) * ttg::lookupThreadsPerWarp(rewriter);
    unsigned numCTAs = ttg::lookupNumCTAs(op);
    auto segments = getSegmentInfos(op, accs, numThreads);
    unsigned payloadBytes = 0;
    for (const auto &seg : segments)
      payloadBytes = std::max(
          payloadBytes, seg.offsetBytes + numThreads * seg.bytesPerThread);
    payloadBytes = llvm::alignTo(payloadBytes, 16u);
    unsigned dstOffset = payloadBytes;
    unsigned barrierOffset = llvm::alignTo(payloadBytes * 2, 8u);

    Value base =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto smemPtrTy =
        ptr_ty(rewriter.getContext(), targetInfo.getSharedAddressSpace());
    Value srcBase = base;
    Value dstBase = b.gep(smemPtrTy, i8_ty, base, b.i32_val(dstOffset),
                          LLVM::GEPNoWrapFlags::inbounds);
    Value barrierPtr = b.gep(smemPtrTy, i8_ty, base, b.i32_val(barrierOffset),
                             LLVM::GEPNoWrapFlags::inbounds);

    Value tid = getThreadId(rewriter, loc);
    Value thread0 = b.icmp_eq(tid, b.i32_val(0));
    Value clusterId = targetInfo.getClusterCTAId(rewriter, loc);

    unsigned numStages = llvm::Log2_32(numCTAs);
    for (unsigned stage = 0; stage < numStages; ++stage) {
      storeAccsToScratch(loc, rewriter, srcBase, segments, tid, accs);
      targetInfo.clusterBarrier(loc, rewriter);

      emitMBarrierInit(rewriter, loc, barrierPtr, thread0);
      ttng::FenceMBarrierInitReleaseClusterOp::create(rewriter, loc);
      ttng::ClusterBarrierOp::create(rewriter, loc, /*relaxed=*/true);

      emitMBarrierExpect(rewriter, loc, barrierPtr, thread0, payloadBytes);
      Value peer = b.xor_(clusterId, b.i32_val(1u << stage));
      Value remoteDst =
          mapaSharedCluster(rewriter, loc, dstBase, peer, thread0);
      Value remoteBarrier =
          mapaSharedCluster(rewriter, loc, barrierPtr, peer, thread0);
      emitBulkCopy(rewriter, loc, thread0, remoteDst, srcBase, payloadBytes,
                   remoteBarrier);
      emitMBarrierWait(rewriter, loc, barrierPtr, /*phase=*/0);

      auto peerVals =
          loadAccsFromScratch(loc, rewriter, dstBase, segments, tid, accs);
      for (unsigned operand = 0; operand < accs.size(); ++operand) {
        for (unsigned reg = 0; reg < accs[operand].size(); ++reg) {
          SmallVector<Value> cur = {peerVals[operand][reg]};
          SmallVector<Value> accum = {accs[operand][reg]};
          accumulate(loc, rewriter, op.getCombineOp(), accum, cur);
          accs[operand][reg] = accum[0];
        }
      }
      targetInfo.clusterBarrier(loc, rewriter);
    }

    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, op.getAxis(),
                                                         str_attr("block"));
    layout = actionRemoveBroadcastedRegs(layout).apply(layout);
    return {std::move(layout), std::move(accs)};
  }

  void storeAccsToScratch(Location loc, ConversionPatternRewriter &rewriter,
                          Value base, ArrayRef<SegmentInfo> segments, Value tid,
                          const SmallVector<SmallVector<Value>> &accs) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemPtrTy =
        ptr_ty(rewriter.getContext(), targetInfo.getSharedAddressSpace());
    for (unsigned operand = 0; operand < segments.size(); ++operand) {
      const auto &seg = segments[operand];
      for (unsigned reg = 0; reg < seg.regsPerThread; ++reg) {
        Value linear =
            b.add(b.mul(tid, b.i32_val(seg.regsPerThread)), b.i32_val(reg));
        Value byteOff = b.add(b.i32_val(seg.offsetBytes),
                              b.mul(linear, b.i32_val(seg.bitWidth / 8)));
        Value ptr = b.gep(smemPtrTy, i8_ty, base, byteOff,
                          LLVM::GEPNoWrapFlags::inbounds);
        targetInfo.storeShared(rewriter, loc, ptr, accs[operand][reg],
                               b.true_val());
      }
    }
  }

  SmallVector<SmallVector<Value>> loadAccsFromScratch(
      Location loc, ConversionPatternRewriter &rewriter, Value base,
      ArrayRef<SegmentInfo> segments, Value tid,
      const SmallVector<SmallVector<Value>> &accsTemplate) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemPtrTy =
        ptr_ty(rewriter.getContext(), targetInfo.getSharedAddressSpace());
    SmallVector<SmallVector<Value>> loaded(accsTemplate.size());
    for (unsigned operand = 0; operand < segments.size(); ++operand) {
      const auto &seg = segments[operand];
      loaded[operand].reserve(seg.regsPerThread);
      for (unsigned reg = 0; reg < seg.regsPerThread; ++reg) {
        Value linear =
            b.add(b.mul(tid, b.i32_val(seg.regsPerThread)), b.i32_val(reg));
        Value byteOff = b.add(b.i32_val(seg.offsetBytes),
                              b.mul(linear, b.i32_val(seg.bitWidth / 8)));
        Value ptr = b.gep(smemPtrTy, i8_ty, base, byteOff,
                          LLVM::GEPNoWrapFlags::inbounds);
        loaded[operand].push_back(targetInfo.loadShared(
            rewriter, loc, ptr, accsTemplate[operand][reg].getType(),
            b.true_val()));
      }
    }
    return loaded;
  }

  void emitMBarrierInit(ConversionPatternRewriter &rewriter, Location loc,
                        Value barrierPtr, Value pred) const {
    PTXBuilder ptxBuilder;
    auto &op = *ptxBuilder.create("@$0 mbarrier.init.shared::cta.b64 [$1], 1;");
    op({ptxBuilder.newOperand(pred, "b"),
        ptxBuilder.newAddrOperand(barrierPtr, "r")},
       /*onlyAttachMLIRArgs=*/true);
    ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
  }

  void emitMBarrierExpect(ConversionPatternRewriter &rewriter, Location loc,
                          Value barrierPtr, Value pred, unsigned bytes) const {
    PTXBuilder ptxBuilder;
    std::string ptx =
        "@$0 mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], " +
        std::to_string(bytes) + ";";
    auto &op = *ptxBuilder.create(ptx);
    op({ptxBuilder.newOperand(pred, "b"),
        ptxBuilder.newAddrOperand(barrierPtr, "r")},
       /*onlyAttachMLIRArgs=*/true);
    ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
  }

  void emitBulkCopy(ConversionPatternRewriter &rewriter, Location loc,
                    Value pred, Value remoteDst, Value localSrc, unsigned bytes,
                    Value remoteBarrier) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    PTXBuilder ptxBuilder;
    auto ptx = "@$0 cp.async.bulk.shared::cluster.shared::cta"
               ".mbarrier::complete_tx::bytes [$1], [$2], $3, [$4];";
    auto &copy = *ptxBuilder.create(ptx);
    SmallVector<PTXBuilder::Operand *> operands{
        ptxBuilder.newOperand(pred, "b"),
        ptxBuilder.newAddrOperand(b.ptrtoint(i32_ty, remoteDst), "r"),
        ptxBuilder.newAddrOperand(b.ptrtoint(i32_ty, localSrc), "r"),
        ptxBuilder.newOperand(b.i32_val(bytes), "r"),
        ptxBuilder.newAddrOperand(b.ptrtoint(i32_ty, remoteBarrier), "r")};
    copy(operands, /*onlyAttachMLIRArgs=*/true);
    ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
  }

  void emitMBarrierWait(ConversionPatternRewriter &rewriter, Location loc,
                        Value barrierPtr, unsigned phase) const {
    std::string ptx = R"(
{
  .reg .pred complete;
waitLoop:
  mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], )" +
                      std::to_string(phase) + R"(;
  @!complete bra.uni waitLoop;
}
)";
    PTXBuilder ptxBuilder;
    auto &wait = *ptxBuilder.create(ptx);
    wait({ptxBuilder.newAddrOperand(barrierPtr, "r")},
         /*onlyAttachMLIRArgs=*/true);
    ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
  }
};

} // namespace

void mlir::triton::NVIDIA::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, targetInfo, benefit);
}
