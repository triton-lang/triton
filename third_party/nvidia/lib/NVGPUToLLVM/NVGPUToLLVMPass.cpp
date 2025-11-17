#include "NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "NVGPUToLLVM/Passes.h"

#include "Dialect/NVGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

#include "nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"

#include "llvm/Support/ErrorHandling.h"

namespace ttn = mlir::triton::nvgpu;
using ttn::Constraints;
using ttn::OperandsAndConstraints;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_CONVERTNVGPUTOLLVM
#include "NVGPUToLLVM/Passes.h.inc"

namespace {

bool isNumber(const std::string &s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

Type getTypeFromConstraint(char constraint, PatternRewriter &rewriter) {
  Type ty;
  if (constraint == 'b')
    ty = IntegerType::get(rewriter.getContext(), 1);
  else if (constraint == 'h')
    ty = IntegerType::get(rewriter.getContext(), 16);
  else if (constraint == 'r')
    ty = IntegerType::get(rewriter.getContext(), 32);
  else if (constraint == 'l')
    ty = IntegerType::get(rewriter.getContext(), 64);
  else if (constraint == 'f')
    ty = Float32Type::get(rewriter.getContext());
  else if (constraint == 'd')
    ty = Float64Type::get(rewriter.getContext());
  else {
    assert(false && "Unsupported constraint");
  }
  return ty;
}

// Converts the given value to the type represented by the constraint
// E.g. if val is of type llvmptr and constraint is 'r', then we convert
// val to i32 using ptrtoint(i32_ty, val)
Value convertToType(Value val, std::string constraint, Location loc,
                    PatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto isConstraintNumber = isNumber(constraint);
  if (!isConstraintNumber) {
    auto ty = getTypeFromConstraint(constraint[0], rewriter);
    if (isa<LLVM::LLVMPointerType>(val.getType())) {
      return b.ptrtoint(ty, val);
    } else {
      assert(val.getType().getIntOrFloatBitWidth() <=
                 ty.getIntOrFloatBitWidth() &&
             "Cannot convert to a smaller type");
      if (val.getType().getIntOrFloatBitWidth() < ty.getIntOrFloatBitWidth())
        return b.zext(ty, val);
    }
  }
  return val;
}

SmallVector<PTXBuilder::Operand *>
getPtxOutputs(const nvgpu::Constraints &outputConstraints,
              PTXBuilder &ptxBuilder) {
  SmallVector<PTXBuilder::Operand *> ptxOutputs;
  for (unsigned i = 0; i < outputConstraints.size(); i++) {
    auto *ptxOutput = ptxBuilder.newOperand(outputConstraints[i]);
    ptxOutputs.push_back(ptxOutput);
  }
  return ptxOutputs;
}

OperandsAndConstraints
unpackOperands(const OperandsAndConstraints &operandsAndConstraints,
               PTXBuilder &ptxBuilder, Location loc,
               PatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  OperandsAndConstraints unpackedOperands;
  for (const auto &[operand, constraint] : operandsAndConstraints) {
    auto llvmStruct = llvm::dyn_cast<LLVM::LLVMStructType>(operand.getType());
    // if a constraint is a number, then we are doing input/output tying
    // if the operand is a struct, then we need to unpack it, and
    // add the constraint to each of the unpacked operands uses the constraint
    // as an offset
    auto isConstraintNumber = isNumber(constraint);
    if (llvmStruct) {
      for (unsigned i = 0; i < llvmStruct.getBody().size(); i++) {
        if (isConstraintNumber) {
          auto constraintInt = std::stoi(constraint) + i;
          unpackedOperands.push_back(
              {b.extract_val(llvmStruct.getBody()[i], operand, i),
               std::to_string(constraintInt)});
        } else {
          unpackedOperands.push_back(
              {b.extract_val(llvmStruct.getBody()[i], operand, i), constraint});
        }
      }
    } else {
      unpackedOperands.push_back({operand, constraint});
    }
  }
  return unpackedOperands;
}

SmallVector<PTXBuilder::Operand *>
getPtxOperands(const OperandsAndConstraints &operandsAndConstraints,
               PTXBuilder &ptxBuilder, Location loc,
               PatternRewriter &rewriter) {
  SmallVector<PTXBuilder::Operand *> ptxOperands;
  auto unpackedOperandsAndConstraints =
      unpackOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
  for (auto &[operand, constraint] : unpackedOperandsAndConstraints) {
    auto convertedOperand = convertToType(operand, constraint, loc, rewriter);
    auto *ptxOperand = ptxBuilder.newOperand(convertedOperand, constraint);
    ptxOperands.push_back(ptxOperand);
  }
  return ptxOperands;
}

std::string patchPtxAsm(Operation *op, std::string ptxAsm) {
  std::vector<std::pair<int, int>> patchLocations;
  std::vector<std::string> patchValues;
  auto start = ptxAsm.find("#", 0);
  while (start != std::string::npos) {
    auto endIterator =
        std::find_if(ptxAsm.begin() + start + 1, ptxAsm.end(),
                     [](unsigned char c) { return !std::isalnum(c); });

    assert(endIterator != ptxAsm.end() && "unexpected asm format");

    auto end = std::distance(ptxAsm.begin(), endIterator);
    auto patchLocation = std::make_pair(start, end);
    patchLocations.push_back(patchLocation);
    auto patchValue = ptxAsm.substr(start + 1, end - start - 1);
    patchValues.push_back(patchValue);
    start = ptxAsm.find("#", end);
  }
  assert(patchLocations.size() == patchValues.size() &&
         "patchLocations and patchValues should have the same size");
  if (patchLocations.size() == 0) {
    return ptxAsm;
  }
  std::string res = "";
  size_t prevStart = 0;
  unsigned i = 0;
  for (auto &[start, end] : patchLocations) {
    res += ptxAsm.substr(prevStart, start - prevStart);
    auto integerAttr = op->getAttrOfType<IntegerAttr>(patchValues[i]);
    auto attr = integerAttr.getInt();
    res += std::to_string(attr);
    prevStart = end;
    i++;
  }
  if (prevStart < ptxAsm.size())
    res += ptxAsm.substr(prevStart, ptxAsm.size() - prevStart);
  return res;
}

template <typename SourceOp>
class NVGPUOpGenericPattern : public OpRewritePattern<SourceOp> {
public:
  explicit NVGPUOpGenericPattern(MLIRContext *context, std::string ptxAsm,
                                 Constraints outputConstraints,
                                 Constraints inputConstraints)
      : OpRewritePattern<SourceOp>(context), ptxAsm(std::move(ptxAsm)),
        outputConstraints(outputConstraints),
        inputConstraints(inputConstraints) {}

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    OperandsAndConstraints operandsAndConstraints;
    for (unsigned i = 0; i < inputConstraints.size(); i++) {
      operandsAndConstraints.push_back(
          {op->getOperand(i), inputConstraints[i]});
    }
    return rewriteAsPtxAsm(op, rewriter, ptxAsm, operandsAndConstraints,
                           outputConstraints);
  }

private:
  std::string ptxAsm;
  Constraints outputConstraints;
  Constraints inputConstraints;
};

class WarpIdOpPattern : public OpRewritePattern<mlir::triton::gpu::WarpIdOp> {
public:
  using OpRewritePattern<mlir::triton::gpu::WarpIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::triton::gpu::WarpIdOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    if (triton::gpu::lookupNumWarps(op) == 1) {
      // If there is only one warp, the warp ID is always 0.
      rewriter.replaceOp(op, b.i32_val(0));
      return success();
    }

    // If this is inside a warp specialize op, compute the relative thread ID
    // within the warp group.
    Value tid = NVVM::ThreadIdXOp::create(rewriter, loc, i32_ty);
    if (std::optional<int> startId =
            getWarpGroupStartThreadId(rewriter.getInsertionBlock()))
      tid = LLVM::SubOp::create(rewriter, loc, tid, b.i32_val(*startId));

    Value warpId = b.udiv(tid, b.i32_val(32));
    // This indicates to PTXAS that the result and its derived values are
    // uniform across the warp. For example, if a branch condition derives from
    // this value, it can be proven to be non-divergent.
    warpId = LLVM::NVIDIA::shuffleIdx(loc, rewriter, warpId, 0);
    rewriter.replaceOp(op, warpId);
    return success();
  }
};

class ClusterCTAIdOpPattern : public OpRewritePattern<ttn::ClusterCTAIdOp> {
  using OpRewritePattern<ttn::ClusterCTAIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::ClusterCTAIdOp op,
                                PatternRewriter &rewriter) const override {
    // TODO Should we pass in the range of the cluster ID?
    // We should benchmark as when doing so for thread_id it regressed lol
    // auto numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(
    //     op->getParentOfType<ModuleOp>());
    auto res = NVVM::ClusterId::create(rewriter, op.getLoc(), i32_ty);
    rewriter.replaceOp(op, res);
    return success();
  }
};

class LoadAcquireOpPattern : public OpRewritePattern<ttn::LoadAcquireOp> {
public:
  using OpRewritePattern<ttn::LoadAcquireOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::LoadAcquireOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type valueTy = op.getType();
    const unsigned valueNBits =
        std::max(8u, (unsigned)getIntOrFloatOrPtrBitWidth(valueTy));
    const size_t maxWordWidth = std::max<size_t>(32, valueNBits);
    const size_t width = std::min((size_t)valueNBits, maxWordWidth);

    const std::string writeConstraint =
        (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");
    PTXBuilder ptxBuilder;
    bool init = true;
    auto *dstOpr = ptxBuilder.newOperand(writeConstraint, init); // =r operation
    auto *addrOpr =
        ptxBuilder.newAddrOperand(op.getAddr(), "l", 0 /* in_off */);
    auto &ld =
        ptxBuilder.create("ld")
            ->global()
            .o("cta", op.getScope() == triton::nvgpu::MemSyncScope::CTA)
            .o("gpu", op.getScope() == triton::nvgpu::MemSyncScope::GPU)
            .o("sys", op.getScope() == triton::nvgpu::MemSyncScope::SYSTEM)
            .o("acquire", op.getSem() == triton::nvgpu::MemSemantic::ACQUIRE)
            .o("relaxed", op.getSem() == triton::nvgpu::MemSemantic::RELAXED)
            .b(width);
    ld(dstOpr, addrOpr).maybePredicate(op.getMask(), "b");

    // Create inline ASM signature
    Type retTy = IntegerType::get(getContext(), width);
    Value ret = ptxBuilder.launch(rewriter, loc, retTy);
    ret = b.bitcast(ret, op.getType());

    rewriter.replaceOp(op, {ret});
    return success();
  }
};

class WGMMAWaitGroupOpPattern : public OpRewritePattern<ttn::WGMMAWaitGroupOp> {
public:
  using OpRewritePattern<ttn::WGMMAWaitGroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::WGMMAWaitGroupOp op,
                                PatternRewriter &rewriter) const override {
    return rewriteAsPtxAsm(op, rewriter, getPtxAsm(op),
                           getOperandsAndConstraints(op),
                           getOutputConstraints(op));
  }

  Constraints getOutputConstraints(ttn::WGMMAWaitGroupOp op) const {
    auto outputStructType = cast<LLVM::LLVMStructType>(op.getType());
    uint32_t numOutputRegs = outputStructType.getBody().size();
    Constraints constraints;
    constraints.reserve(numOutputRegs);
    mlir::DataLayout dl(op->getParentOfType<mlir::ModuleOp>());
    for (auto ty : outputStructType.getBody()) {
      auto bitwidth = dl.getTypeSizeInBits(ty);
      std::string c;
      switch (bitwidth) {
      case 64:
        c = "=l";
        break;
      case 32:
        c = ty.isF32() ? "=f" : "=r";
        break;
      case 16:
        c = "=h";
        break;
      default:
        llvm::report_fatal_error("Unexpected bitwidth in WGMMAWaitGroupOp: " +
                                 Twine(bitwidth));
      }
      constraints.push_back(c);
    }
    return constraints;
  }

  OperandsAndConstraints
  getOperandsAndConstraints(ttn::WGMMAWaitGroupOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto input = op.getInput();
    operandsAndConstraints.push_back({input, "0"});
    return operandsAndConstraints;
  }

  std::string getPtxAsm(ttn::WGMMAWaitGroupOp op) const {
    auto outputStructType = dyn_cast<LLVM::LLVMStructType>(op.getType());
    uint32_t numCRegs = outputStructType.getBody().size();
    std::string args = "";
    uint32_t asmOpIdx = 0;
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
    }
    auto ptxAsm = "// wait for regs: " + args + "\n\t" +
                  "wgmma.wait_group.sync.aligned #pendings;";
    return ptxAsm;
  }
};

class WGMMAOpPattern : public OpRewritePattern<ttn::WGMMAOp> {
public:
  using OpRewritePattern<ttn::WGMMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::WGMMAOp op,
                                PatternRewriter &rewriter) const override {
    return rewriteAsPtxAsm(op, rewriter, getPtxAsm(op),
                           getOperandsAndConstraints(op),
                           getOutputConstraints(op));
  }

  std::vector<std::string> getOutputConstraints(ttn::WGMMAOp op) const {
    // TODO (zahi): Return type must always be a struct for wgmma, currently
    // we rely on the size of output constraints vector to determine whether
    // the output is a struct or not. We should find a way to pass this info
    auto resultType = op.getType();

    auto outputStructType = dyn_cast<LLVM::LLVMStructType>(resultType);
    uint32_t numOutputRegs = outputStructType.getBody().size();
    std::string output =
        outputStructType.getBody().front().isF32() ? "=f" : "=r";
    return std::vector<std::string>(numOutputRegs, output);
  }

  OperandsAndConstraints getOperandsAndConstraints(ttn::WGMMAOp op) const {
    OperandsAndConstraints operandsAndConstraints;
    auto opA = op.getOpA();
    auto opB = op.getOpB();
    auto opC = op.getOpC();
    auto opScaleD = op.getUseC();
    auto typeA = opA.getType();

    auto structTypeA = dyn_cast<LLVM::LLVMStructType>(typeA);

    // TODO (zahi): is this the best way to tie inputs/outputs ?
    if (opC)
      operandsAndConstraints.push_back({opC, "0"});

    if (structTypeA) {
      operandsAndConstraints.push_back({opA, "r"});
    } else {
      operandsAndConstraints.push_back({opA, "l"});
    }

    // Operand B (must be `desc`)
    operandsAndConstraints.push_back({opB, "l"});

    // `scale-d`
    if (op.getOpC())
      operandsAndConstraints.push_back({opScaleD, "b"});

    return operandsAndConstraints;
  }

  std::string getPtxAsm(ttn::WGMMAOp op) const {
    using namespace ttn;
    auto opA = op.getOpA();
    auto opB = op.getOpB();
    auto m = op.getM();
    auto n = op.getN();
    auto k = op.getK();
    auto eltTypeC = op.getEltTypeC();
    auto eltTypeA = op.getEltTypeA();
    auto eltTypeB = op.getEltTypeB();
    auto layoutA = op.getLayoutA();
    auto layoutB = op.getLayoutB();

    // Register checks
    auto typeA = opA.getType();
    auto typeB = opB.getType();
    auto typeOutput = op.getType();
    auto structTypeA = dyn_cast<LLVM::LLVMStructType>(typeA);
    auto structTypeB = dyn_cast<LLVM::LLVMStructType>(typeB);
    auto structTypeOutput = dyn_cast<LLVM::LLVMStructType>(typeOutput);
    assert(!structTypeB && "Operand B can not be registers");
    assert(structTypeOutput && "Output and C operand must be registers");

    // Element type, MNK shape and transposing support check
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma
    bool transA = layoutA == WGMMALayout::col;
    bool transB = layoutB == WGMMALayout::row;
    bool supported = false, needTransArgs = false, floatTypeWGMMA = false;
    assert(m % 8 == 0 && n % 8 == 0 && k % 8 == 0);
    // Below instructions do support transposing, must pass `trans` arguments
    supported |=
        (eltTypeA == WGMMAEltType::f16) && (eltTypeB == WGMMAEltType::f16) &&
        (eltTypeC == WGMMAEltType::f16 || eltTypeC == WGMMAEltType::f32) &&
        (m == 64 && 8 <= n && n <= 256 && k == 16);
    supported |= (eltTypeA == WGMMAEltType::bf16) &&
                 (eltTypeB == WGMMAEltType::bf16) &&
                 (eltTypeC == WGMMAEltType::f32) &&
                 (m == 64 && 8 <= n && n <= 256 && k == 16);
    needTransArgs = supported;
    floatTypeWGMMA = supported;
    // Below instructions do not support transposing
    if (!supported && !transA && !transB) {
      supported |= (eltTypeA == WGMMAEltType::tf32) &&
                   (eltTypeB == WGMMAEltType::tf32) &&
                   (eltTypeC == WGMMAEltType::f32) &&
                   (m == 64 && 8 <= n && n <= 256 && k == 8);
      supported |=
          (eltTypeA == WGMMAEltType::e4m3 || eltTypeA == WGMMAEltType::e5m2) &&
          (eltTypeB == WGMMAEltType::e4m3 || eltTypeB == WGMMAEltType::e5m2) &&
          (eltTypeC == WGMMAEltType::f16 || eltTypeC == WGMMAEltType::f32) &&
          (m == 64 && 8 <= n && n <= 256 && k == 32);
      floatTypeWGMMA = supported;
      // Below instructions are integer-based
      supported |= (eltTypeA == WGMMAEltType::s8) &&
                   (eltTypeB == WGMMAEltType::s8) &&
                   (eltTypeC == WGMMAEltType::s32) &&
                   (m == 64 && 8 <= n && n <= 224 && k == 32);
    }
    assert(supported && "WGMMA type or shape is not supported");

    // Operands
    uint32_t asmOpIdx = 0;
    std::string args = "";

    // Output and operand C
    uint32_t numCRegs = structTypeOutput.getBody().size();

    args += "{";
    for (uint32_t i = 0; i < numCRegs; ++i) {
      args += "$" + std::to_string(asmOpIdx++) + (i == numCRegs - 1 ? "" : ",");
    }
    args += "}, ";

    if (op.getOpC())
      asmOpIdx += numCRegs;

    // Operand A
    if (structTypeA) {
      uint32_t numARegs = structTypeA.getBody().size();
      args += "{";
      for (uint32_t i = 0; i < numARegs; ++i) {
        args +=
            "$" + std::to_string(asmOpIdx++) + (i == numARegs - 1 ? "" : ",");
      }
      args += "}, ";
    } else {
      args += "$" + std::to_string(asmOpIdx++) + ", ";
    }

    // Operand B (must be `desc`)
    args += "$" + std::to_string(asmOpIdx++) + ", ";

    // `scale-d`
    if (op.getOpC())
      args += "$" + std::to_string(asmOpIdx++);
    else
      args += "0";

    // `imm-scale-a`, and `imm-scale-b` are 1 by default only for float-based
    // WGMMA
    if (floatTypeWGMMA)
      args += ", 1, 1";

    // Push `trans-a` and `trans-b` args if needed (determined as constant)
    if (needTransArgs) {
      if (!structTypeA)
        args += ", " + std::to_string(transA);
      args += ", " + std::to_string(transB);
    }

    auto ptxAsm = "wgmma.mma_async.sync.aligned"
                  ".m" +
                  std::to_string(m) + "n" + std::to_string(n) + "k" +
                  std::to_string(k) + "." + stringifyEnum(eltTypeC).str() +
                  "." + stringifyEnum(eltTypeA).str() + "." +
                  stringifyEnum(eltTypeB).str() + " " + args + ";";
    return ptxAsm;
  }
};

static Value createTMAlloc(IRRewriter &rewriter, LLVM::LLVMFuncOp func,
                           size_t size, Value pred, bool twoCTAs) {
  PTXBuilder ptxBuilder;
  Location loc = func.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value sharedMem = mlir::LLVM::getStackPointer(rewriter, func);
  std::string ptxString =
      "@$0 tcgen05.alloc.cta_group::" + std::to_string(twoCTAs ? 2 : 1) +
      ".sync.aligned.shared::cta.b32 [$1], " + std::to_string(size) + ";";

  auto &allocOp = *ptxBuilder.create(ptxString);
  allocOp(
      {ptxBuilder.newOperand(pred, "b"), ptxBuilder.newOperand(sharedMem, "r")},
      /*onlyAttachMLIRArgs=*/true);
  auto voidTy = void_ty(func->getContext());
  ptxBuilder.launch(rewriter, loc, void_ty(func->getContext()));
  NVVM::Barrier0Op::create(rewriter, loc);
  Value address = b.load(i32_ty, sharedMem);
  NVVM::Barrier0Op::create(rewriter, loc);
  address = b.inttoptr(ptr_ty(func.getContext(), 6), address);
  return address;
}

static void createRelinquishAlloc(IRRewriter &rewriter, Location loc,
                                  Value pred, bool twoCTAs) {
  PTXBuilder ptxBuilder;
  std::string ptxString = "@$0 tcgen05.relinquish_alloc_permit.cta_group::" +
                          std::to_string(twoCTAs ? 2 : 1) + ".sync.aligned;";
  auto &f = *ptxBuilder.create(ptxString);
  f({ptxBuilder.newOperand(pred, "b")}, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

void freeTMAlloc(LLVM::LLVMFuncOp func, Value alloc, size_t size, Value pred,
                 bool twoCTAs) {
  func.walk([&](LLVM::ReturnOp ret) {
    OpBuilder b(ret);
    auto ctx = ret->getContext();
    auto loc = ret.getLoc();
    auto voidTy = void_ty(ctx);
    NVVM::Barrier0Op::create(b, loc);
    PTXBuilder ptxBuilder;
    // Calculate the predicate in the inline asm to avoid creating long
    // liveranges.
    std::string ptxString =
        "@$0 tcgen05.dealloc.cta_group::" + std::to_string(twoCTAs ? 2 : 1) +
        ".sync.aligned.b32 $1, " + std::to_string(size) + ";";
    auto &dealloc = *ptxBuilder.create(ptxString);
    dealloc(
        {ptxBuilder.newOperand(pred, "b"), ptxBuilder.newOperand(alloc, "r")},
        /*onlyAttachMLIRArgs=*/true);
    ptxBuilder.launch(b, loc, void_ty(ctx));
  });
}

static Value initTensorMemory(LLVM::LLVMFuncOp func) {
  auto mod = func->getParentOfType<ModuleOp>();
  assert(mod->hasAttr("ttg.tensor_memory_size"));
  size_t size = cast<IntegerAttr>(mod->getAttr("ttg.tensor_memory_size"))
                    .getValue()
                    .getZExtValue();
  if (size == 0)
    return Value();
  IRRewriter rewriter(func.getContext());
  rewriter.setInsertionPointToStart(&func.front());
  auto ctx = mod.getContext();
  auto loc = func.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // A proper error will be raised by the frontend, but to allow compilation to
  // continue we emit a trap.
  if (size > 512) {
    LLVM::Trap::create(rewriter, loc);
    return LLVM::UndefOp::create(rewriter, loc, ptr_ty(ctx, 6));
  }

  bool useTwoCTAs = mlir::triton::nvidia_gpu::getModuleTwoCTAs(mod);
  // This code is only executed by the default warp group.
  Value threadId = NVVM::ThreadIdXOp::create(rewriter, loc, i32_ty);
  Value pred = b.icmp_ult(threadId, b.i32_val(32));
  Value alloc = createTMAlloc(rewriter, func, size, pred, useTwoCTAs);
  createRelinquishAlloc(rewriter, loc, pred, useTwoCTAs);
  // TODO: pred will have a long liverange, we need to check if this is a
  // problem and how it can be fixed.
  freeTMAlloc(func, alloc, size, pred, useTwoCTAs);
  return alloc;
}

static void lowerTensorMemoryAlloc(ModuleOp mod) {
  SmallVector<Operation *> baseOps;
  LLVM::LLVMFuncOp kernel = nullptr;
  mod.walk([&](ttn::TensorMemoryBaseAddress baseOp) {
    baseOps.push_back(baseOp);
    if (!kernel)
      kernel = baseOp->getParentOfType<LLVM::LLVMFuncOp>();
    assert(kernel == baseOp->getParentOfType<LLVM::LLVMFuncOp>() &&
           "TODO: add support for function calls using tmem.");
  });
  if (baseOps.empty())
    return;
  // TODO: Handle cases of matmul used in noinline functions.
  assert(triton::isKernel(kernel));
  Value newBase = initTensorMemory(kernel);
  if (!newBase)
    return;
  for (auto baseOp : baseOps) {
    baseOp->getResult(0).replaceAllUsesWith(newBase);
    baseOp->erase();
  }
}

} // anonymous namespace

class ConvertNVGPUToLLVM
    : public impl::ConvertNVGPUToLLVMBase<ConvertNVGPUToLLVM> {
public:
  using impl::ConvertNVGPUToLLVMBase<
      ConvertNVGPUToLLVM>::ConvertNVGPUToLLVMBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    RewritePatternSet patterns(context);

    patterns.add<ClusterCTAIdOpPattern, WGMMAOpPattern, LoadAcquireOpPattern,
                 WGMMAWaitGroupOpPattern, WarpIdOpPattern>(context);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();

    lowerTensorMemoryAlloc(mod);
    makeAllWarpGroupsIsolatedFromAbove(mod);
  }
};

LogicalResult
nvgpu::rewriteAsPtxAsm(Operation *op, PatternRewriter &rewriter,
                       std::string ptxAsm,
                       const OperandsAndConstraints &operandsAndConstraints,
                       const Constraints &outputConstraints) {
  auto ctx = rewriter.getContext();
  auto loc = op->getLoc();
  ptxAsm = patchPtxAsm(op, std::move(ptxAsm));
  auto hasSideEffects = !isMemoryEffectFree(op);

  PTXBuilder ptxBuilder;
  auto ptxOutputs = getPtxOutputs(outputConstraints, ptxBuilder);
  auto ptxOperands =
      getPtxOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
  SmallVector<PTXBuilder::Operand *> outputsAndOperands = ptxOutputs;
  outputsAndOperands.append(ptxOperands.begin(), ptxOperands.end());
  auto &ptxInstr = *ptxBuilder.create(ptxAsm);
  ptxInstr(outputsAndOperands, /*onlyAttachMLIRArgs=*/true);
  auto retTy =
      op->getNumResults() == 0 ? void_ty(ctx) : op->getResult(0).getType();
  auto res = ptxBuilder.launch(rewriter, loc, retTy,
                               /*hasSideEffects*/ hasSideEffects);
  if (op->getNumResults() == 0) {
    rewriter.eraseOp(op);
  } else {
    rewriter.replaceOp(op, res);
  }

  return success();
}

} // namespace triton
} // namespace mlir
