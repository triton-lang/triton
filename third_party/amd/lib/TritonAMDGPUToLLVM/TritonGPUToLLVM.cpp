#include "TritonGPUToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getSRegValue;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::AMD::TritonGPUToLLVMTypeConverter;
using ::AMD::ConvertTritonGPUOpToLLVMPatternBase;
using ::AMD::ConvertTritonGPUOpToLLVMPattern;

Value llGetPid(int axis, Location loc, ModuleOp moduleOp,
               ConversionPatternRewriter &rewriter) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);
#ifdef USE_ROCM
  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
  Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[axis]);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
#else
  // It is not easy to get the compute capability here, so we use numCTAs to
  // decide the semantic of GetProgramIdOp. If numCTAs = 1, then
  // GetProgramIdOp is converted to "%ctaid", otherwise it is converted to
  // "%clusterid".
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

  std::string sreg = numCTAs == 1 ? "%ctaid." : "%clusterid.";
  sreg.append(1, 'x' + axis); // 0 -> 'x', 1 -> 'y', 2 -> 'z'
  return getSRegValue(rewriter, loc, sreg);
#endif
}

struct ReturnOpConversion : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op.getNumOperands();

    // Currently, Triton kernel function always return nothing.
    // TODO(Superjomn) add support for non-inline device function
    if (numArguments > 0) {
      return rewriter.notifyMatchFailure(
          op, "Only kernel function with nothing returned is supported.");
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                op->getAttrs());
    return success();
  }
};

struct BroadcastOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::BroadcastOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::BroadcastOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Following the order of indices in the legacy code, a broadcast of:
    //   [s(0), s(1) ... s(k-1),    1, s(k+1), s(k+2) ... s(n-1)]
    // =>
    //   [s(0), s(1) ... s(k-1), s(k), s(k+1), s(k+2) ... s(n-1)]
    //
    // logically maps to a broadcast within a thread's scope:
    //   [cta(0)..cta(k-1),     1,cta(k+1)..cta(n-1),spt(0)..spt(k-1),
    //   1,spt(k+1)..spt(n-1)]
    // =>
    //   [cta(0)..cta(k-1),cta(k),cta(k+1)..cta(n-1),spt(0)..spt(k-1),spt(k),spt(k+1)..spt(n-1)]
    //
    // regardless of the order of the layout
    //
    Location loc = op->getLoc();
    Value src = adaptor.getSrc();
    Value result = op.getResult();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType resultTy = op.getType();
    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();

    assert(rank == resultTy.getRank());
    auto order = triton::gpu::getOrder(srcLayout);
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    SmallVector<Value> srcVals =
        getTypeConverter()->unpackLLElements(loc, src, rewriter);

    DenseMap<SmallVector<unsigned>, Value, SmallVectorKeyInfo> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }

    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      for (size_t j = 0; j < srcShape.size(); j++)
        if (srcShape[j] == 1)
          offset[j] = 0;
      resultVals.push_back(srcValues.lookup(offset));
    }

    Value resultStruct =
        getTypeConverter()->packLLElements(loc, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

// The input print op contains:
//  - a "prefix" (string) specified by the user, and
//  - one or more "operands" (tensors).
//
// For each operand, we print all of the values contained in this GPU thread,
// one per line, along with the index of the value in its tensor.
struct PrintOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::PrintOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::PrintOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    Value prefixStr =
        LLVM::addStringToModule(loc, rewriter, "printfPrefix_", op.getPrefix());

    auto getPid = [&](int axis) {
      return llGetPid(axis, loc, op->getParentOfType<ModuleOp>(), rewriter);
    };
    std::array<Value, 3> pid = {getPid(0), getPid(1), getPid(2)};

    // Simple printf of a string without any tensors.
    if (op.getNumOperands() == 0) {
      std::string formatStr;
      llvm::raw_string_ostream os(formatStr);
      os << "pid (" << getFormatSubstr(pid[0]) << ", "
         << getFormatSubstr(pid[1]) << ", " << getFormatSubstr(pid[2]) << ")%s";
      llPrintf(formatStr, {pid[0], pid[1], pid[2], prefixStr}, rewriter);
    } else {
      for (size_t i = 0; i < op.getNumOperands(); i++) {
        // Elements of the tensor that are resident in this GPU thread.
        auto elems = getTypeConverter()->unpackLLElements(
            loc, adaptor.getOperands()[i], rewriter);

        // Get the indices of `elems` within the tensor.  Note that if `elems`
        // has an "interesting" layout, then these will not be in any
        // particularly nice order.

        // Extract the shape of the tensor being printed and use it to figure
        // out how many digits we need for each of the dimensions.
        SmallVector<int, 8> dimWidths;
        SmallVector<SmallVector<Value>> indices;
        if (auto rankedTy =
                op.getOperand(i).getType().dyn_cast<RankedTensorType>()) {
          indices =
              emitIndices(loc, rewriter, rankedTy.getEncoding(), rankedTy);
          for (int64_t dim : rankedTy.getShape()) {
            if (dim > 0) {
              dimWidths.push_back(static_cast<int>(std::ceil(std::log10(dim))));
            } else {
              dimWidths.push_back(0);
            }
          }
        } else {
          // We're printing a scalar.
          assert(elems.size() == 1);
          indices.push_back({});
        }

        if (!elems.empty()) {
          printTensor(prefixStr, /*operand=*/i,
                      /*numOperands=*/op.getNumOperands(), elems, pid, indices,
                      dimWidths, rewriter);
        }
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  void printTensor(Value prefixStr, size_t operand, size_t numOperands,
                   ArrayRef<Value> elems, std::array<Value, 3> pid,
                   ArrayRef<SmallVector<Value>> indices,
                   ArrayRef<int> dimWidths,
                   ConversionPatternRewriter &rewriter) const {
    assert(!elems.empty());
    assert(elems.size() == indices.size());
    assert(dimWidths.size() == indices.front().size());

    size_t rank = dimWidths.size();

    // Format is:
    //   pid (<x>, <y>, <z>) idx (<i1>, <i2>, ...)<prefix> (operand <n>) <elem>
    // where we leave off "(operand <n>)" if there's only one operand.
    //
    // The Python wrapper munges `prefix` so that it prints nicely (e.g. starts
    // with " " and ends with ": ").

    Value formatStrValue;
    for (int i = 0; i < elems.size(); i++) {
      std::string formatStr;
      llvm::raw_string_ostream os(formatStr);

      // nvptx printf can only accept 32 args; if we pass more than that, it
      // will print garbage for the trailing args.
      constexpr int kMaxPrintfOperands = 32;
      SmallVector<Value, kMaxPrintfOperands> printfOperands;

      // TODO(jlebar): We really should pad the pid, but because the max pid is
      // not known at compile-time, this would require nontrivial device-side
      // work.
      os << "pid (";
      for (int j = 0; j < pid.size(); j++) {
        if (j != 0) {
          os << ", ";
        }
        os << getFormatSubstr(pid[j]);
        printfOperands.push_back(pid[j]);
      }
      os << ") ";

      // If `rank` is large enough, we could end up exceeding
      // kMaxPrintfOperands.  In that case, just truncate the index.
      // (Subtract 2 because we're going to add two operands after the index.)
      int maxAllowedRank = kMaxPrintfOperands - printfOperands.size() - 2;

      os << "idx (";
      const auto &index = indices[i];
      for (size_t dim = 0; dim < index.size(); dim++) {
        if (dim != 0) {
          os << ", ";
        }
        if (dim == maxAllowedRank) {
          os << "... (truncated)";
          break;
        }
        os << getFormatSubstr(index[dim], /*width=*/dimWidths[dim]);
        printfOperands.push_back(index[dim]);
      }
      os << ")";

      os << "%s";
      printfOperands.push_back(prefixStr);

      if (numOperands > 1) {
        os << "(operand " << operand << ") ";
      }

      auto elem = elems[i];
      os << getFormatSubstr(elem);
      printfOperands.push_back(elem);

      // It's the same format string each iteration, but it's a lot easier if we
      // construct the format string at the same time as we populate
      // printfOperands.  But we don't want to create BLOCK_SIZE duplicate
      // strings, so we cache the Value.
      if (i == 0) {
        formatStrValue = llPrintf(formatStr, printfOperands, rewriter);
      } else {
        llPrintf(formatStrValue, printfOperands, rewriter);
      }
    }
  }

  std::string getFormatSubstr(Value value,
                              std::optional<int> width = std::nullopt) const {
    std::string prefix = "%";
    if (width.has_value()) {
      prefix += std::to_string(*width);
    }

    Type type = value.getType();
    if (type.isa<LLVM::LLVMPointerType>()) {
      return prefix + "p";
    } else if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
      return prefix + "f";
    } else if (type.isSignedInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return prefix + "lli";
      else
        return prefix + "i";
    } else if (type.isUnsignedInteger() || type.isSignlessInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return prefix + "llu";
      else
        return prefix + "u";
    }
    assert(false && "not supported type");
    return "";
  }

  // declare vprintf(i8*, i8*) as external function
  static LLVM::LLVMFuncOp
  getVprintfDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("vprintf");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    auto *context = rewriter.getContext();

    SmallVector<Type> argsType{ptr_ty(context), ptr_ty(context)};
    auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                             funcType);
  }

  // extend integer to int32, extend float to float64
  // this comes from vprintf alignment requirements.
  static std::pair<Type, Value>
  promoteValue(ConversionPatternRewriter &rewriter, Value value) {
    auto *context = rewriter.getContext();
    auto type = value.getType();
    Value newOp = value;
    Type newType = type;
    auto loc = UnknownLoc::get(context);

    bool bUnsigned = type.isUnsignedInteger();
    if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
      if (bUnsigned) {
        newType = ui32_ty;
        newOp = zext(newType, value);
      } else {
        newType = i32_ty;
        newOp = sext(newType, value);
      }
    } else if (type.isBF16() || type.isF16() || type.isF32()) {
      newType = f64_ty;
      newOp = fpext(newType, value);
    }

    return {newType, newOp};
  }

  // Returns a Value for the format string, which you can reuse.
  static Value llPrintf(StringRef msg, ValueRange args,
                        ConversionPatternRewriter &rewriter) {
    assert(!msg.empty() && "printf with empty string not supported");
    llvm::SmallString<64> msgNewline(msg);
    msgNewline.push_back('\n');
    msgNewline.push_back('\0');
    Value msgValue =
        LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()),
                                rewriter, "printfFormat_", msgNewline);
    llPrintf(msgValue, args, rewriter);
    return msgValue;
  }

  static void llPrintf(Value msg, ValueRange args,
                       ConversionPatternRewriter &rewriter) {
    auto *ctx = rewriter.getContext();
    Type ptr = ptr_ty(ctx);
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto funcOp = getVprintfDeclaration(rewriter);
    auto loc = UnknownLoc::get(ctx);

    Value one = i32_val(1);
    Value zero = i32_val(0);

    Value bufferPtr = null(ptr);

    SmallVector<Value, 16> newArgs;
    if (args.size() >= 1) {
      SmallVector<Type> argTypes;
      for (auto arg : args) {
        Type newType;
        Value newArg;
        std::tie(newType, newArg) = promoteValue(rewriter, arg);
        argTypes.push_back(newType);
        newArgs.push_back(newArg);
      }

      Type structTy = LLVM::LLVMStructType::getLiteral(ctx, argTypes);
      auto allocated =
          rewriter.create<LLVM::AllocaOp>(loc, ptr_ty(ctx), structTy, one,
                                          /*alignment=*/0);

      for (const auto &entry : llvm::enumerate(newArgs)) {
        auto index = i32_val(entry.index());
        auto fieldPtr =
            gep(ptr_ty(ctx), structTy, allocated, ArrayRef<Value>{zero, index});
        store(entry.value(), fieldPtr);
      }
      bufferPtr = bitcast(allocated, ptr);
    }

    SmallVector<Value> operands{msg, bufferPtr};
    call(funcOp, operands);
  }
};

struct AssertOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AssertOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AssertOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto elems = getTypeConverter()->unpackLLElements(
        loc, adaptor.getCondition(), rewriter);
    auto elemTy = elems[0].getType();
    Value condition = int_val(elemTy.getIntOrFloatBitWidth(), 0);
    for (auto elem : elems) {
      if (elemTy.isSignedInteger() || elemTy.isSignlessInteger()) {
        condition =
            or_(condition,
                icmp_eq(elem, rewriter.create<LLVM::ConstantOp>(
                                  loc, elemTy, rewriter.getZeroAttr(elemTy))));
      } else {
        assert(false && "Unsupported type for assert");
        return failure();
      }
    }
    llAssert(op, condition, adaptor.getMessage(), adaptor.getFile(),
             adaptor.getFunc(), adaptor.getLine(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }

  // op: the op at which the assert is inserted. Unlike printf, we need to
  // know about the op to split the block.
  static void llAssert(Operation *op, Value condition, StringRef message,
                       StringRef file, StringRef func, int line,
                       ConversionPatternRewriter &rewriter) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    auto ctx = rewriter.getContext();
    auto loc = op->getLoc();

    // #block1
    // if (condition) {
    //   #block2
    //   __assertfail(message);
    // }
    // #block3
    Block *prevBlock = op->getBlock();
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    auto funcOp = getAssertfailDeclaration(rewriter);
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Value messageString =
        LLVM::addStringToModule(loc, rewriter, "assertMessage_", message);
    Value fileString =
        LLVM::addStringToModule(loc, rewriter, "assertFile_", file);
    Value funcString =
        LLVM::addStringToModule(loc, rewriter, "assertFunc_", func);
    Value lineNumber = i32_val(line);
    Value charSize = int_val(sizeof(size_t) * 8, sizeof(char));

    SmallVector<Value> operands = {messageString, fileString, lineNumber,
                                   funcString, charSize};
    auto ret = call(funcOp, operands);

    // Split a block after the call.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, condition, ifBlock, thenBlock);
  }

  static LLVM::LLVMFuncOp
  getAssertfailDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("__assertfail");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    // void __assert_fail(const char * assertion, const char * file, unsigned
    // int line, const char * function);
    auto *ctx = rewriter.getContext();
    SmallVector<Type> argsType{ptr_ty(ctx), ptr_ty(ctx), i32_ty, ptr_ty(ctx),
                               rewriter.getIntegerType(sizeof(size_t) * 8)};
    auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                             funcType);
  }
};

struct MakeRangeOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::MakeRangeOp> {

  MakeRangeOpConversion(
      TritonGPUToLLVMTypeConverter &converter,
      ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::MakeRangeOp>(
            converter, indexCacheInfo, benefit) {}

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    RankedTensorType ty = op.getType();
    auto shape = ty.getShape();
    auto layout = ty.getEncoding();

    auto elemTy = ty.getElementType();
    assert(elemTy.isInteger(32));
    Value start = createIndexAttrConstant(rewriter, loc, elemTy, op.getStart());
    auto idxs = emitIndices(loc, rewriter, layout, ty);
    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = add(multiDim.value()[0], start);
    }
    Value result =
        getTypeConverter()->packLLElements(loc, retVals, rewriter, ty);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GetProgramIdOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::GetProgramIdOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

#ifdef USE_ROCM
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);

    Value blockId =
        rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[op.getAxisAsInt()]);
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, i32_ty, blockId);
    return success();
#else
    Value programId = llGetPid(op.getAxisAsInt(), op->getLoc(),
                               op->getParentOfType<ModuleOp>(), rewriter);
    rewriter.replaceOp(op, programId);
    return success();
#endif
  }
};

struct GetNumProgramsOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
#ifdef USE_ROCM
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
    Location loc = op->getLoc();
    assert(op.getAxis() < 3);
    Value blockId =
        rewriter.create<::mlir::gpu::GridDimOp>(loc, dims[op.getAxis()]);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);
    return success();
#else
    // It is not easy to get the compute capability here, so we use numCTAs to
    // decide the semantic of GetNumProgramsOp. If numCTAs = 1, then
    // GetNumProgramsOp is converted to "%nctaid", otherwise it is converted to
    // "%nclusterid".
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for GetProgramIdOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Location loc = op->getLoc();
    assert(op.getAxis() < 3);
    std::string sreg = numCTAs == 1 ? "%nctaid." : "%nclusterid.";
    sreg.append(1, 'x' + op.getAxis()); // 0 -> 'x', 1 -> 'y', 2 -> 'z'

    Value numPrograms = getSRegValue(rewriter, loc, sreg);
    rewriter.replaceOp(op, numPrograms);
    return success();
#endif
  }
};

struct AddPtrOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AddPtrOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AddPtrOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto offsetTy = op.getOffset().getType();
    auto resultTensorTy = resultTy.dyn_cast<RankedTensorType>();
    if (resultTensorTy) {
      unsigned elems = getTotalElemsPerThread(resultTy);
      Type elemTy =
          getTypeConverter()->convertType(resultTensorTy.getElementType()
                                              .cast<triton::PointerType>()
                                              .getPointeeType());
      Type ptrTy =
          getTypeConverter()->convertType(resultTensorTy.getElementType());
      auto ptrs = getTypeConverter()->unpackLLElements(loc, adaptor.getPtr(), rewriter);
      auto offsets = getTypeConverter()->unpackLLElements(
          loc, adaptor.getOffset(), rewriter);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(ptrTy, elemTy, ptrs[i], offsets[i]);
      }
      Value view = getTypeConverter()->packLLElements(loc, resultVals, rewriter,
                                                      resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(resultTy.isa<triton::PointerType>());
      auto resultPtrTy = getTypeConverter()->convertType(resultTy);
      auto resultElemTy = getTypeConverter()->convertType(
          resultTy.cast<triton::PointerType>().getPointeeType());
      Value result =
          gep(resultPtrTy, resultElemTy, adaptor.getPtr(), adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

struct AllocTensorOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AllocTensorOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AllocTensorOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AllocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getResult());
    auto resultTy = op.getType().dyn_cast<RankedTensorType>();
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto sharedLayout = resultTy.getEncoding().cast<SharedEncodingAttr>();
    auto order = sharedLayout.getOrder();
    // Workaround for 3D tensors
    // TODO: we need to modify the pipeline pass to give a proper shared
    // encoding to 3D tensors
    SmallVector<unsigned> newOrder;
    if (resultTy.getShape().size() != order.size()) {
      for (auto i = 0; i < order.size(); ++i)
        newOrder.push_back(order[i] + 1);
      newOrder.push_back(0);
    } else {
      newOrder = SmallVector<unsigned>(order.begin(), order.end());
    }

    auto llvmElemTy =
        getTypeConverter()->convertType(resultTy.getElementType());
    auto shapePerCTA = getShapePerCTA(sharedLayout, resultTy.getShape());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, shapePerCTA,
                                      newOrder, loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct DeallocTensorOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::DeallocTensorOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::DeallocTensorOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::DeallocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExtractSliceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::ExtractSliceOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::ExtractSliceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = extract_slice %src[%offsets]
    Location loc = op->getLoc();
    auto srcTy = op.getSrc().getType().dyn_cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    assert(srcLayout && "Unexpected resultLayout in ExtractSliceOpConversion");
    assert(op.hasUnitStride() &&
           "Only unit stride supported by ExtractSliceOpConversion");

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());

    // newBase = base + offset
    // Triton supports either static and dynamic offsets
    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    SmallVector<Value, 4> opOffsetVals;
    SmallVector<Value, 4> offsetVals;
    auto mixedOffsets = op.getMixedOffsets();
    for (auto i = 0, j = 0; i < mixedOffsets.size(); ++i) {
      if (op.isDynamicOffset(i)) {
        // adaptor.getOffsets() returns list of variable offsets. the size of
        // the list may not be the same as mixedOffsets
        opOffsetVals.emplace_back(adaptor.getOffsets()[j]);
        ++j;
      } else
        opOffsetVals.emplace_back(i32_val(op.getStaticOffset(i)));
      offsetVals.emplace_back(add(smemObj.offsets[i], opOffsetVals[i]));
    }
    // Compute the offset based on the original strides of the shared memory
    // object
    auto offset = dot(rewriter, loc, opOffsetVals, smemObj.strides);
    // newShape = rank_reduce(shape)
    // Triton only supports static tensor sizes
    SmallVector<Value, 4> strideVals;
    for (auto i = 0; i < op.getStaticSizes().size(); ++i) {
      if (op.getStaticSize(i) == 1) {
        offsetVals.erase(offsetVals.begin() + i);
      } else {
        strideVals.emplace_back(smemObj.strides[i]);
      }
    }

    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    smemObj =
        SharedMemoryObject(gep(elemPtrTy, llvmElemTy, smemObj.base, offset),
                           llvmElemTy, strideVals, offsetVals);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PTXBuilder ptxBuilder;
    auto &asyncWaitOp = *ptxBuilder.create<>("cp.async.wait_group");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);

    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncCommitGroupOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    PTXBuilder ptxBuilder;
    ptxBuilder.create<>("cp.async.commit_group")->operator()();
    ptxBuilder.launch(rewriter, op.getLoc(), void_ty(op.getContext()));
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncBulkWaitOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AsyncBulkWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncBulkWaitOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncBulkWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PTXBuilder ptxBuilder;
    auto &asyncBulkWaitOp = *ptxBuilder.create<>("cp.async.bulk.wait_group");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncBulkWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);

    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncBulkCommitGroupOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::gpu::AsyncBulkCommitGroupOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncBulkCommitGroupOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncBulkCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    PTXBuilder ptxBuilder;
    ptxBuilder.create<>("cp.async.bulk.commit_group")->operator()();
    ptxBuilder.launch(rewriter, op.getLoc(), void_ty(op.getContext()));
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace


namespace AMD{
void populateTritonGPUToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &moduleAllocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<AllocTensorOpConversion>(typeConverter, moduleAllocation,
                                        benefit);
  patterns.add<DeallocTensorOpConversion>(typeConverter, benefit);
  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, benefit);
  patterns.add<AsyncBulkCommitGroupOpConversion>(typeConverter, benefit);
  patterns.add<AsyncBulkWaitOpConversion>(typeConverter, benefit);
  patterns.add<BroadcastOpConversion>(typeConverter, benefit);
  patterns.add<ExtractSliceOpConversion>(typeConverter, moduleAllocation,
                                         benefit);
  patterns.add<GetProgramIdOpConversion>(typeConverter, benefit);
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<MakeRangeOpConversion>(typeConverter, indexCacheInfo, benefit);
  patterns.add<ReturnOpConversion>(typeConverter, benefit);
  patterns.add<PrintOpConversion>(typeConverter, benefit);
  patterns.add<AssertOpConversion>(typeConverter, benefit);
}

} // namespace mlir::triton
