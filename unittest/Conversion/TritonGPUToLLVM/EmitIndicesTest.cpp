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

#include "gtest/gtest.h"
#include <fstream>
#include <gtest/gtest.h>

#include "DumpLayout.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace triton {
namespace gpu {

//===----------------------------------------------------------------------===//
// EmitIndicesTest
//===----------------------------------------------------------------------===//

MLIRContext *getContext() {
  static MLIRContext *context = [] {
    MLIRContext *context = new MLIRContext();
    context->getOrLoadDialect<TritonGPUDialect>();
    context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context->getOrLoadDialect<mlir::gpu::GPUDialect>();
    context->getOrLoadDialect<mlir::triton::nvgpu::NVGPUDialect>();
    return context;
  }();
  return context;
}

class EmitIndicesTest : public ::testing::Test {
protected:
  EmitIndicesTest() : context(*getContext()) {}

  void runBlocked1dSingleCTA(int size, unsigned sizePerThread,
                             unsigned warpsPerCTA, const std::string &refStr) {
    // If we pass initializer lists to the constructor of BlockedEncodingAttr,
    // there might be multiple constructors matching the same parameter list.
    // For example, the initializer list "order = {0}" can also match the
    // parameter "unsigned numWarps", which is not what we want
    llvm::SmallVector<unsigned> sizePerThread_ = {sizePerThread};
    llvm::SmallVector<unsigned> threadsPerWarp = {32};
    llvm::SmallVector<unsigned> warpsPerCTA_ = {warpsPerCTA};
    llvm::SmallVector<unsigned> order = {0};
    auto layout =
        BlockedEncodingAttr::get(&context, sizePerThread_, threadsPerWarp,
                                 warpsPerCTA_, order, getSingleCTALayout1d());
    runDistributed1d(size, layout, /*multiCTA=*/false, refStr);
  }

  void runBlocked2dSingleCTA(int row, int col,
                             llvm::ArrayRef<unsigned> sizePerThread,
                             llvm::ArrayRef<unsigned> threadsPerWarp,
                             llvm::ArrayRef<unsigned> warpsPerCTA,
                             llvm::ArrayRef<unsigned> order,
                             const std::string &refStr) {
    auto layout =
        BlockedEncodingAttr::get(&context, sizePerThread, threadsPerWarp,
                                 warpsPerCTA, order, getSingleCTALayout2d());
    runDistributed2d(row, col, layout, /*multiCTA=*/false, refStr);
  }

  void runBlocked2dMultiCTA(
      int row, int col, llvm::ArrayRef<unsigned> sizePerThread,
      llvm::ArrayRef<unsigned> threadsPerWarp,
      llvm::ArrayRef<unsigned> warpsPerCTA, llvm::ArrayRef<unsigned> order,
      llvm::ArrayRef<unsigned> CTAsPerCGA, llvm::ArrayRef<unsigned> CTASplitNum,
      llvm::ArrayRef<unsigned> CTAOrder, const std::string &refStr) {
    auto CTALayout =
        CTALayoutAttr::get(&context, CTAsPerCGA, CTASplitNum, CTAOrder);
    auto layout = BlockedEncodingAttr::get(
        &context, sizePerThread, threadsPerWarp, warpsPerCTA, order, CTALayout);
    runDistributed2d(row, col, layout, /*multiCTA=*/true, refStr);
  }

  void runSliceBlockedSingleCTA(int size,
                                llvm::ArrayRef<unsigned> sizePerThread,
                                llvm::ArrayRef<unsigned> threadsPerWarp,
                                llvm::ArrayRef<unsigned> warpsPerCTA,
                                llvm::ArrayRef<unsigned> order,
                                unsigned sliceDim, const std::string &refStr) {
    auto parent =
        BlockedEncodingAttr::get(&context, sizePerThread, threadsPerWarp,
                                 warpsPerCTA, order, getSingleCTALayout2d());
    auto layout = SliceEncodingAttr::get(&context, sliceDim, parent);
    runDistributed1d(size, layout, /*multiCTA=*/false, refStr);
  }

  void runSliceBlockedMultiCTA(int size, llvm::ArrayRef<unsigned> sizePerThread,
                               llvm::ArrayRef<unsigned> threadsPerWarp,
                               llvm::ArrayRef<unsigned> warpsPerCTA,
                               llvm::ArrayRef<unsigned> order,
                               llvm::ArrayRef<unsigned> CTAsPerCGA,
                               llvm::ArrayRef<unsigned> CTASplitNum,
                               llvm::ArrayRef<unsigned> CTAOrder,
                               unsigned sliceDim, const std::string &refStr) {
    auto CTALayout =
        CTALayoutAttr::get(&context, CTAsPerCGA, CTASplitNum, CTAOrder);
    auto parent = BlockedEncodingAttr::get(
        &context, sizePerThread, threadsPerWarp, warpsPerCTA, order, CTALayout);
    auto layout = SliceEncodingAttr::get(&context, sliceDim, parent);
    runDistributed1d(size, layout, /*multiCTA=*/true, refStr);
  }

  void runMmaSingleCTA(int row, int col, unsigned versionMajor,
                       unsigned versionMinor,
                       llvm::ArrayRef<unsigned> warpsPerCTA,
                       llvm::ArrayRef<unsigned> instrShape,
                       const std::string &refStr) {
    auto layout = NvidiaMmaEncodingAttr::get(
        &context, versionMajor, versionMinor, warpsPerCTA,
        getSingleCTALayout2d(), instrShape);
    runDistributed2d(row, col, layout, /*multiCTA=*/false, refStr);
  }

  void runWmmaSingleCTA(int row, int col, llvm::ArrayRef<unsigned> warpsPerCTA,
                        const std::string &refStr) {
    auto layout =
        AMDWmmaEncodingAttr::get(&context, warpsPerCTA, getSingleCTALayout2d());
    runDistributed2d(row, col, layout, /*multiCTA=*/false, refStr);
  }

  void runDotOpSingleCTA(int row, int col, unsigned versionMajor,
                         unsigned versionMinor,
                         llvm::ArrayRef<unsigned> warpsPerCTA,
                         llvm::ArrayRef<unsigned> instrShape, unsigned opIdx,
                         const std::string &refStr) {
    auto parent = NvidiaMmaEncodingAttr::get(
        &context, versionMajor, versionMinor, warpsPerCTA,
        getSingleCTALayout2d(), instrShape);
    auto layout = DotOperandEncodingAttr::get(&context, opIdx, parent, 0);
    runDistributed2d(row, col, layout, /*multiCTA=*/false, refStr);
  }

  void runSharedSingleCTA(int row, int col, bool rowMajor,
                          const std::string &elemTyStr,
                          const std::string &refStr) {
    auto elemTy = getElemTy(elemTyStr);
    auto layout =
        SharedEncodingAttr::get(&context, {row, col}, getMatrixOrder(rowMajor),
                                getSingleCTALayout2d(), elemTy);
    llvm::outs() << layout << "\n";
    runShared(row, col, layout, elemTy, /*multiCTA=*/false, refStr);
  }

private:
  std::string skipSpaces(const std::string &input) {
    std::string output;
    for (char c : input)
      if (c != ' ')
        output += c;
    return output;
  }

  void assertSameStr(const std::string &refStr, const std::string &output) {
    if (refStr != output) {
      llvm::outs() << "RefStr =\n"
                   << refStr << "\n"
                   << "\n"
                   << "Output =\n"
                   << output << "\n";
      FAIL() << "Incorrect output string";
    }
  }

  void runDistributed1d(int size, Attribute layout, bool multiCTA,
                        const std::string &refStr) {
    assertSameStr(skipSpaces(refStr),
                  dumpDistributedLayout(layout, {size}, multiCTA));
  }

  void runDistributed2d(int row, int col, Attribute layout, bool multiCTA,
                        const std::string &refStr) {
    assertSameStr(skipSpaces(refStr),
                  dumpDistributedLayout(layout, {row, col}, multiCTA));
  }

  void runShared(int row, int col, const SharedEncodingAttr &layout,
                 Type elemTy, bool multiCTA, const std::string &refStr) {
    assertSameStr(skipSpaces(refStr),
                  dumpSharedLayout(layout, {row, col}, elemTy, multiCTA));
  }

  CTALayoutAttr getSingleCTALayout1d() {
    return CTALayoutAttr::get(/*context=*/&context, /*CTAsPerCGA=*/{1},
                              /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  }

  CTALayoutAttr getSingleCTALayout2d() {
    return CTALayoutAttr::get(/*context=*/&context, /*CTAsPerCGA=*/{1, 1},
                              /*CTASplitNum=*/{1, 1}, /*CTAOrder=*/{1, 0});
  }

  llvm::SmallVector<unsigned> getMatrixOrder(bool rowMajor) {
    if (rowMajor)
      return {1, 0};
    else
      return {0, 1};
  }

  Type getElemTy(const std::string &elemTyStr) {
    if (elemTyStr == "F16")
      return FloatType::getF16(&context);
    else
      llvm::report_fatal_error("getElemTy not implemented");
    return nullptr;
  }

protected:
  MLIRContext &context;
};

//===----------------------------------------------------------------------===//
// Tests for BlockedEncodingAttr
//===----------------------------------------------------------------------===//

TEST_F(EmitIndicesTest, BlockedLayout_SingleCTA_1D) {
  // clang-format off
  std::string refStr =
      "T0:0,T1:0,T2:0,T3:0,T4:0,T5:0,T6:0,T7:0,T8:0,T9:0,T10:0,T11:0,T12:0,T13:0,T14:0,T15:0,T16:0,T17:0,T18:0,T19:0,T20:0,T21:0,T22:0,T23:0,T24:0,T25:0,T26:0,T27:0,T28:0,T29:0,T30:0,T31:0\n";
  // clang-format on

  runBlocked1dSingleCTA(/*size=*/32, /*sizePerThread*/ 1, /*warpsPerCTA*/ 1,
                        /*refStr=*/refStr);
}

TEST_F(EmitIndicesTest, BlockedLayout_SingleCTA_Order_1_0) {
  // clang-format off
  std::string refStr =
      " T0:0, T1:0, T2:0, T3:0,    T32:0, T33:0, T34:0, T35:0\n"
      " T4:0, T5:0, T6:0, T7:0,    T36:0, T37:0, T38:0, T39:0\n"
      " T8:0, T9:0,T10:0,T11:0,    T40:0, T41:0, T42:0, T43:0\n"
      "T12:0,T13:0,T14:0,T15:0,    T44:0, T45:0, T46:0, T47:0\n"
      "T16:0,T17:0,T18:0,T19:0,    T48:0, T49:0, T50:0, T51:0\n"
      "T20:0,T21:0,T22:0,T23:0,    T52:0, T53:0, T54:0, T55:0\n"
      "T24:0,T25:0,T26:0,T27:0,    T56:0, T57:0, T58:0, T59:0\n"
      "T28:0,T29:0,T30:0,T31:0,    T60:0, T61:0, T62:0, T63:0\n"

      "T64:0,T65:0,T66:0,T67:0,    T96:0, T97:0, T98:0, T99:0\n"
      "T68:0,T69:0,T70:0,T71:0,   T100:0,T101:0,T102:0,T103:0\n"
      "T72:0,T73:0,T74:0,T75:0,   T104:0,T105:0,T106:0,T107:0\n"
      "T76:0,T77:0,T78:0,T79:0,   T108:0,T109:0,T110:0,T111:0\n"
      "T80:0,T81:0,T82:0,T83:0,   T112:0,T113:0,T114:0,T115:0\n"
      "T84:0,T85:0,T86:0,T87:0,   T116:0,T117:0,T118:0,T119:0\n"
      "T88:0,T89:0,T90:0,T91:0,   T120:0,T121:0,T122:0,T123:0\n"
      "T92:0,T93:0,T94:0,T95:0,   T124:0,T125:0,T126:0,T127:0\n";
  // clang-format on

  runBlocked2dSingleCTA(/*row=*/16, /*col=*/8, /*sizePerThread=*/{1, 1},
                        /*threadsPerWarp=*/{8, 4}, /*warpsPerCTA=*/{2, 2},
                        /*order=*/{1, 0}, /*refStr=*/refStr);
}

TEST_F(EmitIndicesTest, BlockedLayout_SingleCTA_Order_0_1) {
  // clang-format off
  std::string refStr =
      " T0:0, T8:0,T16:0,T24:0,    T64:0, T72:0, T80:0, T88:0\n"
      " T1:0, T9:0,T17:0,T25:0,    T65:0, T73:0, T81:0, T89:0\n"
      " T2:0,T10:0,T18:0,T26:0,    T66:0, T74:0, T82:0, T90:0\n"
      " T3:0,T11:0,T19:0,T27:0,    T67:0, T75:0, T83:0, T91:0\n"
      " T4:0,T12:0,T20:0,T28:0,    T68:0, T76:0, T84:0, T92:0\n"
      " T5:0,T13:0,T21:0,T29:0,    T69:0, T77:0, T85:0, T93:0\n"
      " T6:0,T14:0,T22:0,T30:0,    T70:0, T78:0, T86:0, T94:0\n"
      " T7:0,T15:0,T23:0,T31:0,    T71:0, T79:0, T87:0, T95:0\n"

      "T32:0,T40:0,T48:0,T56:0,    T96:0,T104:0,T112:0,T120:0\n"
      "T33:0,T41:0,T49:0,T57:0,    T97:0,T105:0,T113:0,T121:0\n"
      "T34:0,T42:0,T50:0,T58:0,    T98:0,T106:0,T114:0,T122:0\n"
      "T35:0,T43:0,T51:0,T59:0,    T99:0,T107:0,T115:0,T123:0\n"
      "T36:0,T44:0,T52:0,T60:0,   T100:0,T108:0,T116:0,T124:0\n"
      "T37:0,T45:0,T53:0,T61:0,   T101:0,T109:0,T117:0,T125:0\n"
      "T38:0,T46:0,T54:0,T62:0,   T102:0,T110:0,T118:0,T126:0\n"
      "T39:0,T47:0,T55:0,T63:0,   T103:0,T111:0,T119:0,T127:0\n";
  // clang-format on

  runBlocked2dSingleCTA(/*row=*/16, /*col=*/8, /*sizePerThread=*/{1, 1},
                        /*threadsPerWarp=*/{8, 4}, /*warpsPerCTA=*/{2, 2},
                        /*order=*/{0, 1}, /*refStr=*/refStr);
}

TEST_F(EmitIndicesTest, BlockedLayout_SingleCTA_Vectorize) {
  // clang-format off
  std::string refStr =
      " T0:0, T0:1, T0:2, T0:3,    T1:0, T1:1, T1:2, T1:3,    T2:0, T2:1, T2:2, T2:3,    T3:0, T3:1, T3:2, T3:3\n"
      " T4:0, T4:1, T4:2, T4:3,    T5:0, T5:1, T5:2, T5:3,    T6:0, T6:1, T6:2, T6:3,    T7:0, T7:1, T7:2, T7:3\n"
      " T8:0, T8:1, T8:2, T8:3,    T9:0, T9:1, T9:2, T9:3,   T10:0,T10:1,T10:2,T10:3,   T11:0,T11:1,T11:2,T11:3\n"
      "T12:0,T12:1,T12:2,T12:3,   T13:0,T13:1,T13:2,T13:3,   T14:0,T14:1,T14:2,T14:3,   T15:0,T15:1,T15:2,T15:3\n"
      "T16:0,T16:1,T16:2,T16:3,   T17:0,T17:1,T17:2,T17:3,   T18:0,T18:1,T18:2,T18:3,   T19:0,T19:1,T19:2,T19:3\n"
      "T20:0,T20:1,T20:2,T20:3,   T21:0,T21:1,T21:2,T21:3,   T22:0,T22:1,T22:2,T22:3,   T23:0,T23:1,T23:2,T23:3\n"
      "T24:0,T24:1,T24:2,T24:3,   T25:0,T25:1,T25:2,T25:3,   T26:0,T26:1,T26:2,T26:3,   T27:0,T27:1,T27:2,T27:3\n"
      "T28:0,T28:1,T28:2,T28:3,   T29:0,T29:1,T29:2,T29:3,   T30:0,T30:1,T30:2,T30:3,   T31:0,T31:1,T31:2,T31:3\n"

      "T32:0,T32:1,T32:2,T32:3,   T33:0,T33:1,T33:2,T33:3,   T34:0,T34:1,T34:2,T34:3,   T35:0,T35:1,T35:2,T35:3\n"
      "T36:0,T36:1,T36:2,T36:3,   T37:0,T37:1,T37:2,T37:3,   T38:0,T38:1,T38:2,T38:3,   T39:0,T39:1,T39:2,T39:3\n"
      "T40:0,T40:1,T40:2,T40:3,   T41:0,T41:1,T41:2,T41:3,   T42:0,T42:1,T42:2,T42:3,   T43:0,T43:1,T43:2,T43:3\n"
      "T44:0,T44:1,T44:2,T44:3,   T45:0,T45:1,T45:2,T45:3,   T46:0,T46:1,T46:2,T46:3,   T47:0,T47:1,T47:2,T47:3\n"
      "T48:0,T48:1,T48:2,T48:3,   T49:0,T49:1,T49:2,T49:3,   T50:0,T50:1,T50:2,T50:3,   T51:0,T51:1,T51:2,T51:3\n"
      "T52:0,T52:1,T52:2,T52:3,   T53:0,T53:1,T53:2,T53:3,   T54:0,T54:1,T54:2,T54:3,   T55:0,T55:1,T55:2,T55:3\n"
      "T56:0,T56:1,T56:2,T56:3,   T57:0,T57:1,T57:2,T57:3,   T58:0,T58:1,T58:2,T58:3,   T59:0,T59:1,T59:2,T59:3\n"
      "T60:0,T60:1,T60:2,T60:3,   T61:0,T61:1,T61:2,T61:3,   T62:0,T62:1,T62:2,T62:3,   T63:0,T63:1,T63:2,T63:3\n";
  // clang-format on

  runBlocked2dSingleCTA(/*row=*/16, /*col=*/16, /*sizePerThread=*/{1, 4},
                        /*threadsPerWarp=*/{8, 4}, /*warpsPerCTA=*/{2, 1},
                        /*order=*/{1, 0}, /*refStr=*/refStr);
}

// FIXME: These tests are temporarily disabled due to ctaid.x|y|z are swapped
#ifdef TEST_FAILED
TEST_F(EmitIndicesTest, BlockedLayout_MultiCTA_CTAOrder_1_0) {
  // clang-format off
  std::string refStr =
      "CTA0: T0:0,CTA0: T1:0,CTA0: T2:0,CTA0: T3:0,   CTA1: T0:0,CTA1: T1:0,CTA1: T2:0,CTA1: T3:0\n"
      "CTA0: T4:0,CTA0: T5:0,CTA0: T6:0,CTA0: T7:0,   CTA1: T4:0,CTA1: T5:0,CTA1: T6:0,CTA1: T7:0\n"
      "CTA0: T8:0,CTA0: T9:0,CTA0:T10:0,CTA0:T11:0,   CTA1: T8:0,CTA1: T9:0,CTA1:T10:0,CTA1:T11:0\n"
      "CTA0:T12:0,CTA0:T13:0,CTA0:T14:0,CTA0:T15:0,   CTA1:T12:0,CTA1:T13:0,CTA1:T14:0,CTA1:T15:0\n"
      "CTA0:T16:0,CTA0:T17:0,CTA0:T18:0,CTA0:T19:0,   CTA1:T16:0,CTA1:T17:0,CTA1:T18:0,CTA1:T19:0\n"
      "CTA0:T20:0,CTA0:T21:0,CTA0:T22:0,CTA0:T23:0,   CTA1:T20:0,CTA1:T21:0,CTA1:T22:0,CTA1:T23:0\n"
      "CTA0:T24:0,CTA0:T25:0,CTA0:T26:0,CTA0:T27:0,   CTA1:T24:0,CTA1:T25:0,CTA1:T26:0,CTA1:T27:0\n"
      "CTA0:T28:0,CTA0:T29:0,CTA0:T30:0,CTA0:T31:0,   CTA1:T28:0,CTA1:T29:0,CTA1:T30:0,CTA1:T31:0\n"

      "CTA2: T0:0,CTA2: T1:0,CTA2: T2:0,CTA2: T3:0,   CTA3: T0:0,CTA3: T1:0,CTA3: T2:0,CTA3: T3:0\n"
      "CTA2: T4:0,CTA2: T5:0,CTA2: T6:0,CTA2: T7:0,   CTA3: T4:0,CTA3: T5:0,CTA3: T6:0,CTA3: T7:0\n"
      "CTA2: T8:0,CTA2: T9:0,CTA2:T10:0,CTA2:T11:0,   CTA3: T8:0,CTA3: T9:0,CTA3:T10:0,CTA3:T11:0\n"
      "CTA2:T12:0,CTA2:T13:0,CTA2:T14:0,CTA2:T15:0,   CTA3:T12:0,CTA3:T13:0,CTA3:T14:0,CTA3:T15:0\n"
      "CTA2:T16:0,CTA2:T17:0,CTA2:T18:0,CTA2:T19:0,   CTA3:T16:0,CTA3:T17:0,CTA3:T18:0,CTA3:T19:0\n"
      "CTA2:T20:0,CTA2:T21:0,CTA2:T22:0,CTA2:T23:0,   CTA3:T20:0,CTA3:T21:0,CTA3:T22:0,CTA3:T23:0\n"
      "CTA2:T24:0,CTA2:T25:0,CTA2:T26:0,CTA2:T27:0,   CTA3:T24:0,CTA3:T25:0,CTA3:T26:0,CTA3:T27:0\n"
      "CTA2:T28:0,CTA2:T29:0,CTA2:T30:0,CTA2:T31:0,   CTA3:T28:0,CTA3:T29:0,CTA3:T30:0,CTA3:T31:0\n";
  // clang-format on

  runBlocked2dMultiCTA(/*row=*/16, /*col=*/8, /*sizePerThread=*/{1, 1},
                       /*threadsPerWarp=*/{8, 4}, /*warpsPerCTA=*/{1, 1},
                       /*order=*/{1, 0}, /*CTAsPerCGA=*/{2, 2},
                       /*CTASplitNum=*/{2, 2}, /*CTAOrder=*/{1, 0},
                       /*refStr=*/refStr);
}

TEST_F(EmitIndicesTest, BlockedLayout_MultiCTA_CTAOrder_0_1) {
  // clang-format off
  std::string refStr =
      "CTA0: T0:0,CTA0: T1:0,CTA0: T2:0,CTA0: T3:0,   CTA2: T0:0,CTA2: T1:0,CTA2: T2:0,CTA2: T3:0\n"
      "CTA0: T4:0,CTA0: T5:0,CTA0: T6:0,CTA0: T7:0,   CTA2: T4:0,CTA2: T5:0,CTA2: T6:0,CTA2: T7:0\n"
      "CTA0: T8:0,CTA0: T9:0,CTA0:T10:0,CTA0:T11:0,   CTA2: T8:0,CTA2: T9:0,CTA2:T10:0,CTA2:T11:0\n"
      "CTA0:T12:0,CTA0:T13:0,CTA0:T14:0,CTA0:T15:0,   CTA2:T12:0,CTA2:T13:0,CTA2:T14:0,CTA2:T15:0\n"
      "CTA0:T16:0,CTA0:T17:0,CTA0:T18:0,CTA0:T19:0,   CTA2:T16:0,CTA2:T17:0,CTA2:T18:0,CTA2:T19:0\n"
      "CTA0:T20:0,CTA0:T21:0,CTA0:T22:0,CTA0:T23:0,   CTA2:T20:0,CTA2:T21:0,CTA2:T22:0,CTA2:T23:0\n"
      "CTA0:T24:0,CTA0:T25:0,CTA0:T26:0,CTA0:T27:0,   CTA2:T24:0,CTA2:T25:0,CTA2:T26:0,CTA2:T27:0\n"
      "CTA0:T28:0,CTA0:T29:0,CTA0:T30:0,CTA0:T31:0,   CTA2:T28:0,CTA2:T29:0,CTA2:T30:0,CTA2:T31:0\n"

      "CTA1: T0:0,CTA1: T1:0,CTA1: T2:0,CTA1: T3:0,   CTA3: T0:0,CTA3: T1:0,CTA3: T2:0,CTA3: T3:0\n"
      "CTA1: T4:0,CTA1: T5:0,CTA1: T6:0,CTA1: T7:0,   CTA3: T4:0,CTA3: T5:0,CTA3: T6:0,CTA3: T7:0\n"
      "CTA1: T8:0,CTA1: T9:0,CTA1:T10:0,CTA1:T11:0,   CTA3: T8:0,CTA3: T9:0,CTA3:T10:0,CTA3:T11:0\n"
      "CTA1:T12:0,CTA1:T13:0,CTA1:T14:0,CTA1:T15:0,   CTA3:T12:0,CTA3:T13:0,CTA3:T14:0,CTA3:T15:0\n"
      "CTA1:T16:0,CTA1:T17:0,CTA1:T18:0,CTA1:T19:0,   CTA3:T16:0,CTA3:T17:0,CTA3:T18:0,CTA3:T19:0\n"
      "CTA1:T20:0,CTA1:T21:0,CTA1:T22:0,CTA1:T23:0,   CTA3:T20:0,CTA3:T21:0,CTA3:T22:0,CTA3:T23:0\n"
      "CTA1:T24:0,CTA1:T25:0,CTA1:T26:0,CTA1:T27:0,   CTA3:T24:0,CTA3:T25:0,CTA3:T26:0,CTA3:T27:0\n"
      "CTA1:T28:0,CTA1:T29:0,CTA1:T30:0,CTA1:T31:0,   CTA3:T28:0,CTA3:T29:0,CTA3:T30:0,CTA3:T31:0\n";
  // clang-format on

  runBlocked2dMultiCTA(/*row=*/16, /*col=*/8, /*sizePerThread=*/{1, 1},
                       /*threadsPerWarp=*/{8, 4}, /*warpsPerCTA=*/{1, 1},
                       /*order=*/{1, 0}, /*CTAsPerCGA=*/{2, 2},
                       /*CTASplitNum=*/{2, 2}, /*CTAOrder=*/{0, 1},
                       /*refStr=*/refStr);
}

TEST_F(EmitIndicesTest, BlockedLayout_MultiCTA_CTAWrap_Dim1) {
  // clang-format off
  std::string refStr =
      "CTA0: T0:0|CTA1: T0:0,  CTA0: T1:0|CTA1: T1:0,  CTA0: T2:0|CTA1: T2:0,  CTA0: T3:0|CTA1: T3:0\n"
      "CTA0: T4:0|CTA1: T4:0,  CTA0: T5:0|CTA1: T5:0,  CTA0: T6:0|CTA1: T6:0,  CTA0: T7:0|CTA1: T7:0\n"
      "CTA0: T8:0|CTA1: T8:0,  CTA0: T9:0|CTA1: T9:0,  CTA0:T10:0|CTA1:T10:0,  CTA0:T11:0|CTA1:T11:0\n"
      "CTA0:T12:0|CTA1:T12:0,  CTA0:T13:0|CTA1:T13:0,  CTA0:T14:0|CTA1:T14:0,  CTA0:T15:0|CTA1:T15:0\n"
      "CTA0:T16:0|CTA1:T16:0,  CTA0:T17:0|CTA1:T17:0,  CTA0:T18:0|CTA1:T18:0,  CTA0:T19:0|CTA1:T19:0\n"
      "CTA0:T20:0|CTA1:T20:0,  CTA0:T21:0|CTA1:T21:0,  CTA0:T22:0|CTA1:T22:0,  CTA0:T23:0|CTA1:T23:0\n"
      "CTA0:T24:0|CTA1:T24:0,  CTA0:T25:0|CTA1:T25:0,  CTA0:T26:0|CTA1:T26:0,  CTA0:T27:0|CTA1:T27:0\n"
      "CTA0:T28:0|CTA1:T28:0,  CTA0:T29:0|CTA1:T29:0,  CTA0:T30:0|CTA1:T30:0,  CTA0:T31:0|CTA1:T31:0\n"

      "CTA2: T0:0|CTA3: T0:0,  CTA2: T1:0|CTA3: T1:0,  CTA2: T2:0|CTA3: T2:0,  CTA2: T3:0|CTA3: T3:0\n"
      "CTA2: T4:0|CTA3: T4:0,  CTA2: T5:0|CTA3: T5:0,  CTA2: T6:0|CTA3: T6:0,  CTA2: T7:0|CTA3: T7:0\n"
      "CTA2: T8:0|CTA3: T8:0,  CTA2: T9:0|CTA3: T9:0,  CTA2:T10:0|CTA3:T10:0,  CTA2:T11:0|CTA3:T11:0\n"
      "CTA2:T12:0|CTA3:T12:0,  CTA2:T13:0|CTA3:T13:0,  CTA2:T14:0|CTA3:T14:0,  CTA2:T15:0|CTA3:T15:0\n"
      "CTA2:T16:0|CTA3:T16:0,  CTA2:T17:0|CTA3:T17:0,  CTA2:T18:0|CTA3:T18:0,  CTA2:T19:0|CTA3:T19:0\n"
      "CTA2:T20:0|CTA3:T20:0,  CTA2:T21:0|CTA3:T21:0,  CTA2:T22:0|CTA3:T22:0,  CTA2:T23:0|CTA3:T23:0\n"
      "CTA2:T24:0|CTA3:T24:0,  CTA2:T25:0|CTA3:T25:0,  CTA2:T26:0|CTA3:T26:0,  CTA2:T27:0|CTA3:T27:0\n"
      "CTA2:T28:0|CTA3:T28:0,  CTA2:T29:0|CTA3:T29:0,  CTA2:T30:0|CTA3:T30:0,  CTA2:T31:0|CTA3:T31:0\n";
  // clang-format on

  runBlocked2dMultiCTA(/*row=*/16, /*col=*/4, /*sizePerThread=*/{1, 1},
                       /*threadsPerWarp=*/{8, 4}, /*warpsPerCTA=*/{1, 1},
                       /*order=*/{1, 0}, /*CTAsPerCGA=*/{2, 2},
                       /*CTASplitNum=*/{2, 1}, /*CTAOrder=*/{1, 0},
                       /*refStr=*/refStr);
}

TEST_F(EmitIndicesTest, BlockedLayout_MultiCTA_CTAWrap_Dim0) {
  // clang-format off
  std::string refStr =
      "CTA0: T0:0|CTA2: T0:0,CTA0: T1:0|CTA2: T1:0,CTA0: T2:0|CTA2: T2:0,CTA0: T3:0|CTA2: T3:0,  CTA1: T0:0|CTA3: T0:0,CTA1: T1:0|CTA3: T1:0,CTA1: T2:0|CTA3: T2:0,CTA1: T3:0|CTA3: T3:0\n"
      "CTA0: T4:0|CTA2: T4:0,CTA0: T5:0|CTA2: T5:0,CTA0: T6:0|CTA2: T6:0,CTA0: T7:0|CTA2: T7:0,  CTA1: T4:0|CTA3: T4:0,CTA1: T5:0|CTA3: T5:0,CTA1: T6:0|CTA3: T6:0,CTA1: T7:0|CTA3: T7:0\n"
      "CTA0: T8:0|CTA2: T8:0,CTA0: T9:0|CTA2: T9:0,CTA0:T10:0|CTA2:T10:0,CTA0:T11:0|CTA2:T11:0,  CTA1: T8:0|CTA3: T8:0,CTA1: T9:0|CTA3: T9:0,CTA1:T10:0|CTA3:T10:0,CTA1:T11:0|CTA3:T11:0\n"
      "CTA0:T12:0|CTA2:T12:0,CTA0:T13:0|CTA2:T13:0,CTA0:T14:0|CTA2:T14:0,CTA0:T15:0|CTA2:T15:0,  CTA1:T12:0|CTA3:T12:0,CTA1:T13:0|CTA3:T13:0,CTA1:T14:0|CTA3:T14:0,CTA1:T15:0|CTA3:T15:0\n"
      "CTA0:T16:0|CTA2:T16:0,CTA0:T17:0|CTA2:T17:0,CTA0:T18:0|CTA2:T18:0,CTA0:T19:0|CTA2:T19:0,  CTA1:T16:0|CTA3:T16:0,CTA1:T17:0|CTA3:T17:0,CTA1:T18:0|CTA3:T18:0,CTA1:T19:0|CTA3:T19:0\n"
      "CTA0:T20:0|CTA2:T20:0,CTA0:T21:0|CTA2:T21:0,CTA0:T22:0|CTA2:T22:0,CTA0:T23:0|CTA2:T23:0,  CTA1:T20:0|CTA3:T20:0,CTA1:T21:0|CTA3:T21:0,CTA1:T22:0|CTA3:T22:0,CTA1:T23:0|CTA3:T23:0\n"
      "CTA0:T24:0|CTA2:T24:0,CTA0:T25:0|CTA2:T25:0,CTA0:T26:0|CTA2:T26:0,CTA0:T27:0|CTA2:T27:0,  CTA1:T24:0|CTA3:T24:0,CTA1:T25:0|CTA3:T25:0,CTA1:T26:0|CTA3:T26:0,CTA1:T27:0|CTA3:T27:0\n"
      "CTA0:T28:0|CTA2:T28:0,CTA0:T29:0|CTA2:T29:0,CTA0:T30:0|CTA2:T30:0,CTA0:T31:0|CTA2:T31:0,  CTA1:T28:0|CTA3:T28:0,CTA1:T29:0|CTA3:T29:0,CTA1:T30:0|CTA3:T30:0,CTA1:T31:0|CTA3:T31:0\n";
  // clang-format on

  runBlocked2dMultiCTA(
      /*row=*/8, /*col=*/8, /*sizePerThread=*/{1, 1}, /*threadsPerWarp=*/{8, 4},
      /*warpsPerCTA=*/{1, 1}, /*order=*/{1, 0}, /*CTAsPerCGA=*/{2, 2},
      /*CTASplitNum=*/{1, 2}, /*CTAOrder=*/{1, 0}, /*refStr=*/refStr);
}

TEST_F(EmitIndicesTest, BlockedLayout_MultiCTA_CTAWrapBeforeBroadcast_Dim1) {
  // clang-format off
  std::string refStr =
      "CTA0: T0:0|CTA0: T1:0|CTA0: T2:0|CTA0: T3:0  |  CTA1: T0:0|CTA1: T1:0|CTA1: T2:0|CTA1: T3:0\n"
      "CTA0: T4:0|CTA0: T5:0|CTA0: T6:0|CTA0: T7:0  |  CTA1: T4:0|CTA1: T5:0|CTA1: T6:0|CTA1: T7:0\n"
      "CTA0: T8:0|CTA0: T9:0|CTA0:T10:0|CTA0:T11:0  |  CTA1: T8:0|CTA1: T9:0|CTA1:T10:0|CTA1:T11:0\n"
      "CTA0:T12:0|CTA0:T13:0|CTA0:T14:0|CTA0:T15:0  |  CTA1:T12:0|CTA1:T13:0|CTA1:T14:0|CTA1:T15:0\n"
      "CTA0:T16:0|CTA0:T17:0|CTA0:T18:0|CTA0:T19:0  |  CTA1:T16:0|CTA1:T17:0|CTA1:T18:0|CTA1:T19:0\n"
      "CTA0:T20:0|CTA0:T21:0|CTA0:T22:0|CTA0:T23:0  |  CTA1:T20:0|CTA1:T21:0|CTA1:T22:0|CTA1:T23:0\n"
      "CTA0:T24:0|CTA0:T25:0|CTA0:T26:0|CTA0:T27:0  |  CTA1:T24:0|CTA1:T25:0|CTA1:T26:0|CTA1:T27:0\n"
      "CTA0:T28:0|CTA0:T29:0|CTA0:T30:0|CTA0:T31:0  |  CTA1:T28:0|CTA1:T29:0|CTA1:T30:0|CTA1:T31:0\n"

      "CTA2: T0:0|CTA2: T1:0|CTA2: T2:0|CTA2: T3:0  |  CTA3: T0:0|CTA3: T1:0|CTA3: T2:0|CTA3: T3:0\n"
      "CTA2: T4:0|CTA2: T5:0|CTA2: T6:0|CTA2: T7:0  |  CTA3: T4:0|CTA3: T5:0|CTA3: T6:0|CTA3: T7:0\n"
      "CTA2: T8:0|CTA2: T9:0|CTA2:T10:0|CTA2:T11:0  |  CTA3: T8:0|CTA3: T9:0|CTA3:T10:0|CTA3:T11:0\n"
      "CTA2:T12:0|CTA2:T13:0|CTA2:T14:0|CTA2:T15:0  |  CTA3:T12:0|CTA3:T13:0|CTA3:T14:0|CTA3:T15:0\n"
      "CTA2:T16:0|CTA2:T17:0|CTA2:T18:0|CTA2:T19:0  |  CTA3:T16:0|CTA3:T17:0|CTA3:T18:0|CTA3:T19:0\n"
      "CTA2:T20:0|CTA2:T21:0|CTA2:T22:0|CTA2:T23:0  |  CTA3:T20:0|CTA3:T21:0|CTA3:T22:0|CTA3:T23:0\n"
      "CTA2:T24:0|CTA2:T25:0|CTA2:T26:0|CTA2:T27:0  |  CTA3:T24:0|CTA3:T25:0|CTA3:T26:0|CTA3:T27:0\n"
      "CTA2:T28:0|CTA2:T29:0|CTA2:T30:0|CTA2:T31:0  |  CTA3:T28:0|CTA3:T29:0|CTA3:T30:0|CTA3:T31:0\n";
  // clang-format on

  runBlocked2dMultiCTA(/*row=*/16, /*col=*/1, /*sizePerThread=*/{1, 1},
                       /*threadsPerWarp=*/{8, 4}, /*warpsPerCTA=*/{1, 1},
                       /*order=*/{1, 0}, /*CTAsPerCGA=*/{2, 2},
                       /*CTASplitNum=*/{2, 2}, /*CTAOrder=*/{1, 0},
                       /*refStr=*/refStr);
}

TEST_F(EmitIndicesTest, BlockedLayout_MultiCTA_CTAWrapBeforeBroadcast_Dim0) {
  // clang-format off
  std::string refStr =
      "CTA0:T0:0|CTA0: T8:0|CTA0:T16:0|CTA0:T24:0  |  CTA2:T0:0|CTA2: T8:0|CTA2:T16:0|CTA2:T24:0,"
      "CTA0:T1:0|CTA0: T9:0|CTA0:T17:0|CTA0:T25:0  |  CTA2:T1:0|CTA2: T9:0|CTA2:T17:0|CTA2:T25:0,"
      "CTA0:T2:0|CTA0:T10:0|CTA0:T18:0|CTA0:T26:0  |  CTA2:T2:0|CTA2:T10:0|CTA2:T18:0|CTA2:T26:0,"
      "CTA0:T3:0|CTA0:T11:0|CTA0:T19:0|CTA0:T27:0  |  CTA2:T3:0|CTA2:T11:0|CTA2:T19:0|CTA2:T27:0,"
      "CTA0:T4:0|CTA0:T12:0|CTA0:T20:0|CTA0:T28:0  |  CTA2:T4:0|CTA2:T12:0|CTA2:T20:0|CTA2:T28:0,"
      "CTA0:T5:0|CTA0:T13:0|CTA0:T21:0|CTA0:T29:0  |  CTA2:T5:0|CTA2:T13:0|CTA2:T21:0|CTA2:T29:0,"
      "CTA0:T6:0|CTA0:T14:0|CTA0:T22:0|CTA0:T30:0  |  CTA2:T6:0|CTA2:T14:0|CTA2:T22:0|CTA2:T30:0,"
      "CTA0:T7:0|CTA0:T15:0|CTA0:T23:0|CTA0:T31:0  |  CTA2:T7:0|CTA2:T15:0|CTA2:T23:0|CTA2:T31:0,"

      "CTA1:T0:0|CTA1: T8:0|CTA1:T16:0|CTA1:T24:0  |  CTA3:T0:0|CTA3: T8:0|CTA3:T16:0|CTA3:T24:0,"
      "CTA1:T1:0|CTA1: T9:0|CTA1:T17:0|CTA1:T25:0  |  CTA3:T1:0|CTA3: T9:0|CTA3:T17:0|CTA3:T25:0,"
      "CTA1:T2:0|CTA1:T10:0|CTA1:T18:0|CTA1:T26:0  |  CTA3:T2:0|CTA3:T10:0|CTA3:T18:0|CTA3:T26:0,"
      "CTA1:T3:0|CTA1:T11:0|CTA1:T19:0|CTA1:T27:0  |  CTA3:T3:0|CTA3:T11:0|CTA3:T19:0|CTA3:T27:0,"
      "CTA1:T4:0|CTA1:T12:0|CTA1:T20:0|CTA1:T28:0  |  CTA3:T4:0|CTA3:T12:0|CTA3:T20:0|CTA3:T28:0,"
      "CTA1:T5:0|CTA1:T13:0|CTA1:T21:0|CTA1:T29:0  |  CTA3:T5:0|CTA3:T13:0|CTA3:T21:0|CTA3:T29:0,"
      "CTA1:T6:0|CTA1:T14:0|CTA1:T22:0|CTA1:T30:0  |  CTA3:T6:0|CTA3:T14:0|CTA3:T22:0|CTA3:T30:0,"
      "CTA1:T7:0|CTA1:T15:0|CTA1:T23:0|CTA1:T31:0  |  CTA3:T7:0|CTA3:T15:0|CTA3:T23:0|CTA3:T31:0\n";
  // clang-format on

  runBlocked2dMultiCTA(/*row=*/1, /*col=*/16, /*sizePerThread=*/{1, 1},
                       /*threadsPerWarp=*/{4, 8}, /*warpsPerCTA=*/{1, 1},
                       /*order=*/{1, 0}, /*CTAsPerCGA=*/{2, 2},
                       /*CTASplitNum=*/{2, 2}, /*CTAOrder=*/{1, 0},
                       /*refStr=*/refStr);
}

TEST_F(EmitIndicesTest, SliceLayout_MultiCTA) {
  // clang-format off
  std::string refStr =
      "CTA0: T0:0|CTA0: T1:0|CTA0: T2:0|CTA0: T3:0  |  CTA1: T0:0|CTA1: T1:0|CTA1: T2:0|CTA1: T3:0,"
      "CTA0: T4:0|CTA0: T5:0|CTA0: T6:0|CTA0: T7:0  |  CTA1: T4:0|CTA1: T5:0|CTA1: T6:0|CTA1: T7:0,"
      "CTA0: T8:0|CTA0: T9:0|CTA0:T10:0|CTA0:T11:0  |  CTA1: T8:0|CTA1: T9:0|CTA1:T10:0|CTA1:T11:0,"
      "CTA0:T12:0|CTA0:T13:0|CTA0:T14:0|CTA0:T15:0  |  CTA1:T12:0|CTA1:T13:0|CTA1:T14:0|CTA1:T15:0,"
      "CTA0:T16:0|CTA0:T17:0|CTA0:T18:0|CTA0:T19:0  |  CTA1:T16:0|CTA1:T17:0|CTA1:T18:0|CTA1:T19:0,"
      "CTA0:T20:0|CTA0:T21:0|CTA0:T22:0|CTA0:T23:0  |  CTA1:T20:0|CTA1:T21:0|CTA1:T22:0|CTA1:T23:0,"
      "CTA0:T24:0|CTA0:T25:0|CTA0:T26:0|CTA0:T27:0  |  CTA1:T24:0|CTA1:T25:0|CTA1:T26:0|CTA1:T27:0,"
      "CTA0:T28:0|CTA0:T29:0|CTA0:T30:0|CTA0:T31:0  |  CTA1:T28:0|CTA1:T29:0|CTA1:T30:0|CTA1:T31:0,"

      "CTA2: T0:0|CTA2: T1:0|CTA2: T2:0|CTA2: T3:0  |  CTA3: T0:0|CTA3: T1:0|CTA3: T2:0|CTA3: T3:0,"
      "CTA2: T4:0|CTA2: T5:0|CTA2: T6:0|CTA2: T7:0  |  CTA3: T4:0|CTA3: T5:0|CTA3: T6:0|CTA3: T7:0,"
      "CTA2: T8:0|CTA2: T9:0|CTA2:T10:0|CTA2:T11:0  |  CTA3: T8:0|CTA3: T9:0|CTA3:T10:0|CTA3:T11:0,"
      "CTA2:T12:0|CTA2:T13:0|CTA2:T14:0|CTA2:T15:0  |  CTA3:T12:0|CTA3:T13:0|CTA3:T14:0|CTA3:T15:0,"
      "CTA2:T16:0|CTA2:T17:0|CTA2:T18:0|CTA2:T19:0  |  CTA3:T16:0|CTA3:T17:0|CTA3:T18:0|CTA3:T19:0,"
      "CTA2:T20:0|CTA2:T21:0|CTA2:T22:0|CTA2:T23:0  |  CTA3:T20:0|CTA3:T21:0|CTA3:T22:0|CTA3:T23:0,"
      "CTA2:T24:0|CTA2:T25:0|CTA2:T26:0|CTA2:T27:0  |  CTA3:T24:0|CTA3:T25:0|CTA3:T26:0|CTA3:T27:0,"
      "CTA2:T28:0|CTA2:T29:0|CTA2:T30:0|CTA2:T31:0  |  CTA3:T28:0|CTA3:T29:0|CTA3:T30:0|CTA3:T31:0\n";
  // clang-format on

  runSliceBlockedMultiCTA(/*size=*/16, /*sizePerThread=*/{1, 1},
                          /*threadsPerWarp=*/{8, 4}, /*warpsPerCTA=*/{1, 1},
                          /*order=*/{1, 0}, /*CTAsPerCGA=*/{2, 2},
                          /*CTASplitNum=*/{2, 2}, /*CTAOrder=*/{1, 0},
                          /*sliceDim=*/1, /*refStr=*/refStr);
}

//===----------------------------------------------------------------------===//
// Tests for SharedEncodingAttr
//===----------------------------------------------------------------------===//

TEST_F(EmitIndicesTest, SharedLayout) {
  // clang-format off
  std::string refStr =
      "(0: 0),(0: 1),(0: 2),(0: 3),(0: 4),(0: 5),(0: 6),(0: 7),(0: 8),(0: 9),(0:10),(0:11),(0:12),(0:13),(0:14),(0:15),(0:16),(0:17),(0:18),(0:19),(0:20),(0:21),(0:22),(0:23),(0:24),(0:25),(0:26),(0:27),(0:28),(0:29),(0:30),(0:31)\n"
      "(1: 0),(1: 1),(1: 2),(1: 3),(1: 4),(1: 5),(1: 6),(1: 7),(1: 8),(1: 9),(1:10),(1:11),(1:12),(1:13),(1:14),(1:15),(1:16),(1:17),(1:18),(1:19),(1:20),(1:21),(1:22),(1:23),(1:24),(1:25),(1:26),(1:27),(1:28),(1:29),(1:30),(1:31)\n"
      "(2: 8),(2: 9),(2:10),(2:11),(2:12),(2:13),(2:14),(2:15),(2: 0),(2: 1),(2: 2),(2: 3),(2: 4),(2: 5),(2: 6),(2: 7),(2:24),(2:25),(2:26),(2:27),(2:28),(2:29),(2:30),(2:31),(2:16),(2:17),(2:18),(2:19),(2:20),(2:21),(2:22),(2:23)\n"
      "(3: 8),(3: 9),(3:10),(3:11),(3:12),(3:13),(3:14),(3:15),(3: 0),(3: 1),(3: 2),(3: 3),(3: 4),(3: 5),(3: 6),(3: 7),(3:24),(3:25),(3:26),(3:27),(3:28),(3:29),(3:30),(3:31),(3:16),(3:17),(3:18),(3:19),(3:20),(3:21),(3:22),(3:23)\n"
      "(4:16),(4:17),(4:18),(4:19),(4:20),(4:21),(4:22),(4:23),(4:24),(4:25),(4:26),(4:27),(4:28),(4:29),(4:30),(4:31),(4: 0),(4: 1),(4: 2),(4: 3),(4: 4),(4: 5),(4: 6),(4: 7),(4: 8),(4: 9),(4:10),(4:11),(4:12),(4:13),(4:14),(4:15)\n"
      "(5:16),(5:17),(5:18),(5:19),(5:20),(5:21),(5:22),(5:23),(5:24),(5:25),(5:26),(5:27),(5:28),(5:29),(5:30),(5:31),(5: 0),(5: 1),(5: 2),(5: 3),(5: 4),(5: 5),(5: 6),(5: 7),(5: 8),(5: 9),(5:10),(5:11),(5:12),(5:13),(5:14),(5:15)\n"
      "(6:24),(6:25),(6:26),(6:27),(6:28),(6:29),(6:30),(6:31),(6:16),(6:17),(6:18),(6:19),(6:20),(6:21),(6:22),(6:23),(6: 8),(6: 9),(6:10),(6:11),(6:12),(6:13),(6:14),(6:15),(6: 0),(6: 1),(6: 2),(6: 3),(6: 4),(6: 5),(6: 6),(6: 7)\n"
      "(7:24),(7:25),(7:26),(7:27),(7:28),(7:29),(7:30),(7:31),(7:16),(7:17),(7:18),(7:19),(7:20),(7:21),(7:22),(7:23),(7: 8),(7: 9),(7:10),(7:11),(7:12),(7:13),(7:14),(7:15),(7: 0),(7: 1),(7: 2),(7: 3),(7: 4),(7: 5),(7: 6),(7: 7)\n";
  // clang-format on

  runSharedSingleCTA(/*row=*/8, /*col=*/32, /*rowMajor=*/true,
                     /*elemTyStr=*/"F16", /*refStr=*/refStr);
}

TEST_F(EmitIndicesTest, LayoutVisualizer_Blocked) {
  CTALayoutAttr CTALayout =
      CTALayoutAttr::get(/*context=*/&context, /*CTAsPerCGA=*/{2, 2},
                         /*CTASplitNum=*/{2, 2}, /*CTAOrder=*/{1, 0});

  Attribute blockedLayout = BlockedEncodingAttr::get(
      /*context=*/&context, /*sizePerThread=*/{1, 4},
      /*threadsPerWarp=*/{2, 16},
      /*warpsPerCTA=*/{4, 1}, /*order=*/{1, 0}, /*CTALayout=*/CTALayout);

  llvm::SmallVector<int64_t> shape = {/*row=*/128, /*col=*/128};

  std::ofstream ofs("blockedLayout.csv");
  ofs << dumpDistributedLayout(blockedLayout, shape, /*multiCTA=*/true);
}

TEST_F(EmitIndicesTest, LayoutVisualizer_Shared) {
  CTALayoutAttr CTALayout =
      CTALayoutAttr::get(/*context=*/&context, /*CTAsPerCGA=*/{1, 1},
                         /*CTASplitNum=*/{1, 1}, /*CTAOrder=*/{1, 0});

  Attribute sharedLayout = SharedEncodingAttr::get(
      /*context=*/&context, /*vec=*/1, /*perPhase=*/2, /*maxPhase=*/8,
      /*order=*/{0, 1}, /*CTALayout=*/CTALayout);

  llvm::SmallVector<int64_t> shape = {/*row=*/16, /*col=*/16};
  Type elemTy = FloatType::getF16(&context);

  std::ofstream ofs("sharedLayout.csv");
  ofs << dumpSharedLayout(sharedLayout, shape, elemTy, /*multiCTA=*/false);
}
#endif

//===----------------------------------------------------------------------===//
// Tests for SliceEncodingAttr
//===----------------------------------------------------------------------===//

TEST_F(EmitIndicesTest, SliceLayout_SingleCTA_SliceDim1) {
  // clang-format off
  std::string refStr =
      " T0:0| T1:0| T2:0| T3:0| T4:0| T5:0| T6:0| T7:0,"
      " T8:0| T9:0|T10:0|T11:0|T12:0|T13:0|T14:0|T15:0,"
      "T16:0|T17:0|T18:0|T19:0|T20:0|T21:0|T22:0|T23:0,"
      "T24:0|T25:0|T26:0|T27:0|T28:0|T29:0|T30:0|T31:0\n";
  // clang-format on

  runSliceBlockedSingleCTA(/*size=*/4, /*sizePerThread=*/{1, 1},
                           /*threadsPerWarp=*/{4, 8}, /*warpsPerCTA=*/{1, 1},
                           /*order=*/{1, 0}, /*sliceDim=*/1, /*refStr=*/refStr);
}

TEST_F(EmitIndicesTest, SliceLayout_SingleCTA_SliceDim0) {
  // clang-format off
  std::string refStr =
      "T0:0| T8:0|T16:0|T24:0,"
      "T1:0| T9:0|T17:0|T25:0,"
      "T2:0|T10:0|T18:0|T26:0,"
      "T3:0|T11:0|T19:0|T27:0,"
      "T4:0|T12:0|T20:0|T28:0,"
      "T5:0|T13:0|T21:0|T29:0,"
      "T6:0|T14:0|T22:0|T30:0,"
      "T7:0|T15:0|T23:0|T31:0\n";
  // clang-format on

  runSliceBlockedSingleCTA(/*size=*/8, /*sizePerThread=*/{1, 1},
                           /*threadsPerWarp=*/{4, 8}, /*warpsPerCTA=*/{1, 1},
                           /*order=*/{1, 0}, /*sliceDim=*/0, /*refStr=*/refStr);
}

//===----------------------------------------------------------------------===//
// Tests for NvidiaMmaEncodingAttr
//===----------------------------------------------------------------------===//

TEST_F(EmitIndicesTest, MmaLayout) {
  // clang-format off
  std::string refStr =
      " T0:0, T0:1, T1:0, T1:1, T2:0, T2:1, T3:0, T3:1\n"
      " T4:0, T4:1, T5:0, T5:1, T6:0, T6:1, T7:0, T7:1\n"
      " T8:0, T8:1, T9:0, T9:1,T10:0,T10:1,T11:0,T11:1\n"
      "T12:0,T12:1,T13:0,T13:1,T14:0,T14:1,T15:0,T15:1\n"
      "T16:0,T16:1,T17:0,T17:1,T18:0,T18:1,T19:0,T19:1\n"
      "T20:0,T20:1,T21:0,T21:1,T22:0,T22:1,T23:0,T23:1\n"
      "T24:0,T24:1,T25:0,T25:1,T26:0,T26:1,T27:0,T27:1\n"
      "T28:0,T28:1,T29:0,T29:1,T30:0,T30:1,T31:0,T31:1\n"
      " T0:2, T0:3, T1:2, T1:3, T2:2, T2:3, T3:2, T3:3\n"
      " T4:2, T4:3, T5:2, T5:3, T6:2, T6:3, T7:2, T7:3\n"
      " T8:2, T8:3, T9:2, T9:3,T10:2,T10:3,T11:2,T11:3\n"
      "T12:2,T12:3,T13:2,T13:3,T14:2,T14:3,T15:2,T15:3\n"
      "T16:2,T16:3,T17:2,T17:3,T18:2,T18:3,T19:2,T19:3\n"
      "T20:2,T20:3,T21:2,T21:3,T22:2,T22:3,T23:2,T23:3\n"
      "T24:2,T24:3,T25:2,T25:3,T26:2,T26:3,T27:2,T27:3\n"
      "T28:2,T28:3,T29:2,T29:3,T30:2,T30:3,T31:2,T31:3\n";
  // clang-format on

  runMmaSingleCTA(/*row=*/16, /*col=*/8, /*versionMajor=*/2, /*versionMinor=*/1,
                  /*warpsPerCTA=*/{1, 1}, /*instrShape=*/{16, 8},
                  /*refStr=*/refStr);
}

//===----------------------------------------------------------------------===//
// Tests for AMDWmmaEncodingAttr
//===----------------------------------------------------------------------===//

TEST_F(EmitIndicesTest, WmmaLayout) {
  // clang-format off
  std::string refStr =
      "T0:0,T1:0,T2:0,T3:0,T4:0,T5:0,T6:0,T7:0,T8:0,T9:0,T10:0,T11:0,T12:0,T13:0,T14:0,T15:0\n"
      "T16:0,T17:0,T18:0,T19:0,T20:0,T21:0,T22:0,T23:0,T24:0,T25:0,T26:0,T27:0,T28:0,T29:0,T30:0,T31:0\n"
      "T0:1,T1:1,T2:1,T3:1,T4:1,T5:1,T6:1,T7:1,T8:1,T9:1,T10:1,T11:1,T12:1,T13:1,T14:1,T15:1\n"
      "T16:1,T17:1,T18:1,T19:1,T20:1,T21:1,T22:1,T23:1,T24:1,T25:1,T26:1,T27:1,T28:1,T29:1,T30:1,T31:1\n"
      "T0:2,T1:2,T2:2,T3:2,T4:2,T5:2,T6:2,T7:2,T8:2,T9:2,T10:2,T11:2,T12:2,T13:2,T14:2,T15:2\n"
      "T16:2,T17:2,T18:2,T19:2,T20:2,T21:2,T22:2,T23:2,T24:2,T25:2,T26:2,T27:2,T28:2,T29:2,T30:2,T31:2\n"
      "T0:3,T1:3,T2:3,T3:3,T4:3,T5:3,T6:3,T7:3,T8:3,T9:3,T10:3,T11:3,T12:3,T13:3,T14:3,T15:3\n"
      "T16:3,T17:3,T18:3,T19:3,T20:3,T21:3,T22:3,T23:3,T24:3,T25:3,T26:3,T27:3,T28:3,T29:3,T30:3,T31:3\n"
      "T0:4,T1:4,T2:4,T3:4,T4:4,T5:4,T6:4,T7:4,T8:4,T9:4,T10:4,T11:4,T12:4,T13:4,T14:4,T15:4\n"
      "T16:4,T17:4,T18:4,T19:4,T20:4,T21:4,T22:4,T23:4,T24:4,T25:4,T26:4,T27:4,T28:4,T29:4,T30:4,T31:4\n"
      "T0:5,T1:5,T2:5,T3:5,T4:5,T5:5,T6:5,T7:5,T8:5,T9:5,T10:5,T11:5,T12:5,T13:5,T14:5,T15:5\n"
      "T16:5,T17:5,T18:5,T19:5,T20:5,T21:5,T22:5,T23:5,T24:5,T25:5,T26:5,T27:5,T28:5,T29:5,T30:5,T31:5\n"
      "T0:6,T1:6,T2:6,T3:6,T4:6,T5:6,T6:6,T7:6,T8:6,T9:6,T10:6,T11:6,T12:6,T13:6,T14:6,T15:6\n"
      "T16:6,T17:6,T18:6,T19:6,T20:6,T21:6,T22:6,T23:6,T24:6,T25:6,T26:6,T27:6,T28:6,T29:6,T30:6,T31:6\n"
      "T0:7,T1:7,T2:7,T3:7,T4:7,T5:7,T6:7,T7:7,T8:7,T9:7,T10:7,T11:7,T12:7,T13:7,T14:7,T15:7\n"
      "T16:7,T17:7,T18:7,T19:7,T20:7,T21:7,T22:7,T23:7,T24:7,T25:7,T26:7,T27:7,T28:7,T29:7,T30:7,T31:7\n";
  // clang-format on

  runWmmaSingleCTA(/*row=*/16, /*col=*/16,
                   /*warpsPerCTA=*/{1, 1},
                   /*refStr=*/refStr);
}

//===----------------------------------------------------------------------===//
// The following unittests are tools for Triton developers to visualize layouts.
// You can modify parameters and shapes here to create your own layout and
// tensor. The output will be saved into a csv file which can be opened with
// Microsoft Excel.
//===----------------------------------------------------------------------===//

TEST_F(EmitIndicesTest, LayoutVisualizer_Slice) {
  CTALayoutAttr CTALayout =
      CTALayoutAttr::get(/*context=*/&context, /*CTAsPerCGA=*/{1, 1},
                         /*CTASplitNum=*/{1, 1}, /*CTAOrder=*/{1, 0});

  Attribute blockedLayout = BlockedEncodingAttr::get(
      /*context=*/&context, /*sizePerThread=*/{1, 1}, /*threadsPerWarp=*/{4, 8},
      /*warpsPerCTA=*/{1, 1}, /*order=*/{1, 0}, /*CTALayout=*/CTALayout);

  Attribute sliceLayout = SliceEncodingAttr::get(
      /*context=*/&context, /*dim=*/1, /*parent=*/blockedLayout);

  llvm::SmallVector<int64_t> shape = {4};

  std::ofstream ofs("sliceLayout.csv");
  ofs << dumpDistributedLayout(sliceLayout, shape, /*multiCTA=*/false);
}

TEST_F(EmitIndicesTest, LayoutVisualizer_Mma) {
  CTALayoutAttr CTALayout =
      CTALayoutAttr::get(/*context=*/&context, /*CTAsPerCGA=*/{1, 1},
                         /*CTASplitNum=*/{1, 1}, /*CTAOrder=*/{1, 0});

  Attribute mmaLayout = NvidiaMmaEncodingAttr::get(
      /*context=*/&context, /*versionMajor=*/2, /*versionMinor=*/1,
      /*warpsPerCTA=*/{1, 1}, /*CTALayout=*/CTALayout, /*instrShape=*/{16, 8});

  llvm::SmallVector<int64_t> shape = {/*row=*/16, /*col=*/8};

  std::ofstream ofs("mmaLayout.csv");
  ofs << dumpDistributedLayout(mmaLayout, shape, /*multiCTA=*/false);
}

TEST_F(EmitIndicesTest, LayoutVisualizer_Wmma) {
  CTALayoutAttr CTALayout =
      CTALayoutAttr::get(/*context=*/&context, /*CTAsPerCGA=*/{1, 1},
                         /*CTASplitNum=*/{1, 1}, /*CTAOrder=*/{1, 0});

  Attribute wmmaLayout = AMDWmmaEncodingAttr::get(
      /*context=*/&context,
      /*warpsPerCTA=*/{1, 1}, /*CTALayout=*/CTALayout);

  llvm::SmallVector<int64_t> shape = {/*row=*/16, /*col=*/16};

  std::ofstream ofs("WmmaLayout.csv");
  ofs << dumpDistributedLayout(wmmaLayout, shape, /*multiCTA=*/false);
}

// Checks that result of emitIndices with and without linear layouts is the
// same.
//
// This is only for "distributed" layouts, i.e. layouts whose values are stored
// in registers distributed among threads in blocks.
template <typename LayoutT, typename ParamsT>
class DistributedLLTest : public EmitIndicesTest,
                          public ::testing::WithParamInterface<ParamsT> {
protected:
  void DoIt();
};

template <typename LayoutT, typename ParamsT>
void DistributedLLTest<LayoutT, ParamsT>::DoIt() {
  ParamsT params = this->GetParam();
  LayoutT legacyLayout = params.getEncoding();
  auto type = RankedTensorType::get(params.shape, FloatType::getF16(&context),
                                    legacyLayout);

  int threadsPerWarp = product(triton::gpu::getThreadsPerWarp(legacyLayout));
  int numThreads = product(triton::gpu::getThreadsPerWarp(legacyLayout)) *
                   product(triton::gpu::getWarpsPerCTA(legacyLayout));

  // Can't call getCTAsPerCGA on a SliceEncodingAttr.  But all we care about is
  // the total number of CTAs, which we can just as easily get from the slice
  // layout's parent.
  Attribute nonSliceLayout = legacyLayout;
  while (auto sliceLayout = dyn_cast<SliceEncodingAttr>(nonSliceLayout)) {
    nonSliceLayout = sliceLayout.getParent();
  }
  int numCTAs = product(triton::gpu::getCTAsPerCGA(nonSliceLayout));

  mlir::OpBuilder builder(&context);
  Location loc = UnknownLoc::get(&context);
  auto mlirModule = mlir::ModuleOp::create(loc);
  auto func = builder.create<mlir::triton::FuncOp>(
      loc, "test_func", builder.getFunctionType({}, {}));
  mlirModule.push_back(func);
  auto *block = func.addEntryBlock();
  IRRewriter rewriter(&context);
  rewriter.setInsertionPointToStart(block);

  NVIDIA::TargetInfo target(90);
  auto llIndices = emitIndicesUsingLinearLayouts(
      loc, rewriter, target, legacyLayout, type, /*withCTAOffset=*/true);
  auto legacyIndices = emitIndices(loc, rewriter, target, legacyLayout, type,
                                   /*withCTAOffset=*/true, /*allowLL=*/false);

  // This test takes a long time if we check all indices.  But for linear
  // layouts, we really should only need to check powers of 2.  We wrap the
  // loops in this `iterate` function so we can easily change between checking
  // all indices and just the powers of 2.
  constexpr bool checkAllElems = false;
  bool stopIterating = false;
  auto iterate = [&](int n, auto fn) {
    if (checkAllElems) {
      for (int i = 0; i < n && !stopIterating; i++) {
        fn(i);
      }
    } else {
      if (n > 0) {
        fn(0);
      }
      for (int i = 0; (1 << i) < n && !stopIterating; i++) {
        fn(1 << i);
      }
    }
  };

  // We don't need to print a lot of failures because we also print our guess as
  // to the correct linear layout at the end.
  constexpr int kMaxFailures = 4;
  int64_t numFailures = 0;
  bool passedInitialChecks = false;

  // Wrap these tests in a lambda so failed ASSERTs exit the loop but don't exit
  // the whole test.
  [&] {
    ASSERT_TRUE(llIndices.has_value());
    ASSERT_EQ(llIndices->size(), legacyIndices.size());
    passedInitialChecks = true;

    iterate(llIndices->size(), [&](int i) {
      SCOPED_TRACE("Register " + std::to_string(i));
      ASSERT_EQ((*llIndices)[i].size(), legacyIndices[i].size());
      iterate((*llIndices)[i].size(), [&](int j) {
        SCOPED_TRACE("Dimension " + std::to_string(j));
        iterate(numCTAs, [&](int ctaId) {
          SCOPED_TRACE("CTA " + std::to_string(ctaId));
          iterate(numThreads, [&](int tid) {
            SCOPED_TRACE("Thread " + std::to_string(tid));
            int llValue = evalValue((*llIndices)[i][j], ctaId, tid);
            int legacyValue = evalValue(legacyIndices[i][j], ctaId, tid);
            EXPECT_EQ(llValue, legacyValue);
            if (llValue != legacyValue) {
              ++numFailures;
            }
            if (numFailures > kMaxFailures) {
              llvm::errs() << "Too many failures, aborting\n";
              stopIterating = true;
            }
          });
        });
      });
    });
  }();

  // If there was a failure, try to infer what the correct linear layout should
  // have been.  This assumes that the legacy layout itself is linear, of
  // course!
  if (!passedInitialChecks || numFailures > 0) {
    llvm::errs() << "Linear layout was\n"
                 << toLinearLayout(params.shape, params.getEncoding()) << "\n";

    llvm::errs() << "But based on the legacy layout, the LL should be:\n\n";

    llvm::errs() << "LinearLayout({\n";
    llvm::errs() << "  {S(\"register\"), {\n";
    for (int reg = 1; reg < legacyIndices.size(); reg *= 2) {
      llvm::errs() << "    {" << join(legacyIndices[reg], ", ", [](Value v) {
        return evalValue(v, /*ctaId=*/0, /*tid=*/0);
      }) << "},\n";
    }
    llvm::errs() << "  }},\n";

    llvm::errs() << "  {S(\"lane\"), {\n";
    for (int tid = 1; tid < numThreads; tid *= 2) {
      if (tid == threadsPerWarp) {
        llvm::errs() << "  }},\n";
        llvm::errs() << "  {S(\"warp\"), {\n";
      }
      llvm::errs() << "    {" << join(legacyIndices[0], ", ", [&](Value v) {
        return evalValue(v, /*ctaId=*/0, tid);
      }) << "},\n";
    }
    llvm::errs() << "  }},\n";
    llvm::errs() << "  {S(\"block\"), {\n";
    for (int ctaId = 1; ctaId < numCTAs; ctaId *= 2) {
      llvm::errs() << "    {" << join(legacyIndices[0], ", ", [&](Value v) {
        return evalValue(v, ctaId, /*tid=*/0);
      }) << "},\n";
    }
    llvm::errs() << "  }}\n";
    llvm::errs() << "}, {"
                 << triton::join(llvm::seq(type.getRank()), ", ",
                                 [](int dim) {
                                   return "S(\"dim" + std::to_string(dim) +
                                          "\")";
                                 })
                 << "})\n";
  }
}

struct BlockedLLTestParams {
  std::vector<int64_t> shape;
  std::vector<unsigned> sizePerThread;
  std::vector<unsigned> threadsPerWarp;
  std::vector<unsigned> warpsPerCTA;
  std::vector<unsigned> order;
  std::vector<unsigned> CTAsPerCGA;
  std::vector<unsigned> CTASplitNum;
  std::vector<unsigned> CTAOrder;

  BlockedEncodingAttr getEncoding() const {
    return BlockedEncodingAttr::get(
        getContext(), sizePerThread, threadsPerWarp, warpsPerCTA, order,
        CTALayoutAttr::get(getContext(), CTAsPerCGA, CTASplitNum, CTAOrder));
  }
};

std::ostream &operator<<(std::ostream &os, const BlockedLLTestParams &params) {
  std::string str;
  llvm::raw_string_ostream llvm_os(str);
  llvm_os << "shape=" << triton::join(params.shape, "x")
          << ", encoding=" << params.getEncoding();
  os << str;
  return os;
}

class BlockedLLTest
    : public DistributedLLTest<BlockedEncodingAttr, BlockedLLTestParams> {};

TEST_P(BlockedLLTest, DoIt) { DoIt(); }

INSTANTIATE_TEST_SUITE_P(TestCases, BlockedLLTest,
                         ::testing::ValuesIn(std::vector<BlockedLLTestParams>({
                             {
                                 .shape = {128, 16},
                                 .sizePerThread = {1, 4},
                                 .threadsPerWarp = {8, 4},
                                 .warpsPerCTA = {4, 1},
                                 .order = {1, 0},
                                 .CTAsPerCGA = {2, 2},
                                 .CTASplitNum = {2, 1},
                                 .CTAOrder = {1, 0},
                             },
                             {
                                 .shape = {1, 128},
                                 .sizePerThread = {8, 1},
                                 .threadsPerWarp = {8, 4},
                                 .warpsPerCTA = {1, 4},
                                 .order = {0, 1},
                                 .CTAsPerCGA = {1, 2},
                                 .CTASplitNum = {1, 2},
                                 .CTAOrder = {1, 0},
                             },
                             {
                                 .shape = {64, 1},
                                 .sizePerThread = {8, 1},
                                 .threadsPerWarp = {8, 4},
                                 .warpsPerCTA = {1, 4},
                                 .order = {0, 1},
                                 .CTAsPerCGA = {1, 2},
                                 .CTASplitNum = {1, 2},
                                 .CTAOrder = {1, 0},
                             },
                             {
                                 .shape = {128, 1},
                                 .sizePerThread = {1, 8},
                                 .threadsPerWarp = {4, 8},
                                 .warpsPerCTA = {4, 1},
                                 .order = {1, 0},
                                 .CTAsPerCGA = {1, 2},
                                 .CTASplitNum = {1, 1},
                                 .CTAOrder = {1, 0},
                             },
                             {
                                 .shape = {1, 64},
                                 .sizePerThread = {1, 8},
                                 .threadsPerWarp = {4, 8},
                                 .warpsPerCTA = {4, 1},
                                 .order = {1, 0},
                                 .CTAsPerCGA = {1, 2},
                                 .CTASplitNum = {1, 1},
                                 .CTAOrder = {1, 0},
                             },
                             {
                                 .shape = {128, 1},
                                 .sizePerThread = {1, 1},
                                 .threadsPerWarp = {1, 32},
                                 .warpsPerCTA = {2, 2},
                                 .order = {1, 0},
                                 .CTAsPerCGA = {1, 2},
                                 .CTASplitNum = {1, 2},
                                 .CTAOrder = {1, 0},
                             },
                             {
                                 .shape = {1, 128},
                                 .sizePerThread = {1, 1},
                                 .threadsPerWarp = {1, 32},
                                 .warpsPerCTA = {2, 2},
                                 .order = {1, 0},
                                 .CTAsPerCGA = {1, 2},
                                 .CTASplitNum = {1, 2},
                                 .CTAOrder = {1, 0},
                             },
                             {
                                 .shape = {1},
                                 .sizePerThread = {1},
                                 .threadsPerWarp = {32},
                                 .warpsPerCTA = {4},
                                 .order = {0},
                                 .CTAsPerCGA = {2},
                                 .CTASplitNum = {2},
                                 .CTAOrder = {0},
                             },
                             {
                                 .shape = {128, 128},
                                 .sizePerThread = {2, 2},
                                 .threadsPerWarp = {4, 8},
                                 .warpsPerCTA = {2, 2},
                                 .order = {0, 1},
                                 .CTAsPerCGA = {2, 2},
                                 .CTASplitNum = {2, 2},
                                 .CTAOrder = {0, 1},
                             },
                             {
                                 .shape = {1024, 128},
                                 .sizePerThread = {2, 2},
                                 .threadsPerWarp = {4, 8},
                                 .warpsPerCTA = {2, 2},
                                 .order = {1, 0},
                                 .CTAsPerCGA = {2, 2},
                                 .CTASplitNum = {2, 2},
                                 .CTAOrder = {1, 0},
                             },
                         })));

struct NvidiaMmaLLTestParams {
  std::vector<int64_t> shape;
  unsigned versionMajor;
  unsigned versionMinor;
  std::vector<unsigned> warpsPerCTA;
  std::vector<unsigned> instrShape;
  std::vector<unsigned> CTAsPerCGA;
  std::vector<unsigned> CTASplitNum;
  std::vector<unsigned> CTAOrder;

  NvidiaMmaEncodingAttr getEncoding() const {
    return NvidiaMmaEncodingAttr::get(
        getContext(), versionMajor, versionMinor, warpsPerCTA,
        CTALayoutAttr::get(getContext(), CTAsPerCGA, CTASplitNum, CTAOrder),
        instrShape);
  }
};

std::ostream &operator<<(std::ostream &os,
                         const NvidiaMmaLLTestParams &params) {
  std::string str;
  llvm::raw_string_ostream llvm_os(str);
  llvm_os << "shape=" << triton::join(params.shape, "x")
          << ", encoding=" << params.getEncoding();
  os << str;
  return os;
}

class NvidiaMmaLLTest
    : public DistributedLLTest<NvidiaMmaEncodingAttr, NvidiaMmaLLTestParams> {};

TEST_P(NvidiaMmaLLTest, DoIt) { DoIt(); }

INSTANTIATE_TEST_SUITE_P(
    MMAv2, NvidiaMmaLLTest,
    ::testing::ValuesIn(std::vector<NvidiaMmaLLTestParams>({
        {
            .shape = {16, 8},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {1, 1},
            .instrShape = {16, 8},
            .CTAsPerCGA = {1, 1},
            .CTASplitNum = {1, 1},
            .CTAOrder = {1, 0},
        },
        {
            .shape = {32, 32},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {1, 1},
            .instrShape = {16, 8},
            .CTAsPerCGA = {1, 1},
            .CTASplitNum = {1, 1},
            .CTAOrder = {1, 0},
        },
        {
            .shape = {128, 8},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {1, 1},
            .instrShape = {16, 8},
            .CTAsPerCGA = {1, 1},
            .CTASplitNum = {1, 1},
            .CTAOrder = {1, 0},
        },
        {
            .shape = {16, 128},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {1, 1},
            .instrShape = {16, 8},
            .CTAsPerCGA = {1, 1},
            .CTASplitNum = {1, 1},
            .CTAOrder = {1, 0},
        },
        {
            .shape = {32, 32},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {2, 2},
            .instrShape = {16, 8},
            .CTAsPerCGA = {1, 1},
            .CTASplitNum = {1, 1},
            .CTAOrder = {1, 0},
        },
        {
            .shape = {16, 8},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {2, 2},
            .instrShape = {16, 8},
            .CTAsPerCGA = {1, 1},
            .CTASplitNum = {1, 1},
            .CTAOrder = {1, 0},
        },
        {
            .shape = {16, 512},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {2, 2},
            .instrShape = {16, 8},
            .CTAsPerCGA = {1, 1},
            .CTASplitNum = {1, 1},
            .CTAOrder = {1, 0},
        },
        {
            .shape = {512, 8},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {2, 2},
            .instrShape = {16, 8},
            .CTAsPerCGA = {1, 1},
            .CTASplitNum = {1, 1},
            .CTAOrder = {1, 0},
        },
        {
            .shape = {512, 512},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {2, 2},
            .instrShape = {16, 8},
            .CTAsPerCGA = {1, 1},
            .CTASplitNum = {1, 1},
            .CTAOrder = {1, 0},
        },
        {
            // Legacy emitIndices seems to do implicit duplication in the last
            // two dims, but not in the others.  That is, this test works
            // because shape[0] == warpsPerCTA[0] * CTASplitNum[0], but if you
            // increase shape[0] to 32, then the legacy layout will not increase
            // its size, whereas the linear layout will.  I think this is a bug
            // in the legacy layout.
            .shape = {16, 128, 128},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {16, 1, 1},
            .instrShape = {1, 16, 8},
            .CTAsPerCGA = {1, 1, 1},
            .CTASplitNum = {1, 1, 1},
            .CTAOrder = {2, 1, 0},
        },
        {
            .shape = {16 * 4, 128, 128},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {16, 1, 1},
            .instrShape = {1, 16, 8},
            .CTAsPerCGA = {4, 1, 1},
            .CTASplitNum = {4, 1, 1},
            .CTAOrder = {2, 1, 0},
        },
        {
            .shape = {16 * 4, 128, 128},
            .versionMajor = 2,
            .versionMinor = 0,
            .warpsPerCTA = {16, 1, 1},
            .instrShape = {1, 16, 8},
            .CTAsPerCGA = {4, 2, 2},
            .CTASplitNum = {4, 2, 1},
            .CTAOrder = {2, 1, 0},
        },
    })));

std::vector<NvidiaMmaLLTestParams> makeNvidiaMmaV3TestCases() {
  std::vector<NvidiaMmaLLTestParams> testCases;
  auto addTests = [&](ArrayRef<unsigned> instrShape, unsigned warpsPerCGA_dim0,
                      ArrayRef<std::vector<int64_t>> shapes) {
    for (const auto &shape : shapes) {
      for (unsigned wpc0 : {4, 8}) {
        for (unsigned wpc1 : {1, 2, 4, 8}) {
          testCases.push_back({
              .shape = shape,
              .versionMajor = 3,
              .versionMinor = 0,
              .warpsPerCTA = {wpc0, wpc1},
              .instrShape = instrShape,
              .CTAsPerCGA = {1, 1},
              .CTASplitNum = {1, 1},
              .CTAOrder = {1, 0},
          });
        }
      }
    }
  };

  // These shapes were captured from grep'ing the TTGIR generated by Triton unit
  // tests.
  addTests({16, 16, 8}, 4, {{16, 16}, {32, 16}, {32, 32}, {64, 64}});
  addTests({16, 16, 16}, 4, {{64, 16}, {128, 16}, {128, 128}});
  addTests({16, 16, 32}, 4, {{64, 16}, {128, 16}});
  addTests({16, 32, 8}, 4, {{64, 32}, {128, 32}});
  addTests({16, 32, 16}, 4, {{64, 32}, {64, 64}, {256, 64}});
  addTests({16, 64, 8}, 4, {{64, 64}, {128, 64}});
  addTests({16, 64, 16}, 4, {{64, 64}, {128, 64}});
  addTests({16, 64, 32}, 4, {{64, 64}});
  addTests({16, 128, 8}, 4, {{64, 128}, {128, 128}});
  addTests({16, 128, 16}, 4, {{64, 128}, {128, 128}});
  addTests({16, 128, 16}, 8, {{64, 128}, {128, 128}});
  addTests({16, 128, 32}, 8, {{64, 128}, {128, 128}});
  addTests({16, 256, 8}, 8, {{128, 256}});
  addTests({16, 256, 16}, 8, {{128, 256}});
  addTests({16, 256, 32}, 8, {{128, 256}});

  // Shapes 1xN and Nx1 appear in IR, but legacy emitIndices cannot handle them.
  // They appear in IR like the following.
  //
  //   #mma = #nvidia_mma<{versionMajor=3, versionMinor=0,
  //                       warpsPerCTA=[4, 1], instrShape=[16, 64, 16]}>
  //   %a : tensor<64xf16, #slice<dim=0, parent=#mma>>
  //   %b = tt.expand_dims %a : tensor<1x64xf16, #mma>
  //   %c = arith.extf %b : tensor<1x64xf32, #mma>
  //   %d = tt.broadcast %c : tensor<64x64xf32, #mma>
  //
  // TODO(jlebar): For now we don't test these layouts.  Once we have slice
  // layout working, we can add support, since their layouts should match that
  // of emitIndices for the corresponding slice layout.

  return testCases;
}

INSTANTIATE_TEST_SUITE_P(MMAv3, NvidiaMmaLLTest,
                         ::testing::ValuesIn(makeNvidiaMmaV3TestCases()));

struct SliceLLTestParams {
  std::vector<int64_t> shape;
  int64_t sliceDim;
  std::variant<BlockedLLTestParams, NvidiaMmaLLTestParams> parent;

  SliceEncodingAttr getEncoding() const {
    return std::visit(
        [&](const auto &parentParams) {
          return SliceEncodingAttr::get(getContext(), sliceDim,
                                        parentParams.getEncoding());
        },
        parent);
  }
};

std::ostream &operator<<(std::ostream &os, const SliceLLTestParams &params) {
  std::string str;
  llvm::raw_string_ostream llvm_os(str);
  llvm_os << "shape=" << triton::join(params.shape, "x")
          << ", encoding=" << params.getEncoding();
  os << str;
  return os;
}

class SliceVsLinearLayoutsTest
    : public DistributedLLTest<SliceEncodingAttr, SliceLLTestParams> {};

TEST_P(SliceVsLinearLayoutsTest, DoIt) { DoIt(); }

INSTANTIATE_TEST_SUITE_P(TestCases, SliceVsLinearLayoutsTest,
                         ::testing::ValuesIn(
                             std::vector<SliceLLTestParams>(
                                 {
                                     {
                                         .shape = {128},
                                         .sliceDim = 0,
                                         .parent =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {2, 4},
                                                 .threadsPerWarp = {4, 2},
                                                 .warpsPerCTA = {2, 2},
                                                 .order = {1, 0},
                                                 .CTAsPerCGA = {2, 2},
                                                 .CTASplitNum = {2, 2},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {128},
                                         .sliceDim = 1,
                                         .parent =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {2, 4},
                                                 .threadsPerWarp = {4, 2},
                                                 .warpsPerCTA = {2, 2},
                                                 .order = {1, 0},
                                                 .CTAsPerCGA = {2, 2},
                                                 .CTASplitNum = {2, 2},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },

                                     {
                                         .shape = {32},
                                         .sliceDim = 1,
                                         .parent =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 1},
                                                 .threadsPerWarp = {32, 1},
                                                 .warpsPerCTA = {4, 1},
                                                 .order = {0, 1},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {32},
                                         .sliceDim = 0,
                                         .parent =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 1},
                                                 .threadsPerWarp = {32, 1},
                                                 .warpsPerCTA = {4, 1},
                                                 .order = {0, 1},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {32},
                                         .sliceDim = 1,
                                         .parent =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 4},
                                                 .threadsPerWarp = {8, 4},
                                                 .warpsPerCTA = {2, 2},
                                                 .order = {0, 1},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {32},
                                         .sliceDim = 0,
                                         .parent =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 4},
                                                 .threadsPerWarp = {8, 4},
                                                 .warpsPerCTA = {2, 2},
                                                 .order = {0, 1},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {1},
                                         .sliceDim = 0,
                                         .parent =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 4},
                                                 .threadsPerWarp = {8, 4},
                                                 .warpsPerCTA = {2, 2},
                                                 .order = {0, 1},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },

                                     {
                                         .shape = {16},
                                         .sliceDim = 0,
                                         .parent =
                                             NvidiaMmaLLTestParams{
                                                 .versionMajor = 2,
                                                 .versionMinor = 0,
                                                 .warpsPerCTA = {2, 2},
                                                 .instrShape = {16, 8},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {128},
                                         .sliceDim = 0,
                                         .parent =
                                             NvidiaMmaLLTestParams{
                                                 .versionMajor = 2,
                                                 .versionMinor = 0,
                                                 .warpsPerCTA = {2, 2},
                                                 .instrShape = {16, 8},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {16},
                                         .sliceDim = 1,
                                         .parent =
                                             NvidiaMmaLLTestParams{
                                                 .versionMajor = 2,
                                                 .versionMinor = 0,
                                                 .warpsPerCTA = {2, 2},
                                                 .instrShape = {16, 8},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {128},
                                         .sliceDim = 1,
                                         .parent =
                                             NvidiaMmaLLTestParams{
                                                 .versionMajor = 2,
                                                 .versionMinor = 0,
                                                 .warpsPerCTA = {2, 2},
                                                 .instrShape = {16, 8},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {128},
                                         .sliceDim = 0,
                                         .parent =
                                             NvidiaMmaLLTestParams{
                                                 .versionMajor = 3,
                                                 .versionMinor = 0,
                                                 .warpsPerCTA = {4, 4},
                                                 .instrShape = {16, 16, 16},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                 })));

struct LoadSharedToDistributedLLTestParams {
  std::vector<int64_t> shape;
  unsigned shmemVec;
  unsigned shmemPerPhase;
  unsigned shmemMaxPhase;
  std::vector<unsigned> shmemOrder;
  std::vector<unsigned> shmemCTAsPerCGA;
  std::vector<unsigned> shmemCTASplitNum;
  std::vector<unsigned> shmemCTAOrder;
  bool shmemHasLeadingOffset;
  std::vector<unsigned> shmemStrides;
  std::variant<BlockedLLTestParams, NvidiaMmaLLTestParams> dst;

  Attribute getSrcEncoding() const {
    return SharedEncodingAttr::get(
        getContext(), shmemVec, shmemPerPhase, shmemMaxPhase, shmemOrder,
        CTALayoutAttr::get(getContext(), shmemCTAsPerCGA, shmemCTASplitNum,
                           shmemCTAOrder),
        shmemHasLeadingOffset);
  }

  Attribute getDstEncoding() const {
    return std::visit(
        [&](const auto &dstParams) -> Attribute {
          return dstParams.getEncoding();
        },
        dst);
  }
};

std::ostream &operator<<(std::ostream &os,
                         const LoadSharedToDistributedLLTestParams &params) {
  std::string str;
  llvm::raw_string_ostream llvm_os(str);
  llvm_os << "shape=" << triton::join(params.shape, "x")
          << ", src=" << params.getSrcEncoding()
          << ", dst=" << params.getDstEncoding()
          << ", strides=" << triton::join(params.shmemStrides, "x");
  os << str;
  return os;
}

class LoadSharedToDistributedLLTest
    : public ::testing::TestWithParam<LoadSharedToDistributedLLTestParams> {};

TEST_P(LoadSharedToDistributedLLTest, DoIt) {
  MLIRContext *ctx = getContext();

  auto params = GetParam();

  Attribute srcEncoding = params.getSrcEncoding();
  Attribute dstEncoding = params.getDstEncoding();

  mlir::OpBuilder builder(ctx);
  Location loc = UnknownLoc::get(ctx);
  auto mlirModule = mlir::ModuleOp::create(loc);
  mlirModule->setAttr(
      "triton_gpu.num-warps",
      builder.getI32IntegerAttr(product(getWarpsPerCTA(dstEncoding))));

  auto func = builder.create<mlir::triton::FuncOp>(
      loc, "test_func", builder.getFunctionType({}, {}));
  mlirModule.push_back(func);
  auto *block = func.addEntryBlock();
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPointToStart(block);

  NVIDIA::TargetInfo target(90);

  int rank = params.shape.size();
  Type elemLlvmTy = builder.getI32Type();
  auto dstTy = RankedTensorType::get(params.shape, elemLlvmTy, dstEncoding);
  auto srcTy = MemDescType::get(params.shape, elemLlvmTy, srcEncoding,
                                SharedMemorySpaceAttr::get(ctx));
  Value base = i32_val(1000000);
  SmallVector<Value> shmemStridesValues;
  for (auto stride : params.shmemStrides) {
    shmemStridesValues.push_back(i32_val(stride));
  }
  SharedMemoryObject shmemObj =
      SharedMemoryObject(base, elemLlvmTy, shmemStridesValues,
                         std::vector<Value>(rank, i32_val(0)));

  auto getPtrsAndVecOffs = [&](bool allowLLs,
                               SmallVector<std::pair<Value, int64_t>> &ret) {
    SmallVector<Value> vals = loadSharedToDistributed(
        dstTy, srcTy, elemLlvmTy, shmemObj, loc, rewriter, target, allowLLs);
    for (Value v : vals) {
      auto ee = dyn_cast<LLVM::ExtractElementOp>(v.getDefiningOp());
      ASSERT_TRUE(!!ee);
      auto addr =
          dyn_cast<LLVM::LoadOp>(ee.getVector().getDefiningOp()).getAddr();
      ASSERT_TRUE(!!addr);
      auto pos = dyn_cast<LLVM::ConstantOp>(ee.getPosition().getDefiningOp());
      ASSERT_TRUE(!!pos);
      auto posAttr = dyn_cast<IntegerAttr>(pos.getValue());
      ASSERT_TRUE(!!posAttr);
      ret.push_back({addr, posAttr.getInt()});
    }
  };

  SmallVector<std::pair<Value, int64_t>> ptrsAndVecOffsLegacy;
  getPtrsAndVecOffs(/*allowLLs=*/false, ptrsAndVecOffsLegacy);
  SmallVector<std::pair<Value, int64_t>> ptrsAndVecOffsLL;
  getPtrsAndVecOffs(/*allowLLs=*/true, ptrsAndVecOffsLL);

  // This test takes a long time if we check all indices.  But for linear
  // layouts, we really should only need to check powers of 2.  We wrap the
  // loops in this `iterate` function so we can easily change between checking
  // all indices and just the powers of 2.
  constexpr bool checkAllElems = false;
  bool stopIterating = false;
  auto iterate = [&](int n, auto fn) {
    if (checkAllElems) {
      for (int i = 0; i < n && !stopIterating; i++) {
        fn(i);
      }
    } else {
      if (n > 0 && !stopIterating) {
        fn(0);
      }
      for (int i = 0; (1 << i) < n && !stopIterating; i++) {
        fn(1 << i);
      }
    }
  };

  const int numCTAs = product(getCTAsPerCGA(dstEncoding));
  const int threadsPerCTA = product(getThreadsPerWarp(dstEncoding)) *
                            product(getWarpsPerCTA(dstEncoding));
  int numFailures = 0;
  constexpr int kMaxFailures = 128;
  ASSERT_EQ(ptrsAndVecOffsLegacy.size(), ptrsAndVecOffsLL.size());
  for (int i = 0; i < ptrsAndVecOffsLegacy.size(); i++) {
    SCOPED_TRACE("Register " + std::to_string(i));
    EXPECT_EQ(ptrsAndVecOffsLegacy[i].second, ptrsAndVecOffsLL[i].second);
    iterate(numCTAs, [&](int ctaId) {
      SCOPED_TRACE("CTA " + std::to_string(ctaId));
      iterate(threadsPerCTA, [&](int threadId) {
        SCOPED_TRACE("Thread " + std::to_string(threadId));
        int llValue = evalValue(ptrsAndVecOffsLL[i].first, ctaId, threadId);
        int legacyValue =
            evalValue(ptrsAndVecOffsLegacy[i].first, ctaId, threadId);
        EXPECT_EQ(llValue, legacyValue);
        if (llValue != legacyValue) {
          ++numFailures;
        }
        if (numFailures > kMaxFailures) {
          llvm::errs() << "Too many failures, aborting\n";
          stopIterating = true;
        }
      });
    });
  }
}

INSTANTIATE_TEST_SUITE_P(TestCases, LoadSharedToDistributedLLTest,
                         ::testing::ValuesIn(
                             std::vector<LoadSharedToDistributedLLTestParams>(
                                 {
                                     {
                                         .shape = {128},
                                         .shmemVec = 1,
                                         .shmemPerPhase = 1,
                                         .shmemMaxPhase = 1,
                                         .shmemOrder = {0},
                                         .shmemCTAsPerCGA = {1},
                                         .shmemCTASplitNum = {1},
                                         .shmemCTAOrder = {0},
                                         .shmemHasLeadingOffset = false,
                                         .shmemStrides = {1},
                                         .dst =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1},
                                                 .threadsPerWarp = {32},
                                                 .warpsPerCTA = {4},
                                                 .order = {0},
                                                 .CTAsPerCGA = {1},
                                                 .CTASplitNum = {1},
                                                 .CTAOrder = {0},
                                             },
                                     },
                                     {
                                         .shape = {128, 128},
                                         .shmemVec = 1,
                                         .shmemPerPhase = 1,
                                         .shmemMaxPhase = 1,
                                         .shmemOrder = {0, 1},
                                         .shmemCTAsPerCGA = {1, 1},
                                         .shmemCTASplitNum = {1, 1},
                                         .shmemCTAOrder = {1, 0},
                                         .shmemHasLeadingOffset = false,
                                         .shmemStrides = {128, 1},
                                         .dst =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 1},
                                                 .threadsPerWarp = {8, 4},
                                                 .warpsPerCTA = {4, 4},
                                                 .order = {1, 0},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {128, 128},
                                         .shmemVec = 1,
                                         .shmemPerPhase = 2,
                                         .shmemMaxPhase = 8,
                                         .shmemOrder = {0, 1},
                                         .shmemCTAsPerCGA = {1, 1},
                                         .shmemCTASplitNum = {1, 1},
                                         .shmemCTAOrder = {1, 0},
                                         .shmemHasLeadingOffset = false,
                                         .shmemStrides = {128, 1},
                                         .dst =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 1},
                                                 .threadsPerWarp = {4, 8},
                                                 .warpsPerCTA = {4, 4},
                                                 .order = {1, 0},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {128, 128},
                                         .shmemVec = 1,
                                         .shmemPerPhase = 2,
                                         .shmemMaxPhase = 8,
                                         .shmemOrder = {0, 1},
                                         .shmemCTAsPerCGA = {1, 1},
                                         .shmemCTASplitNum = {1, 1},
                                         .shmemCTAOrder = {1, 0},
                                         .shmemHasLeadingOffset = false,
                                         .shmemStrides = {128, 1},
                                         .dst =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 1},
                                                 .threadsPerWarp = {1, 32},
                                                 .warpsPerCTA = {4, 2},
                                                 .order = {1, 0},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {16, 16},
                                         .shmemVec = 1,
                                         .shmemPerPhase = 1,
                                         .shmemMaxPhase = 1,
                                         .shmemOrder = {1, 0},
                                         .shmemCTAsPerCGA = {1, 1},
                                         .shmemCTASplitNum = {1, 1},
                                         .shmemCTAOrder = {1, 0},
                                         .shmemHasLeadingOffset = false,
                                         .shmemStrides = {1024, 1},
                                         .dst =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 1},
                                                 .threadsPerWarp = {4, 8},
                                                 .warpsPerCTA = {1, 1},
                                                 .order = {1, 0},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {64, 64},
                                         .shmemVec = 1,
                                         .shmemPerPhase = 1,
                                         .shmemMaxPhase = 1,
                                         .shmemOrder = {1, 0},
                                         .shmemCTAsPerCGA = {1, 1},
                                         .shmemCTASplitNum = {1, 1},
                                         .shmemCTAOrder = {1, 0},
                                         .shmemHasLeadingOffset = false,
                                         .shmemStrides = {64, 1},
                                         .dst =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {8, 1},
                                                 .threadsPerWarp = {16, 2},
                                                 .warpsPerCTA = {1, 4},
                                                 .order = {0, 1},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {128, 128},
                                         .shmemVec = 4,
                                         .shmemPerPhase = 4,
                                         .shmemMaxPhase = 2,
                                         .shmemOrder = {0, 1},
                                         .shmemCTAsPerCGA = {1, 1},
                                         .shmemCTASplitNum = {1, 1},
                                         .shmemCTAOrder = {1, 0},
                                         .shmemHasLeadingOffset = true,
                                         // The legacy layout code assumes that
                                         // the strides in the row/col
                                         // dimensions are "dense" and match the
                                         // order.  LLs can handle other
                                         // strides, but obviously it won't
                                         // match the legacy code, so we can't
                                         // test it here.
                                         .shmemStrides = {1, 128},
                                         .dst =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 1},
                                                 .threadsPerWarp = {32, 32},
                                                 .warpsPerCTA = {4, 4},
                                                 .order = {1, 0},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {128, 128},
                                         .shmemVec = 4,
                                         .shmemPerPhase = 4,
                                         .shmemMaxPhase = 2,
                                         .shmemOrder = {1, 0},
                                         .shmemCTAsPerCGA = {1, 1},
                                         .shmemCTASplitNum = {1, 1},
                                         .shmemCTAOrder = {1, 0},
                                         .shmemHasLeadingOffset = true,
                                         .shmemStrides = {128, 1},
                                         .dst =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 1},
                                                 .threadsPerWarp = {32, 32},
                                                 .warpsPerCTA = {4, 4},
                                                 .order = {1, 0},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {128, 128},
                                         .shmemVec = 4,
                                         .shmemPerPhase = 2,
                                         .shmemMaxPhase = 4,
                                         .shmemOrder = {0, 1},
                                         .shmemCTAsPerCGA = {1, 1},
                                         .shmemCTASplitNum = {1, 1},
                                         .shmemCTAOrder = {1, 0},
                                         .shmemHasLeadingOffset = true,
                                         .shmemStrides = {1, 128},
                                         .dst =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 1},
                                                 .threadsPerWarp = {32, 32},
                                                 .warpsPerCTA = {4, 4},
                                                 .order = {1, 0},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                     {
                                         .shape = {128, 128},
                                         .shmemVec = 4,
                                         .shmemPerPhase = 1,
                                         .shmemMaxPhase = 8,
                                         .shmemOrder = {0, 1},
                                         .shmemCTAsPerCGA = {1, 1},
                                         .shmemCTASplitNum = {1, 1},
                                         .shmemCTAOrder = {1, 0},
                                         .shmemHasLeadingOffset = true,
                                         .shmemStrides = {1, 128},
                                         .dst =
                                             BlockedLLTestParams{
                                                 .sizePerThread = {1, 1},
                                                 .threadsPerWarp = {32, 32},
                                                 .warpsPerCTA = {4, 4},
                                                 .order = {1, 0},
                                                 .CTAsPerCGA = {1, 1},
                                                 .CTASplitNum = {1, 1},
                                                 .CTAOrder = {1, 0},
                                             },
                                     },
                                 })));

} // namespace gpu
} // namespace triton
} // namespace mlir

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
