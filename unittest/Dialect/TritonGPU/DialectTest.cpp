#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>

#include "mlir/AsmParser/AsmParser.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Signals.h"

namespace {

template <typename T> std::string stringifyLLVMType(const T &t) {
  std::string str;
  llvm::raw_string_ostream ros(str);
  ros << t;
  return str;
}
} // namespace

namespace mlir {
// gtest printer for mlir::Attribute.  This must live in namespace mlir in order
// for it to be found via ADL.
void PrintTo(const Attribute &attr, std::ostream *os) {
  *os << stringifyLLVMType(attr);
}
} // namespace mlir

namespace mlir::triton::gpu {
namespace {

std::vector<DistributedEncodingTrait>
createDistributedEncodings(MLIRContext &ctx) {
  // Assorted distributed encodings to run tests on
  // Define a tensor shape
  auto rank = 2;
  SmallVector<SmallVector<unsigned>> orders = {{0, 1}, {1, 0}};
  SmallVector<triton::gpu::CTALayoutAttr> ctaLayouts = {
      triton::gpu::CTALayoutAttr::getDefault(&ctx, rank),
      triton::gpu::CTALayoutAttr::get(&ctx, {4, 2}, {2, 2}, {1, 0}),
  };
  std::vector<DistributedEncodingTrait> distributedEncodings;

  // Create blocked and slice(blocked) encodings
  {
    SmallVector<unsigned> sizePerThread = {4, 4};
    SmallVector<unsigned> threadsPerWarp = {4, 8};
    SmallVector<unsigned> warpsPerCTA = {2, 2};

    for (auto ctaLayout : ctaLayouts) {
      for (const auto &order : orders) {
        auto blockedEncoding = triton::gpu::BlockedEncodingAttr::get(
            &ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);
        distributedEncodings.push_back(blockedEncoding);
        distributedEncodings.push_back(
            triton::gpu::SliceEncodingAttr::get(&ctx, 0, blockedEncoding));
      }
    }
  }

  // Create an MMAv2 and DotOperandEncodingAttr (MMAv3 doesn't support linear
  // layouts yet)
  {
    for (auto versionMajor : {2, 3}) {
      unsigned versionMinor = 0;
      auto kWidth = 2;
      SmallVector<unsigned> warpsPerCTA{4, 2};
      auto instrShape = versionMajor == 2 ? SmallVector<unsigned>{16, 8}
                                          : SmallVector<unsigned>{16, 32, 16};
      auto mma = triton::gpu::NvidiaMmaEncodingAttr::get(
          &ctx, versionMajor, versionMinor, warpsPerCTA, ctaLayouts[0],
          instrShape);
      distributedEncodings.push_back(mma);
      // Create an opIdx=0 and opIdx=1 encoding
      for (unsigned opIdx = 0; opIdx < 2; ++opIdx) {
        if (opIdx == 1 && versionMajor == 3) {
          // MMAv3 doesn't support register operand on the rhs
          continue;
        }
        distributedEncodings.push_back(
            triton::gpu::DotOperandEncodingAttr::get(&ctx, opIdx, mma, kWidth));
      }
    }
  }
  return distributedEncodings;
}

std::string strReplace(std::string s, const std::string &from,
                       const std::string &to) {
  size_t start_pos = 0;
  while ((start_pos = s.find(from, start_pos)) != std::string::npos) {
    s.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  return s;
}

// We use some abbreviations when spelling out MLIR types.
std::string expandTyStr(std::string s) {
  s = strReplace(s, "T<", "tensor<");
  s = strReplace(s, "#B", "#ttg.blocked");
  s = strReplace(s, "spt", "sizePerThread");
  s = strReplace(s, "tpw", "threadsPerWarp");
  s = strReplace(s, "wpc", "warpsPerCTA");
  s = strReplace(s, "ord", "order");
  return s;
}

// Advances a multidimensional index.  Returns true if we wrapped around to the
// beginning.
bool advance(MutableArrayRef<unsigned> idx, ArrayRef<unsigned> shape,
             ArrayRef<unsigned> order) {
  for (int dim : order) {
    if (idx[dim] < shape[dim] - 1) {
      idx[dim]++;
      return false;
    }
    idx[dim] = 0;
  }
  return true;
}

// Gets a flat index from a multidimensional index.
int64_t getFlatIdx(ArrayRef<unsigned> idx, ArrayRef<unsigned> shape,
                   ArrayRef<unsigned> order) {
  int64_t flatIdx = 0;
  int64_t stride = 1;
  for (int i = 0; i < idx.size(); i++) {
    flatIdx += idx[order[i]] * stride;
    stride *= shape[order[i]];
  }
  return flatIdx;
}

class InferLayoutTest : public ::testing::Test {
public:
  InferLayoutTest()
      : inferLayout(
            ctx.getOrLoadDialect<TritonGPUDialect>()
                ->getRegisteredInterface<DialectInferLayoutInterface>()) {}

protected:
  static MLIRContext ctx;

  DialectInferLayoutInterface *inferLayout;
};

/*static*/ MLIRContext InferLayoutTest::ctx;

void testReshape(RankedTensorType srcTy, RankedTensorType dstTy,
                 std::optional<BlockedEncodingAttr> expectedDstEnc,
                 DialectInferLayoutInterface *inferLayout,
                 bool longErrors = true) {

  MLIRContext *ctx = srcTy.getContext();

  // Capture any errors from calling inferReshapeNoOpReorderEncoding, so we can
  // print them if we expected the reshape to succeed but it failed.
  std::vector<std::string> diags;
  Attribute inferredEnc;
  LogicalResult result = success();
  {
    ScopedDiagnosticHandler scopedHandler(
        ctx, [&](Diagnostic &diag) { diags.push_back("  - " + diag.str()); });
    result = inferLayout->inferReshapeOpEncoding(
        srcTy.getShape(), srcTy.getEncoding(), dstTy.getShape(), inferredEnc,
        UnknownLoc::get(ctx));
  }

  // We expect the reshape to succeed as long as the inputs have the same
  // number of elements
  EXPECT_TRUE(succeeded(result))
      << "Expected reshape to succeed, but it didn't!  Error(s):\n"
      << join(diags, "\n");

  if (auto expectedEnc = dstTy.getEncoding()) {
    EXPECT_EQ(inferredEnc, expectedEnc);
  }

  // We know that infer(srcShape, srcEnc, dstShape) => dstEnc.  Check that it
  // works the other way around too: infer(dstShape, dstEnc, srcShape) =>
  // srcEnc.  (This is an invariant of the inference function.)
  // Even more, we check that the inferred encoding is structurally the same as
  // the src encoding, showing that the inference is consistent.
  {
    std::vector<std::string> diags;
    ScopedDiagnosticHandler scopedHandler(
        ctx, [&](Diagnostic &diag) { diags.push_back("  - " + diag.str()); });
    Attribute inferredSrcEnc;
    auto result = inferLayout->inferReshapeOpEncoding(
        dstTy.getShape(), inferredEnc, srcTy.getShape(), inferredSrcEnc,
        UnknownLoc::get(ctx));
    EXPECT_TRUE(succeeded(result))
        << "Inverse encoding inference (" << triton::join(dstTy.getShape(), "x")
        << " " << stringifyLLVMType(inferredEnc) << " -> "
        << triton::join(srcTy.getShape(), "x") << "failed:\n"
        << join(diags, "\n");
    auto srcLinear = toLinearLayout(srcTy.getShape(), srcTy.getEncoding());
    auto inferredSrcLinear = toLinearLayout(srcTy.getShape(), inferredSrcEnc);
    EXPECT_EQ(inferredSrcLinear, srcLinear)
        << "Inverse encoding inference (" << triton::join(dstTy.getShape(), "x")
        << " " << stringifyLLVMType(inferredEnc) << " -> "
        << triton::join(srcTy.getShape(), "x")
        << " gave the wrong result.  Expected " << srcLinear.toString()
        << " but "
        << "got " << inferredSrcLinear.toString() << ".\n";
  }

  // The funtional characterisation of resize is that, if we have a srcLayout
  // and a dstLayout, then the flattened layouts are views of the same data
  // when considered as C-contiguous.
  auto makeFlattenedCContig = [](ArrayRef<int64_t> shape, Attribute layout) {
    auto ctx = layout.getContext();
    auto linear = toLinearLayout(shape, layout);
    auto dims = standardOutDimNames(ctx, shape.size());
    std::reverse(dims.begin(), dims.end());
    return linear.transposeOuts(dims).reshapeOuts(
        {{dims.back(), linear.getTotalOutDimSize()}});
  };
  EXPECT_EQ(makeFlattenedCContig(srcTy.getShape(), srcTy.getEncoding()),
            makeFlattenedCContig(dstTy.getShape(), inferredEnc));
}

class InferReshapeOpEncodingTest
    : public InferLayoutTest,
      public ::testing::WithParamInterface<
          std::tuple<std::string /*srcTy*/, std::string /*dstTy*/>> {};

TEST_P(InferReshapeOpEncodingTest, DoIt) {
  std::string srcTyStr = expandTyStr(std::get<0>(GetParam()));
  std::string dstTyStr = expandTyStr(std::get<1>(GetParam()));

  auto src = mlir::parseType(srcTyStr, &ctx);
  if (!src)
    FAIL() << "Could not parse source type: " << srcTyStr;

  auto dst = mlir::parseType(dstTyStr, &ctx);
  if (!dst)
    FAIL() << "Could not parse destination type: " << dstTyStr;

  std::optional<BlockedEncodingAttr> expectedDstEnc;
  if (auto dstEnc = cast<RankedTensorType>(dst).getEncoding()) {
    expectedDstEnc = cast<BlockedEncodingAttr>(dstEnc);
  }

  testReshape(cast<RankedTensorType>(src), cast<RankedTensorType>(dst),
              expectedDstEnc, inferLayout, /*longErrors=*/true);
}

// A testcase of {a, b, c} means:
//  - if `c` is false, check that a reshape from shape+encoding `a` to shape `b`
//    is deemed impossible.
//  - else if `c` is true:
//    - check that a reshape from shape+encoding `a` to shape `b` yields an
//      encoding that makes the reshape a nop, and
//    - if b has an encoding, check that the inferred encoding matches b's.
INSTANTIATE_TEST_SUITE_P(
    Reshapes, InferReshapeOpEncodingTest,
    ::testing::ValuesIn(std::vector<std::tuple<std::string /*srcTy*/,
                                               std::string /*dstTy*/>>({
        // Use raw strings in here so clang-format doesn't try to wrap them.
        {R"(T<128x64xf32, #B<{spt=[1,1], tpw=[1,32], wpc=[1,1], ord=[1,0]}>>)",
         R"(T<8192xf32,   #B<{spt=[1],   tpw=[32],   wpc=[1],   ord=[0]}>>)"},

        {R"(T<128xf32,  #B<{spt=[4],   tpw=[32],   wpc=[1],   ord=[0]}>>)",
         R"(T<32x4xf32, #B<{spt=[1,4], tpw=[32,1], wpc=[1,1], ord=[1,0]}>>)"},

        {R"(T<128xf32,  #B<{spt=[4],   tpw=[32],   wpc=[1],   ord=[0]}>>)",
         R"(T<16x8xf32, #B<{spt=[1,4], tpw=[16,2], wpc=[1,1], ord=[1,0]}>>)"},

        {R"(T<32x32xf32, #B<{spt=[2,2], tpw=[32,1], wpc=[1,1], ord=[1,0]}>>)",
         "T<1024xf32>"},

        {R"(T<32x4xf32,     #B<{spt=[1,4],     tpw=[32,1],     wpc=[1,1],     ord=[1,0]}>>)",
         R"(T<2x16x2x2xf32, #B<{spt=[1,1,2,2], tpw=[2,16,1,1], wpc=[1,1,1,1], ord=[3,2,1,0]}>>)"},

        {R"(T<4x32xf32,     #B<{spt=[4,1],     tpw=[1,32],     wpc=[1,1],     ord=[0,1]}>>)",
         R"(T<2x2x2x16xf32, #B<{spt=[2,2,1,1], tpw=[1,1,2,16], wpc=[1,1,1,1], ord=[1,0,3,2]}>>)"},

        {R"(T<32x32xf32,     #B<{spt=[4,4],     tpw=[4,8],     wpc=[1,1],     ord=[1,0]}>>)",
         R"(T<2x16x2x16xf32, #B<{spt=[1,4,1,4], tpw=[1,4,2,4], wpc=[1,1,1,1], ord=[3,2,1,0]}>>)"},

        {R"(T<32x32xf32,     #B<{spt=[4,4],     tpw=[4,8],     wpc=[1,1],     ord=[1,0]}>>)",
         R"(T<16x2x16x2xf32, #B<{spt=[2,2,2,2], tpw=[4,1,8,1], wpc=[1,1,1,1], ord=[3,2,1,0]}>>)"},

        {R"(T<32x32xf32, #B<{spt=[4,4], tpw=[4,8], wpc=[1,1], ord=[0,1]}>>)",
         R"(T<16x2x16x2xf32>)"},

        // nop reshape, but the block size is 2x larger than the tensor.
        {R"(T<4x2x2x4xf32, #B<{spt=[2,1,1,2], tpw=[2,1,1,2], wpc=[2,2,1,1], ord=[0,3,1,2]}>>)",
         R"(T<4x2x2x4xf32, #B<{spt=[2,1,1,2], tpw=[2,1,1,2], wpc=[2,2,1,1], ord=[0,3,1,2]}>>)"},

        {R"(T<2x4x2x4xf32, #B<{spt=[1,2,2,1], tpw=[1,2,1,2], wpc=[1,2,2,1], ord=[2,1,0,3]}>>)",
         R"(T<4x2x2x4xf32>)"},

        {R"(T<1x2x2x4xf32, #B<{spt=[1,32,4,4], tpw=[4,4,16,16], wpc=[8,8,8,1], ord=[0,1,2,3]}>>)",
         R"(T<2x2x4x1xf32>)"},

        {R"(T<2x2x2x2xf32, #B<{spt=[2,2,2,2], tpw=[1,1,1,1], wpc=[1,1,1,1], ord=[1,0,3,2]}>>)",
         R"(T<4x4xf32>)"},

        {R"(T<16x8xf32, #B<{spt=[1,2], tpw=[2,4], wpc=[2,1], ord=[1,0]}>>)",
         R"(T<128xf32>)"},

        {R"(T<16x1x8xf32, #B<{spt=[8,1,1], tpw=[2,1,1], wpc=[1,1,8], ord=[2,1,0]}>>)",
         R"(T<128x1xf32>)"},

        {R"(T<16x1x8xf32, #B<{spt=[1,1,8], tpw=[2,1,1], wpc=[8,1,1], ord=[2,1,0]}>>)",
         R"(T<128x1xf32>)"},

        {R"(T<32x32xf32, #B<{spt=[1,2], tpw=[1,8], wpc=[1,1], ord=[1,0]}>>)",
         R"(T<1024xf32>)"},

        {R"(T<4x4xf32, #B<{spt=[1,1], tpw=[2,4], wpc=[2,1], ord=[0,1]}>>)",
         R"(T<16xf32>)"},

        {R"(T<32xf32,   #B<{spt=[2],   tpw=[32],   wpc=[2],   ord=[0]}>>)",
         R"(T<16x2xf32, #B<{spt=[1,2], tpw=[32,1], wpc=[2,1], ord=[1,0]}>>)"},

        {R"(T<2x1x2xf32, #B<{spt=[2,1,1], tpw=[2,1,2], wpc=[4,1,8], ord=[2,1,0]}>>)",
         R"(T<2x2xf32,   #B<{spt=[2,1],   tpw=[2,2],   wpc=[4,8],   ord=[1,0]}>>)"},
    })));

class Fp4ToFpOpTest : public ::testing::Test {
public:
  Fp4ToFpOpTest() { ctx.getOrLoadDialect<TritonGPUDialect>(); }

protected:
  MLIRContext ctx;
};

TEST_F(Fp4ToFpOpTest, Fp4ToFpOpLayoutPropagation) {
  SmallVector<SmallVector<int64_t>> shapes = {{64, 128}, {256, 1024}};
  auto distributedEncodings = createDistributedEncodings(ctx);
  auto *inferLayout =
      ctx.getOrLoadDialect<TritonGPUDialect>()
          ->getRegisteredInterface<DialectInferLayoutInterface>();

  for (auto enc : distributedEncodings) {
    for (auto shape : shapes) {
      if (auto sliceEncoding = dyn_cast<triton::gpu::SliceEncodingAttr>(enc)) {
        shape.erase(shape.begin() + sliceEncoding.getDim());
      }
      auto rank = shape.size();
      auto axis = rank - 1;
      // Test that we can do a round trip from src to dst encoding and back.
      Attribute dstEnc;
      LogicalResult result = inferLayout->inferFp4ToFpOpEncoding(
          shape, axis, enc, dstEnc, /*fwdInference=*/true, std::nullopt);
      EXPECT_TRUE(succeeded(result));
      Attribute newSrcEnc;
      auto newShape = shape;
      newShape[axis] *= 2;
      result = inferLayout->inferFp4ToFpOpEncoding(
          newShape, axis, dstEnc, newSrcEnc, /*fwdInference=*/false,
          std::nullopt);
      EXPECT_TRUE(succeeded(result));
      // Structural equality.
      EXPECT_EQ(toLinearLayout(shape, newSrcEnc), toLinearLayout(shape, enc));
      // We'll have equality iff dstEnc is a legacy encoding.
      if (!isa<LinearEncodingAttr>(dstEnc)) {
        EXPECT_EQ(newSrcEnc, enc);
      }
    }
  }
}

class JoinOpTest : public ::testing::Test {
public:
  JoinOpTest() { ctx.getOrLoadDialect<TritonGPUDialect>(); }

protected:
  MLIRContext ctx;
};

TEST_F(JoinOpTest, JoinOpLayoutPropagation) {
  SmallVector<SmallVector<int64_t>> shapes = {{64, 128}, {256, 1024}};
  auto distributedEncodings = createDistributedEncodings(ctx);
  auto *inferLayout =
      ctx.getOrLoadDialect<TritonGPUDialect>()
          ->getRegisteredInterface<DialectInferLayoutInterface>();

  for (auto enc : distributedEncodings) {
    for (auto shape : shapes) {
      if (auto sliceEncoding = dyn_cast<triton::gpu::SliceEncodingAttr>(enc)) {
        shape.erase(shape.begin() + sliceEncoding.getDim());
      }
      auto rank = shape.size();
      // Join only supports Linear or Blocked
      auto linear = LinearEncodingAttr::get(&ctx, toLinearLayout(shape, enc));
      // Test that we can do a round trip from src to dst encoding and back.
      Attribute dstEnc;
      LogicalResult result = inferLayout->inferDefaultJoinOpEncoding(
          linear, dstEnc, shape, std::nullopt);
      EXPECT_TRUE(succeeded(result));
      Attribute newSrcEnc;
      auto newShape = shape;
      newShape.push_back(2);
      result = inferLayout->inferSplitOpEncoding(dstEnc, newSrcEnc, newShape,
                                                 std::nullopt);
      EXPECT_TRUE(succeeded(result));
      // Structural equality.
      EXPECT_EQ(toLinearLayout(shape, newSrcEnc), toLinearLayout(shape, enc));
      // We'll have equality iff dstEnc is a legacy encoding.
      if (!isa<LinearEncodingAttr>(dstEnc)) {
        EXPECT_EQ(newSrcEnc, enc);
      }

      // We test against this decomposition:
      // newShape = shape
      // newShape[axis] *= 2
      // rank = len(shape)
      // transShape = list(range(rank))
      // transShape.insert(axis + 1, rank)
      // join(enc, enc).trans(transShape).reshape(newShape)
      auto axis = rank - 1;
      auto transPerm = llvm::to_vector(llvm::seq<int32_t>(0, rank));
      transPerm.insert(transPerm.begin() + axis + 1, rank);
      Attribute joinedEnc;
      result = inferLayout->inferDefaultJoinOpEncoding(enc, joinedEnc, shape,
                                                       std::nullopt);
      auto joinShape = shape;
      joinShape.push_back(2);
      assert(succeeded(result));
      Attribute transEnc;
      result = inferLayout->inferTransOpEncoding(joinedEnc, joinShape,
                                                 transPerm, transEnc);
      assert(succeeded(result));
      SmallVector<int64_t> transShape;
      for (auto i : transPerm) {
        transShape.push_back(joinShape[i]);
      }
      Attribute reshapedEnc;
      result = inferLayout->inferReshapeOpEncoding(
          transShape, transEnc, newShape, reshapedEnc, std::nullopt);
      assert(succeeded(result));
      // The layouts should be structurally the same
      // but reshapeEnc will likely be a LinearEncodingAttr
      EXPECT_EQ(toLinearLayout(newShape, reshapedEnc),
                toLinearLayout(newShape, dstEnc));
    }
  }
}

class AMDLayoutTest : public ::testing::Test {
public:
  AMDLayoutTest() {
    ctx.getOrLoadDialect<TritonGPUDialect>();
    ctaLayout =
        triton::gpu::CTALayoutAttr::get(&ctx, ctaPerCGA, ctaSplit, ctaOrder);
    f16Ty = Float16Type::get(&ctx);
  }

  triton::gpu::DotOperandEncodingAttr
  createDotOperand(int idx, Attribute parent, int kWidth) {
    return triton::gpu::DotOperandEncodingAttr::get(&ctx, idx, parent, kWidth);
  }

protected:
  MLIRContext ctx;
  const SmallVector<unsigned> ctaPerCGA{1, 1, 1};
  const SmallVector<unsigned> ctaSplit{1, 1, 1};
  const SmallVector<unsigned> ctaOrder{2, 1, 0};
  triton::gpu::CTALayoutAttr ctaLayout;
  Type f16Ty;
};

class AMDMfmaLayoutTest : public AMDLayoutTest {
public:
  AMDMfmaLayoutTest() = default;

  triton::gpu::AMDMfmaEncodingAttr createMFMA(int mDim, int nDim,
                                              ArrayRef<unsigned> warpsPerCTA) {
    return triton::gpu::AMDMfmaEncodingAttr::get(
        &ctx, /*versionMajor=*/2, /*versionMinor=*/0, warpsPerCTA, mDim, nDim,
        /*isTransposed=*/false, ctaLayout);
  }

  triton::gpu::AMDMfmaEncodingAttr
  createTransposedMFMA(int mDim, int nDim, ArrayRef<unsigned> warpsPerCTA) {
    return triton::gpu::AMDMfmaEncodingAttr::get(
        &ctx, /*versionMajor=*/2, /*versionMinor=*/0, warpsPerCTA, mDim, nDim,
        /*isTransposed=*/true, ctaLayout);
  }
};

class LinearEncodingTest : public ::testing::Test {
public:
  LinearEncodingTest() { ctx.getOrLoadDialect<TritonGPUDialect>(); }

protected:
  MLIRContext ctx;
};

TEST_F(LinearEncodingTest, DistributedEncodingToLinearEncoding) {
  // Define a tensor shape
  auto rank = 2;
  SmallVector<SmallVector<int64_t>> shapes = {{64, 128}, {256, 1024}};
  std::vector<DistributedEncodingTrait> distributedEncodings =
      createDistributedEncodings(ctx);

  auto n = distributedEncodings.size();
  for (auto i = 0; i < n; ++i) {
    if (auto blocked = dyn_cast<triton::gpu::BlockedEncodingAttr>(
            distributedEncodings[i])) {
      for (unsigned opIdx = 0; opIdx < 2; ++opIdx) {
        distributedEncodings.push_back(
            triton::gpu::DotOperandEncodingAttr::get(&ctx, opIdx, blocked, 0));
      }
    }
  }

  auto is_dot_op_with_block_parent = [](Attribute layout) {
    auto dot_layout = dyn_cast<triton::gpu::DotOperandEncodingAttr>(layout);
    return dot_layout &&
           isa<triton::gpu::BlockedEncodingAttr>(dot_layout.getParent());
  };

  for (const auto &distributedEncoding : distributedEncodings) {
    for (auto shape : shapes) {
      if (auto sliceEncoding =
              dyn_cast<triton::gpu::SliceEncodingAttr>(distributedEncoding)) {
        shape.erase(shape.begin() + sliceEncoding.getDim());
      }

      // Create LinearEncodingAttr from the LinearLayout
      auto linearLayout = distributedEncoding.toLinearLayout(shape);
      auto linearEncoding =
          triton::gpu::LinearEncodingAttr::get(&ctx, linearLayout);

      // Test that the canonical form of the LinearLayout is indeed canonical
      // by expanding it to the original shape
      auto expandedLL = linearEncoding.toLinearLayout(shape);
      ASSERT_EQ(linearLayout, expandedLL);

      // Test that methods of DistributedEncoding return the same values
      Type eltTy = Float32Type::get(&ctx);

      ASSERT_EQ(distributedEncoding.getTotalElemsPerThread(shape),
                linearEncoding.getTotalElemsPerThread(shape));
      ASSERT_EQ(distributedEncoding.getElemsPerThread(shape),
                linearEncoding.getElemsPerThread(shape));
      if (!is_dot_op_with_block_parent(distributedEncoding)) {
        ASSERT_EQ(distributedEncoding.getRepOrder(),
                  linearEncoding.getRepOrder());
      }

      // block level
      // SliceEncoding is not well-defined for CGAs
      if (!isa<triton::gpu::SliceEncodingAttr>(distributedEncoding)) {
        auto baseEncoding = cast<LayoutEncodingTrait>(distributedEncoding);
        ASSERT_EQ(baseEncoding.getCTASplitNum(),
                  linearEncoding.getCTASplitNum());
        ASSERT_EQ(baseEncoding.getCTAsPerCGA(), baseEncoding.getCTAsPerCGA());
        // If we are not using CGAs, the order is meaningless
        auto useCGA =
            baseEncoding.getCTAsPerCGA() != SmallVector<unsigned>(rank, 1);
        if (useCGA && !is_dot_op_with_block_parent(distributedEncoding)) {
          ASSERT_EQ(baseEncoding.getCTAOrder(), linearEncoding.getCTAOrder());
        }
      }
    }
  }
}
} // namespace
} // namespace mlir::triton::gpu

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
