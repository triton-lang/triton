#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Partitioning.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Signals.h"

template <typename T> static std::string stringifyLLVMType(const T &t) {
  std::string str;
  llvm::raw_string_ostream ros(str);
  ros << t;
  return str;
}

namespace mlir {
// gtest printer for mlir::Attribute.  This must live in namespace mlir in order
// for it to be found via ADL.
static void PrintTo(const Attribute &attr, std::ostream *os) {
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
  SmallVector<triton::gpu::CGAEncodingAttr> cgaLayouts = {
      triton::gpu::CGAEncodingAttr::get1CTALayout(&ctx, rank),
      triton::gpu::CGAEncodingAttr::fromSplitParams(&ctx, {4, 2}, {2, 2},
                                                    {1, 0}),
  };
  std::vector<DistributedEncodingTrait> distributedEncodings;

  // Create blocked and slice(blocked) encodings
  {
    SmallVector<unsigned> sizePerThread = {4, 4};
    SmallVector<unsigned> threadsPerWarp = {4, 8};
    SmallVector<unsigned> warpsPerCTA = {2, 2};

    for (auto cgaLayout : cgaLayouts) {
      for (const auto &order : orders) {
        auto blockedEncoding = triton::gpu::BlockedEncodingAttr::get(
            &ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order, cgaLayout);
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
          &ctx, versionMajor, versionMinor, warpsPerCTA, cgaLayouts[0],
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
        /*allowReorder=*/false, UnknownLoc::get(ctx));
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
        /*allowReorder=*/false, UnknownLoc::get(ctx));
    EXPECT_TRUE(succeeded(result))
        << "Inverse encoding inference (" << triton::join(dstTy.getShape(), "x")
        << " " << stringifyLLVMType(inferredEnc) << " -> "
        << triton::join(srcTy.getShape(), "x") << "failed:\n"
        << join(diags, "\n");
    auto srcLinear = toLinearLayout(srcTy);
    auto inferredSrcLinear = toLinearLayout(srcTy.getShape(), inferredSrcEnc);
    EXPECT_EQ(inferredSrcLinear, srcLinear)
        << "Inverse encoding inference (" << triton::join(dstTy.getShape(), "x")
        << " " << stringifyLLVMType(inferredEnc) << " -> "
        << triton::join(srcTy.getShape(), "x")
        << " gave the wrong result.  Expected " << srcLinear.toString()
        << " but "
        << "got " << inferredSrcLinear.toString() << ".\n";
  }

  // The functional characterisation of resize is that, if we have a srcLayout
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

class ShapePerCTATest : public ::testing::Test {
public:
  ShapePerCTATest() { ctx.getOrLoadDialect<TritonGPUDialect>(); }

protected:
  MLIRContext ctx;
};

TEST_F(ShapePerCTATest, ShapePerCTA) {
  // Equal length
  SmallVector<unsigned> CTASplitNum = {2, 4};
  SmallVector<int64_t> shape = {64, 128};
  auto shapePerCTA = getShapePerCTA(CTASplitNum, shape);
  auto expectedShapePerCTA = SmallVector<int64_t>{32, 32};
  EXPECT_EQ(shapePerCTA.size(), shape.size());
  EXPECT_EQ(shapePerCTA, expectedShapePerCTA);

  // rank(shape) < rank(CTASplitNum)
  CTASplitNum = {2, 4, 8};
  shapePerCTA = getShapePerCTA(CTASplitNum, shape);
  expectedShapePerCTA = SmallVector<int64_t>{16, 16};
  EXPECT_EQ(shapePerCTA.size(), shape.size());
  EXPECT_EQ(shapePerCTA, expectedShapePerCTA);

  // rank(shape) > rank(CTASplitNum)
  CTASplitNum = {2};
  shapePerCTA = getShapePerCTA(CTASplitNum, shape);
  expectedShapePerCTA = SmallVector<int64_t>{64, 64};
  EXPECT_EQ(shapePerCTA.size(), shape.size());
  EXPECT_EQ(shapePerCTA, expectedShapePerCTA);
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
      result = inferLayout->inferTransOpEncoding(
          joinedEnc, joinShape, transPerm, transEnc, /*loc=*/{});
      assert(succeeded(result));
      SmallVector<int64_t> transShape;
      for (auto i : transPerm) {
        transShape.push_back(joinShape[i]);
      }
      Attribute reshapedEnc;
      result = inferLayout->inferReshapeOpEncoding(
          transShape, transEnc, newShape, reshapedEnc,
          /*allowReorder=*/false, std::nullopt);
      assert(succeeded(result));
      // The layouts should be structurally the same
      // but reshapeEnc will likely be a LinearEncodingAttr
      EXPECT_EQ(toLinearLayout(newShape, reshapedEnc),
                toLinearLayout(newShape, dstEnc));
    }
  }
}

class LinearEncodingTest : public ::testing::Test {
public:
  LinearEncodingTest() { ctx.getOrLoadDialect<TritonGPUDialect>(); }

protected:
  MLIRContext ctx;
};

TEST_F(LinearEncodingTest, MemDescIndexInfersRankReducedSharedLayout) {
  auto srcType = dyn_cast<MemDescType>(
      parseType("!ttg.memdesc<8x64xf32, "
                "#ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], "
                "[0, 16], [0, 32], [1, 0], [2, 8], [4, 16]]}, alignment = 8>, "
                "#ttg.shared_memory, mutable>",
                &ctx));
  ASSERT_TRUE(srcType);

  MemDescType resultType;
  ASSERT_TRUE(succeeded(MemDescIndexOp::inferResultType(
      srcType, resultType, UnknownLoc::get(&ctx))));
  EXPECT_EQ(resultType.getShape(), ArrayRef<int64_t>({64}));
  EXPECT_EQ(resultType.getAllocShape(), ArrayRef<int64_t>({8, 64}));

  auto resultEncoding =
      dyn_cast<SharedLinearEncodingAttr>(resultType.getEncoding());
  ASSERT_TRUE(resultEncoding);
  EXPECT_TRUE(resultEncoding.getHasIndexPhase());
  EXPECT_EQ(resultEncoding.getIndexPhaseMask(), 24u);
  auto outDimSizes =
      llvm::to_vector(resultEncoding.getLinearLayout().getOutDimSizes());
  EXPECT_THAT(outDimSizes, testing::ElementsAre(64));
}

TEST_F(LinearEncodingTest, MemDescIndexPreservesSwizzledSharedAddresses) {
  auto srcType = dyn_cast<MemDescType>(
      parseType("!ttg.memdesc<8x64xf32, "
                "#ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, "
                "order = [1, 0]}>, #ttg.shared_memory, mutable>",
                &ctx));
  ASSERT_TRUE(srcType);

  MemDescType resultType;
  ASSERT_TRUE(succeeded(MemDescIndexOp::inferResultType(
      srcType, resultType, UnknownLoc::get(&ctx))));
  auto resultEncoding =
      dyn_cast<SharedLinearEncodingAttr>(resultType.getEncoding());
  ASSERT_TRUE(resultEncoding);
  EXPECT_EQ(resultEncoding.getIndexPhaseMask(), 24u);
  EXPECT_EQ(resultEncoding.getAlignment(),
            cast<SharedEncodingTrait>(srcType.getEncoding()).getAlignment());

  OpBuilder builder(&ctx);
  Block block;
  auto loc = builder.getUnknownLoc();
  Value source = block.addArgument(srcType, loc);
  Value indexValue = block.addArgument(builder.getI32Type(), loc);
  builder.setInsertionPointToStart(&block);
  auto indexOp =
      MemDescIndexOp::create(builder, loc, resultType, source, indexValue);
  auto offsetLayout = indexOp.getLogicalIndexOffsetLayout();
  auto srcInverse = toLinearLayout(srcType).invert();
  auto dstLayout = toLinearLayout(resultType);
  auto dstInverse = dstLayout.invert();
  auto srcDims = standardOutDimNames(&ctx, srcType.getRank());
  auto dstDims = standardOutDimNames(&ctx, resultType.getRank());
  auto kOffset = StringAttr::get(&ctx, "offset");
  auto kBlock = StringAttr::get(&ctx, "block");

  for (int32_t index = 0; index < srcType.getDimSize(0); ++index) {
    auto offsets = offsetLayout.apply({{srcDims[0], index}, {srcDims[1], 0}});
    ASSERT_EQ(offsets[0].first, StringAttr::get(&ctx, "base"));
    ASSERT_EQ(offsets[1].first, StringAttr::get(&ctx, "phase"));
    int32_t high = offsets[0].second;
    int32_t low = offsets[1].second;

    auto resultLogicalOffset =
        dstLayout.apply({{kOffset, low}, {kBlock, 0}})[0].second;
    auto recoveredLow =
        dstInverse.apply({{dstDims[0], resultLogicalOffset}})[0].second;
    ASSERT_EQ(recoveredLow, low);

    for (int32_t coord = 0; coord < resultType.getDimSize(0); ++coord) {
      int32_t srcOffset =
          srcInverse.apply({{srcDims[0], index}, {srcDims[1], coord}})[0]
              .second;
      int32_t dstOffset = dstInverse.apply({{dstDims[0], coord}})[0].second;
      EXPECT_EQ(srcOffset, high + (dstOffset ^ low))
          << "index=" << index << ", coord=" << coord;
    }
  }
}

TEST_F(LinearEncodingTest, MemDescIndexDistinguishesSyntheticBufferDimension) {
  auto srcType = dyn_cast<MemDescType>(
      parseType("!ttg.memdesc<2x8x64xf32, "
                "#ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], "
                "[0, 16], [0, 32], [1, 0], [2, 8], [4, 16]]}, alignment = 8>, "
                "#ttg.shared_memory, mutable>",
                &ctx));
  ASSERT_TRUE(srcType);

  MemDescType resultType;
  ASSERT_TRUE(succeeded(MemDescIndexOp::inferResultType(
      srcType, resultType, UnknownLoc::get(&ctx))));
  EXPECT_EQ(resultType.getShape(), ArrayRef<int64_t>({8, 64}));
  EXPECT_EQ(resultType.getAllocShape(), ArrayRef<int64_t>({8, 64}));
  EXPECT_EQ(resultType.getEncoding(), srcType.getEncoding());
  EXPECT_FALSE(MemDescIndexOp::hasLogicalSharedIndexProvenance(resultType));
}

TEST_F(LinearEncodingTest, MemDescIndexPreservesZeroPhaseProvenance) {
  auto srcType = dyn_cast<MemDescType>(
      parseType("!ttg.memdesc<8x64xf32, "
                "#ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], "
                "[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]]}, alignment = 8>, "
                "#ttg.shared_memory, mutable>",
                &ctx));
  ASSERT_TRUE(srcType);

  MemDescType resultType;
  ASSERT_TRUE(succeeded(MemDescIndexOp::inferResultType(
      srcType, resultType, UnknownLoc::get(&ctx))));
  auto resultEncoding =
      dyn_cast<SharedLinearEncodingAttr>(resultType.getEncoding());
  ASSERT_TRUE(resultEncoding);
  EXPECT_TRUE(resultEncoding.getHasIndexPhase());
  EXPECT_EQ(resultEncoding.getIndexPhaseMask(), 0u);
  EXPECT_TRUE(MemDescIndexOp::hasLogicalSharedIndexProvenance(resultType));
}

TEST_F(LinearEncodingTest, MemDescIndexSingletonLayoutRoundtrip) {
  auto kOffset = StringAttr::get(&ctx, "offset");
  auto kBlock = StringAttr::get(&ctx, "block");
  for (unsigned rank : {1u, 2u, 3u}) {
    SCOPED_TRACE(rank);
    SmallVector<int64_t> srcShape(rank + 1, 1);
    srcShape[0] = 2;
    std::vector<int32_t> basis(rank + 1, 0);
    basis[0] = 1;
    LinearLayout layout({{kOffset, {basis}}, {kBlock, {}}},
                        standardOutDimNames(&ctx, rank + 1));
    auto encoding = SharedLinearEncodingAttr::get(&ctx, layout, 16);
    auto srcType = MemDescType::get(srcShape, Float32Type::get(&ctx), encoding,
                                    SharedMemorySpaceAttr::get(&ctx), true);

    MemDescType resultType;
    ASSERT_TRUE(succeeded(MemDescIndexOp::inferResultType(
        srcType, resultType, UnknownLoc::get(&ctx))));
    auto resultEncoding =
        cast<SharedLinearEncodingAttr>(resultType.getEncoding());
    EXPECT_EQ(resultEncoding.getRank(), rank);
    EXPECT_EQ(resultEncoding.getLinearLayout().getTotalInDimSizeLog2(), 0);
    EXPECT_TRUE(resultEncoding.getHasIndexPhase());
    EXPECT_EQ(resultEncoding.getIndexPhaseMask(), 0u);
    EXPECT_EQ(parseType(stringifyLLVMType(resultType), &ctx), resultType);
  }
}

TEST_F(LinearEncodingTest, ReplaceUsesSingletonMemDescIndexRoundtrip) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect, TritonGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #synthetic = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
    #logical = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
    #smem = #ttg.shared_memory
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func @singleton_index(%replacement: !ttg.memdesc<2x1xf32, #logical, #smem, mutable>, %index: i32) {
        %copy = ttg.local_alloc : () -> !ttg.memdesc<2x1xf32, #synthetic, #smem, mutable>
        %view = ttg.memdesc_index %copy[%index] : !ttg.memdesc<2x1xf32, #synthetic, #smem, mutable> -> !ttg.memdesc<1xf32, #synthetic, #smem, mutable>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto func = module->lookupSymbol<triton::FuncOp>("singleton_index");
  LocalAllocOp copy;
  func.walk([&](LocalAllocOp op) { copy = op; });
  ASSERT_TRUE(copy);

  OpBuilder builder(&context);
  bool replaced =
      replaceUsesAndPropagateType(builder, copy, func.getArgument(0));
  ASSERT_TRUE(replaced);
  ASSERT_TRUE(succeeded(verify(*module)));

  MemDescIndexOp index;
  func.walk([&](MemDescIndexOp op) { index = op; });
  ASSERT_TRUE(index);
  EXPECT_EQ(index.getSrc(), func.getArgument(0));
  auto encoding = cast<SharedLinearEncodingAttr>(index.getType().getEncoding());
  EXPECT_EQ(encoding.getRank(), 1u);
  EXPECT_TRUE(encoding.getHasIndexPhase());
  EXPECT_EQ(encoding.getIndexPhaseMask(), 0u);

  std::string serialized;
  llvm::raw_string_ostream stream(serialized);
  module->print(stream);
  auto roundtrip = parseSourceString<ModuleOp>(serialized, &context);
  ASSERT_TRUE(roundtrip);
  EXPECT_TRUE(succeeded(verify(*roundtrip)));
  roundtrip->walk(
      [&](MemDescIndexOp op) { EXPECT_EQ(op.getType(), index.getType()); });
}

TEST_F(LinearEncodingTest, LogicalIndexPhaseRejectsDynamicParentDimension) {
  auto validType = dyn_cast<MemDescType>(parseType(
      "!ttg.memdesc<64xf32, "
      "#ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, "
      "alignment = 8, hasIndexPhase = true, indexPhaseMask = 0>, "
      "#ttg.shared_memory, mutable, 2x64>",
      &ctx));
  ASSERT_TRUE(validType);

  SmallVector<int64_t> allocShape = {ShapedType::kDynamic, 64};
  std::vector<std::string> diagnostics;
  MemDescType dynamicType;
  {
    ScopedDiagnosticHandler handler(
        &ctx, [&](Diagnostic &diag) { diagnostics.push_back(diag.str()); });
    dynamicType = MemDescType::getChecked(
        [&]() { return emitError(UnknownLoc::get(&ctx)); }, &ctx,
        validType.getShape(), validType.getElementType(),
        validType.getEncoding(), validType.getMemorySpace(),
        validType.getMutableMemory(), ArrayRef<int64_t>(allocShape));
  }

  EXPECT_FALSE(dynamicType);
  EXPECT_THAT(diagnostics,
              testing::Contains(testing::HasSubstr(
                  "leading allocation dimension must be a positive power of "
                  "two")));
}

TEST_F(LinearEncodingTest, ReplaceUsesRejectsNestedLogicalMemDescIndex) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect, TritonGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
    #plain = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 8], [4, 16]]}, alignment = 8>
    #indexed = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8, hasIndexPhase = true, indexPhaseMask = 24>
    #replacement = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]]}, alignment = 8, hasIndexPhase = true, indexPhaseMask = 0>
    #smem = #ttg.shared_memory
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func @reject_nested_index(
          %replacement: !ttg.memdesc<8x64xf32, #replacement, #smem, mutable, 2x8x64>,
          %plain_replacement: !ttg.memdesc<8x64xf32, #plain, #smem, mutable>,
          %index: i32) {
        %copy = ttg.local_alloc : () -> !ttg.memdesc<8x64xf32, #plain, #smem, mutable>
        %view = ttg.memdesc_index %copy[%index] : !ttg.memdesc<8x64xf32, #plain, #smem, mutable> -> !ttg.memdesc<64xf32, #indexed, #smem, mutable, 8x64>
        %value = ttg.local_load %view : !ttg.memdesc<64xf32, #indexed, #smem, mutable, 8x64> -> tensor<64xf32, #blocked>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  auto func = module->lookupSymbol<triton::FuncOp>("reject_nested_index");
  ASSERT_TRUE(func);
  LocalAllocOp copy;
  func.walk([&](LocalAllocOp op) { copy = op; });
  ASSERT_TRUE(copy);

  OpBuilder builder(&context);
  EXPECT_FALSE(replaceUsesAndPropagateType(builder, copy, func.getArgument(0)));

  SmallVector<MemDescIndexOp> indexes;
  func.walk([&](MemDescIndexOp op) { indexes.push_back(op); });
  ASSERT_EQ(indexes.size(), 1u);
  EXPECT_EQ(indexes.front().getSrc(), copy.getResult());
  EXPECT_TRUE(succeeded(verify(*module)));

  EXPECT_TRUE(replaceUsesAndPropagateType(builder, copy, func.getArgument(1)));
  indexes.clear();
  func.walk([&](MemDescIndexOp op) { indexes.push_back(op); });
  ASSERT_EQ(indexes.size(), 1u);
  EXPECT_EQ(indexes.front().getSrc(), func.getArgument(1));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(LinearEncodingTest, ReplaceUsesRejectsUnsupportedPhaseConsumer) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect, TritonGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
    #plain = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8>
    #phase = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8, hasIndexPhase = true, indexPhaseMask = 0>
    #smem = #ttg.shared_memory
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func @reject_unsupported_consumer(
          %replacement: !ttg.memdesc<64xf32, #phase, #smem, mutable, 2x64>,
          %src: tensor<64x!tt.ptr<f32>, #blocked>) {
        %copy = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #plain, #smem, mutable>
        %token = ttg.async_copy_global_to_local %src, %copy : tensor<64x!tt.ptr<f32>, #blocked> -> <64xf32, #plain, #smem, mutable>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  auto func =
      module->lookupSymbol<triton::FuncOp>("reject_unsupported_consumer");
  ASSERT_TRUE(func);
  LocalAllocOp copy;
  AsyncCopyGlobalToLocalOp asyncCopy;
  func.walk([&](LocalAllocOp op) { copy = op; });
  func.walk([&](AsyncCopyGlobalToLocalOp op) { asyncCopy = op; });
  ASSERT_TRUE(copy && asyncCopy);

  OpBuilder builder(&context);
  EXPECT_FALSE(replaceUsesAndPropagateType(builder, copy, func.getArgument(0)));
  EXPECT_EQ(asyncCopy.getResult(), copy.getResult());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(LinearEncodingTest, ReplaceUsesRejectsLogicalIndexPhaseThroughReshape) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect, TritonGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #plain = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8>
    #zero_phase = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8, hasIndexPhase = true, indexPhaseMask = 0>
    #phase = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8, hasIndexPhase = true, indexPhaseMask = 24>
    #smem = #ttg.shared_memory
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func @reject_phase_reshape(
          %zero_phase: !ttg.memdesc<64xf32, #zero_phase, #smem, mutable, 2x64>,
          %phase: !ttg.memdesc<64xf32, #phase, #smem, mutable, 8x64>,
          %plain: !ttg.memdesc<64xf32, #plain, #smem, mutable>) {
        %copy = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #plain, #smem, mutable>
        %view = ttg.memdesc_reshape %copy : !ttg.memdesc<64xf32, #plain, #smem, mutable> -> !ttg.memdesc<64xf32, #plain, #smem, mutable>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  auto func = module->lookupSymbol<triton::FuncOp>("reject_phase_reshape");
  ASSERT_TRUE(func);
  LocalAllocOp copy;
  MemDescReshapeOp reshape;
  func.walk([&](LocalAllocOp op) { copy = op; });
  func.walk([&](MemDescReshapeOp op) { reshape = op; });
  ASSERT_TRUE(copy && reshape);

  OpBuilder builder(&context);
  for (Value replacement : func.getArguments().take_front(2)) {
    bool replaced = replaceUsesAndPropagateType(builder, copy, replacement);
    ASSERT_FALSE(replaced);
    EXPECT_EQ(reshape.getSrc(), copy.getResult());
    EXPECT_TRUE(succeeded(verify(*module)));
  }

  bool replaced =
      replaceUsesAndPropagateType(builder, copy, func.getArgument(2));
  EXPECT_TRUE(replaced);
  func.walk([&](MemDescReshapeOp op) {
    EXPECT_EQ(op.getSrc(), func.getArgument(2));
  });
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(LinearEncodingTest, ReplaceUsesRejectsInvalidPropagatedSubslice) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect, TritonGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #synthetic = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0]]}, alignment = 8>
    #logical = #ttg.shared_linear<{offset = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 1, 0], [0, 2, 0], [0, 4, 0], [1, 0, 0], [2, 0, 0]]}, alignment = 8>
    #smem = #ttg.shared_memory
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func @reject_invalid_subslice(
          %replacement: !ttg.memdesc<4x8x16xf32, #logical, #smem, mutable>) {
        %copy = ttg.local_alloc : () -> !ttg.memdesc<4x8x16xf32, #synthetic, #smem, mutable>
        %view = ttg.memdesc_subslice %copy[1, 0, 0] : !ttg.memdesc<4x8x16xf32, #synthetic, #smem, mutable> -> !ttg.memdesc<2x8x16xf32, #synthetic, #smem, mutable, 4x8x16>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  auto func = module->lookupSymbol<triton::FuncOp>("reject_invalid_subslice");
  ASSERT_TRUE(func);
  LocalAllocOp copy;
  MemDescSubsliceOp subslice;
  func.walk([&](LocalAllocOp op) { copy = op; });
  func.walk([&](MemDescSubsliceOp op) { subslice = op; });
  ASSERT_TRUE(copy && subslice);

  OpBuilder builder(&context);
  EXPECT_FALSE(replaceUsesAndPropagateType(builder, copy, func.getArgument(0)));
  SmallVector<MemDescSubsliceOp> subslices;
  func.walk([&](MemDescSubsliceOp op) { subslices.push_back(op); });
  ASSERT_EQ(subslices.size(), 1u);
  EXPECT_EQ(subslices.front().getSrc(), copy.getResult());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(LinearEncodingTest,
       ReplaceUsesRejectsInferredIndexPhaseInAutomaticPartition) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect, TritonGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #synthetic = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8>
    #logical = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 8], [4, 16]]}, alignment = 8>
    #smem = #ttg.shared_memory
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func @reject_partitioned_index(
          %replacement: !ttg.memdesc<8x64xf32, #logical, #smem, mutable>, %index: i32) {
        %copy = ttg.local_alloc : () -> !ttg.memdesc<8x64xf32, #synthetic, #smem, mutable>
        %view = ttg.memdesc_index %copy[%index] {ttg.partition = array<i32: 1>} : !ttg.memdesc<8x64xf32, #synthetic, #smem, mutable> -> !ttg.memdesc<64xf32, #synthetic, #smem, mutable>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto func = module->lookupSymbol<triton::FuncOp>("reject_partitioned_index");
  ASSERT_TRUE(func);
  LocalAllocOp copy;
  MemDescIndexOp index;
  func.walk([&](LocalAllocOp op) { copy = op; });
  func.walk([&](MemDescIndexOp op) { index = op; });
  ASSERT_TRUE(copy && index);
  auto oldIndexType = index.getType();
  // An unused result ensures that checking only downstream consumers cannot
  // catch the new phase on the partitioned index itself.
  ASSERT_TRUE(index->use_empty());

  OpBuilder builder(&context);
  bool replaced =
      replaceUsesAndPropagateType(builder, copy, func.getArgument(0));
  ASSERT_FALSE(replaced);
  EXPECT_EQ(index.getSrc(), copy.getResult());
  EXPECT_EQ(index.getType(), oldIndexType);
  EXPECT_TRUE(succeeded(verify(*module)));

  index->removeAttr(kPartitionAttrName);
  replaced = replaceUsesAndPropagateType(builder, copy, func.getArgument(0));
  ASSERT_TRUE(replaced);
  func.walk([&](MemDescIndexOp op) { index = op; });
  EXPECT_EQ(index.getSrc(), func.getArgument(0));
  auto encoding = cast<SharedLinearEncodingAttr>(index.getType().getEncoding());
  EXPECT_TRUE(encoding.getHasIndexPhase());
  EXPECT_EQ(encoding.getIndexPhaseMask(), 24u);
  EXPECT_TRUE(succeeded(verify(*module)));
}

static OwningOpRef<ModuleOp>
parseWarpSpecializeMemDescIndexModule(MLIRContext &context) {
  return parseSourceString<ModuleOp>(R"mlir(
    #plain = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 8], [4, 16]]}, alignment = 8>
    #indexed = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8, hasIndexPhase = true, indexPhaseMask = 24>
    #replacement = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]]}, alignment = 8, hasIndexPhase = true, indexPhaseMask = 0>
    #smem = #ttg.shared_memory
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func @through_warp_specialize(
          %replacement: !ttg.memdesc<8x64xf32, #replacement, #smem, mutable, 2x8x64>,
          %plain_replacement: !ttg.memdesc<8x64xf32, #plain, #smem>,
          %index: i32) {
        %copy = ttg.local_alloc : () -> !ttg.memdesc<8x64xf32, #plain, #smem, mutable>
        ttg.warp_specialize(%copy, %index)
        default {
          ttg.warp_yield
        }
        partition0(%capture: !ttg.memdesc<8x64xf32, #plain, #smem, mutable>, %partition_index: i32) num_warps(4) {
          %view = ttg.memdesc_index %capture[%partition_index] : !ttg.memdesc<8x64xf32, #plain, #smem, mutable> -> !ttg.memdesc<64xf32, #indexed, #smem, mutable, 8x64>
          ttg.warp_return
        } : (!ttg.memdesc<8x64xf32, #plain, #smem, mutable>, i32) -> ()
        tt.return
      }
    }
  )mlir",
                                     &context);
}

static OwningOpRef<ModuleOp>
parseWarpSpecializePhaseConsumerModule(MLIRContext &context) {
  return parseSourceString<ModuleOp>(R"mlir(
    #blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    #plain = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8>
    #phase = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8, hasIndexPhase = true, indexPhaseMask = 0>
    #smem = #ttg.shared_memory
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func @phase_through_warp_specialize(
          %replacement: !ttg.memdesc<64xf32, #phase, #smem, mutable, 2x64>) {
        %copy = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #plain, #smem, mutable>
        ttg.warp_specialize(%copy)
        default {
          ttg.warp_yield
        }
        partition0(%capture: !ttg.memdesc<64xf32, #plain, #smem, mutable>) num_warps(4) {
          %value = ttg.local_load %capture : !ttg.memdesc<64xf32, #plain, #smem, mutable> -> tensor<64xf32, #blocked>
          ttg.warp_return
        } : (!ttg.memdesc<64xf32, #plain, #smem, mutable>) -> ()
        tt.return
      }
    }
  )mlir",
                                     &context);
}

TEST_F(LinearEncodingTest,
       ReplaceUsesPropagatesLogicalIndexPhaseThroughWarpSpecialize) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect, TritonGPUDialect>();
  auto module = parseWarpSpecializePhaseConsumerModule(context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  auto func =
      module->lookupSymbol<triton::FuncOp>("phase_through_warp_specialize");
  ASSERT_TRUE(func);
  LocalAllocOp copy;
  WarpSpecializePartitionsOp partitions;
  LocalLoadOp load;
  func.walk([&](LocalAllocOp op) { copy = op; });
  func.walk([&](WarpSpecializePartitionsOp op) { partitions = op; });
  func.walk([&](LocalLoadOp op) { load = op; });
  ASSERT_TRUE(copy && partitions && load);

  OpBuilder builder(&context);
  EXPECT_TRUE(replaceUsesAndPropagateType(builder, copy, func.getArgument(0)));
  EXPECT_EQ(partitions.getExplicitCaptures().front(), func.getArgument(0));
  auto partitionArg = partitions->getRegion(0).getArgument(0);
  EXPECT_EQ(partitionArg.getType(), func.getArgument(0).getType());
  EXPECT_EQ(load.getSrc(), partitionArg);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(LinearEncodingTest,
       ReplaceUsesRejectsIndexPhaseInAutomaticPartitionConsumer) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect, TritonGPUDialect>();
  auto module = parseWarpSpecializePhaseConsumerModule(context);
  ASSERT_TRUE(module);
  auto func =
      module->lookupSymbol<triton::FuncOp>("phase_through_warp_specialize");
  ASSERT_TRUE(func);
  LocalAllocOp copy;
  WarpSpecializePartitionsOp partitions;
  LocalLoadOp load;
  func.walk([&](LocalAllocOp op) { copy = op; });
  func.walk([&](WarpSpecializePartitionsOp op) { partitions = op; });
  func.walk([&](LocalLoadOp op) { load = op; });
  ASSERT_TRUE(copy && partitions && load);

  OpBuilder builder(&context);
  load->setAttr(kPartitionAttrName, builder.getDenseI32ArrayAttr({1}));
  ASSERT_TRUE(succeeded(verify(*module)));
  auto capture = partitions->getRegion(0).getArgument(0);
  auto oldCaptureType = capture.getType();

  // The replacement has zero phase mask, but its provenance must still be
  // rejected before changing either the capture or the nested consumer.
  bool replaced =
      replaceUsesAndPropagateType(builder, copy, func.getArgument(0));
  ASSERT_FALSE(replaced);
  EXPECT_EQ(partitions.getExplicitCaptures().front(), copy.getResult());
  EXPECT_EQ(capture.getType(), oldCaptureType);
  EXPECT_EQ(load.getSrc(), capture);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(LinearEncodingTest,
       ReplaceUsesRejectsNestedLogicalMemDescIndexInWarpSpecialize) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect, TritonGPUDialect>();
  auto module = parseWarpSpecializeMemDescIndexModule(context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  auto func = module->lookupSymbol<triton::FuncOp>("through_warp_specialize");
  ASSERT_TRUE(func);
  LocalAllocOp copy;
  WarpSpecializePartitionsOp partitions;
  MemDescIndexOp index;
  func.walk([&](LocalAllocOp op) { copy = op; });
  func.walk([&](WarpSpecializePartitionsOp op) { partitions = op; });
  func.walk([&](MemDescIndexOp op) { index = op; });
  ASSERT_TRUE(copy && partitions && index);

  OpBuilder builder(&context);
  EXPECT_FALSE(replaceUsesAndPropagateType(builder, copy, func.getArgument(0)));
  EXPECT_EQ(partitions.getExplicitCaptures().front(), copy.getResult());
  EXPECT_EQ(partitions->getRegion(0).getArgument(0).getType(), copy.getType());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(LinearEncodingTest,
       ReplaceUsesPropagatesMemDescIndexThroughWarpSpecialize) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect, TritonGPUDialect>();
  auto module = parseWarpSpecializeMemDescIndexModule(context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  auto func = module->lookupSymbol<triton::FuncOp>("through_warp_specialize");
  ASSERT_TRUE(func);
  LocalAllocOp copy;
  WarpSpecializePartitionsOp partitions;
  MemDescIndexOp index;
  func.walk([&](LocalAllocOp op) { copy = op; });
  func.walk([&](WarpSpecializePartitionsOp op) { partitions = op; });
  func.walk([&](MemDescIndexOp op) { index = op; });
  ASSERT_TRUE(copy && partitions && index);

  OpBuilder builder(&context);
  EXPECT_TRUE(replaceUsesAndPropagateType(builder, copy, func.getArgument(1)));
  EXPECT_EQ(partitions.getExplicitCaptures().front(), func.getArgument(1));
  auto partitionArg = partitions->getRegion(0).getArgument(0);
  EXPECT_EQ(partitionArg.getType(), func.getArgument(1).getType());

  MemDescType expectedType;
  ASSERT_TRUE(succeeded(MemDescIndexOp::inferResultType(
      cast<MemDescType>(partitionArg.getType()), expectedType)));
  SmallVector<MemDescIndexOp> indexes;
  func.walk([&](MemDescIndexOp op) { indexes.push_back(op); });
  ASSERT_EQ(indexes.size(), 1u);
  EXPECT_EQ(indexes.front().getSrc(), partitionArg);
  EXPECT_EQ(indexes.front().getType(), expectedType);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(LinearEncodingTest, ReplaceUsesRejectsWarpYieldResultTypeChange) {
  MLIRContext context;
  context.loadDialect<triton::TritonDialect, TritonGPUDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #plain = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8>
    #phase = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]]}, alignment = 8, hasIndexPhase = true, indexPhaseMask = 24>
    #smem = #ttg.shared_memory
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
      tt.func @preserve_warp_yield_types(
          %phase: !ttg.memdesc<64xf32, #phase, #smem, mutable, 8x64>,
          %plain: !ttg.memdesc<64xf32, #plain, #smem, mutable>, %marker: i32) {
        %scalar, %desc = ttg.warp_specialize()
        default {
          %copy = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #plain, #smem, mutable>
          ttg.warp_yield %marker, %copy : i32, !ttg.memdesc<64xf32, #plain, #smem, mutable>
        }
        partition0() num_warps(4) {
          ttg.warp_return
        } : () -> (i32, !ttg.memdesc<64xf32, #plain, #smem, mutable>)
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto func = module->lookupSymbol<triton::FuncOp>("preserve_warp_yield_types");
  ASSERT_TRUE(func);
  LocalAllocOp copy;
  WarpYieldOp yield;
  func.walk([&](LocalAllocOp op) { copy = op; });
  func.walk([&](WarpYieldOp op) { yield = op; });
  ASSERT_TRUE(copy && yield);

  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);
  OpBuilder builder(&context);
  bool replaced =
      replaceUsesAndPropagateType(builder, copy, func.getArgument(0));
  ASSERT_FALSE(replaced);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  EXPECT_EQ(before, after);
  EXPECT_TRUE(succeeded(verify(*module)));

  replaced = replaceUsesAndPropagateType(builder, copy, func.getArgument(1));
  ASSERT_TRUE(replaced);
  EXPECT_EQ(yield.getOperand(0), func.getArgument(2));
  EXPECT_EQ(yield.getOperand(1), func.getArgument(1));
  EXPECT_EQ(yield.getParentOp().getResultTypes()[1],
            func.getArgument(1).getType());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(LinearEncodingTest, MaybeLinearToCGAEncodingAttr) {
  auto s = [&](StringRef name) { return StringAttr::get(&ctx, name); };
  auto dim0 = s("dim0"), dim1 = s("dim1");
  LinearLayout cga({{s("block"), {{1, 0}, {0, 1}}}}, {dim0, dim1});
  LinearLayout cta({{s("offset"), {{1, 0}, {2, 0}, {1, 1}, {0, 2}}}},
                   {dim0, dim1});
  auto result = maybeLinearToCGAEncodingAttr(cta * cga);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->getLinearLayout(), cga);
}

TEST_F(LinearEncodingTest, MaybeLinearToCGAEncodingAttrRejectsInvalidFactors) {
  auto s = [&](StringRef name) { return StringAttr::get(&ctx, name); };
  // A block bit lies below a register bit, so no CTA * CGA factorization
  // exists.
  LinearLayout interleaved({{s("register"), {{1}, {4}}}, {s("block"), {{2}}}},
                           {s("dim0")});
  EXPECT_TRUE(failed(maybeLinearToCGAEncodingAttr(interleaved)));

  // A block basis overlaps the intra-CTA bits, so division fails.
  LinearLayout overlapping({{s("offset"), {{1}}}, {s("block"), {{3}}}},
                           {s("dim0")});
  EXPECT_TRUE(failed(maybeLinearToCGAEncodingAttr(overlapping)));

  // Division succeeds, but the quotient is not a valid CGA encoding.
  LinearLayout swizzled({{s("offset"), {{1}}}, {s("block"), {{6}, {4}}}},
                        {s("dim0")});
  EXPECT_TRUE(failed(maybeLinearToCGAEncodingAttr(swizzled)));
}

TEST_F(LinearEncodingTest, MaybeLinearToCGAEncodingAttrPreservesBroadcastBits) {
  auto s = [&](StringRef name) { return StringAttr::get(&ctx, name); };
  LinearLayout layout(
      {{s("register"), {{1}, {2}, {4}}}, {s("block"), {{0}, {8}, {0}, {16}}}},
      {s("dim0")});
  auto result = maybeLinearToCGAEncodingAttr(layout);
  ASSERT_TRUE(succeeded(result));
  LinearLayout expected({{s("block"), {{0}, {1}, {0}, {2}}}}, {s("dim0")});
  EXPECT_EQ(result->getLinearLayout(), expected);
}

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
        auto baseCGALayout = baseEncoding.getCGALayout();
        auto linearCGALayout = linearEncoding.getCGALayout();
        ASSERT_EQ(baseCGALayout.getCTASplitNum(),
                  linearCGALayout.getCTASplitNum());
        ASSERT_EQ(baseCGALayout.getCTAsPerCGA(),
                  linearCGALayout.getCTAsPerCGA());
        // If we are not using CGAs, the order is meaningless
        auto useCGA =
            baseCGALayout.getCTAsPerCGA() != SmallVector<unsigned>(rank, 1);
        if (useCGA && !is_dot_op_with_block_parent(distributedEncoding)) {
          ASSERT_EQ(baseCGALayout.getCTAOrder(), linearCGALayout.getCTAOrder());
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
