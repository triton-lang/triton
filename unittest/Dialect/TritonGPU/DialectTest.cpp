#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>

#include "mlir/AsmParser/AsmParser.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
  s = strReplace(s, "#B", "#triton_gpu.blocked");
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

// Represents the many indices of one element of a tensor with a
// BlockedEncoding.
//
// The purpose of this class is we can say, if two MultiIdx's have the same
// flatFoo values before and after a reshape, then the same GPU thread contains
// the same element (and the reshape is a nop, at least for that element).
struct MultiIdx {
  using Vec = SmallVector<unsigned, 5>;

  // Logical index into the tensor.
  Vec idx;

  // If the tensor's encoding has e.g. numPerThread = [2,2], then idxInThread
  // tells us which of the four elements per thread this is.  Same for idxInWarp
  // and idxInCTA.
  Vec idxInThread;
  Vec idxInWarp;
  Vec idxInCTA;

  // If the tensor's encoding defines a block of size [x,y,z], the tensor itself
  // may be larger than this, comprising multiple blocks.  This tells us which
  // block we're in.
  Vec idxOuter;

  // flatIdx is flattened according to the tensor's logical order (i.e. ignoring
  // the encoding).  The others are flattened according to the tensor's physical
  // encoding.
  int64_t flatIdx;
  int64_t flatIdxInThread;
  int64_t flatIdxInWarp;
  int64_t flatIdxInCTA;
  int64_t flatIdxOuter;
};

bool sameFlatIdxs(const MultiIdx &a, const MultiIdx &b) {
  return a.flatIdx == b.flatIdx && //
         a.flatIdxInThread == b.flatIdxInThread &&
         a.flatIdxInWarp == b.flatIdxInWarp &&
         a.flatIdxInCTA == b.flatIdxInCTA && //
         a.flatIdxOuter == b.flatIdxOuter;
}

std::string multiIdxsToString(ArrayRef<std::unique_ptr<MultiIdx>> idxs) {
  std::stringstream ss;
  for (const auto &idxPtr : idxs) {
    const MultiIdx &idx = *idxPtr;
    ss //
        << "  [" << triton::join(idx.idx, ",") << "] (" << idx.flatIdx << ") "
        << "elem=[" << triton::join(idx.idxInThread, ",") << "] ("
        << idx.flatIdxInThread << ") "
        << "thread=[" << triton::join(idx.idxInWarp, ",") << "] ("
        << idx.flatIdxInWarp << ") "
        << "warp=[" << triton::join(idx.idxInCTA, ",") << "] ("
        << idx.flatIdxInCTA << ") "
        << "outer=[" << triton::join(idx.idxOuter, ",") << "] ("
        << idx.flatIdxOuter << ")\n";
  }
  return ss.str();
}

std::vector<std::unique_ptr<MultiIdx>> getMultiIdxs(ArrayRef<unsigned> shape,
                                                    BlockedEncodingAttr enc) {
  using Vec = MultiIdx::Vec;

  const unsigned rank = shape.size();
  auto sizePerThread = enc.getSizePerThread();
  auto threadsPerWarp = enc.getThreadsPerWarp();
  auto warpsPerCTA = enc.getWarpsPerCTA();
  auto order = enc.getOrder();

  Vec numBlocks;
  for (int i = 0; i < rank; i++) {
    numBlocks.push_back(ceil<unsigned>(
        shape[i], sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i]));
  }

  Vec idxInThread(rank, 0);
  Vec idxInWarp(rank, 0);
  Vec idxInCTA(rank, 0);
  Vec idxOuter(rank, 0);

  int64_t nElems = product(sizePerThread) * product(threadsPerWarp) *
                   product(warpsPerCTA) * product(numBlocks);

  // We eventually sort this array, and if the elements are plain MultiIdx
  // elements rather than pointers, we have to swap them, which ends up being
  // expensive.
  std::vector<std::unique_ptr<MultiIdx>> elems;
  elems.reserve(nElems);

  for (int64_t i = 0; i < nElems; i++) {
    auto e = std::make_unique<MultiIdx>();
    e->idxInThread = idxInThread;
    e->idxInWarp = idxInWarp;
    e->idxInCTA = idxInCTA;
    e->idxOuter = idxOuter;

    for (int i = 0; i < rank; i++) {
      e->idx.push_back(    //
          idxInThread[i] + //
          idxInWarp[i] * sizePerThread[i] +
          idxInCTA[i] * sizePerThread[i] * threadsPerWarp[i] +
          idxOuter[i] * sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i]);
    }

    e->flatIdxInThread = getFlatIdx(e->idxInThread, sizePerThread, order);
    e->flatIdxInWarp = getFlatIdx(e->idxInWarp, threadsPerWarp, order);
    e->flatIdxInCTA = getFlatIdx(e->idxInCTA, warpsPerCTA, order);
    e->flatIdxOuter = getFlatIdx(e->idxOuter, numBlocks, order);
    e->flatIdx = getFlatIdx(e->idx, shape,
                            llvm::to_vector(llvm::reverse(llvm::seq(rank))));

    elems.push_back(std::move(e));

    if (advance(idxInThread, sizePerThread, order)) {
      if (advance(idxInWarp, threadsPerWarp, order)) {
        if (advance(idxInCTA, warpsPerCTA, order)) {
          advance(idxOuter, numBlocks, order);
        }
      }
    }
  }
  llvm::sort(elems, [](const std::unique_ptr<MultiIdx> &a,
                       const std::unique_ptr<MultiIdx> &b) {
    return a->flatIdx < b->flatIdx;
  });
  return elems;
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

// The optional outparam couldReshape tells the caller whether the reshape
// worked.  You might want this to be a return value instead, but gtest ASSERT
// and FAIL have an implicit `return`, so only work in fns that return void.
void testReshape(RankedTensorType srcTy, RankedTensorType dstTy,
                 std::optional<BlockedEncodingAttr> expectedDstEnc,
                 std::optional<bool> expectSuccess,
                 DialectInferLayoutInterface *inferLayout,
                 bool longErrors = true, bool *couldReshape = nullptr) {
  std::unique_ptr<bool> couldReshapeStorage;
  if (!couldReshape) {
    couldReshapeStorage = std::make_unique<bool>();
    couldReshape = couldReshapeStorage.get();
  }
  *couldReshape = false;

  MLIRContext *ctx = srcTy.getContext();
  ASSERT_TRUE(expectSuccess || !dstTy.getEncoding())
      << "dstTy shouldn't have an expected encoding if we're expecting the "
         "reshape to be impossible!";

  // Capture any errors from calling inferReshapeNoOpReorderEncoding, so we can
  // print them if we expected the reshape to succeed but it failed.
  std::vector<std::string> diags;
  Attribute inferredEnc;
  LogicalResult result = success();
  {
    ScopedDiagnosticHandler scopedHandler(
        ctx, [&](Diagnostic &diag) { diags.push_back("  - " + diag.str()); });
    result = inferLayout->inferReshapeOpNoReorderEncoding(
        srcTy.getShape(), srcTy.getEncoding(), dstTy.getShape(), inferredEnc,
        UnknownLoc::get(ctx));
  }

  if (!expectSuccess.has_value() && !succeeded(result)) {
    // We didn't know whether or not it was supposed to succeed, and it didn't.
    // Test passes!
    return;
  }

  if (expectSuccess.has_value() && !*expectSuccess) {
    EXPECT_FALSE(succeeded(result))
        << "Expected reshape to be impossible, but got dst encoding: "
        << stringifyLLVMType(inferredEnc);
    *couldReshape = true;
    return;
  }

  if (!succeeded(result)) {
    FAIL() << "Expected reshape to succeed, but it didn't!  Error(s):\n"
           << join(diags, "\n");
  }
  if (auto expectedEnc = dstTy.getEncoding()) {
    EXPECT_EQ(inferredEnc, expectedEnc);
  }

  // We know that infer(srcShape, srcEnc, dstShape) => dstEnc.  Check that it
  // works the other way around too: infer(dstShape, dstEnc, srcShape) =>
  // srcEnc.  (This is an invariant of the inference function.)
  {
    std::vector<std::string> diags;
    ScopedDiagnosticHandler scopedHandler(
        ctx, [&](Diagnostic &diag) { diags.push_back("  - " + diag.str()); });
    Attribute inferredSrcEnc;
    auto result = inferLayout->inferReshapeOpNoReorderEncoding(
        dstTy.getShape(), inferredEnc, srcTy.getShape(), inferredSrcEnc,
        UnknownLoc::get(ctx));
    EXPECT_TRUE(succeeded(result))
        << "Inverse encoding inference (" << triton::join(dstTy.getShape(), "x")
        << " " << stringifyLLVMType(inferredEnc) << " -> "
        << triton::join(srcTy.getShape(), "x") << "failed:\n"
        << join(diags, "\n");
    if (succeeded(result)) {
      EXPECT_EQ(inferredSrcEnc, srcTy.getEncoding())
          << "Inverse encoding inference ("
          << triton::join(dstTy.getShape(), "x") << " "
          << stringifyLLVMType(inferredEnc) << " -> "
          << triton::join(srcTy.getShape(), "x")
          << " gave the wrong result.  Expected "
          << stringifyLLVMType(srcTy.getEncoding()) << " but got "
          << stringifyLLVMType(inferredSrcEnc) << ".\n";
    }
  }

  std::vector<std::unique_ptr<MultiIdx>> srcMultiIdxs =
      getMultiIdxs(SmallVector<unsigned>(srcTy.getShape()),
                   mlir::cast<BlockedEncodingAttr>(srcTy.getEncoding()));

  std::vector<std::unique_ptr<MultiIdx>> dstMultiIdxs =
      getMultiIdxs(SmallVector<unsigned>(dstTy.getShape()),
                   mlir::cast<BlockedEncodingAttr>(inferredEnc));

  if (srcMultiIdxs.size() != dstMultiIdxs.size() ||
      !llvm::all_of(llvm::zip_equal(srcMultiIdxs, dstMultiIdxs),
                    [](const auto &pair) {
                      const auto &[a, b] = pair;
                      return sameFlatIdxs(*a, *b);
                    })) {
    SCOPED_TRACE(longErrors ? "dst indices:\n" + multiIdxsToString(dstMultiIdxs)
                            : "");
    SCOPED_TRACE(longErrors ? "src indices:\n" + multiIdxsToString(srcMultiIdxs)
                            : "");
    ADD_FAILURE() << "Reified indices do not match for encodings:\n"
                  << "  src: [" << triton::join(srcTy.getShape(), "x") << "] "
                  << stringifyLLVMType(srcTy.getEncoding()) << "\n"
                  << "  dst: [" << triton::join(dstTy.getShape(), "x") << "] "
                  << stringifyLLVMType(inferredEnc);
  } else {
    *couldReshape = true;
  }
}

class InferReshapeOpNoReorderEncodingTest
    : public InferLayoutTest,
      public ::testing::WithParamInterface<
          std::tuple<std::string /*srcTy*/, std::string /*dstTy*/,
                     bool /*expectSuccess*/>> {};

TEST_P(InferReshapeOpNoReorderEncodingTest, DoIt) {
  std::string srcTyStr = expandTyStr(std::get<0>(GetParam()));
  std::string dstTyStr = expandTyStr(std::get<1>(GetParam()));
  bool expectSuccess = std::get<2>(GetParam());

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
              expectedDstEnc, expectSuccess, inferLayout, /*longErrors=*/true);
}

// A testcase of {a, b, c} means:
//  - if `c` is false, check that a reshape from shape+encoding `a` to shape `b`
//    is deemed impossible.
//  - else if `c` is true:
//    - check that a reshape from shape+encoding `a` to shape `b` yields an
//      encoding that makes the reshape a nop, and
//    - if b has an encoding, check that the inferred encoding matches b's.
INSTANTIATE_TEST_SUITE_P(
    Reshapes, InferReshapeOpNoReorderEncodingTest,
    ::testing::ValuesIn(std::vector<
                        std::tuple<std::string /*srcTy*/, std::string /*dstTy*/,
                                   bool /*expectSuccess*/>>({
        // Use raw strings in here so clang-format doesn't try to wrap them.
        {R"(T<128x64xf32, #B<{spt=[1,1], tpw=[1,32], wpc=[1,1], ord=[1,0]}>>)",
         R"(T<8192xf32,   #B<{spt=[1],   tpw=[32],   wpc=[1],   ord=[0]}>>)",
         true},

        {R"(T<128xf32,  #B<{spt=[4],   tpw=[32],   wpc=[1],   ord=[0]}>>)",
         R"(T<32x4xf32, #B<{spt=[1,4], tpw=[32,1], wpc=[1,1], ord=[1,0]}>>)",
         true},

        {R"(T<128xf32,  #B<{spt=[4],   tpw=[32],   wpc=[1],   ord=[0]}>>)",
         R"(T<16x8xf32, #B<{spt=[1,4], tpw=[16,2], wpc=[1,1], ord=[1,0]}>>)",
         true},

        {R"(T<32x32xf32, #B<{spt=[2,2], tpw=[32,1], wpc=[1,1], ord=[1,0]}>>)",
         "T<128xf32>", false},

        {R"(T<32x4xf32,     #B<{spt=[1,4],     tpw=[32,1],     wpc=[1,1],     ord=[1,0]}>>)",
         R"(T<2x16x2x2xf32, #B<{spt=[1,1,2,2], tpw=[2,16,1,1], wpc=[1,1,1,1], ord=[3,2,1,0]}>>)",
         true},

        {R"(T<4x32xf32,     #B<{spt=[4,1],     tpw=[1,32],     wpc=[1,1],     ord=[0,1]}>>)",
         R"(T<2x2x2x16xf32, #B<{spt=[2,2,1,1], tpw=[1,1,2,16], wpc=[1,1,1,1], ord=[1,0,3,2]}>>)",
         true},

        {R"(T<32x32xf32,     #B<{spt=[4,4],     tpw=[4,8],     wpc=[1,1],     ord=[1,0]}>>)",
         R"(T<2x16x2x16xf32, #B<{spt=[1,4,1,4], tpw=[1,4,2,4], wpc=[1,1,1,1], ord=[3,2,1,0]}>>)",
         true},

        {R"(T<32x32xf32,     #B<{spt=[4,4],     tpw=[4,8],     wpc=[1,1],     ord=[1,0]}>>)",
         R"(T<16x2x16x2xf32, #B<{spt=[2,2,2,2], tpw=[4,1,8,1], wpc=[1,1,1,1], ord=[3,2,1,0]}>>)",
         true},

        {R"(T<32x32xf32, #B<{spt=[4,4], tpw=[4,8], wpc=[1,1], ord=[0,1]}>>)",
         R"(T<16x2x16x2xf32>)", true},

        // nop reshape, but the block size is 2x larger than the tensor.
        {R"(T<4x2x2x4xf32, #B<{spt=[2,1,1,2], tpw=[2,1,1,2], wpc=[2,2,1,1], ord=[0,3,1,2]}>>)",
         R"(T<4x2x2x4xf32, #B<{spt=[2,1,1,2], tpw=[2,1,1,2], wpc=[2,2,1,1], ord=[0,3,1,2]}>>)",
         true},

        {R"(T<2x4x2x4xf32, #B<{spt=[1,2,2,1], tpw=[1,2,1,2], wpc=[1,2,2,1], ord=[2,1,0,3]}>>)",
         R"(T<4x2x2x4xf32>)", false},

        {R"(T<1x2x2x4xf32, #B<{spt=[1,32,4,4], tpw=[4,4,16,16], wpc=[8,8,8,1], ord=[0,1,2,3]}>>)",
         R"(T<2x2x4x1xf32>)", false},

        {R"(T<2x2x2x2xf32, #B<{spt=[2,2,2,2], tpw=[1,1,1,1], wpc=[1,1,1,1], ord=[1,0,3,2]}>>)",
         R"(T<4x4xf32>)", true},

        {R"(T<16x8xf32, #B<{spt=[1,2], tpw=[2,4], wpc=[2,1], ord=[1,0]}>>)",
         R"(T<128xf32>)", true},

        {R"(T<16x1x8xf32, #B<{spt=[8,1,1], tpw=[2,1,1], wpc=[1,1,8], ord=[2,1,0]}>>)",
         R"(T<128x1xf32>)", false},

        {R"(T<16x1x8xf32, #B<{spt=[1,1,8], tpw=[2,1,1], wpc=[8,1,1], ord=[2,1,0]}>>)",
         R"(T<128x1xf32>)", true},

        {R"(T<32x32xf32, #B<{spt=[1,2], tpw=[1,8], wpc=[1,1], ord=[1,0]}>>)",
         R"(T<1024xf32>)", true},

        {R"(T<4x4xf32, #B<{spt=[1,1], tpw=[2,4], wpc=[2,1], ord=[0,1]}>>)",
         R"(T<16xf32>)", false},

        {R"(T<32xf32,   #B<{spt=[2],   tpw=[32],   wpc=[2],   ord=[0]}>>)",
         R"(T<16x2xf32, #B<{spt=[1,2], tpw=[32,1], wpc=[2,1], ord=[1,0]}>>)",
         true},

        {R"(T<2x1x2xf32, #B<{spt=[2,1,1], tpw=[2,1,2], wpc=[4,1,8], ord=[2,1,0]}>>)",
         R"(T<2x2xf32,   #B<{spt=[2,1],   tpw=[2,2],   wpc=[4,8],   ord=[1,0]}>>)",
         true},
    })));

TEST_F(InferLayoutTest, FuzzReshape) {
  const int numTests = 1000; // Increase to get more coverage.

  std::minstd_rand rng(/*seed=*/0);
  auto randPow2Vec = [&](int rank, int maxPow2) {
    SmallVector<unsigned> ret;
    for (int i = 0; i < rank; i++) {
      int pow2 = std::uniform_int_distribution<unsigned>(0, maxPow2)(rng);
      if (pow2 == maxPow2 && maxPow2 > 0) {
        maxPow2--;
      }
      ret.push_back(1 << pow2);
    }
    return ret;
  };

  int numSuccess = 0;
  for (int i = 0; i < numTests; i++) {
    SCOPED_TRACE("iteration " + std::to_string(i));
    int rank = std::uniform_int_distribution<int>(1, 4)(rng);

    SmallVector<int64_t> srcShape(
        convertType<int64_t>(randPow2Vec(rank, /*maxPow2=*/4)));
    SmallVector<int64_t> dstShape = srcShape;
    std::shuffle(dstShape.begin(), dstShape.end(), rng);

    // Optionally merge some dimensions in dst.
    for (int i = 1; i < dstShape.size(); i++) {
      if (std::uniform_real_distribution<float>(0, 1)(rng) > 1.0 / rank) {
        dstShape[i - 1] *= dstShape[i];
        dstShape.erase(dstShape.begin() + i);
        i--;
      }
    }

    SmallVector<unsigned> sizePerThread = randPow2Vec(rank, /*maxPow2=*/3);
    SmallVector<unsigned> threadsPerWarp = randPow2Vec(rank, /*maxPow2=*/3);
    SmallVector<unsigned> warpsPerCTA = randPow2Vec(rank, /*maxPow2=*/3);

    SmallVector<unsigned> order(llvm::to_vector(llvm::seq<unsigned>(rank)));
    std::shuffle(order.begin(), order.end(), rng);

    auto ctaLayout = CTALayoutAttr::get(
        &ctx, SmallVector<unsigned>(rank, 1), SmallVector<unsigned>(rank, 1),
        llvm::to_vector(llvm::reverse(llvm::seq<unsigned>(rank))));

    auto srcTy = RankedTensorType::get(
        srcShape, FloatType::getF32(&ctx),
        BlockedEncodingAttr::get(&ctx, sizePerThread, threadsPerWarp,
                                 warpsPerCTA, order, ctaLayout));
    auto dstTy = RankedTensorType::get(dstShape, FloatType::getF32(&ctx));

    bool couldReshape = false;
    testReshape(srcTy, dstTy, /*expectedDstEnc=*/std::nullopt,
                /*expectSuccess=*/std::nullopt, inferLayout,
                /*longErrors=*/false, &couldReshape);
    if (couldReshape)
      numSuccess++;
  }

  // We don't expect or want 100% success, but if only a tiny fraction of tests
  // actually exercise the successful reshape logic, then that gives us bad
  // coverage.  I'm currently getting 35% success, which seems good enough,
  // especially since the successful cases take a lot longer to run because of
  // the MultiIdx checks (so we're spending most of our time on successful
  // cases, even if they're only 1/3 of the iterations).
  //
  // Run ctest with --verbose to see this output.  For example:
  //   $ cd python/build/cmake.blah.blah
  //   $ ninja
  //   $ $(git rev-parse --show-toplevel)/.venv/bin/ctest --verbose
  printf("Fuzz success rate: %d/%d = %.2f%%\n", numSuccess, numTests,
         100.0 * numSuccess / numTests);
}

class AMDMfmaLayoutTest : public ::testing::Test {
public:
  AMDMfmaLayoutTest() {
    ctx.getOrLoadDialect<TritonGPUDialect>();
    ctaLayout =
        triton::gpu::CTALayoutAttr::get(&ctx, ctaPerCGA, ctaSplit, ctaOrder);
    f16Ty = FloatType::getF16(&ctx);
  }

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

protected:
  MLIRContext ctx;
  const SmallVector<unsigned> ctaPerCGA{1, 1, 1};
  const SmallVector<unsigned> ctaSplit{1, 1, 1};
  const SmallVector<unsigned> ctaOrder{2, 1, 0};
  triton::gpu::CTALayoutAttr ctaLayout;
  Type f16Ty;
};

TEST_F(AMDMfmaLayoutTest, mfma32) {
  auto mfma2d = createMFMA(32, 32, {2, 4});
  ASSERT_THAT(mfma2d.getThreadOrder(), testing::ElementsAre(1u, 0u));
  ASSERT_THAT(mfma2d.getWarpOrder(), testing::ElementsAre(1u, 0u));

  auto tmfma2d = createTransposedMFMA(32, 32, {2, 4});
  ASSERT_THAT(tmfma2d.getThreadOrder(), testing::ElementsAre(0u, 1u));
  ASSERT_THAT(tmfma2d.getWarpOrder(), testing::ElementsAre(1u, 0u));

  auto mfma3d = createMFMA(32, 32, {2, 4, 1});
  ASSERT_THAT(mfma3d.getThreadOrder(), testing::ElementsAre(2u, 1u, 0u));
  ASSERT_THAT(mfma3d.getWarpOrder(), testing::ElementsAre(2u, 1u, 0u));

  auto tmfma3d = createTransposedMFMA(32, 32, {2, 4, 1});
  ASSERT_THAT(tmfma3d.getThreadOrder(), testing::ElementsAre(1u, 2u, 0u));
  ASSERT_THAT(tmfma3d.getWarpOrder(), testing::ElementsAre(2u, 1u, 0u));
}

TEST_F(AMDMfmaLayoutTest, mfma16) {
  auto mfma2d = createMFMA(16, 16, {2, 4});
  ASSERT_THAT(mfma2d.getThreadOrder(), testing::ElementsAre(1u, 0u));
  ASSERT_THAT(mfma2d.getWarpOrder(), testing::ElementsAre(1u, 0u));

  auto tmfma2d = createTransposedMFMA(16, 16, {2, 4});
  ASSERT_THAT(tmfma2d.getThreadOrder(), testing::ElementsAre(0u, 1u));
  ASSERT_THAT(tmfma2d.getWarpOrder(), testing::ElementsAre(1u, 0u));

  auto mfma3d = createMFMA(16, 16, {2, 4, 1});
  ASSERT_THAT(mfma3d.getThreadOrder(), testing::ElementsAre(2u, 1u, 0u));
  ASSERT_THAT(mfma3d.getWarpOrder(), testing::ElementsAre(2u, 1u, 0u));

  auto tmfma3d = createTransposedMFMA(16, 16, {2, 4, 1});
  ASSERT_THAT(tmfma3d.getThreadOrder(), testing::ElementsAre(1u, 2u, 0u));
  ASSERT_THAT(tmfma3d.getWarpOrder(), testing::ElementsAre(2u, 1u, 0u));
}

} // anonymous namespace
} // namespace mlir::triton::gpu

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
