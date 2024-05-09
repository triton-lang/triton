#include "triton/Tools/LinearLayout.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/MathExtras.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iterator>

namespace mlir {
std::ostream &operator<<(std::ostream &os, StringAttr str) {
  os << str.str();
  return os;
}
} // namespace mlir

namespace mlir::triton {
namespace {

using ::llvm::to_vector;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

using BasesT = LinearLayout::BasesT;

std::vector<int32_t> standardBasis(int32_t size) {
  assert(llvm::isPowerOf2_32(size));
  std::vector<int32_t> ret;
  for (int i = 1; i < size; i *= 2) {
    ret.push_back(i);
  }
  return ret;
}

std::vector<int32_t> zerosLike(int32_t size) {
  assert(llvm::isPowerOf2_32(size));
  return std::vector<int32_t>(llvm::Log2_32(size));
}

class LinearLayoutTest : public ::testing::Test {
public:
  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(LinearLayoutTest, Empty) {
  LinearLayout layout = LinearLayout::empty();
  EXPECT_THAT(layout.getBases(), IsEmpty());
  EXPECT_THAT(to_vector(layout.getInDimNames()), IsEmpty());
  EXPECT_THAT(to_vector(layout.getOutDimNames()), IsEmpty());
}

TEST_F(LinearLayoutTest, Identity1D) {
  LinearLayout layout =
      LinearLayout::identity1D(32, S("testIns"), S("testOuts"));
  EXPECT_THAT(layout, LinearLayout({
                          {S("testIns"), {{S("testOuts"), standardBasis(32)}}},
                      }));
  EXPECT_THAT(to_vector(layout.getInDimNames()), ElementsAre(S("testIns")));
  EXPECT_THAT(to_vector(layout.getOutDimNames()), ElementsAre(S("testOuts")));
  EXPECT_THAT(layout.getInDimSizeLog2(S("testIns")), 5);
  EXPECT_THAT(layout.getOutDimSizeLog2(S("testOuts")), 5);
}

TEST_F(LinearLayoutTest, Identity1DSize1) {
  LinearLayout layout =
      LinearLayout::identity1D(1, S("testIns"), S("testOuts"));
  EXPECT_EQ(layout, LinearLayout({{S("testIns"), {{S("testOuts"), {}}}}}));
  EXPECT_THAT(to_vector(layout.getInDimNames()), ElementsAre(S("testIns")));
  EXPECT_THAT(to_vector(layout.getOutDimNames()), ElementsAre(S("testOuts")));
  EXPECT_THAT(layout.getInDimSizeLog2(S("testIns")), 0);
  EXPECT_THAT(layout.getOutDimSizeLog2(S("testOuts")), 0);
}

TEST_F(LinearLayoutTest, Zeros1D) {
  LinearLayout layout = LinearLayout::zeros1D(32, S("ins"), S("outs"));
  EXPECT_EQ(layout, LinearLayout({{S("ins"), {{S("outs"), zerosLike(32)}}}}));
}

TEST_F(LinearLayoutTest, MultiplyIdentity) {
  LinearLayout prod = LinearLayout::identity1D(16, S("in"), S("out")) *
                      LinearLayout::identity1D(32, S("in"), S("out"));
  EXPECT_EQ(prod,
            LinearLayout({
                {S("in"), {{S("out"), {1, 2, 4, 8, 16, 32, 64, 128, 256}}}},
            }));
  EXPECT_THAT(to_vector(prod.getInDimNames()), ElementsAre(S("in")));
  EXPECT_THAT(to_vector(prod.getOutDimNames()), ElementsAre(S("out")));
}

TEST_F(LinearLayoutTest, MultiplyDisjoint) {
  LinearLayout prod = LinearLayout::identity1D(32, S("in1"), S("out1")) *
                      LinearLayout::identity1D(16, S("in2"), S("out2"));
  EXPECT_EQ(prod,
            LinearLayout({
                {S("in1"),
                 {{S("out1"), standardBasis(32)}, {S("out2"), zerosLike(32)}}},
                {S("in2"),
                 {{S("out1"), zerosLike(16)}, {S("out2"), standardBasis(16)}}},
            }));
  EXPECT_THAT(to_vector(prod.getInDimNames()), ElementsAre(S("in1"), S("in2")));
  EXPECT_THAT(to_vector(prod.getOutDimNames()),
              ElementsAre(S("out1"), S("out2")));
}

TEST_F(LinearLayoutTest, MultiplyByEmpty) {
  LinearLayout prod =
      LinearLayout::empty() * LinearLayout::identity1D(32, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout::identity1D(32, S("in"), S("out")));
}

TEST_F(LinearLayoutTest, MultiplyByZeros) {
  LinearLayout prod = LinearLayout::identity1D(8, S("in"), S("out")) *
                      LinearLayout::zeros1D(16, S("in"), S("out"));
  EXPECT_EQ(prod,
            LinearLayout({{S("in"), {{S("out"), {1, 2, 4, 0, 0, 0, 0}}}}}));
}

TEST_F(LinearLayoutTest, MultiplyZerosByDegenerate) {
  LinearLayout prod = LinearLayout::zeros1D(16, S("in"), S("out1")) *
                      LinearLayout({{S("in"), {{S("out2"), {}}}}});
  EXPECT_EQ(prod, LinearLayout({{S("in"),
                                 {{S("out1"), zerosLike(16)},
                                  {S("out2"), {{zerosLike(16)}}}}}}));
}

TEST_F(LinearLayoutTest, MultiplyEmptyIdentityAndZeros) {
  LinearLayout prod = LinearLayout::identity1D(0, S("in"), S("out")) *
                      LinearLayout::zeros1D(4, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout({{S("in"), {{S("out"), {0, 0}}}}}));
}

TEST_F(LinearLayoutTest, MultiplyOverlapping) {
  LinearLayout prod = LinearLayout::identity1D(4, S("in"), S("out1")) *
                      LinearLayout::identity1D(8, S("in"), S("out2"));
  EXPECT_EQ(prod,
            LinearLayout({{
                S("in"),
                {{S("out1"), {1, 2, 0, 0, 0}}, {S("out2"), {0, 0, 1, 2, 4}}},
            }}));
}

TEST_F(LinearLayoutTest, TimesEquals) {
  LinearLayout prod = LinearLayout::empty();
  prod *= LinearLayout::identity1D(32, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout::identity1D(32, S("in"), S("out")));
}

TEST_F(LinearLayoutTest, GetOutDimSizeLog2) {
  LinearLayout layout({
      {S("in0"), {{S("dim0"), {0, 0, 0}}}},
      {S("in1"), {{S("dim0"), {1, 2}}}},
  });
  EXPECT_EQ(layout.getOutDimSizeLog2(S("dim0")), 2);
}

TEST_F(LinearLayoutTest, TransposeOuts) {
  LinearLayout layout = (LinearLayout::identity1D(32, S("in1"), S("out1")) *
                         LinearLayout::identity1D(16, S("in2"), S("out2")))
                            .transposeOuts({S("out2"), S("out1")});
  EXPECT_THAT(to_vector(layout.getOutDimNames()),
              ElementsAre(S("out2"), S("out1")));
  EXPECT_EQ(layout,
            LinearLayout({
                {S("in1"),
                 {{S("out2"), zerosLike(32)}, {S("out1"), standardBasis(32)}}},
                {S("in2"),
                 {{S("out2"), standardBasis(16)}, {S("out1"), zerosLike(16)}}},
            }));
}

TEST_F(LinearLayoutTest, TransposeIns) {
  LinearLayout layout = (LinearLayout::identity1D(32, S("in1"), S("out1")) *
                         LinearLayout::identity1D(16, S("in2"), S("out2")))
                            .transposeIns({S("in2"), S("in1")});
  EXPECT_THAT(to_vector(layout.getInDimNames()),
              ElementsAre(S("in2"), S("in1")));
  EXPECT_EQ(
      layout,
      LinearLayout(
          {{S("in2"),
            {{S("out1"), zerosLike(16)}, {S("out2"), standardBasis(16)}}},
           {S("in1"),
            {{S("out1"), standardBasis(32)}, {S("out2"), zerosLike(32)}}}}));
}

TEST_F(LinearLayoutTest, EmptyToString) {
  // Mostly I just want to make sure it doesn't crash.
  EXPECT_EQ(LinearLayout::empty().toString(), "(empty layout)\n");
}

TEST_F(LinearLayoutTest, Apply) {
  LinearLayout layout({
      {S("in1"), {{S("out1"), {4, 2, 1}}, {S("out2"), {2, 1, 0}}}},
      {S("in2"), {{S("out1"), {1, 2}}, {S("out2"), {2, 1}}}},
  });
  EXPECT_THAT(layout.apply({{S("in1"), 0}, {S("in2"), 0}}),
              ElementsAre(Pair(S("out1"), 0), Pair(S("out2"), 0)));
  EXPECT_THAT(layout.apply({{S("in2"), 0}, {S("in1"), 1}}),
              ElementsAre(Pair(S("out1"), 4), Pair(S("out2"), 2)));
  EXPECT_THAT(layout.apply({{S("in2"), 1}, {S("in1"), 0}}),
              ElementsAre(Pair(S("out1"), 1), Pair(S("out2"), 2)));
}

// This is really more of a benchmark than a test.  We're checking that it
// doesn't take so long to run that a human notices and says "hmm".  :)
TEST_F(LinearLayoutTest, ConstructLargeLayout) {
  std::vector<int32_t> pows2;
  for (int i = 0; i < 25; i++) {
    pows2.push_back(1 << i);
  }
  LinearLayout layout({{S("in"), {{S("out"), pows2}}}});
}

} // anonymous namespace
} // namespace mlir::triton
