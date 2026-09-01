#include "mlir/IR/TypeUtilities.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"

namespace mlir {
using namespace triton;
using namespace triton::gpu;
namespace {

// A known bit is affine in the bits of the tensor's row-major logical index.
// Unknown bits remain actual gather-index operands, never guessed constants.
struct IndexBit {
  uint64_t variables;
  bool constant;
  bool operator==(const IndexBit &rhs) const {
    return variables == rhs.variables && constant == rhs.constant;
  }
};
using Bit = std::optional<IndexBit>;
using IndexBits = SmallVector<Bit>;

static Bit xorBit(Bit a, Bit b) {
  if (!a || !b)
    return std::nullopt;
  return IndexBit{a->variables ^ b->variables, a->constant != b->constant};
}

static Bit andBit(Bit a, Bit b) {
  if (a && !a->variables)
    return a->constant ? b : a;
  if (b && !b->variables)
    return b->constant ? a : b;
  return a == b ? a : std::nullopt;
}

static Bit orBit(Bit a, Bit b) {
  Bit one = IndexBit{0, true};
  return xorBit(andBit(xorBit(a, one), xorBit(b, one)), one);
}

static IndexBits constantBits(const APInt &value) {
  IndexBits bits;
  for (unsigned i = 0; i < value.getBitWidth(); ++i)
    bits.push_back(IndexBit{0, value[i]});
  return bits;
}

static IndexBits addBits(IndexBits a, const IndexBits &b) {
  Bit carry = IndexBit{0, false};
  for (unsigned i = 0; i < a.size(); ++i) {
    Bit sum = xorBit(a[i], b[i]);
    Bit next = orBit(andBit(a[i], b[i]), andBit(sum, carry));
    a[i] = xorBit(sum, carry);
    carry = next;
  }
  return a;
}

static SmallVector<unsigned> logicalBitOffsets(RankedTensorType type) {
  SmallVector<unsigned> offsets(type.getRank());
  unsigned offset = 0;
  for (int dim = type.getRank() - 1; dim >= 0; --dim) {
    offsets[dim] = offset;
    offset += llvm::Log2_64(type.getDimSize(dim));
  }
  return offsets;
}

class GatherIndexAnalysis {
public:
  IndexBits get(Value value) {
    auto it = cache.find(value);
    if (it != cache.end())
      return it->second;
    IndexBits bits = analyze(value);
    cache.try_emplace(value, bits);
    return bits;
  }

private:
  IndexBits analyze(Value value) {
    unsigned width = getElementTypeOrSelf(value.getType()).getIntOrFloatBitWidth();
    IndexBits unknown(width);
    Operation *op = value.getDefiningOp();
    if (!op)
      return unknown;
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
        return constantBits(integer.getValue());
      auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
      if (!dense)
        return unknown;
      if (dense.isSplat())
        return constantBits(dense.getSplatValue<APInt>());
      auto values = dense.getValues<APInt>();
      IndexBits result = constantBits(*values.begin());
      for (unsigned coord = 0; coord < llvm::Log2_64(dense.getNumElements());
           ++coord) {
        APInt delta = *(values.begin() + (uint64_t{1} << coord)) ^ *values.begin();
        for (unsigned bit = 0; bit < width; ++bit)
          result[bit]->variables |= uint64_t(delta[bit]) << coord;
      }
      for (auto [index, element] : llvm::enumerate(values))
        for (unsigned bit = 0; bit < width; ++bit)
          if (result[bit] && (llvm::popcount(result[bit]->variables & index) % 2 !=
                             (element[bit] != result[bit]->constant)))
            result[bit] = std::nullopt;
      return result;
    }
    if (auto range = dyn_cast<MakeRangeOp>(op)) {
      IndexBits bits(width, IndexBit{0, false});
      for (unsigned bit = 0; bit < llvm::Log2_64(range.getType().getNumElements());
           ++bit)
        bits[bit] = IndexBit{uint64_t{1} << bit, false};
      return addBits(bits, constantBits(APInt(width, range.getStart())));
    }
    if (isa<SplatOp, ExpandDimsOp, ConvertLayoutOp>(op))
      return get(op->getOperand(0));
    if (auto reshape = dyn_cast<ReshapeOp>(op))
      return reshape.getAllowReorder() ? unknown : get(reshape.getSrc());
    if (isa<BroadcastOp, TransOp>(op)) {
      auto srcType = cast<RankedTensorType>(op->getOperand(0).getType());
      auto dstType = cast<RankedTensorType>(value.getType());
      auto srcOffsets = logicalBitOffsets(srcType);
      auto dstOffsets = logicalBitOffsets(dstType);
      SmallVector<unsigned> permutation(llvm::Log2_64(srcType.getNumElements()));
      auto transpose = dyn_cast<TransOp>(op);
      for (unsigned dst = 0; dst < dstType.getRank(); ++dst) {
        unsigned src = transpose ? transpose.getOrder()[dst] : dst;
        for (unsigned bit = 0; bit < llvm::Log2_64(srcType.getDimSize(src)); ++bit)
          permutation[srcOffsets[src] + bit] = dstOffsets[dst] + bit;
      }
      IndexBits bits = get(op->getOperand(0));
      for (Bit &bit : bits) {
        if (!bit)
          continue;
        uint64_t variables = 0;
        for (auto [src, dst] : llvm::enumerate(permutation))
          variables |= ((bit->variables >> src) & 1) << dst;
        bit->variables = variables;
      }
      return bits;
    }
    if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(op)) {
      IndexBits bits = get(op->getOperand(0));
      Bit extension =
          isa<arith::ExtSIOp>(op) ? bits.back() : Bit(IndexBit{0, false});
      bits.resize(width, extension);
      return bits;
    }
    if (isa<arith::XOrIOp, arith::AndIOp, arith::OrIOp, arith::AddIOp>(op)) {
      IndexBits a = get(op->getOperand(0)), b = get(op->getOperand(1));
      if (isa<arith::AddIOp>(op))
        return addBits(a, b);
      for (unsigned i = 0; i < width; ++i)
        a[i] = isa<arith::XOrIOp>(op) ? xorBit(a[i], b[i])
               : isa<arith::AndIOp>(op) ? andBit(a[i], b[i])
                                       : orBit(a[i], b[i]);
      return a;
    }
    if (isa<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp>(op)) {
      IndexBits bits = get(op->getOperand(0)), shift = get(op->getOperand(1));
      APInt amount(width, 0);
      for (unsigned i = 0; i < width; ++i) {
        if (!shift[i] || shift[i]->variables)
          return unknown;
        if (shift[i]->constant)
          amount.setBit(i);
      }
      if (amount.uge(width))
        return unknown;
      int distance = amount.getZExtValue();
      Bit extension =
          isa<arith::ShRSIOp>(op) ? bits.back() : Bit(IndexBit{0, false});
      IndexBits result(width, extension);
      for (int i = 0; i < int(width); ++i) {
        int source = isa<arith::ShLIOp>(op) ? i - distance : i + distance;
        if (source >= 0 && source < int(width))
          result[i] = bits[source];
      }
      return result;
    }
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      Bit condition = get(select.getCondition()).front();
      IndexBits a = get(select.getTrueValue()), b = get(select.getFalseValue());
      if (condition && !condition->variables)
        return condition->constant ? a : b;
      for (unsigned i = 0; i < width; ++i)
        a[i] = a[i] == b[i]
                   ? a[i]
                   : xorBit(b[i], andBit(condition, xorBit(a[i], b[i])));
      return a;
    }
    return unknown;
  }

  DenseMap<Value, IndexBits> cache;
};

// Map the receiving physical position and actual index to the desired logical
// source coordinate. The extra constant input has value one. Affine index bits
// are substituted, retaining only unknown bits as independent index inputs.
static LinearLayout getGatherAccess(GatherOp op) {
  MLIRContext *ctx = op.getContext();
  auto name = [&](StringRef value) { return StringAttr::get(ctx, value); };
  auto srcType = op.getSrc().getType();
  auto idxType = op.getIndices().getType();
  unsigned axis = op.getAxis();
  unsigned axisBits = llvm::Log2_64(srcType.getDimSize(axis));
  IndexBits bits = GatherIndexAnalysis().get(op.getIndices());
  bits.resize(axisBits, IndexBit{0, false});
  auto layout = toLinearLayout(idxType).removeZeroBasesAlongDim(name("register"));
  auto bases = layout.getBases();
  auto offsets = logicalBitOffsets(idxType);
  for (auto &[dim, vectors] : bases) {
    for (auto &vector : vectors) {
      uint64_t logical = 0;
      for (unsigned d = 0; d < idxType.getRank(); ++d)
        logical |= uint64_t(vector[d]) << offsets[d];
      vector[axis] = 0;
      for (unsigned bit = 0; bit < axisBits; ++bit)
        if (bits[bit] && llvm::popcount(bits[bit]->variables & logical) % 2)
          vector[axis] |= 1 << bit;
    }
  }
  auto &indices = bases[name("index")];
  for (unsigned bit = 0; bit < axisBits; ++bit) {
    std::vector<int32_t> vector(idxType.getRank(), 0);
    if (!bits[bit])
      vector[axis] = 1 << bit;
    indices.push_back(std::move(vector));
  }
  std::vector<int32_t> constant(idxType.getRank(), 0);
  for (unsigned bit = 0; bit < axisBits; ++bit)
    if (bits[bit] && bits[bit]->constant)
      constant[axis] |= 1 << bit;
  bases[name("constant")] = {std::move(constant)};
  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  for (auto [dim, size] : llvm::enumerate(srcType.getShape()))
    outDims.push_back({name(("dim" + Twine(dim)).str()), size});
  return LinearLayout(std::move(bases), outDims, /*requireSurjective=*/false);
}

// Solve for the source register (and, if needed, lane), holding the other
// physical coordinates equal to the receiver's. Unlike pseudoinversion, this
// can use any replica in the receiving thread/warp, including dependent bases.
static std::optional<LinearLayout>
getLocalGatherLayout(const LinearLayout &source, const LinearLayout &access,
                     unsigned numMovingDims) {
  MLIRContext *ctx = source.getInDimNames().begin()->getContext();
  auto name = [&](StringRef value) { return StringAttr::get(ctx, value); };
  SmallVector<StringAttr> dims = {name("register"), name("lane"), name("warp"),
                                 name("block")};
  auto variableDims = ArrayRef<StringAttr>(dims).take_front(numMovingDims);
  auto fixedDims = ArrayRef<StringAttr>(dims).drop_front(numMovingDims);

  auto bases = access.getBases();
  for (StringAttr dim : fixedDims) {
    auto &vectors = bases[dim];
    if (vectors.size() != source.getInDimSizeLog2(dim))
      return std::nullopt;
    for (auto [bit, vector] : llvm::enumerate(vectors))
      for (auto [coordinate, value] : llvm::enumerate(vector))
        value ^= source.getBasis(dim, bit)[coordinate];
  }
  LinearLayout residual(std::move(bases), access.getOutDims(),
                        /*requireSurjective=*/false);
  return lstsq(source.sublayout(variableDims,
                               llvm::to_vector(source.getOutDimNames())),
               residual);
}

} // namespace

GatherLoweringHelper::GatherLoweringHelper(GatherOp op) {
  auto source = toLinearLayout(op.getSrc().getType())
                    .removeZeroBasesAlongDim(StringAttr::get(op.getContext(),
                                                            "register"));
  auto access = getGatherAccess(op);
  warpLocalLayout = getLocalGatherLayout(source, access, /*numMovingDims=*/1);
  if (!warpLocalLayout)
    warpLocalLayout = getLocalGatherLayout(source, access, /*numMovingDims=*/2);
  ctaLocal = warpLocalLayout.has_value() ||
             getLocalGatherLayout(source, access, /*numMovingDims=*/3).has_value();
  auto srcType = op.getSrc().getType();
  auto idxType = op.getIndices().getType();
  auto name = [&](StringRef value) {
    return StringAttr::get(op.getContext(), value);
  };
  if (!warpLocalLayout && idxType.getDimSize(op.getAxis()) == 1) {
    auto columns =
        source.resizeOutDim(name(("dim" + Twine(op.getAxis())).str()), 1);
    auto indices =
        toLinearLayout(idxType).removeZeroBasesAlongDim(name("register"));
    auto selector = getLocalGatherLayout(indices, columns, /*numMovingDims=*/1);
    if (selector && selector->sublayoutIsZero(
                        {name("lane"), name("warp"), name("block")},
                        name("register"))) {
      auto registers = selector->sublayout(name("register"), name("register"));
      for (unsigned reg = 0; reg < source.getInDimSize(name("register")); ++reg)
        sourceToIndex.push_back(
            registers.apply({{name("register"), reg}}).front().second);
    }
  }
  int64_t scratchElements = sourceToIndex.empty() ? srcType.getNumElements()
                                                 : idxType.getNumElements();
  scratchSize = warpLocalLayout
                    ? 0
                    : scratchElements *
                          llvm::divideCeil(srcType.getElementTypeBitWidth(), 8u);
}

} // namespace mlir
