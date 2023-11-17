#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

namespace triton {
class LoadOp;
class StoreOp;
class FuncOp;
namespace gpu {
class SharedEncodingAttr;
}
} // namespace triton

SmallVector<unsigned, 3> mmaVersionToInstrShape(int version,
                                                const ArrayRef<int64_t> &shape,
                                                RankedTensorType type);

/// Returns true if the Load is for TMA
bool isLoadFromTensorPtr(triton::LoadOp op);

/// Returns true if the store is for TMA
bool isStoreToTensorPtr(triton::StoreOp op);

/// Return the first consumer of v
Operation *getFirstUser(Value v);

/// Return the proper SharedEncodingAttr according to shape/order
triton::gpu::SharedEncodingAttr getSharedEncoding(RankedTensorType tensorTy);

/* Dump Triton IR in graphviz dot format.
 *
 * You can override `onValue` and `onOperation` in a subclass to mark
 * specific Values and Operations. The below subclass
 * GraphLayoutMarker is an example.
 *
 * Default NodeInfo for Value nodes:
 *   {{"shape": "box"},
 *    {"style", "filled"},
 *    {"fillcolor", "white"},
 *    {"label", shapeStr}}
 *
 * Default NodeInfo for Operation nodes:
 *   {{"shape": "ellipse"},
 *    {"style", "filled"},
 *    {"fillcolor", "white"},
 *    {"label", operationName}}
 *
 * If the key "label" is not set by `onValue` or `onOperation`, default labels
 * will be generated. For Value node, the default label is the shape string and
 * for Operation node, it is the operation name.
 *
 * Reference:
 *   https://graphviz.org/doc/info/shapes.html
 *   https://graphviz.org/doc/info/colors.html
 *
 * Usage:
 *   C++:   GraphDumper().dumpToFile(func, "func.dot");
 *   Shell: dot -Tjpg func.dot -o func.jpg
 */
class GraphDumper {
public:
  using NodeInfo = std::map<std::string, std::string>;

  // Override this function to mark specific Values
  virtual NodeInfo onValue(Value value) const;
  // Override this function to mark specific Operations
  virtual NodeInfo onOperation(Operation *op) const;

  std::string dump(triton::FuncOp func) const;
  void dumpToFile(triton::FuncOp func, const std::string &filename) const;

protected:
  std::string getShapeStr(const Type &type) const;

  std::string getUniqueId(Value value) const;
  std::string getUniqueId(Operation *op) const;

  std::string emitNode(const std::string &id, const NodeInfo style) const;
  std::string emitEdge(const std::string &srcId,
                       const std::string &destId) const;

  std::string emitValueNode(Value value) const;
  std::string emitOperationNode(Operation *op) const;
};

/* A subclass of GraphDumper that marks different layout kinds in different
 * colors.*/
class GraphLayoutMarker : public GraphDumper {
public:
  NodeInfo onValue(Value value) const override;

protected:
  std::string getColor(const Type &type) const;
};

// Infers the encoding of the result of op given the source encoding.
std::optional<Attribute> inferDstEncoding(Operation *op, Attribute encoding);

// Infers the encoding of the source of op given the result encoding.
std::optional<Attribute> inferSrcEncoding(Operation *op, Attribute encoding);

bool isExpensiveLoadOrStore(Operation *op);

bool canFoldIntoConversion(Operation *op, Attribute targetEncoding);

// Replace ForOp with a new ForOp with extra operands. The YieldOp is not
// updated and needs to be updated separatly for the loop to be correct.
scf::ForOp replaceForOpWithNewSignature(OpBuilder &rewriter, scf::ForOp loop,
                                        ValueRange newIterOperands);

Operation *cloneWithInferType(mlir::OpBuilder &rewriter, Operation *op,
                              IRMapping &mapping);

// Get backward slice of tensor values starting from the root node along with
// encoding propagation.
LogicalResult getConvertBackwardSlice(
    Value root, SetVector<Value> &slice, Attribute rootEncoding,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation = nullptr);

// Populate pattern to remove dead cycles in ForOp.
void populateForOpDeadArgumentElimination(RewritePatternSet &patterns);

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order);

SmallVector<Value> delinearize(OpBuilder &b, Location loc, unsigned linear,
                               ArrayRef<unsigned> shape);

SmallVector<Value> delinearize(OpBuilder &b, Location loc, Value linear,
                               ArrayRef<unsigned> shape);
Value linearize(OpBuilder &b, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape, ArrayRef<unsigned> order);

Value linearize(OpBuilder &b, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape);
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_UTILITY_H_
