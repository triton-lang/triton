#include "cpu/include/Analysis/TensorPtrShapeInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::triton::cpu {

TensorPtrShapeInfo TensorPtrShapeInfo::join(const TensorPtrShapeInfo &lhs,
                                            const TensorPtrShapeInfo &rhs) {
  // If one argument is not initialized, return the other.
  if (lhs.getRank() == 0)
    return rhs;
  if (rhs.getRank() == 0)
    return lhs;
  assert(lhs.getRank() == rhs.getRank());

  SmallVector<int64_t> shape(lhs.getShape());
  SmallVector<int64_t> strides(lhs.getStrides());
  for (int64_t i = 0; i < lhs.getRank(); ++i) {
    if (shape[i] != rhs.getSize(i))
      shape[i] = ShapedType::kDynamic;
    if (strides[i] != rhs.getStride(i))
      strides[i] = ShapedType::kDynamic;
  }
  return TensorPtrShapeInfo(shape, strides);
}

namespace {

template <class T>
void initPessimisticStateFromFunc(int argNumber, T funcOp,
                                  SmallVectorImpl<int64_t> &shape,
                                  SmallVectorImpl<int64_t> &strides) {
  auto loadFromAttr = [&](std::string_view attrName,
                          SmallVectorImpl<int64_t> &out) {
    Attribute attr = funcOp.getArgAttr(argNumber, attrName);
    if (auto dense_attr = dyn_cast_or_null<DenseElementsAttr>(attr)) {
      auto vals = dense_attr.getValues<int>();
      out = SmallVector<int64_t>(vals.begin(), vals.end());
    }
  };
  loadFromAttr("tt.shape", shape);
  loadFromAttr("tt.strides", strides);
}

TensorPtrShapeInfo getPessimisticValueState(Value value) {
  int rank = 0;
  if (triton::isTensorPointerType(value.getType()))
    rank = cast<RankedTensorType>(getPointeeType(value.getType())).getRank();

  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;

  BlockArgument blockArg = dyn_cast<BlockArgument>(value);

  if (blockArg && blockArg.getOwner()->isEntryBlock()) {
    Operation *op = blockArg.getOwner()->getParentOp();
    if (auto fun = dyn_cast<FunctionOpInterface>(op))
      initPessimisticStateFromFunc(blockArg.getArgNumber(), fun, shape,
                                   strides);
    // llvm codegen check alignment to generate vector load/store
    // would be nice if this wasn't the case
    else if (auto fun = dyn_cast<LLVM::LLVMFuncOp>(op))
      initPessimisticStateFromFunc(blockArg.getArgNumber(), fun, shape,
                                   strides);
  } else if (Operation *op = value.getDefiningOp()) {
    if (isa<RegionBranchOpInterface>(op)) {
      // scf::ForOp, scf::IfOp, scf::WhileOp
      // Control flow operations are initialized with "unknown" state.
    } else {
      // Other operations are conservatively initialized with dynamic
      // shape and strides unless they have specified.
      if (Attribute attr = op->getDiscardableAttr("tt.shape")) {
        auto vals = cast<DenseElementsAttr>(attr).getValues<int>();
        shape = SmallVector<int64_t>(vals.begin(), vals.end());
      } else {
        shape.insert(shape.end(), rank, ShapedType::kDynamic);
      }
      if (Attribute attr = op->getDiscardableAttr("tt.strides")) {
        auto vals = cast<DenseElementsAttr>(attr).getValues<int>();
        strides = SmallVector<int64_t>(vals.begin(), vals.end());
      } else {
        strides.insert(strides.end(), rank, ShapedType::kDynamic);
      }
    }
  }

  return TensorPtrShapeInfo(shape, strides);
}

class ShapeInfoAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                              dataflow::Lattice<TensorPtrShapeInfo>> {
private:
  void
  setToEntryState(dataflow::Lattice<TensorPtrShapeInfo> *lattice) override {
    propagateIfChanged(
        lattice, lattice->join(getPessimisticValueState(lattice->getPoint())));
  }

public:
  ShapeInfoAnalysis(DataFlowSolver &solver);
  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<TensorPtrShapeInfo>>::getLatticeElement;
  using FuncShapeInfoMapT = DenseMap<FunctionOpInterface, TensorPtrShapeInfo>;

  void visitOperation(
      Operation *op,
      ArrayRef<const dataflow::Lattice<TensorPtrShapeInfo> *> operands,
      ArrayRef<dataflow::Lattice<TensorPtrShapeInfo> *> results) override;
};

ShapeInfoAnalysis::ShapeInfoAnalysis(DataFlowSolver &solver)
    : dataflow::SparseForwardDataFlowAnalysis<
          dataflow::Lattice<TensorPtrShapeInfo>>(solver) {}

SmallVector<int64_t> copyConstOrDynamic(OperandRange ops) {
  SmallVector<int64_t> res;
  for (auto op : ops) {
    if (auto cstOp = op.getDefiningOp<arith::ConstantOp>()) {
      auto intAttr = dyn_cast<IntegerAttr>(cstOp.getValue());
      assert(intAttr);
      res.push_back(intAttr.getInt());
    } else {
      res.push_back(ShapedType::kDynamic);
    }
  }
  return res;
}

void ShapeInfoAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const dataflow::Lattice<TensorPtrShapeInfo> *> operands,
    ArrayRef<dataflow::Lattice<TensorPtrShapeInfo> *> results) {
  // TODO: For sure not the right way to do this
  // but why is scf.if not initialized otherwise?
  for (auto op : operands)
    if (op->getValue().getRank() == 0)
      setToEntryState((dataflow::Lattice<TensorPtrShapeInfo> *)op);

  TensorPtrShapeInfo res;
  // Tensor pointers are only produced by MakeTensorPtrOp which has
  // shape and strides as its args, and AdvanceOp which preserves
  // shape and strides of the input pointer.
  if (auto makePtrOp = dyn_cast<MakeTensorPtrOp>(op)) {
    SmallVector<int64_t> shape = copyConstOrDynamic(makePtrOp.getShape());
    SmallVector<int64_t> strides = copyConstOrDynamic(makePtrOp.getStrides());
    res = TensorPtrShapeInfo(shape, strides);
  } else if (auto advOp = dyn_cast<AdvanceOp>(op)) {
    res = operands[0]->getValue();
  }

  // join all lattice elements
  for (auto *result : results)
    propagateIfChanged(result, result->join(res));
}

} // namespace

void ModuleTensorPtrShapeInfoAnalysis::initialize(FunctionOpInterface funcOp) {
  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  ShapeInfoAnalysis *analysis = solver->load<ShapeInfoAnalysis>();
  if (failed(solver->initializeAndRun(funcOp)))
    return;
  auto *shapeInfoMap = getFuncData(funcOp);
  auto updateShapeInfoMap = [&](Value value) {
    auto shapeInfo = analysis->getLatticeElement(value)->getValue();
    TensorPtrShapeInfo curShapeInfo;
    if (shapeInfoMap->count(value)) {
      curShapeInfo =
          TensorPtrShapeInfo::join(shapeInfo, shapeInfoMap->lookup(value));
    } else {
      curShapeInfo = shapeInfo;
    }
    (*shapeInfoMap)[value] = curShapeInfo;
  };
  funcOp.walk([&](Operation *op) {
    for (auto value : op->getResults()) {
      updateShapeInfoMap(value);
    }
  });
  funcOp.walk([&](Block *block) {
    for (auto value : block->getArguments()) {
      updateShapeInfoMap(value);
    }
  });
}

void ModuleTensorPtrShapeInfoAnalysis::update(CallOpInterface callOp,
                                              FunctionOpInterface callee) {
  auto caller = callOp->getParentOfType<FunctionOpInterface>();
  auto *shapeInfoMap = getFuncData(caller);
  for (auto entry : llvm::enumerate(callOp->getOperands())) {
    auto index = entry.index();
    auto value = entry.value();
    auto setAttrFn = [&](StringRef attrName, ArrayRef<int64_t> value) {
      SmallVector<int64_t> curValue(value);
      if (auto attr =
              callee.getArgAttrOfType<DenseElementsAttr>(index, attrName)) {
        auto oldValue = cast<DenseElementsAttr>(attr).getValues<int>();
        assert(oldValue.size() == curValue.size());
        for (size_t i = 0; i < curValue.size(); ++i)
          if (curValue[i] != oldValue[i])
            curValue[i] = ShapedType::kDynamic;
      }
      auto attr = DenseElementsAttr::get(
          VectorType::get(curValue.size(),
                          IntegerType::get(callee.getContext(), 64)),
          ArrayRef<int64_t>(curValue));
      callee.setArgAttr(index, attrName, attr);
    };
    auto shapeInfo = shapeInfoMap->lookup(value);
    if (shapeInfo.getRank()) {
      setAttrFn("tt.shape", shapeInfo.getShape());
      setAttrFn("tt.strides", shapeInfo.getStrides());
    }
  }
}

} // namespace mlir::triton::cpu
