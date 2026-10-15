#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOWERSTORAGETYPES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// This list is deliberately explicit: a new operation must define its native
// type semantics before storage lowering can expose integer values to it.
bool supportsStorageTypes(Operation *op) {
  if (op->getName().getStringRef() == "arith.constant") {
    // Typed E8M0 constants can occur in saved IR after bitcast folding. Lower
    // their encoded bits, never their numerical value, to integer storage.
    Attribute value = op->getAttr("value");
    if (auto scalar = dyn_cast_or_null<FloatAttr>(value))
      return isa<Float8E8M0FNUType>(scalar.getType());
    if (auto elements = dyn_cast_or_null<DenseElementsAttr>(value))
      return isa<Float8E8M0FNUType>(elements.getElementType());
    return false;
  }
  return llvm::StringSwitch<bool>(op->getName().getStringRef())
      .Cases({"tt.func", "tt.call", "tt.return"}, true)
      .Cases({"scf.for", "scf.if", "scf.while", "scf.yield", "scf.condition"},
             true)
      .Cases({"cf.br", "cf.cond_br", "cf.switch"}, true)
      .Cases({"arith.select", "arith.bitcast", "ub.poison"}, true)
      .Cases({"tt.bitcast", "tt.int_to_ptr", "tt.ptr_to_int", "tt.addptr"},
             true)
      .Cases({"tt.load", "tt.store", "tt.splat", "tt.unsplat"}, true)
      .Cases({"tt.expand_dims", "tt.reshape", "tt.broadcast", "tt.trans"}, true)
      .Cases({"tt.join", "tt.split", "tt.gather"}, true)
      .Cases({"tt.reduce", "tt.reduce.return", "tt.scan", "tt.scan.return"},
             true)
      .Cases({"tt.map_elementwise", "tt.map_elementwise.return"}, true)
      .Cases({"tt.make_tensor_descriptor", "tt.descriptor_load",
              "tt.descriptor_store", "tt.descriptor_gather",
              "tt.descriptor_scatter"},
             true)
      .Case("tt.dot_scaled", true)
      .Cases({"gluon.set_auto_layout", "ttg.convert_layout"}, true)
      .Cases({"ttg.local_alloc", "ttg.local_dealloc", "ttg.local_load",
              "ttg.local_store", "ttg.local_gather", "ttg.local_scatter"},
             true)
      .Cases({"ttg.memdesc_index", "ttg.memdesc_subslice", "ttg.memdesc_trans",
              "ttg.memdesc_reshape", "ttg.memdesc_reinterpret"},
             true)
      .Cases({"ttg.async_copy_global_to_local", "ttg.global_scratch_alloc"},
             true)
      .Cases({"ttg.warp_specialize", "ttg.warp_specialize.partitions",
              "ttg.warp_yield", "ttg.warp_return"},
             true)
      .Case("ttg.fp4_to_fp", true)
      .Cases({"ttng.tmem_alloc", "ttng.tmem_load", "ttng.tmem_store",
              "ttng.tmem_copy", "ttng.tmem_subslice"},
             true)
      .Cases({"ttng.async_shared_store", "ttng.async_tma_copy_global_to_local",
              "ttng.async_tma_copy_local_to_global", "ttng.async_tma_gather",
              "ttng.async_tma_scatter"},
             true)
      .Cases({"ttng.reinterpret_tensor_descriptor", "ttng.tc_gen5_mma_scaled",
              "ttng.packed_arith"},
             true)
      .Cases({"amdg.extract_slice", "amdg.concat", "amdg.in_thread_transpose"},
             true)
      .Cases({"amdg.scaled_upcast_fp4", "amdg.scaled_upcast_fp8",
              "amdg.scaled_downcast_fp4", "amdg.scaled_downcast_fp8"},
             true)
      .Cases({"amdg.buffer_load", "amdg.buffer_store",
              "amdg.buffer_load_to_local", "amdg.local_load_packed_transposed"},
             true)
      .Cases({"amdg.async_copy_local_to_global",
              "amdg.async_tdm_copy_global_to_local",
              "amdg.async_tdm_fused_copy_global_to_local",
              "amdg.async_tdm_copy_local_to_global"},
             true)
      .Cases({"amdg.async_tdm_gather", "amdg.async_tdm_scatter",
              "amdg.update_tensor_descriptor", "amdg.tdm_prefetch"},
             true)
      .Default(false);
}

bool usesStorageTypes(Operation *op) {
  AttrTypeWalker walker;
  walker.addWalk([](Type type) {
    return isLowPrecisionStorageType(type) ? WalkResult::interrupt()
                                           : WalkResult::advance();
  });
  for (Type type : op->getOperandTypes())
    if (walker.walk(type).wasInterrupted())
      return true;
  for (Type type : op->getResultTypes())
    if (walker.walk(type).wasInterrupted())
      return true;
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (BlockArgument arg : block.getArguments())
        if (walker.walk(arg.getType()).wasInterrupted())
          return true;
  return walker.walk(op->getAttrDictionary()).wasInterrupted();
}

struct TritonLowerStorageTypes
    : impl::TritonLowerStorageTypesBase<TritonLowerStorageTypes> {
  using TritonLowerStorageTypesBase::TritonLowerStorageTypesBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    StringRef compilationTarget = target;
    if (compilationTarget.empty())
      if (auto attr = mod->getAttrOfType<StringAttr>("ttg.target"))
        compilationTarget = attr.getValue();
    bool hasStorageTypes = false;
    WalkResult legal = mod.walk([&](Operation *op) {
      if (!usesStorageTypes(op))
        return WalkResult::advance();
      hasStorageTypes = true;
      // AMD WMMA selects each operand's scale format independently. NVIDIA
      // scaled MMA shares one format, so check it before E8M0 becomes i8.
      // An unspecified target must not silently permit that reinterpretation.
      if (auto dot = dyn_cast<DotScaledOp>(op);
          dot && !compilationTarget.starts_with("hip:")) {
        auto isNativeScale = [](Value scale) {
          return scale &&
                 isa<Float8E8M0FNUType>(getElementTypeOrSelf(scale.getType()));
        };
        if (isNativeScale(dot.getAScale()) || isNativeScale(dot.getBScale())) {
          for (Value scale : {dot.getAScale(), dot.getBScale()}) {
            if (!scale)
              continue;
            Type elementType = getElementTypeOrSelf(scale.getType());
            if (!elementType.isInteger(8) &&
                !isa<Float8E8M0FNUType>(elementType)) {
              op->emitOpError("native E8M0 scales cannot be paired with a "
                              "different scale format on this target");
              return WalkResult::interrupt();
            }
          }
        }
      }
      if (supportsStorageTypes(op))
        return WalkResult::advance();
      op->emitOpError("does not support native low-precision storage types; "
                      "use an explicit bitcast to operate on their bits");
      return WalkResult::interrupt();
    });
    if (legal.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    if (!hasStorageTypes)
      return;
    // In particular, check copy element types and packed operand/scale roles
    // before erasing the information those checks need.
    if (failed(verify(mod))) {
      signalPassFailure();
      return;
    }

    // packed_arith already uses E8M0 as an instruction selector, unlike scaled
    // dot's explicitly specified formats. Keep its E8M0 operands typed at the
    // instruction boundary; treating those operands as i8 would select FP4.
    SmallVector<std::pair<OpOperand *, Type>> e8m0Operands;
    mod.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "ttng.packed_arith")
        return;
      for (OpOperand &operand : op->getOpOperands())
        if (isa<Float8E8M0FNUType>(
                getElementTypeOrSelf(operand.get().getType())))
          e8m0Operands.emplace_back(&operand, operand.get().getType());
    });

    AttrTypeReplacer replacer;
    replacer.addReplacement([](FloatAttr attr) -> std::optional<Attribute> {
      if (isa<Float8E8M0FNUType>(attr.getType()))
        return IntegerAttr::get(IntegerType::get(attr.getContext(), 8),
                                attr.getValue().bitcastToAPInt());
      return std::nullopt;
    });
    replacer.addReplacement(
        [](DenseElementsAttr attr) -> std::optional<Attribute> {
          if (isa<Float8E8M0FNUType>(attr.getElementType()))
            return attr.bitcast(IntegerType::get(attr.getContext(), 8));
          return std::nullopt;
        });
    replacer.addReplacement([](Type type) -> std::optional<Type> {
      if (isLowPrecisionStorageType(type))
        return IntegerType::get(type.getContext(), 8);
      return std::nullopt;
    });
    replacer.recursivelyReplaceElementsIn(mod, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
    for (auto [operand, type] : e8m0Operands) {
      Operation *op = operand->getOwner();
      OpBuilder builder(op);
      operand->set(
          BitcastOp::create(builder, op->getLoc(), type, operand->get()));
    }
  }
};

} // namespace
} // namespace mlir::triton
