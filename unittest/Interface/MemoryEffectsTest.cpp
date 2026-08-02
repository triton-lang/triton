#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

#include <gtest/gtest.h>

namespace mlir {

TEST(Interface, SharedMemoryResourcesRetainParent) {
  auto *shared = triton::gpu::SharedMemory::get();
  MLIRContext context;
  Block block;
  BlockArgument value = block.addArgument(IntegerType::get(&context, 32),
                                          UnknownLoc::get(&context));
  for (auto *resource : {static_cast<SideEffects::Resource *>(
                             triton::gpu::GenericSharedMemory::get()),
                         static_cast<SideEffects::Resource *>(
                             triton::gpu::AsyncSharedMemory::get()),
                         static_cast<SideEffects::Resource *>(
                             triton::gpu::BarrierSharedMemory::get())}) {
    EXPECT_TRUE(resource->isSubresourceOf(shared));

    SmallVector<MemoryEffects::EffectInstance> effects;
    triton::gpu::addSharedMemoryEffects<MemoryEffects::Write>(effects, value,
                                                              resource);
    ASSERT_EQ(effects.size(), 2u);
    EXPECT_EQ(effects[0].getResource(), shared);
    EXPECT_EQ(effects[1].getResource(), resource);
  }
}

TEST(Interface, SharedMemoryEffectsAreClassified) {
  DialectRegistry registry;
  registry
      .insert<triton::gpu::TritonGPUDialect,
              triton::instrument::TritonInstrumentDialect,
              triton::nvidia_gpu::TritonNvidiaGPUDialect,
              triton::amdgpu::TritonAMDGPUDialect, triton::nvws::NVWSDialect,
              triton::proton::gpu::ProtonGPUDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr unsigned numValues = 128;
  auto dim0 = StringAttr::get(&context, "dim0");
  auto layout = triton::LinearLayout::identity1D(
                    1, StringAttr::get(&context, "offset"), dim0) *
                triton::LinearLayout::identity1D(
                    1, StringAttr::get(&context, "block"), dim0);
  auto encoding = triton::gpu::SharedLinearEncodingAttr::get(
      &context, std::move(layout), /*layoutAlignment=*/1);
  auto memDescType = triton::gpu::MemDescType::get(
      {1}, IntegerType::get(&context, 32), encoding,
      triton::gpu::SharedMemorySpaceAttr::get(&context),
      /*mutableMemory=*/true, /*allocShape=*/{1});
  Block block;
  SmallVector<Value> memDescOperands, tensorOperands, i1Operands, tokenOperands;
  for (unsigned i = 0; i < numValues; ++i) {
    memDescOperands.push_back(
        block.addArgument(memDescType, UnknownLoc::get(&context)));
    tensorOperands.push_back(block.addArgument(
        RankedTensorType::get({1}, IntegerType::get(&context, 32)),
        UnknownLoc::get(&context)));
    i1Operands.push_back(block.addArgument(IntegerType::get(&context, 1),
                                           UnknownLoc::get(&context)));
    tokenOperands.push_back(block.addArgument(
        triton::gpu::AsyncTokenType::get(&context), UnknownLoc::get(&context)));
  }
  SmallVector<Type> resultTypes(numValues, memDescType);
  auto isClassifiedResource = [](SideEffects::Resource *resource) {
    return resource == triton::gpu::GenericSharedMemory::get() ||
           resource == triton::gpu::AsyncSharedMemory::get() ||
           resource == triton::gpu::BarrierSharedMemory::get();
  };
  auto sameEffect = [](const MemoryEffects::EffectInstance &lhs,
                       const MemoryEffects::EffectInstance &rhs) {
    return lhs.getEffect() == rhs.getEffect() &&
           lhs.getValue() == rhs.getValue() &&
           lhs.getSymbolRef() == rhs.getSymbolRef() &&
           lhs.getParameters() == rhs.getParameters() &&
           lhs.getStage() == rhs.getStage() &&
           lhs.getEffectOnFullRegion() == rhs.getEffectOnFullRegion();
  };

  for (StringRef dialect :
       {triton::gpu::TritonGPUDialect::getDialectNamespace(),
        triton::instrument::TritonInstrumentDialect::getDialectNamespace(),
        triton::nvidia_gpu::TritonNvidiaGPUDialect::getDialectNamespace(),
        triton::amdgpu::TritonAMDGPUDialect::getDialectNamespace(),
        triton::nvws::NVWSDialect::getDialectNamespace(),
        triton::proton::gpu::ProtonGPUDialect::getDialectNamespace()}) {
    for (RegisteredOperationName opName :
         context.getRegisteredOperationsByDialect(dialect)) {
      if (!opName.hasInterface<MemoryEffectOpInterface>())
        continue;
      OperationState state(UnknownLoc::get(&context), opName.getStringRef());
      SmallVector<Value> opOperands = memDescOperands;
      if (opName.getStringRef() ==
              triton::gpu::LocalAllocOp::getOperationName() ||
          opName.getStringRef() ==
              triton::nvidia_gpu::TMEMAllocOp::getOperationName()) {
        opOperands = tensorOperands;
      } else if (opName.getStringRef() ==
                 triton::nvidia_gpu::TCGen5MMAOp::getOperationName()) {
        opOperands[3] = tokenOperands[3];
        opOperands[4] = i1Operands[4];
        opOperands[5] = i1Operands[5];
        opOperands[7] = i1Operands[7];
      } else if (opName.getStringRef() ==
                 triton::nvidia_gpu::TCGen5MMAScaledOp::getOperationName()) {
        opOperands[3] = tokenOperands[3];
        opOperands[6] = i1Operands[6];
        opOperands[7] = i1Operands[7];
        opOperands[9] = i1Operands[9];
      }
      state.addOperands(opOperands);
      state.addTypes(resultTypes);
      OwningOpRef<Operation *> op(Operation::create(state));
      NamedAttrList inherentAttrs;
      opName.populateInherentAttrs(op.get(), inherentAttrs);
      for (StringRef name : {"operandSegmentSizes", "resultSegmentSizes"}) {
        auto segments =
            dyn_cast_or_null<DenseI32ArrayAttr>(inherentAttrs.get(name));
        if (!segments)
          continue;
        SmallVector<int32_t> sizes(segments.size(), 1);
        opName.setInherentAttr(op.get(), StringAttr::get(&context, name),
                               DenseI32ArrayAttr::get(&context, sizes));
      }

      SmallVector<MemoryEffects::EffectInstance> effects;
      cast<MemoryEffectOpInterface>(op.get()).getEffects(effects);
      for (const MemoryEffects::EffectInstance &effect : effects) {
        if (effect.getResource() != triton::gpu::SharedMemory::get() ||
            !isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect()))
          continue;
        EXPECT_EQ(llvm::count_if(effects,
                                 [&](const auto &candidate) {
                                   return isClassifiedResource(
                                              candidate.getResource()) &&
                                          sameEffect(effect, candidate);
                                 }),
                  1u)
            << opName.getStringRef().str();
      }
    }
  }
}

} // namespace mlir
