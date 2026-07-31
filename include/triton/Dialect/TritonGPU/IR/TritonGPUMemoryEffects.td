#ifndef TRITON_GPU_MEMORY_EFFECTS
#define TRITON_GPU_MEMORY_EFFECTS

include "mlir/Interfaces/SideEffectInterfaces.td"

def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;
def GenericSharedMemory : Resource<"::mlir::triton::gpu::GenericSharedMemory">;
def AsyncSharedMemory : Resource<"::mlir::triton::gpu::AsyncSharedMemory">;
def BarrierSharedMemory : Resource<"::mlir::triton::gpu::BarrierSharedMemory">;

// Keep the parent effect for consumers that identify shared memory by its
// original resource, and add exactly one specialized classification.
defvar GenericSharedRead = [MemRead<SharedMemory>, MemRead<GenericSharedMemory>];
defvar GenericSharedWrite =
    [MemWrite<SharedMemory>, MemWrite<GenericSharedMemory>];
defvar GenericSharedReadWrite =
    !listconcat(GenericSharedRead, GenericSharedWrite);
defvar AsyncSharedRead = [MemRead<SharedMemory>, MemRead<AsyncSharedMemory>];
defvar AsyncSharedWrite = [MemWrite<SharedMemory>, MemWrite<AsyncSharedMemory>];
defvar BarrierSharedRead = [MemRead<SharedMemory>, MemRead<BarrierSharedMemory>];
defvar BarrierSharedWrite =
    [MemWrite<SharedMemory>, MemWrite<BarrierSharedMemory>];
defvar BarrierSharedReadWrite =
    !listconcat(BarrierSharedRead, BarrierSharedWrite);

class SharedMemoryEffectCount<list<OpVariableDecorator> decorators,
                              string resource, string effect> {
  int value = !size(!filter(decorator, decorators,
                    !if(!isa<MemoryEffect>(decorator),
                        !and(!eq(!cast<MemoryEffect>(decorator).resource,
                                 resource),
                             !eq(!cast<MemoryEffect>(decorator).effect,
                                 effect)), 0)));
}

class ClassifiedSharedMemoryEffects<list<OpVariableDecorator> decorators> {
  defvar read = "::mlir::MemoryEffects::Read";
  defvar write = "::mlir::MemoryEffects::Write";
  defvar sharedReads =
      SharedMemoryEffectCount<decorators, SharedMemory.name, read>.value;
  defvar sharedWrites =
      SharedMemoryEffectCount<decorators, SharedMemory.name, write>.value;
  defvar classifiedReads = !add(
      !add(SharedMemoryEffectCount<decorators, GenericSharedMemory.name,
                                   read>.value,
           SharedMemoryEffectCount<decorators, AsyncSharedMemory.name,
                                   read>.value),
      SharedMemoryEffectCount<decorators, BarrierSharedMemory.name, read>.value);
  defvar classifiedWrites = !add(
      !add(SharedMemoryEffectCount<decorators, GenericSharedMemory.name,
                                   write>.value,
           SharedMemoryEffectCount<decorators, AsyncSharedMemory.name,
                                   write>.value),
      SharedMemoryEffectCount<decorators, BarrierSharedMemory.name,
                              write>.value);
  bit value = !and(!or(!eq(sharedReads, 0), !eq(classifiedReads, 1)),
                   !or(!eq(sharedWrites, 0), !eq(classifiedWrites, 1)));
}

class ClassifiedSharedMemoryOperand<dag operands, int index> {
  defvar operand = !getdagarg<OpVariable>(operands, index);
  bit value = !if(!initialized(operand),
                  ClassifiedSharedMemoryEffects<
                      !cast<OpVariable>(operand).decorators>.value,
                  1);
}

class ClassifiedSharedMemoryTrait<list<Trait> traits, int index> {
  defvar trait = traits[index];
  bit value = !if(!isa<MemoryEffects>(trait),
                  ClassifiedSharedMemoryEffects<
                      !cast<MemoryEffects>(trait).effects>.value,
                  1);
}

class VerifySharedMemoryEffects<dag operands, list<Trait> traits> {
  assert !foldl(1, !range(!size(operands)), result, index,
                !and(result,
                     ClassifiedSharedMemoryOperand<operands, index>.value)),
      "shared-memory reads and writes require exactly one generic, async, or barrier effect";
  assert !foldl(1, !range(!size(traits)), result, index,
                !and(result,
                     ClassifiedSharedMemoryTrait<traits, index>.value)),
      "shared-memory operation effects require exactly one generic, async, or barrier effect";
}

#endif
