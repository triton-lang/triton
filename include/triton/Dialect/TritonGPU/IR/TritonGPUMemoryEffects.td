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

#endif
