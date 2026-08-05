#ifndef TRITON_GPU_MEMORY_EFFECTS
#define TRITON_GPU_MEMORY_EFFECTS

include "mlir/Interfaces/SideEffectInterfaces.td"

def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;
def GenericSharedMemory : Resource<
    "::mlir::triton::gpu::SharedMemory",
    [{::mlir::triton::gpu::SharedMemoryAccessKindAttr::get(
        getContext(), ::mlir::triton::gpu::SharedMemoryAccessKind::Generic)}]>;
def AsyncSharedMemory : Resource<
    "::mlir::triton::gpu::SharedMemory",
    [{::mlir::triton::gpu::SharedMemoryAccessKindAttr::get(
        getContext(), ::mlir::triton::gpu::SharedMemoryAccessKind::Async)}]>;
def BarrierSharedMemory : Resource<
    "::mlir::triton::gpu::SharedMemory",
    [{::mlir::triton::gpu::SharedMemoryAccessKindAttr::get(
        getContext(), ::mlir::triton::gpu::SharedMemoryAccessKind::Barrier)}]>;

defvar GenericSharedRead = [MemRead<GenericSharedMemory>];
defvar GenericSharedWrite = [MemWrite<GenericSharedMemory>];
defvar GenericSharedReadWrite =
    !listconcat(GenericSharedRead, GenericSharedWrite);
defvar AsyncSharedRead = [MemRead<AsyncSharedMemory>];
defvar AsyncSharedWrite = [MemWrite<AsyncSharedMemory>];
defvar BarrierSharedRead = [MemRead<BarrierSharedMemory>];
defvar BarrierSharedWrite = [MemWrite<BarrierSharedMemory>];
defvar BarrierSharedReadWrite =
    !listconcat(BarrierSharedRead, BarrierSharedWrite);

#endif
