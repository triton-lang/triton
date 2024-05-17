#pragma once

// This file will add older hip functions used in the versioning system
// Find the deprecated functions and structs in hip_device.cpp

// This struct is also kept in hip_device.cpp
typedef struct hipDeviceProp_tR0000 {
  char name[256];            ///< Device name.
  size_t totalGlobalMem;     ///< Size of global memory region (in bytes).
  size_t sharedMemPerBlock;  ///< Size of shared memory region (in bytes).
  int regsPerBlock;          ///< Registers per block.
  int warpSize;              ///< Warp size.
  int maxThreadsPerBlock;    ///< Max work items per work group or workgroup max size.
  int maxThreadsDim[3];      ///< Max number of threads in each dimension (XYZ) of a block.
  int maxGridSize[3];        ///< Max grid dimensions (XYZ).
  int clockRate;             ///< Max clock frequency of the multiProcessors in khz.
  int memoryClockRate;       ///< Max global memory clock frequency in khz.
  int memoryBusWidth;        ///< Global memory bus width in bits.
  size_t totalConstMem;      ///< Size of shared memory region (in bytes).
  int major;  ///< Major compute capability.  On HCC, this is an approximation and features may
              ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
              ///< feature caps.
  int minor;  ///< Minor compute capability.  On HCC, this is an approximation and features may
              ///< differ from CUDA CC.  See the arch feature flags for portable ways to query
              ///< feature caps.
  int multiProcessorCount;          ///< Number of multi-processors (compute units).
  int l2CacheSize;                  ///< L2 cache size.
  int maxThreadsPerMultiProcessor;  ///< Maximum resident threads per multi-processor.
  int computeMode;                  ///< Compute mode.
  int clockInstructionRate;  ///< Frequency in khz of the timer used by the device-side "clock*"
                             ///< instructions.  New for HIP.
  hipDeviceArch_t arch;      ///< Architectural feature flags.  New for HIP.
  int concurrentKernels;     ///< Device can possibly execute multiple kernels concurrently.
  int pciDomainID;           ///< PCI Domain ID
  int pciBusID;              ///< PCI Bus ID.
  int pciDeviceID;           ///< PCI Device ID.
  size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per Multiprocessor.
  int isMultiGpuBoard;                      ///< 1 if device is on a multi-GPU board, 0 if not.
  int canMapHostMemory;                     ///< Check whether HIP can map host memory
  int gcnArch;                              ///< DEPRECATED: use gcnArchName instead
  char gcnArchName[256];                    ///< AMD GCN Arch Name.
  int integrated;                           ///< APU vs dGPU
  int cooperativeLaunch;                    ///< HIP device supports cooperative launch
  int cooperativeMultiDeviceLaunch;         ///< HIP device supports cooperative launch on multiple
                                            ///< devices
  int maxTexture1DLinear;                   ///< Maximum size for 1D textures bound to linear memory
  int maxTexture1D;                         ///< Maximum number of elements in 1D images
  int maxTexture2D[2];  ///< Maximum dimensions (width, height) of 2D images, in image elements
  int maxTexture3D[3];  ///< Maximum dimensions (width, height, depth) of 3D images, in image
                        ///< elements
  unsigned int* hdpMemFlushCntl;  ///< Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
  unsigned int* hdpRegFlushCntl;  ///< Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
  size_t memPitch;                ///< Maximum pitch in bytes allowed by memory copies
  size_t textureAlignment;        ///< Alignment requirement for textures
  size_t texturePitchAlignment;   ///< Pitch alignment requirement for texture references bound to
                                  ///< pitched memory
  int kernelExecTimeoutEnabled;   ///< Run time limit for kernels executed on the device
  int ECCEnabled;                 ///< Device has ECC support enabled
  int tccDriver;                  ///< 1:If device is Tesla device using TCC driver, else 0
  int cooperativeMultiDeviceUnmatchedFunc;       ///< HIP device supports cooperative launch on
                                                 ///< multiple
                                                 /// devices with unmatched functions
  int cooperativeMultiDeviceUnmatchedGridDim;    ///< HIP device supports cooperative launch on
                                                 ///< multiple
                                                 /// devices with unmatched grid dimensions
  int cooperativeMultiDeviceUnmatchedBlockDim;   ///< HIP device supports cooperative launch on
                                                 ///< multiple
                                                 /// devices with unmatched block dimensions
  int cooperativeMultiDeviceUnmatchedSharedMem;  ///< HIP device supports cooperative launch on
                                                 ///< multiple
                                                 /// devices with unmatched shared memories
  int isLargeBar;                                ///< 1: if it is a large PCI bar device, else 0
  int asicRevision;                              ///< Revision of the GPU in this device
  int managedMemory;                   ///< Device supports allocating managed memory on this system
  int directManagedMemAccessFromHost;  ///< Host can directly access managed memory on the device
                                       ///< without migration
  int concurrentManagedAccess;  ///< Device can coherently access managed memory concurrently with
                                ///< the CPU
  int pageableMemoryAccess;     ///< Device supports coherently accessing pageable memory
                                ///< without calling hipHostRegister on it
  int pageableMemoryAccessUsesHostPageTables;  ///< Device accesses pageable memory via the host's
                                               ///< page tables
} hipDeviceProp_tR0000;


#ifdef __cplusplus
extern "C" {
#endif

hipError_t hipGetDevicePropertiesR0000(hipDeviceProp_tR0000* prop, int device);
hipError_t hipChooseDeviceR0000(int* device, const hipDeviceProp_tR0000* prop);

#ifdef __cplusplus
}
#endif
