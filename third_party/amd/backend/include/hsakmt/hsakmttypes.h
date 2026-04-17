/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _HSAKMTTYPES_H_
#define _HSAKMTTYPES_H_

// the definitions and THUNK API are version specific - define the version
// numbers here
#define HSAKMT_VERSION_MAJOR 0
#define HSAKMT_VERSION_MINOR 99

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN64) || defined(_WINDOWS) || defined(_WIN32)

#if defined(_WIN32)
#define HSAKMTAPI __stdcall
#else
#define HSAKMTAPI
#endif

typedef unsigned char HSAuint8;
typedef char HSAint8;
typedef unsigned short HSAuint16;
typedef signed short HSAint16;
typedef unsigned __int32 HSAuint32;
typedef signed __int32 HSAint32;
typedef signed __int64 HSAint64;
typedef unsigned __int64 HSAuint64;

#elif defined(__linux__) || defined(__APPLE__)

#include <stdbool.h>
#include <stdint.h>

#define HSAKMTAPI

typedef uint8_t HSAuint8;
typedef int8_t HSAint8;
typedef uint16_t HSAuint16;
typedef int16_t HSAint16;
typedef uint32_t HSAuint32;
typedef int32_t HSAint32;
typedef int64_t HSAint64;
typedef uint64_t HSAuint64;

#endif

typedef void *HSA_HANDLE;
typedef HSAuint64 HSA_QUEUEID;
// An HSA_QUEUEID that is never a valid queue ID.
#define INVALID_QUEUEID 0xFFFFFFFFFFFFFFFFULL

// A PID that is never a valid process ID.
#define INVALID_PID 0xFFFFFFFF

// // A HSA_NODEID that is never a valid node ID.
#define INVALID_NODEID 0xFFFFFFFF

// This is included in order to force the alignments to be 4 bytes so that
// it avoids extra padding added by the compiler when a 64-bit binary is
// generated.
#pragma pack(push, hsakmttypes_h, 4)

//
// HSA STATUS codes returned by the KFD Interfaces
//

typedef enum _HSAKMT_STATUS {
  HSAKMT_STATUS_SUCCESS = 0, // Operation successful
  HSAKMT_STATUS_ERROR = 1,   // General error return if not otherwise specified
  HSAKMT_STATUS_DRIVER_MISMATCH =
      2, // User mode component is not compatible with kernel HSA driver

  HSAKMT_STATUS_INVALID_PARAMETER =
      3,                            // KFD identifies input parameters invalid
  HSAKMT_STATUS_INVALID_HANDLE = 4, // KFD identifies handle parameter invalid
  HSAKMT_STATUS_INVALID_NODE_UNIT =
      5, // KFD identifies node or unit parameter invalid

  HSAKMT_STATUS_NO_MEMORY =
      6, // No memory available (when allocating queues or memory)
  HSAKMT_STATUS_BUFFER_TOO_SMALL =
      7, // A buffer needed to handle a request is too small

  HSAKMT_STATUS_NOT_IMPLEMENTED =
      10, // KFD function is not implemented for this set of paramters
  HSAKMT_STATUS_NOT_SUPPORTED =
      11,                         // KFD function is not supported on this node
  HSAKMT_STATUS_UNAVAILABLE = 12, // KFD function is not available currently on
                                  // this node (but may be at a later time)
  HSAKMT_STATUS_OUT_OF_RESOURCES =
      13, // KFD function request exceeds the resources currently available.

  HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED = 20, // KFD driver path not opened
  HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR =
      21, // user-kernel mode communication failure
  HSAKMT_STATUS_KERNEL_ALREADY_OPENED = 22, // KFD driver path already opened
  HSAKMT_STATUS_HSAMMU_UNAVAILABLE =
      23, // ATS/PRI 1.1 (Address Translation Services) not available
          // (IOMMU driver not installed or not-available)

  HSAKMT_STATUS_WAIT_FAILURE = 30, // The wait operation failed
  HSAKMT_STATUS_WAIT_TIMEOUT = 31, // The wait operation timed out

  HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED =
      35,                                   // Memory buffer already registered
  HSAKMT_STATUS_MEMORY_NOT_REGISTERED = 36, // Memory buffer not registered
  HSAKMT_STATUS_MEMORY_ALIGNMENT = 37,      // Memory parameter not aligned

} HSAKMT_STATUS;

//
// HSA KFD interface version information. Calling software has to validate that
// it meets the minimum interface version as described in the API specification.
// All future structures will be extended in a backward compatible fashion.
//

typedef struct _HsaVersionInfo {
  HSAuint32
      KernelInterfaceMajorVersion; // supported kernel interface major version
  HSAuint32
      KernelInterfaceMinorVersion; // supported kernel interface minor version
} HsaVersionInfo;

//
// HSA Topology Discovery Infrastructure structure definitions.
// The infrastructure implementation is based on design specified in the Kernel
// HSA Driver ADD The discoverable data is retrieved from ACPI structures in the
// platform infrastructure, as defined in the "Heterogeneous System Architecture
// Detail Topology" specification.
//
// The following structure is returned on a call to
// hsaKmtAcquireSystemProperties() as output. When the call is made within a
// process context, a "snapshot" of the topology information is taken within the
// KFD to avoid any changes during the enumeration process. The Snapshot is
// released when hsaKmtReleaseSystemProperties() is called or when the process
// exits or is terminated.
//

typedef struct _HsaSystemProperties {
  HSAuint32 NumNodes; // the number of "H-NUMA" memory nodes.
                      // each node represents a discoverable node of the system
                      // All other enumeration is done on a per-node basis

  HSAuint32
      PlatformOem; // identifies HSA platform, reflects the OEMID in the CRAT
  HSAuint32 PlatformId; // HSA platform ID, reflects OEM TableID in the CRAT
  HSAuint32
      PlatformRev; // HSA platform revision, reflects Platform Table Revision ID
} HsaSystemProperties;

typedef union {
  HSAuint32 Value;
  struct {
    unsigned int uCode : 10;   // ucode packet processor version
    unsigned int Major : 6;    // GFXIP Major engine version
    unsigned int Minor : 8;    // GFXIP Minor engine version
    unsigned int Stepping : 8; // GFXIP Stepping info
  } ui32;
} HSA_ENGINE_ID;

typedef union {
  HSAuint32 Value;
  struct {
    unsigned int uCodeSDMA : 10; // ucode version SDMA engine
    unsigned int uCodeRes : 10;  // ucode version (reserved)
    unsigned int Reserved : 12;  // Reserved, must be 0
  };
} HSA_ENGINE_VERSION;

typedef union {
  HSAuint32 Value;
  struct {
    unsigned int HotPluggable : 1; // the node may be removed by some system
                                   // action (event will be sent)
    unsigned int
        HSAMMUPresent : 1; // This node has an ATS/PRI 1.1 compatible
                           // translation agent in the system (e.g. IOMMUv2)
    unsigned int
        SharedWithGraphics : 1; // this HSA nodes' GPU function is also used for
                                // OS primary graphics render (= UI)
    unsigned int QueueSizePowerOfTwo : 1; // This node GPU requires the queue
                                          // size to be a power of 2 value
    unsigned int QueueSize32bit : 1; // This node GPU requires the queue size to
                                     // be less than 4GB
    unsigned int
        QueueIdleEvent : 1; // This node GPU supports notification on Queue Idle
    unsigned int
        VALimit : 1; // This node GPU has limited VA range for platform
                     // (typical 40bit). Affects shared VM use for 64bit apps
    unsigned int WatchPointsSupported : 1; // Indicates if Watchpoints are
                                           // available on the node.
    unsigned int WatchPointsTotalBits : 4; // Watchpoints available. To
                                           // determine the number use 2^value

    unsigned int
        DoorbellType : 2; // 0: This node has pre-1.0 doorbell characteristic
                          // 1: This node has 1.0 doorbell characteristic
                          // 2,3: reserved for future use
    unsigned int AQLQueueDoubleMap : 1;  // The unit needs a VA “double map”
    unsigned int DebugTrapSupported : 1; // Indicates if Debug Trap is supported
                                         // on the node.
    unsigned int WaveLaunchTrapOverrideSupported : 1; // Indicates if Wave
                                                      // Launch Trap Override is
                                                      // supported on the node.
    unsigned int WaveLaunchModeSupported : 1; // Indicates if Wave Launch Mode
                                              // is supported on the node.
    unsigned int PreciseMemoryOperationsSupported : 1; // Indicates if Precise
                                                       // Memory Operations are
                                                       // supported on the node.
    unsigned int DEPRECATED_SRAM_EDCSupport : 1; // Old buggy user mode depends
                                                 // on this being 0
    unsigned int Mem_EDCSupport : 1; // Indicates if GFX internal DRAM/HBM
                                     // EDC/ECC functionality is active
    unsigned int RASEventNotify : 1; // Indicates if GFX extended RASFeatures
                                     // and RAS EventNotify status is available
    unsigned int ASICRevision : 4; // Indicates the ASIC revision of the chip on
                                   // this node.
    unsigned int SRAM_EDCSupport : 1; // Indicates if GFX internal SRAM EDC/ECC
                                      // functionality is active
    unsigned int SVMAPISupported : 1; // Whether or not the SVM API is supported
    unsigned int CoherentHostAccess : 1; // Whether or not device memory can be
                                         // coherently accessed by the host CPU
    unsigned int DebugSupportedFirmware : 1;        // Indicates if HWS firmware
                                                    // supports GPU debugging
    unsigned int PreciseALUOperationsSupported : 1; // Indicates if precise ALU
                                                    // operations are supported
                                                    // for GPU debugging
    unsigned int
        PerQueueResetSupported : 1; // Indicates per-queue reset supported
  } ui32;
} HSA_CAPABILITY;

typedef union {
  HSAuint32 Value;
  struct {
    unsigned int PerSDMAQueueResetSupported : 1; // Indicates per-sdma queue
                                                 // reset supported
    unsigned int AqlEmulationPm4_ : 1; // Indicates device uses AQL emulation
                                       // via PM4 packets
    unsigned int Reserved : 30;        // Reserved
  } ui32;
} HSA_CAPABILITY2;

// Debug Properties and values
// HSA runtime may expose a subset of the capabilities outlined to the applicati
typedef union {
  HSAuint64 Value;
  struct {
    HSAuint64 WatchAddrMaskLoBit : 4; // Only bits
                                      // WatchAddrMaskLoBit..WatchAddrMaskHiBit
                                      // of the
    HSAuint64 WatchAddrMaskHiBit : 6; // watch address mask are used.
                                      // 0 is the least significant bit.
    HSAuint64 DispatchInfoAlwaysValid : 1;    // 0 if control of TTMP setup is
                                              // controlled on a per process
                                              // basis and is not always enabled
                                              // 1 if TTMP setup is always
                                              // enabled
    HSAuint64 AddressWatchpointShareKind : 1; // whether the address watchpoint
                                              //     is per process or shared
                                              //     with all proccesses
                                              // 0 if shared or unsuppoted
                                              //    (unsupported indicated by
                                              //    address_watchpoint_count ==
                                              //    0) All current devices have
                                              //    shared watchpoints
                                              // 1 if unshared
    HSAuint64 Reserved : 52;                  //
  };
} HSA_DEBUG_PROPERTIES;

//
// HSA node properties. This structure is an output parameter of
// hsaKmtGetNodeProperties() The application or runtime can use the information
// herein to size the topology management structures Unless there is some very
// weird setup, there is at most one "GPU" device (with a certain number of
// throughput compute units (= SIMDs) associated with a H-NUMA node.
//

#define HSA_PUBLIC_NAME_SIZE 64 // Marketing name string size

typedef struct _HsaNodeProperties {
  HSAuint32 NumCPUCores; // # of latency (= CPU) cores present on this HSA node.
                         // This value is 0 for a HSA node with no such cores,
                         // e.g a "discrete HSA GPU"
  HSAuint32
      NumFComputeCores; // # of HSA throughtput (= GPU) FCompute cores ("SIMD")
                        // present in a node. This value is 0 if no FCompute
                        // cores are present (e.g. pure "CPU node").
  HSAuint32
      NumNeuralCores; // # of HSA neural processing units (= AIE) present in a
                      // node. This value is 0 if there are no NeuralCores.
  HSAuint32 NumMemoryBanks; // # of discoverable memory bank affinity properties
                            // on this "H-NUMA" node.
  HSAuint32 NumCaches; // # of discoverable cache affinity properties on this
                       // "H-NUMA"  node.

  HSAuint32 NumIOLinks; // # of discoverable IO link affinity properties of this
                        // node connecting to other nodes.

  HSAuint32 CComputeIdLo; // low value of the logical processor ID of the
                          // latency (= CPU) cores available on this node
  HSAuint32 FComputeIdLo; // low value of the logical processor ID of the
                          // throughput (= GPU) units available on this node

  HSA_CAPABILITY Capability;   // see above
  HSA_CAPABILITY2 Capability2; // see above

  HSAuint32
      MaxWavesPerSIMD; // This identifies the max. number of launched waves per
                       // SIMD. If NumFComputeCores is 0, this value is ignored.
  HSAuint32
      LDSSizeInKB; // Size of Local Data Store in Kilobytes per SIMD Wavefront
  HSAuint32 GDSSizeInKB; // Size of Global Data Store in Kilobytes shared across
                         // SIMD Wavefronts

  HSAuint32 WaveFrontSize; // Number of SIMD cores per wavefront executed,
                           // typically 64, may be 32 or a different value for
                           // some HSA based architectures

  HSAuint32 NumShaderBanks; // Number of Shader Banks or Shader Engines, typical
                            // values are 1 or 2

  HSAuint32 NumArrays;     // Number of SIMD arrays per engine
  HSAuint32 NumCUPerArray; // Number of Compute Units (CU) per SIMD array
  HSAuint32 NumSIMDPerCU;  // Number of SIMD representing a Compute Unit (CU)

  HSAuint32 MaxSlotsScratchCU; // Number of temp. memory ("scratch") wave slots
                               // available to access, may be 0 if HW has no
                               // restrictions

  HSA_ENGINE_ID
  EngineId; // Identifier (rev) of the GPU uEngine or Firmware, may be 0
  HSA_ENGINE_ID OverrideEngineId; // Identifier (rev) of the Overrided GPU
                                  // uEngine or Firmware, may be 0

  HSAuint16 VendorId; // GPU vendor id; 0 on latency (= CPU)-only nodes
  HSAuint16 DeviceId; // GPU device id; 0 on latency (= CPU)-only nodes

  HSAuint32 LocationId; // GPU BDF (Bus/Device/function number) - identifies the
                        // device location in the overall system
  HSAuint64 LocalMemSize;              // Local memory size
  HSAuint32 MaxEngineClockMhzFCompute; // maximum engine clocks for CPU and
  HSAuint32 MaxEngineClockMhzCCompute; // GPU function, including any boost
                                       // caopabilities,
  HSAint32 DrmRenderMinor;             // DRM render device minor device number
  HSAuint16 MarketingName[HSA_PUBLIC_NAME_SIZE]; // Public name of the "device"
                                                 // on the node (board or APU
                                                 // name). Unicode string
  HSAuint8 AMDName[HSA_PUBLIC_NAME_SIZE]; // CAL Name of the "device", ASCII
  HSA_ENGINE_VERSION uCodeEngineVersions;
  HSA_DEBUG_PROPERTIES DebugProperties; // Debug properties of this node.
  HSAuint64 HiveID; // XGMI Hive the GPU node belongs to in the system. It is an
                    // opaque and static number hash created by the PSP
  HSAuint32 NumSdmaEngines;     // number of PCIe optimized SDMA engines
  HSAuint32 NumSdmaXgmiEngines; // number of XGMI optimized SDMA engines

  HSAuint8 NumSdmaQueuesPerEngine; // number of SDMA queue per one engine
  HSAuint8 NumCpQueues;            // number of Compute queues
  HSAuint8 NumGws;                 // number of GWS barriers
  HSAuint8 Integrated; // 0 - discrete GPU, 1 - integrated GPU (including small
                       // APU and APP APU)

  HSAuint32 Domain;   // PCI domain of the GPU
  HSAuint64 UniqueID; // Globally unique immutable id

  HSAuint32 VGPRSizePerCU; // VGPR size in bytes per CU
  HSAuint32 SGPRSizePerCU; // SGPR size in bytes per CU

  HSAuint32 NumXcc;   // Number of XCC
  HSAuint32 KFDGpuID; // GPU Hash ID generated by KFD

  HSAuint32 FamilyID; // GPU family id

  HSAuint32 CwsrSize;     // Size of the CWSR
  HSAuint32 CtlStackSize; // Size of the control stack

  HSAuint32 LuidLowPart;  // Windows Locally Unique Identifier Low 4 bytes
  HSAuint32 LuidHighPart; // Windows Locally Unique Identifier High 4 bytes
  HSAuint64 WallClockKHz; // Wall Clock Frequency in KHz
} HsaNodeProperties;

typedef enum _HSA_HEAPTYPE {
  HSA_HEAPTYPE_SYSTEM = 0,
  HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC =
      1, // CPU "visible" part of GPU device local memory (for discrete GPU)
  HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE =
      2, // CPU "invisible" part of GPU device local memory (for discrete GPU)
         // All HSA accessible memory is per definition "CPU visible"
         // "Private memory" is relevant for graphics interop only.
  HSA_HEAPTYPE_GPU_GDS = 3,     // GPU internal memory (GDS)
  HSA_HEAPTYPE_GPU_LDS = 4,     // GPU internal memory (LDS)
  HSA_HEAPTYPE_GPU_SCRATCH = 5, // GPU special memory (scratch)
  HSA_HEAPTYPE_DEVICE_SVM = 6,  // sys-memory mapped by device page tables
  HSA_HEAPTYPE_MMIO_REMAP = 7,  // remapped mmio, such as hdp flush registers

  HSA_HEAPTYPE_NUMHEAPTYPES,
  HSA_HEAPTYPE_SIZE = 0xFFFFFFFF
} HSA_HEAPTYPE;

typedef union {
  HSAuint32 MemoryProperty;
  struct {
    unsigned int
        HotPluggable : 1; // the memory may be removed by some system action,
                          // memory should be used for temporary data
    unsigned int NonVolatile : 1; // memory content is preserved across a
                                  // power-off cycle.
    unsigned int Reserved : 30;
  } ui32;
} HSA_MEMORYPROPERTY;

//
// Discoverable HSA Memory properties.
// The structure is the output parameter of the hsaKmtGetNodeMemoryProperties()
// function
//

typedef struct _HsaMemoryProperties {
  HSA_HEAPTYPE HeapType; // system or frame buffer,
  union {
    HSAuint64 SizeInBytes; // physical memory size of the memory range in bytes
    struct {
      HSAuint32 SizeInBytesLow;  // physical memory size of the memory range in
                                 // bytes (lower 32bit)
      HSAuint32 SizeInBytesHigh; // physical memory size of the memory range in
                                 // bytes (higher 32bit)
    } ui32;
  };
  HSA_MEMORYPROPERTY Flags; // See definitions above

  HSAuint32 Width; // memory width - the number of parallel bits of the memory
                   // interface
  HSAuint32
      MemoryClockMax; // memory clock for the memory, this allows computing the
                      // available bandwidth to the memory when needed
  HSAuint64
      VirtualBaseAddress; // if set to value != 0, indicates the virtual base
                          // address of the memory in process virtual space
} HsaMemoryProperties;

//
// Discoverable Cache Properties. (optional).
// The structure is the output parameter of the hsaKmtGetNodeMemoryProperties()
// function Any of the parameters may be 0 (= not defined)
//

#define HSA_CPU_SIBLINGS 256
#define HSA_PROCESSORID_ALL 0xFFFFFFFF

typedef union {
  HSAuint32 Value;
  struct {
    unsigned int Data : 1;
    unsigned int Instruction : 1;
    unsigned int CPU : 1;
    unsigned int HSACU : 1;
    unsigned int Reserved : 28;
  } ui32;
} HsaCacheType;

typedef struct _HaCacheProperties {
  HSAuint32 ProcessorIdLow; // Identifies the processor number

  HSAuint32 CacheLevel;         // Integer representing level: 1, 2, 3, 4, etc
  HSAuint32 CacheSize;          // Size of the cache
  HSAuint32 CacheLineSize;      // Cache line size in bytes
  HSAuint32 CacheLinesPerTag;   // Cache lines per Cache Tag
  HSAuint32 CacheAssociativity; // Cache Associativity
  HSAuint32 CacheLatency;       // Cache latency in ns
  HsaCacheType CacheType;
  HSAuint32 SiblingMap[HSA_CPU_SIBLINGS];
} HsaCacheProperties;

//
// Discoverable CPU Compute Properties. (optional).
// The structure is the output parameter of the hsaKmtGetCComputeProperties()
// function Any of the parameters may be 0 (= not defined)
//

typedef struct _HsaCComputeProperties {
  HSAuint32 SiblingMap[HSA_CPU_SIBLINGS];
} HsaCComputeProperties;

//
// Discoverable IoLink Properties (optional).
// The structure is the output parameter of the hsaKmtGetIoLinkProperties()
// function. Any of the parameters may be 0 (= not defined)
//

typedef enum _HSA_IOLINKTYPE {
  HSA_IOLINKTYPE_UNDEFINED = 0,
  HSA_IOLINKTYPE_HYPERTRANSPORT = 1,
  HSA_IOLINKTYPE_PCIEXPRESS = 2,
  HSA_IOLINKTYPE_AMBA = 3,
  HSA_IOLINKTYPE_MIPI = 4,
  HSA_IOLINK_TYPE_QPI_1_1 = 5,
  HSA_IOLINK_TYPE_RESERVED1 = 6,
  HSA_IOLINK_TYPE_RESERVED2 = 7,
  HSA_IOLINK_TYPE_RAPID_IO = 8,
  HSA_IOLINK_TYPE_INFINIBAND = 9,
  HSA_IOLINK_TYPE_RESERVED3 = 10,
  HSA_IOLINK_TYPE_XGMI = 11,
  HSA_IOLINK_TYPE_XGOP = 12,
  HSA_IOLINK_TYPE_GZ = 13,
  HSA_IOLINK_TYPE_ETHERNET_RDMA = 14,
  HSA_IOLINK_TYPE_RDMA_OTHER = 15,
  HSA_IOLINK_TYPE_OTHER = 16,
  HSA_IOLINKTYPE_NUMIOLINKTYPES,
  HSA_IOLINKTYPE_SIZE = 0xFFFFFFFF
} HSA_IOLINKTYPE;

typedef union {
  HSAuint32 LinkProperty;
  struct {
    unsigned int Override : 1; // bus link properties are determined by this
                               // structure not by the HSA_IOLINKTYPE. The other
                               // flags are valid only if this bit is set to one
    unsigned int NonCoherent : 1;    // The link doesn't support coherent
                                     // transactions memory accesses across must
                                     // not be set to "host cacheable"!
    unsigned int NoAtomics32bit : 1; // The link doesn't support 32bit-wide
                                     // atomic transactions
    unsigned int NoAtomics64bit : 1; // The link doesn't support 64bit-wide
                                     // atomic transactions
    unsigned int
        NoPeerToPeerDMA : 1; // The link doesn't allow device P2P access
    unsigned int Reserved : 27;
  } ui32;
} HSA_LINKPROPERTY;

typedef struct _HsaIoLinkProperties {
  HSA_IOLINKTYPE IoLinkType; // see above
  HSAuint32 VersionMajor;    // Bus interface version (optional)
  HSAuint32 VersionMinor;    // Bus interface version (optional)

  HSAuint32 NodeFrom; //
  HSAuint32 NodeTo;   //

  HSAuint32 Weight; // weight factor (derived from CDIT)

  HSAuint32 MinimumLatency; // minimum cost of time to transfer (rounded to ns)
  HSAuint32 MaximumLatency; // maximum cost of time to transfer (rounded to ns)
  HSAuint32 MinimumBandwidth; // minimum interface Bandwidth in MB/s
  HSAuint32 MaximumBandwidth; // maximum interface Bandwidth in MB/s
  HSAuint32 RecTransferSize;  // recommended transfer size to reach maximum
                              // bandwidth in Bytes
  HSAuint32 RecSdmaEngIdMask; // recommended sdma engine IDs to reach maximum
                              // bandwidth
  HSA_LINKPROPERTY
  Flags; // override flags (may be active for specific platforms)
} HsaIoLinkProperties;

//
// Memory allocation definitions for the KFD HSA interface
//

typedef struct _HsaMemFlags {
  union {
    struct {
      unsigned int NonPaged : 1;    // default = 0: pageable memory
      unsigned int CachePolicy : 2; // see HSA_CACHING_TYPE
      unsigned int ReadOnly : 1;    // default = 0: Read/Write memory
      unsigned int PageSize : 2;    // see HSA_PAGE_SIZE
      unsigned int HostAccess : 1;  // default = 0: GPU access only
      unsigned int
          NoSubstitute : 1; // default = 0: if specific memory is not available
                            // on node (e.g. on discrete GPU local), allocation
                            // may fall back to system memory node 0 memory (=
                            // always available). Otherwise no allocation is
                            // possible.
      unsigned int
          GDSMemory : 1; // default = 0: If set, the allocation will occur in
                         // GDS heap. HostAccess must be 0, all other flags
                         // (except NoSubstitute) should be 0 when setting this
                         // entry to 1. GDS allocation may fail due to limited
                         // resources. Application code is required to work
                         // without any allocated GDS memory using regular
                         // memory. Allocation fails on any node without GPU
                         // function.
      unsigned int
          Scratch : 1; // default = 0: If set, the allocation will occur in GPU
                       // "scratch area". HostAccess must be 0, all other flags
                       // (except NoSubstitute) should be 0 when setting this
                       // entry to 1. Scratch allocation may fail due to limited
                       // resources. Application code is required to work
                       // without any allocation. Allocation fails on any node
                       // without GPU function.
      unsigned int
          AtomicAccessFull : 1; // default = 0: If set, the memory will be
                                // allocated and mapped to allow atomic ops
                                // processing. On AMD APU, this will use the ATC
                                // path on system memory, irrespective of the
                                // NonPaged flag setting (= if NonPaged is set,
                                // the memory is pagelocked but mapped through
                                // IOMMUv2 instead of GPUVM). All atomic ops
                                // must be supported on this memory.
      unsigned int
          AtomicAccessPartial : 1; // default = 0: See above for
                                   // AtomicAccessFull description, however
                                   // focused on AMD discrete GPU that support
                                   // PCIe atomics; the memory allocation is
                                   // mapped to allow for PCIe atomics to
                                   // operate on system memory, irrespective of
                                   // NonPaged set or the presence of an ATC
                                   // path in the system. The atomic operations
                                   // supported are limited to SWAP,
                                   // CompareAndSwap (CAS) and FetchAdd (this
                                   // PCIe op allows both atomic increment and
                                   // decrement via 2-complement arithmetic),
                                   // which are the only atomic ops directly
                                   // supported in PCI Express. On AMD APU,
                                   // setting this flag will allocate the same
                                   // type of memory as AtomicAccessFull, but it
                                   // will be considered compatible with
                                   // discrete GPU atomic operations access.
      unsigned int
          ExecuteAccess : 1; // default = 0: Identifies if memory is primarily
                             // used for data or accessed for executable code
                             // (e.g. queue memory) by the host CPU or the
                             // device. Influences the page attribute setting
                             // within the allocation
      unsigned int
          CoarseGrain : 1; // default = 0: The memory can be accessed assuming
                           // cache coherency maintained by link infrastructure
                           // and HSA agents. 1: memory consistency needs to be
                           // enforced at synchronization points at dispatch or
                           // other software enforced synchronization
                           // boundaries.
      unsigned int
          AQLQueueMemory : 1; // default = 0; If 1: The caller indicates that
                              // the memory will be used as AQL queue memory.
                              // The KFD will ensure that the memory returned is
                              // allocated in the optimal memory location and
                              // optimal alignment requirements
      unsigned int FixedAddress : 1; // Allocate memory at specified virtual
                                     // address. Fail if address is not free.
      unsigned int
          NoNUMABind : 1; // Don't bind system memory to a specific NUMA node
      unsigned int Uncached : 1;  // Caching flag for fine-grained memory on A+A
                                  // HW platform
      unsigned int NoAddress : 1; // only do vram allocation, return a handle,
                                  // not allocate virtual address.
      unsigned int OnlyAddress : 1; // only do virtal address allocation without
                                    // vram allocation.
      unsigned int
          ExtendedCoherent : 1; // system-scope coherence on atomic instructions
      unsigned int
          GTTAccess : 1; // default = 0; If 1: The caller indicates this memory
                         // will be mapped to GART for MES KFD will allocate GTT
                         // memory with the Preferred_node set as gpu_id for
                         // GART mapping
      unsigned int Contiguous : 1; // Allocate contiguous VRAM
      unsigned int
          ExecuteBlit : 1; // default = 0; If 1: The caller indicates that the
                           // memory is for blit kernel object.
      unsigned int
          QueueObject : 1; // AQL queue object, used in windows for CPU access
                           // to get the read pointer from amd_queue_t
      unsigned int Reserved : 7;

    } ui32;
    HSAuint32 Value;
  };
} HsaMemFlags;

typedef struct _HsaMemMapFlags {
  union {
    struct {
      unsigned int Reserved1 : 1;   //
      unsigned int CachePolicy : 2; // see HSA_CACHING_TYPE
      unsigned int ReadOnly : 1;    // memory is not modified while mapped
                                    // allows migration scale-out
      unsigned int PageSize : 2;    // see HSA_PAGE_SIZE, hint to use
                                    // this page size if possible and
                                    // smaller than default
      unsigned int HostAccess : 1;  // default = 0: GPU access only
      unsigned int Migrate : 1;     // Hint: Allows migration to local mem
                                    // of mapped GPU(s), instead of mapping
                                    // physical location
      unsigned int Probe : 1;       // default = 0: Indicates that a range
                                    // will be mapped by the process soon,
                                    // but does not initiate a map operation
                                    // may trigger eviction of nonessential
                                    // data from the memory, reduces latency
                                    // “cleanup hint” only, may be ignored
      unsigned int Reserved : 23;
    } ui32;
    HSAuint32 Value;
  };
} HsaMemMapFlags;

typedef struct _HsaGraphicsResourceInfo {
  void *MemoryAddress;           // For use in hsaKmtMapMemoryToGPU(Nodes)
  HSAuint64 SizeInBytes;         // Buffer size
  const void *Metadata;          // Pointer to metadata owned by Thunk
  HSAuint32 MetadataSizeInBytes; // Size of metadata
  HSAuint32 NodeId;              // GPU exported the buffer
} HsaGraphicsResourceInfo;

typedef enum _HSA_CACHING_TYPE {
  HSA_CACHING_CACHED = 0,
  HSA_CACHING_NONCACHED = 1,
  HSA_CACHING_WRITECOMBINED = 2,
  HSA_CACHING_RESERVED = 3,
  HSA_CACHING_NUM_CACHING,
  HSA_CACHING_SIZE = 0xFFFFFFFF
} HSA_CACHING_TYPE;

typedef enum _HSA_PAGE_SIZE {
  HSA_PAGE_SIZE_4KB = 0,
  HSA_PAGE_SIZE_64KB = 1, // 64KB pages, not generally available in systems
  HSA_PAGE_SIZE_2MB = 2,
  HSA_PAGE_SIZE_1GB = 3, // 1GB pages, not generally available in systems
} HSA_PAGE_SIZE;

typedef enum _HSA_DEVICE {
  HSA_DEVICE_CPU = 0,
  HSA_DEVICE_GPU = 1,
  MAX_HSA_DEVICE = 2
} HSA_DEVICE;

typedef enum _HSA_QUEUE_PRIORITY {
  HSA_QUEUE_PRIORITY_MINIMUM = -3,
  HSA_QUEUE_PRIORITY_LOW = -2,
  HSA_QUEUE_PRIORITY_BELOW_NORMAL = -1,
  HSA_QUEUE_PRIORITY_NORMAL = 0,
  HSA_QUEUE_PRIORITY_ABOVE_NORMAL = 1,
  HSA_QUEUE_PRIORITY_HIGH = 2,
  HSA_QUEUE_PRIORITY_MAXIMUM = 3,
  HSA_QUEUE_PRIORITY_NUM_PRIORITY,
  HSA_QUEUE_PRIORITY_SIZE = 0xFFFFFFFF
} HSA_QUEUE_PRIORITY;

typedef enum _HSA_QUEUE_TYPE {
  HSA_QUEUE_COMPUTE = 1, // AMD PM4 compatible Compute Queue
  HSA_QUEUE_SDMA = 2, // PCIe optimized SDMA Queue, used for data transport and
                      // format conversion (e.g. (de-)tiling, etc).
  HSA_QUEUE_MULTIMEDIA_DECODE = 3, // reserved, for HSA multimedia decode queue
  HSA_QUEUE_MULTIMEDIA_ENCODE = 4, // reserved, for HSA multimedia encode queue
  HSA_QUEUE_SDMA_XGMI = 5,         // XGMI optimized SDMA Queue
  HSA_QUEUE_SDMA_BY_ENG_ID = 6,    // Queue with specified SDMA engine ID

  // the following values indicate a queue type permitted to reference OS
  // graphics resources through the interoperation API. See [5] "HSA Graphics
  // Interoperation specification" for more details on use of such resources.

  HSA_QUEUE_COMPUTE_OS = 11, // AMD PM4 compatible Compute Queue
  HSA_QUEUE_SDMA_OS = 12,    // SDMA Queue, used for data transport and format
                             // conversion (e.g. (de-)tiling, etc).
  HSA_QUEUE_MULTIMEDIA_DECODE_OS =
      13, // reserved, for HSA multimedia decode queue
  HSA_QUEUE_MULTIMEDIA_ENCODE_OS =
      14, // reserved, for HSA multimedia encode queue

  HSA_QUEUE_COMPUTE_AQL = 21, // HSA AQL packet compatible Compute Queue
  HSA_QUEUE_DMA_AQL = 22,     // HSA AQL packet compatible DMA Queue
  HSA_QUEUE_DMA_AQL_XGMI =
      23, // HSA AQL packet compatible XGMI optimized DMA Queue

  // more types in the future

  HSA_QUEUE_TYPE_SIZE = 0xFFFFFFFF // aligns to 32bit enum
} HSA_QUEUE_TYPE;

/**
  The user context save area is page aligned. The HsaUserContextSaveAreaHeader
  header starts at offset 0. Space for a user space copy of the control stack
  comes next and is immediately followed by the user space wave save state. The
  start of the user space wave save state is page aligned. The debugger reserved
  area comes next and is 64 byte aligned.

  The user context save area is valid for the duration that the associated
  queue exists. When a context save occurs, the HsaUserContextSaveAreaHeader
  header will be updated with information about the context save. The context
  save area is not modified by any other operation, including a context resume.
 */

typedef struct {
  HSAuint32 ControlStackOffset; // Byte offset from start of user context
                                // save area to the last saved top (lowest
                                // address) of control stack data. Must be
                                // 4 byte aligned.
  HSAuint32 ControlStackSize;   // Byte size of the last saved control stack
                                // data. Must be 4 byte aligned.
  HSAuint32 WaveStateOffset;    // Byte offset from start of user context save
                                // area to the last saved base (lowest address)
                                // of wave state data. Must be 4 byte aligned.
  HSAuint32 WaveStateSize;      // Byte size of the last saved wave state data.
                                // Must be 4 byte aligned.
  HSAuint32 DebugOffset;        // Byte offset from start of the user context
                                // save area to the memory reserved for the
                                // debugger. Must be 64 byte aligned.
  HSAuint32 DebugSize;          // Byte size of the memory reserved for the
                                // debugger. Must be 64 byte aligned.
  volatile HSAint64 *ErrorReason; // Address of the HSA signal payload for
                                  // reporting the error reason bitmask.
                                  // Must be 4 byte aligned.
  HSAuint32 ErrorEventId;         // Event ID used for exception signalling.
                                  // Must be 4 byte aligned.
  HSAuint32 Reserved1;
} HsaUserContextSaveAreaHeader;

typedef struct {
  HSAuint32 QueueDetailError;     // HW specific queue error state
  HSAuint32 QueueTypeExtended;    // HW specific queue type info.
                                  // 0 = no information
  HSAuint32 NumCUAssigned;        // size of *CUMaskInfo bit array, Multiple
                                  // of 32, 0 = no information
  HSAuint32 *CUMaskInfo;          // runtime/system CU assignment for realtime
                                  // queue & reserved CU priority. Ptr to
                                  // bit-array, each bit represents one CU.
                                  // NULL = no information
  HSAuint32 *UserContextSaveArea; // reference to user space context save area
  HSAuint64 SaveAreaSizeInBytes;  // Must be 4-Byte aligned
  HSAuint32 *ControlStackTop;     // ptr to the TOS
  HSAuint64 ControlStackUsedInBytes; // Must be 4-Byte aligned
  HsaUserContextSaveAreaHeader *SaveAreaHeader;
  HSAuint64 Reserved2; // runtime/system CU assignment
} HsaQueueInfo;

typedef struct _HsaQueueResource {
  HSA_QUEUEID QueueId; /** queue ID */
  /** Doorbell address to notify HW of a new dispatch */
  union {
    HSAuint32 *Queue_DoorBell;
    HSAuint64 *Queue_DoorBell_aql;
    HSAuint64 QueueDoorBell;
  };

  /** virtual address to notify HW of queue write ptr value */
  union {
    HSAuint32 *Queue_write_ptr;
    HSAuint64 *Queue_write_ptr_aql;
    HSAuint64 QueueWptrValue;
  };

  /** virtual address updated by HW to indicate current read location */
  union {
    HSAuint32 *Queue_read_ptr;
    HSAuint64 *Queue_read_ptr_aql;
    HSAuint64 QueueRptrValue;
  };

  volatile HSAint64 *ErrorReason; /** exception bits signal payload */
} HsaQueueResource;

// TEMPORARY structure definition - to be used only on "Triniti + Southern
// Islands" platform
typedef struct _HsaQueueReport {
  HSAuint32 VMID;      // Required on SI to dispatch IB in primary ring
  void *QueueAddress;  // virtual address of UM mapped compute ring
  HSAuint64 QueueSize; // size of the UM mapped compute ring
} HsaQueueReport;

typedef enum _HSA_DBG_WAVEOP {
  HSA_DBG_WAVEOP_HALT = 1,   // Halts a wavefront
  HSA_DBG_WAVEOP_RESUME = 2, // Resumes a wavefront
  HSA_DBG_WAVEOP_KILL = 3,   // Kills a wavefront
  HSA_DBG_WAVEOP_DEBUG = 4,  // Causes wavefront to enter debug mode
  HSA_DBG_WAVEOP_TRAP = 5,   // Causes wavefront to take a trap
  HSA_DBG_NUM_WAVEOP = 5,
  HSA_DBG_MAX_WAVEOP = 0xFFFFFFFF
} HSA_DBG_WAVEOP;

typedef enum _HSA_DBG_WAVEMODE {
  HSA_DBG_WAVEMODE_SINGLE = 0, // send command to a single wave
  // Broadcast to all wavefronts of all processes is not supported for HSA user
  // mode
  HSA_DBG_WAVEMODE_BROADCAST_PROCESS =
      2, // send to waves within current process
  HSA_DBG_WAVEMODE_BROADCAST_PROCESS_CU =
      3, // send to waves within current process on CU
  HSA_DBG_NUM_WAVEMODE = 3,
  HSA_DBG_MAX_WAVEMODE = 0xFFFFFFFF
} HSA_DBG_WAVEMODE;

typedef enum _HSA_DBG_WAVEMSG_TYPE {
  HSA_DBG_WAVEMSG_AUTO = 0,
  HSA_DBG_WAVEMSG_USER = 1,
  HSA_DBG_WAVEMSG_ERROR = 2,
  HSA_DBG_NUM_WAVEMSG,
  HSA_DBG_MAX_WAVEMSG = 0xFFFFFFFF
} HSA_DBG_WAVEMSG_TYPE;

typedef enum _HSA_DBG_WATCH_MODE {
  HSA_DBG_WATCH_READ = 0,    // Read operations only
  HSA_DBG_WATCH_NONREAD = 1, // Write or Atomic operations only
  HSA_DBG_WATCH_ATOMIC = 2,  // Atomic Operations only
  HSA_DBG_WATCH_ALL = 3,     // Read, Write or Atomic operations
  HSA_DBG_WATCH_NUM
} HSA_DBG_WATCH_MODE;

typedef enum _HSA_DBG_TRAP_OVERRIDE {
  HSA_DBG_TRAP_OVERRIDE_OR =
      0, // Bitwise OR exception mask with HSA_DBG_TRAP_MASK
  HSA_DBG_TRAP_OVERRIDE_REPLACE =
      1, // Replace exception mask with HSA_DBG_TRAP_MASK
  HSA_DBG_TRAP_OVERRIDE_NUM
} HSA_DBG_TRAP_OVERRIDE;

typedef enum _HSA_DBG_TRAP_MASK {
  HSA_DBG_TRAP_MASK_FP_INVALID = 1,          // Floating point invalid operation
  HSA_DBG_TRAP_MASK_FP_INPUT_DENOMAL = 2,    // Floating point input denormal
  HSA_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO = 4,   // Floating point divide by zero
  HSA_DBG_TRAP_MASK_FP_OVERFLOW = 8,         // Floating point overflow
  HSA_DBG_TRAP_MASK_FP_UNDERFLOW = 16,       // Floating point underflow
  HSA_DBG_TRAP_MASK_FP_INEXACT = 32,         // Floating point inexact
  HSA_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO = 64, // Integer divide by zero
  HSA_DBG_TRAP_MASK_DBG_ADDRESS_WATCH = 128, // Debug address watch
  HSA_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION = 256 // Memory violation
} HSA_DBG_TRAP_MASK;

typedef enum _HSA_DBG_TRAP_EXCEPTION_CODE {
  HSA_DBG_EC_NONE = 0,
  /* per queue */
  HSA_DBG_EC_QUEUE_WAVE_ABORT = 1,
  HSA_DBG_EC_QUEUE_WAVE_TRAP = 2,
  HSA_DBG_EC_QUEUE_WAVE_MATH_ERROR = 3,
  HSA_DBG_EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION = 4,
  HSA_DBG_EC_QUEUE_WAVE_MEMORY_VIOLATION = 5,
  HSA_DBG_EC_QUEUE_WAVE_APERTURE_VIOLATION = 6,
  HSA_DBG_EC_QUEUE_PACKET_DISPATCH_DIM_INVALID = 16,
  HSA_DBG_EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID = 17,
  HSA_DBG_EC_QUEUE_PACKET_DISPATCH_CODE_INVALID = 18,
  HSA_DBG_EC_QUEUE_PACKET_RESERVED = 19,
  HSA_DBG_EC_QUEUE_PACKET_UNSUPPORTED = 20,
  HSA_DBG_EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID = 21,
  HSA_DBG_EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID = 22,
  HSA_DBG_EC_QUEUE_PACKET_VENDOR_UNSUPPORTED = 23,
  HSA_DBG_EC_QUEUE_PREEMPTION_ERROR = 30,
  HSA_DBG_EC_QUEUE_NEW = 31,
  /* per device */
  HSA_DBG_EC_DEVICE_QUEUE_DELETE = 32,
  HSA_DBG_EC_DEVICE_MEMORY_VIOLATION = 33,
  HSA_DBG_EC_DEVICE_RAS_ERROR = 34,
  HSA_DBG_EC_DEVICE_FATAL_HALT = 35,
  HSA_DBG_EC_DEVICE_NEW = 36,
  /* per process */
  HSA_DBG_EC_PROCESS_RUNTIME = 48,
  HSA_DBG_EC_PROCESS_DEVICE_REMOVE = 49,
  HSA_DBG_EC_MAX
} HSA_DBG_TRAP_EXCEPTION_CODE;

/* Mask generated by ecode defined in enum above. */
#define HSA_EC_MASK(ecode) (1ULL << (ecode - 1))

typedef enum _HSA_DBG_WAVE_LAUNCH_MODE {
  HSA_DBG_WAVE_LAUNCH_MODE_NORMAL = 0, // Wavefront launched normally.
  HSA_DBG_WAVE_LAUNCH_MODE_HALT = 1,   // Wavefront launched in halted mode.
  HSA_DBG_WAVE_LAUNCH_MODE_KILL =
      2, // Wavefront is launched but immediately
         // terminated before executing any instructions.
  HSA_DBG_WAVE_LAUNCH_MODE_SINGLE_STEP =
      3, // Wavefront is launched in single step (debug)
         // mode. If debug trap is enabled by
         // hsaKmtDbgEnableDebugTrap() then causes a
         // trap after executing each instruction,
         // otherwise behaves the same as
         // HSA_DBG_WAVE_LAUNCH_MODE_NORMAL.
  HSA_DBG_WAVE_LAUNCH_MODE_DISABLE = 4, // Disable launching any new waves.
  HSA_DBG_WAVE_LAUNCH_MODE_NUM
} HSA_DBG_WAVE_LAUNCH_MODE;

/**
 *    There are no flags currently defined.
 */
typedef enum HSA_DBG_NODE_CONTROL {
  HSA_DBG_NODE_CONTROL_FLAG_MAX = 0x01
} HSA_DBG_NODE_CONTROL;

#define HSA_RUNTIME_ENABLE_CAPS_SUPPORTS_CORE_DUMP_MASK 0x80000000

// This structure is hardware specific and may change in the future
typedef struct _HsaDbgWaveMsgAMDGen2 {
  HSAuint32 Value;
  HSAuint32 Reserved2;

} HsaDbgWaveMsgAMDGen2;

typedef union _HsaDbgWaveMessageAMD {
  HsaDbgWaveMsgAMDGen2 WaveMsgInfoGen2;
  // for future HsaDbgWaveMsgAMDGen3;
} HsaDbgWaveMessageAMD;

typedef struct _HsaDbgWaveMessage {
  void *MemoryVA; // ptr to associated host-accessible data
  HsaDbgWaveMessageAMD DbgWaveMsg;
} HsaDbgWaveMessage;

//
// HSA sync primitive, Event and HW Exception notification API definitions
// The API functions allow the runtime to define a so-called sync-primitive, a
// SW object combining a user-mode provided "syncvar" and a scheduler event that
// can be signaled through a defined GPU interrupt. A syncvar is a process
// virtual memory location of a certain size that can be accessed by CPU and GPU
// shader code within the process to set and query the content within that
// memory. The definition of the content is determined by the HSA runtime and
// potentially GPU shader code interfacing with the HSA runtime. The syncvar
// values may be commonly written through an PM4 WRITE_DATA packet in the user
// mode instruction stream. The OS scheduler event is typically associated and
// signaled by an interrupt issued by the GPU, but other HSA system interrupt
// conditions from other HW (e.g. IOMMUv2) may be surfaced by the KFD by this
// mechanism, too.
//

// these are the new definitions for events
typedef enum _HSA_EVENTTYPE {
  HSA_EVENTTYPE_SIGNAL = 0,            // user-mode generated GPU signal
  HSA_EVENTTYPE_NODECHANGE = 1,        // HSA node change (attach/detach)
  HSA_EVENTTYPE_DEVICESTATECHANGE = 2, // HSA device state change( start/stop )
  HSA_EVENTTYPE_HW_EXCEPTION = 3,      // GPU shader exception event
  HSA_EVENTTYPE_SYSTEM_EVENT = 4,      // GPU SYSCALL with parameter info
  HSA_EVENTTYPE_DEBUG_EVENT = 5,       // GPU signal for debugging
  HSA_EVENTTYPE_PROFILE_EVENT = 6,     // GPU signal for profiling
  HSA_EVENTTYPE_QUEUE_EVENT = 7,       // GPU signal queue idle state (EOP pm4)
  HSA_EVENTTYPE_MEMORY = 8, // GPU signal for signaling memory access faults and
                            // memory subsystem issues
  //...
  HSA_EVENTTYPE_MAXID,
  HSA_EVENTTYPE_TYPE_SIZE = 0xFFFFFFFF
} HSA_EVENTTYPE;

//
// Definitions for types of pending debug events
//
typedef enum _HSA_DEBUG_EVENT_TYPE {
  HSA_DEBUG_EVENT_TYPE_NONE = 0,
  HSA_DEBUG_EVENT_TYPE_TRAP = 1,
  HSA_DEBUG_EVENT_TYPE_VMFAULT = 2,
  HSA_DEBUG_EVENT_TYPE_TRAP_VMFAULT = 3
} HSA_DEBUG_EVENT_TYPE;

typedef HSAuint32 HSA_EVENTID;

//
// Subdefinitions for various event types: Syncvar
//

typedef struct _HsaSyncVar {
  union {
    void *UserData;             // pointer to user mode data
    HSAuint64 UserDataPtrValue; // 64bit compatibility of value
  } SyncVar;
  HSAuint64 SyncVarSize;
} HsaSyncVar;

//
// Subdefinitions for various event types: NodeChange
//

typedef enum _HSA_EVENTTYPE_NODECHANGE_FLAGS {
  HSA_EVENTTYPE_NODECHANGE_ADD = 0,
  HSA_EVENTTYPE_NODECHANGE_REMOVE = 1,
  HSA_EVENTTYPE_NODECHANGE_SIZE = 0xFFFFFFFF
} HSA_EVENTTYPE_NODECHANGE_FLAGS;

typedef struct _HsaNodeChange {
  HSA_EVENTTYPE_NODECHANGE_FLAGS
  Flags; // HSA node added/removed on the platform
} HsaNodeChange;

//
// Sub-definitions for various event types: DeviceStateChange
//

typedef enum _HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS {
  HSA_EVENTTYPE_DEVICESTATUSCHANGE_START = 0, // device started (and available)
  HSA_EVENTTYPE_DEVICESTATUSCHANGE_STOP =
      1, // device stopped (i.e. unavailable)
  HSA_EVENTTYPE_DEVICESTATUSCHANGE_SIZE = 0xFFFFFFFF
} HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS;

typedef struct _HsaDeviceStateChange {
  HSAuint32 NodeId;  // F-NUMA node that contains the device
  HSA_DEVICE Device; // device type: GPU or CPU
  HSA_EVENTTYPE_DEVICESTATECHANGE_FLAGS Flags; // event flags
} HsaDeviceStateChange;

//
// Sub-definitions for various event types: Memory exception
//

typedef enum _HSA_EVENTID_MEMORYFLAGS {
  HSA_EVENTID_MEMORY_RECOVERABLE =
      0, // access fault, recoverable after page adjustment
  HSA_EVENTID_MEMORY_FATAL_PROCESS =
      1, // memory access requires process context destruction, unrecoverable
  HSA_EVENTID_MEMORY_FATAL_VM =
      2, // memory access requires all GPU VA context destruction, unrecoverable
} HSA_EVENTID_MEMORYFLAGS;

typedef struct _HsaAccessAttributeFailure {
  unsigned int NotPresent : 1; // Page not present or supervisor privilege
  unsigned int ReadOnly : 1;   // Write access to a read-only page
  unsigned int NoExecute : 1;  // Execute access to a page marked NX
  unsigned int GpuAccess : 1;  // Host access only
  unsigned int ECC : 1;        // RAS ECC failure (notification of DRAM ECC -
                               // non-recoverable - error, if supported by HW)
  unsigned int Imprecise : 1;  // Can't determine the exact fault address
  unsigned int
      ErrorType : 3; // Indicates RAS errors or other errors causing the access
                     // to GPU to fail 0 = no RAS error, 1 = ECC_SRAM, 2 =
                     // Link_SYNFLOOD (poison), 3 = GPU hang (not attributable
                     // to a specific cause), other values reserved
  unsigned int Reserved : 23; // must be 0
} HsaAccessAttributeFailure;

// data associated with HSA_EVENTID_MEMORY
typedef struct _HsaMemoryAccessFault {
  HSAuint32 NodeId; // H-NUMA node that contains the device where the memory
                    // access occurred
  HSAuint64 VirtualAddress;          // virtual address this occurred on
  HsaAccessAttributeFailure Failure; // failure attribute
  HSA_EVENTID_MEMORYFLAGS Flags;     // event flags
} HsaMemoryAccessFault;

typedef enum _HSA_EVENTID_HW_EXCEPTION_CAUSE {
  HSA_EVENTID_HW_EXCEPTION_GPU_HANG = 0, // GPU Hang
  HSA_EVENTID_HW_EXCEPTION_ECC = 1,      // SRAM ECC error
} HSA_EVENTID_HW_EXCEPTION_CAUSE;

// data associated with HSA_EVENTID_HW_EXCEPTION
typedef struct _HsaHwException {
  HSAuint32 NodeId; // Node Id where the memory exception occured
  HSAuint32 ResetType;
  HSAuint32 MemoryLost;
  HSA_EVENTID_HW_EXCEPTION_CAUSE ResetCause;
} HsaHwException;

typedef struct _HsaEventData {
  HSA_EVENTTYPE EventType; // event type

  union {
    // return data associated with HSA_EVENTTYPE_SIGNAL and other events
    HsaSyncVar SyncVar;

    // data associated with HSA_EVENTTYPE_NODE_CHANGE
    HsaNodeChange NodeChangeState;

    // data associated with HSA_EVENTTYPE_DEVICE_STATE_CHANGE
    HsaDeviceStateChange DeviceState;

    // data associated with HSA_EVENTTYPE_MEMORY
    HsaMemoryAccessFault MemoryAccessFault;

    // data associated with HSA_EVENTTYPE_HW_EXCEPTION
    HsaHwException HwException;
  } EventData;

  // the following data entries are internal to the KFD & thunk itself.

  HSAuint64 HWData1; // internal thunk store for Event data  (OsEventHandle)
  HSAuint64 HWData2; // internal thunk store for Event data  (HWAddress)
  HSAuint32 HWData3; // internal thunk store for Event data  (HWData)
} HsaEventData;

typedef struct _HsaEventDescriptor {
  HSA_EVENTTYPE EventType; // event type to allocate
  HSAuint32 NodeId;   // H-NUMA node containing GPU device that is event source
  HsaSyncVar SyncVar; // pointer to user mode syncvar data,
                      // syncvar->UserDataPtrValue may be NULL
} HsaEventDescriptor;

typedef struct _HsaEvent {
  HSA_EVENTID EventId;
  HsaEventData EventData;
} HsaEvent;

typedef enum _HsaEventTimeout {
  HSA_EVENTTIMEOUT_IMMEDIATE = 0,
  HSA_EVENTTIMEOUT_INFINITE = 0xFFFFFFFF
} HsaEventTimeOut;

typedef struct _HsaClockCounters {
  HSAuint64 GPUClockCounter;
  HSAuint64 CPUClockCounter;
  HSAuint64 SystemClockCounter;
  HSAuint64 SystemClockFrequencyHz;
} HsaClockCounters;

#ifndef DEFINE_GUID
typedef struct _HSA_UUID {
  HSAuint32 Data1;
  HSAuint16 Data2;
  HSAuint16 Data3;
  HSAuint8 Data4[8];
} HSA_UUID;

#define HSA_DEFINE_UUID(name, dw, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8)      \
  static const HSA_UUID name = {dw, w1, w2, {b1, b2, b3, b4, b5, b6, b7, b8}}
#else
#define HSA_UUID GUID
#define HSA_DEFINE_UUID DEFINE_GUID
#endif

// HSA_UUID that identifies the GPU ColorBuffer (CB) block
// {9ba429c6-af2d-4b38-b349-157271beac6a}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_CB, 0x9ba429c6, 0xaf2d, 0x4b38, 0xb3, 0x49,
                0x15, 0x72, 0x71, 0xbe, 0xac, 0x6a);

// HSA_UUID that identifies the GPU (CPF) block
// {2b0ad2b5-1c43-4f46-a7bc-e119411ea6c9}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_CPF, 0x2b0ad2b5, 0x1c43, 0x4f46, 0xa7,
                0xbc, 0xe1, 0x19, 0x41, 0x1e, 0xa6, 0xc9);

// HSA_UUID that identifies the GPU (CPG) block
// {590ec94d-20f0-448f-8dff-316c679de7ff
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_CPG, 0x590ec94d, 0x20f0, 0x448f, 0x8d,
                0xff, 0x31, 0x6c, 0x67, 0x9d, 0xe7, 0xff);

// HSA_UUID that identifies the GPU (DB) block
// {3d1a47fc-0013-4ed4-8306-822ca0b7a6c2
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_DB, 0x3d1a47fc, 0x0013, 0x4ed4, 0x83, 0x06,
                0x82, 0x2c, 0xa0, 0xb7, 0xa6, 0xc2);

// HSA_UUID that identifies the GPU (GDS) block
// {f59276ec-2526-4bf8-8ec0-118f77700dc9
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_GDS, 0xf59276ec, 0x2526, 0x4bf8, 0x8e,
                0xc0, 0x11, 0x8f, 0x77, 0x70, 0x0d, 0xc9);

// HSA_UUID that identifies the GPU (GRBM) block
// {8f00933c-c33d-4801-97b7-7007f78573ad
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_GRBM, 0x8f00933c, 0xc33d, 0x4801, 0x97,
                0xb7, 0x70, 0x07, 0xf7, 0x85, 0x73, 0xad);

// HSA_UUID that identifies the GPU (GRBMSE) block
// {34ebd8d7-7c8b-4d15-88fa-0e4e4af59ac1
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_GRBMSE, 0x34ebd8d7, 0x7c8b, 0x4d15, 0x88,
                0xfa, 0x0e, 0x4e, 0x4a, 0xf5, 0x9a, 0xc1);

// HSA_UUID that identifies the GPU (IA) block
// {34276944-4264-4fcd-9d6e-ae264582ec51
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_IA, 0x34276944, 0x4264, 0x4fcd, 0x9d, 0x6e,
                0xae, 0x26, 0x45, 0x82, 0xec, 0x51);

// HSA_UUID that identifies the GPU Memory Controller (MC) block
// {13900B57-4956-4D98-81D0-68521937F59C
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_MC, 0x13900b57, 0x4956, 0x4d98, 0x81, 0xd0,
                0x68, 0x52, 0x19, 0x37, 0xf5, 0x9c);

// HSA_UUID that identifies the GPU (PASC) block
// {b0e7fb5d-0efc-4744-b516-5d23dc1fd56c
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_PASC, 0xb0e7fb5d, 0x0efc, 0x4744, 0xb5,
                0x16, 0x5d, 0x23, 0xdc, 0x1f, 0xd5, 0x6c);

// HSA_UUID that identifies the GPU (PASU) block
// {9a152b6a-1fad-45f2-a5bf-f163826bd0cd
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_PASU, 0x9a152b6a, 0x1fad, 0x45f2, 0xa5,
                0xbf, 0xf1, 0x63, 0x82, 0x6b, 0xd0, 0xcd);

// HSA_UUID that identifies the GPU (SPI) block
// {eda81044-d62c-47eb-af89-4f6fbf3b38e0
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_SPI, 0xeda81044, 0xd62c, 0x47eb, 0xaf,
                0x89, 0x4f, 0x6f, 0xbf, 0x3b, 0x38, 0xe0);

// HSA_UUID that identifies the GPU (SRBM) block
// {9f8040e0-6830-4019-acc8-463c9e445b89
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_SRBM, 0x9f8040e0, 0x6830, 0x4019, 0xac,
                0xc8, 0x46, 0x3c, 0x9e, 0x44, 0x5b, 0x89);

// GUID that identifies the GPU Shader Sequencer (SQ) block
// {B5C396B6-D310-47E4-86FC-5CC3043AF508}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_SQ, 0xb5c396b6, 0xd310, 0x47e4, 0x86, 0xfc,
                0x5c, 0xc3, 0x4, 0x3a, 0xf5, 0x8);

// HSA_UUID that identifies the GPU (SX) block
// {bdb8d737-43cc-4162-be52-51cfb847beaf}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_SX, 0xbdb8d737, 0x43cc, 0x4162, 0xbe, 0x52,
                0x51, 0xcf, 0xb8, 0x47, 0xbe, 0xaf);

// HSA_UUID that identifies the GPU (TA) block
// {c01ee43d-ad92-44b1-8ab9-be5e696ceea7}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_TA, 0xc01ee43d, 0xad92, 0x44b1, 0x8a, 0xb9,
                0xbe, 0x5e, 0x69, 0x6c, 0xee, 0xa7);

// HSA_UUID that identifies the GPU TextureCache (TCA) block
// {333e393f-e147-4f49-a6d1-60914c7086b0}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_TCA, 0x333e393f, 0xe147, 0x4f49, 0xa6,
                0xd1, 0x60, 0x91, 0x4c, 0x70, 0x86, 0xb0);

// HSA_UUID that identifies the GPU TextureCache (TCC) block
// {848ce855-d805-4566-a8ab-73e884cc6bff}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_TCC, 0x848ce855, 0xd805, 0x4566, 0xa8,
                0xab, 0x73, 0xe8, 0x84, 0xcc, 0x6b, 0xff);

// HSA_UUID that identifies the GPU (TCP) block
// {e10a013b-17d4-4bf5-b089-429591059b60}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_TCP, 0xe10a013b, 0x17d4, 0x4bf5, 0xb0,
                0x89, 0x42, 0x95, 0x91, 0x05, 0x9b, 0x60);

// HSA_UUID that identifies the GPU (TCS) block
// {4126245c-4d96-4d1a-8aed-a939d4cc8ec9}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_TCS, 0x4126245c, 0x4d96, 0x4d1a, 0x8a,
                0xed, 0xa9, 0x39, 0xd4, 0xcc, 0x8e, 0xc9);

// HSA_UUID that identifies the GPU (TD) block
// {7d7c0fe4-fe41-4fea-92c9-4544d7706dc6}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_TD, 0x7d7c0fe4, 0xfe41, 0x4fea, 0x92, 0xc9,
                0x45, 0x44, 0xd7, 0x70, 0x6d, 0xc6);

// HSA_UUID that identifies the GPU (VGT) block
// {0b6a8cb7-7a01-409f-a22c-3014854f1359}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_VGT, 0x0b6a8cb7, 0x7a01, 0x409f, 0xa2,
                0x2c, 0x30, 0x14, 0x85, 0x4f, 0x13, 0x59);

// HSA_UUID that identifies the GPU (WD) block
// {0e176789-46ed-4b02-972a-916d2fac244a}
HSA_DEFINE_UUID(HSA_PROFILEBLOCK_AMD_WD, 0x0e176789, 0x46ed, 0x4b02, 0x97, 0x2a,
                0x91, 0x6d, 0x2f, 0xac, 0x24, 0x4a);

typedef enum _HSA_PROFILE_TYPE {
  HSA_PROFILE_TYPE_PRIVILEGED_IMMEDIATE =
      0, // immediate access counter (KFD access only)
  HSA_PROFILE_TYPE_PRIVILEGED_STREAMING =
      1, // streaming counter, HW continuously
         // writes to memory on updates (KFD access only)
  HSA_PROFILE_TYPE_NONPRIV_IMMEDIATE = 2, // user-queue accessible counter
  HSA_PROFILE_TYPE_NONPRIV_STREAMING = 3, // user-queue accessible counter
  //...
  HSA_PROFILE_TYPE_NUM,

  HSA_PROFILE_TYPE_SIZE = 0xFFFFFFFF // In order to align to 32-bit value
} HSA_PROFILE_TYPE;

typedef struct _HsaCounterFlags {
  union {
    struct {
      unsigned int Global : 1;     // counter is global
                                   // (not tied to VMID/WAVE/CU, ...)
      unsigned int Resettable : 1; // counter can be reset by SW
                                   // (always to 0?)
      unsigned int ReadOnly : 1;   // counter is read-only
                                   // (but may be reset, if indicated)
      unsigned int Stream : 1;     // counter has streaming capability
                                   // (after trigger, updates buffer)
      unsigned int Reserved : 28;
    } ui32;
    HSAuint32 Value;
  };
} HsaCounterFlags;

typedef struct _HsaCounter {
  HSA_PROFILE_TYPE Type;       // specifies the counter type
  HSAuint64 CounterId;         // indicates counter register offset
  HSAuint32 CounterSizeInBits; // indicates relevant counter bits
  HSAuint64 CounterMask;       // bitmask for counter value (if applicable)
  HsaCounterFlags Flags;       // Property flags (see above)
  HSAuint32 BlockIndex;        // identifies block the counter belongs to,
                               // value may be 0 to NumBlocks
} HsaCounter;

typedef struct _HsaCounterBlockProperties {
  HSA_UUID BlockId;        // specifies the block location
  HSAuint32 NumCounters;   // How many counters are available?
                           // (sizes Counters[] array below)
  HSAuint32 NumConcurrent; // How many counter slots are available
                           // in block?
  HsaCounter Counters[1];  // Start of counter array
                           // (NumCounters elements total)
} HsaCounterBlockProperties;

typedef struct _HsaCounterProperties {
  HSAuint32 NumBlocks;     // How many profilable block are available?
                           // (sizes Blocks[] array below)
  HSAuint32 NumConcurrent; // How many blocks slots can be queried
                           // concurrently by HW?
  HsaCounterBlockProperties Blocks[1]; // Start of block array
                                       // (NumBlocks elements total)
} HsaCounterProperties;

typedef HSAuint64 HSATraceId;

typedef struct _HsaPmcTraceRoot {
  HSAuint64 TraceBufferMinSizeBytes; // (page aligned)
  HSAuint32 NumberOfPasses;
  HSATraceId TraceId;
} HsaPmcTraceRoot;

typedef struct _HsaGpuTileConfig {
  HSAuint32 *TileConfig;
  HSAuint32 *MacroTileConfig;
  HSAuint32 NumTileConfigs;
  HSAuint32 NumMacroTileConfigs;

  HSAuint32 GbAddrConfig;

  HSAuint32 NumBanks;
  HSAuint32 NumRanks;
  /* 9 dwords on 64-bit system */
  HSAuint32 Reserved[7]; /* Round up to 16 dwords for future extension */
} HsaGpuTileConfig;

typedef enum _HSA_POINTER_TYPE {
  HSA_POINTER_UNKNOWN = 0,
  HSA_POINTER_ALLOCATED =
      1, // Allocated with hsaKmtAllocMemory (except scratch)
  HSA_POINTER_REGISTERED_USER = 2,     // Registered user pointer
  HSA_POINTER_REGISTERED_GRAPHICS = 3, // Registered graphics buffer
  HSA_POINTER_REGISTERED_SHARED = 4,   // Registered shared buffer (IPC)
                                       // (hsaKmtRegisterGraphicsToNodes)
  HSA_POINTER_RESERVED_ADDR = 5        // address-only reservation VA
} HSA_POINTER_TYPE;

typedef struct _HsaPointerInfo {
  HSA_POINTER_TYPE Type;      // Pointer type
  HSAuint32 Node;             // Node where the memory is located
  HsaMemFlags MemFlags;       // HsaMemFlags used to alloc memory
  void *CPUAddress;           // Start address for CPU access
  HSAuint64 GPUAddress;       // Start address for GPU access
  HSAuint64 SizeInBytes;      // Size in bytes
  HSAuint32 NRegisteredNodes; // Number of nodes the memory is registered to
  HSAuint32 NMappedNodes;     // Number of nodes the memory is mapped to
  const HSAuint32 *RegisteredNodes; // Array of registered nodes
  const HSAuint32 *MappedNodes;     // Array of mapped nodes
  void *UserData;                   // User data associated with the memory
} HsaPointerInfo;

typedef HSAuint32 HsaSharedMemoryHandle[8];

typedef struct _HsaMemoryRange {
  void *MemoryAddress;   // Pointer to GPU memory
  HSAuint64 SizeInBytes; // Size of above memory
} HsaMemoryRange;

typedef enum _HSA_SVM_FLAGS {
  HSA_SVM_FLAG_HOST_ACCESS = 0x00000001, // Guarantee host access to memory
  HSA_SVM_FLAG_COHERENT =
      0x00000002, // Fine grained coherency between all devices with access
  HSA_SVM_FLAG_HIVE_LOCAL =
      0x00000004, // Use any GPU in same hive as preferred device
  HSA_SVM_FLAG_GPU_RO = 0x00000008,   // GPUs only read, allows replication
  HSA_SVM_FLAG_GPU_EXEC = 0x00000010, // Allow execution on GPU
  HSA_SVM_FLAG_GPU_READ_MOSTLY =
      0x00000020, // GPUs mostly read, may allow similar optimizations as RO,
                  // but writes fault
  HSA_SVM_FLAG_GPU_ALWAYS_MAPPED =
      0x00000040, // Keep GPU memory mapping always valid as if XNACK is disable
  HSA_SVM_FLAG_EXT_COHERENT = 0x00000080, //  Fine grained coherency between all
                                          //  devices using device-scope atomics
} HSA_SVM_FLAGS;

typedef enum _HSA_SVM_ATTR_TYPE {
  HSA_SVM_ATTR_PREFERRED_LOC, // gpuid of the preferred location, 0 for
                              // system memory, INVALID_NODEID for
                              // "don't care"
  HSA_SVM_ATTR_PREFETCH_LOC,  // gpuid of the prefetch location, 0 for
                              // system memory. Setting this triggers an
                              // immediate prefetch (migration)
  HSA_SVM_ATTR_ACCESS,
  HSA_SVM_ATTR_ACCESS_IN_PLACE,
  HSA_SVM_ATTR_NO_ACCESS,  // specify memory access for the gpuid given
                           // by the attribute value
  HSA_SVM_ATTR_SET_FLAGS,  // bitmask of flags to set (see HSA_SVM_FLAGS)
  HSA_SVM_ATTR_CLR_FLAGS,  // bitmask of flags to clear
  HSA_SVM_ATTR_GRANULARITY // migration granularity (log2 num pages)
} HSA_SVM_ATTR_TYPE;

typedef struct _HSA_SVM_ATTRIBUTE {
  HSAuint32 type;  // attribute type (see enum HSA_SVM_ATTR_TYPE)
  HSAuint32 value; // attribute value
} HSA_SVM_ATTRIBUTE;

typedef enum _HSA_SMI_EVENT {
  HSA_SMI_EVENT_NONE = 0,    /* not used */
  HSA_SMI_EVENT_VMFAULT = 1, /* event start counting at 1 */
  HSA_SMI_EVENT_THERMAL_THROTTLE = 2,
  HSA_SMI_EVENT_GPU_PRE_RESET = 3,
  HSA_SMI_EVENT_GPU_POST_RESET = 4,
  HSA_SMI_EVENT_MIGRATE_START = 5,
  HSA_SMI_EVENT_MIGRATE_END = 6,
  HSA_SMI_EVENT_PAGE_FAULT_START = 7,
  HSA_SMI_EVENT_PAGE_FAULT_END = 8,
  HSA_SMI_EVENT_QUEUE_EVICTION = 9,
  HSA_SMI_EVENT_QUEUE_RESTORE = 10,
  HSA_SMI_EVENT_UNMAP_FROM_GPU = 11,
  HSA_SMI_EVENT_INDEX_MAX = 12,

  /*
   * max event number, as a flag bit to get events from all processes,
   * this requires super user permission, otherwise will not be able to
   * receive event from any process. Without this flag to receive events
   * from same process.
   */
  HSA_SMI_EVENT_ALL_PROCESS = 64
} HSA_EVENT_TYPE;

typedef enum _HSA_MIGRATE_TRIGGERS {
  HSA_MIGRATE_TRIGGER_PREFETCH,
  HSA_MIGRATE_TRIGGER_PAGEFAULT_GPU,
  HSA_MIGRATE_TRIGGER_PAGEFAULT_CPU,
  HSA_MIGRATE_TRIGGER_TTM_EVICTION
} HSA_MIGRATE_TRIGGERS;

typedef enum _HSA_QUEUE_EVICTION_TRIGGERS {
  HSA_QUEUE_EVICTION_TRIGGER_SVM,
  HSA_QUEUE_EVICTION_TRIGGER_USERPTR,
  HSA_QUEUE_EVICTION_TRIGGER_TTM,
  HSA_QUEUE_EVICTION_TRIGGER_SUSPEND,
  HSA_QUEUE_EVICTION_CRIU_CHECKPOINT,
  HSA_QUEUE_EVICTION_CRIU_RESTORE
} HSA_QUEUE_EVICTION_TRIGGERS;

typedef enum _HSA_SVM_UNMAP_TRIGGERS {
  HSA_SVM_UNMAP_TRIGGER_MMU_NOTIFY,
  HSA_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE,
  HSA_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU
} HSA_SVM_UNMAP_TRIGGERS;

#define HSA_SMI_EVENT_MASK_FROM_INDEX(i) (1ULL << ((i) - 1))
#define HSA_SMI_EVENT_MSG_SIZE 96

typedef void *HsaAMDGPUDeviceHandle;

typedef HSAuint32 HsaPcSamplingTraceId;

typedef enum _HSA_PC_SAMPLING_METHOD_KIND {
  HSA_PC_SAMPLING_METHOD_KIND_HOSTTRAP_V1 = 1,
  HSA_PC_SAMPLING_METHOD_KIND_STOCHASTIC_V1,
} HSA_PC_SAMPLING_METHOD_KIND;

typedef enum _HSA_PC_SAMPLING_UNITS {
  HSA_PC_SAMPLING_UNIT_INTERVAL_MICROSECONDS,
  HSA_PC_SAMPLING_UNIT_INTERVAL_CYCLES,
  HSA_PC_SAMPLING_UNIT_INTERVAL_INSTRUCTIONS,
} HSA_PC_SAMPLING_UNIT_INTERVAL;

typedef struct _HsaPcSamplingInfo {
  HSAuint64 value;
  HSAuint64 value_min;
  HSAuint64 value_max;
  HSAuint64 flags;
  HSA_PC_SAMPLING_METHOD_KIND method;
  HSA_PC_SAMPLING_UNIT_INTERVAL units;
} HsaPcSamplingInfo;

typedef union {
  HSAuint32 Value;
  struct {
    unsigned int requiresVAddr : 1; // Requires virtual address
    unsigned int kmtHandle : 1;     // Handle is a KMT handle
  } ui32;
} HSA_REGISTER_MEM_FLAGS;

#pragma pack(pop, hsakmttypes_h)

typedef enum _HsaAisFlags {
  HSA_AIS_READ = 0x1,
  HSA_AIS_WRITE = 0x2
} HsaAisFlags;

/* memory object handle used for translating drm BO object*/
typedef struct _HsaMemoryObjectHandle *HsaMemoryObjectHandle;

/* Access Permissions for memory mapping */
typedef enum _HsaMemoryMapFlags {
  HSA_MEMORY_ACCESS_NONE = 0,
  HSA_MEMORY_ACCESS_RO = 1,
  HSA_MEMORY_ACCESS_WO = 2,
  HSA_MEMORY_ACCESS_RW = 3
} HsaMemoryMapFlags;

/* Handle type for import */
typedef enum _HsaExternalHandleType {
  HSA_EXTERNAL_HANDLE_GEM_FLINK_NAME = 0,
  HSA_EXTERNAL_HANDLE_KMS = 1,
  HSA_EXTERNAL_HANDLE_DMA_BUF = 2
} HsaExternalHandleType;

typedef struct _HsaExternalHandleDesc {
  HsaAMDGPUDeviceHandle
      device_handle;          // GPU device handle (used for import only)
  HSAint32 fd;                // dmabuf fd
  HsaExternalHandleType type; // handle type
  void *mem;          // existing buffer address (for windows and WSL only)
  HSAuint32 metadata; // Used for IPC handles
} HsaExternalHandleDesc;

typedef struct _HsaHandleImportResult {
  HsaMemoryObjectHandle buf_handle; // Thunk buffer object handle
  HSAuint64 alloc_size;             // allocation size for import
  HSAuint32 metadata;               // Used for IPC handles
} HsaHandleImportResult;

typedef struct _HsaMemoryExportResult {
  HSAint32 fd; // dmabuf fd
} HsaMemoryExportResult;

typedef struct _HsaHandleImportFlags {
  struct {
    unsigned int IPCHandle : 1;      // Handle type is IPC
    unsigned int SysMem : 1;         // Memory type is System Memory
    unsigned int UpdateMetadata : 1; // Update metadata with IPC handle
    unsigned int Reserved : 29;
  } ui32;
} HsaHandleImportFlags;

typedef struct _HsaStructureSizes {
  HSAuint16 StructureSizes; // sizeof(HsaStructureSizes) used for check overflow
  HSAuint16 SizeOfHsaNodeProperties; // sizeof(HsaNodeProperties)
  HSAuint16 Reserved[6];
} HsaStructureSizes;

#ifdef __cplusplus
} // extern "C"
#endif

#endif //_HSAKMTTYPES_H_
