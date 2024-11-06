// automatically generated
/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef INC_HIP_OSTREAM_OPS_H_
#define INC_HIP_OSTREAM_OPS_H_

#include <hip/hip_runtime.h>
#include <hip/hip_deprecated.h>
#include "roctracer.h"

#ifdef __cplusplus
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>

namespace roctracer {
namespace hip_support {
static int HIP_depth_max = 1;
static int HIP_depth_max_cnt = 0;
static std::string HIP_structs_regex = "";
// begin ostream ops for HIP 
// basic ostream ops
namespace detail {
  inline static void print_escaped_string(std::ostream& out, const char *v, size_t len) {
    out << '"'; 
    for (size_t i = 0; i < len && v[i]; ++i) {
      switch (v[i]) {
      case '\"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\\b"; break;
      case '\f': out << "\\\f"; break;
      case '\n': out << "\\\n"; break;
      case '\r': out << "\\\r"; break;
      case '\t': out << "\\\t"; break;
      default:
        if (std::isprint((unsigned char)v[i])) std::operator<<(out, v[i]);
        else {
          std::ios_base::fmtflags flags(out.flags());
          out << "\\x" << std::setfill('0') << std::setw(2) << std::hex << (unsigned int)(unsigned char)v[i];
          out.flags(flags);
        }
        break;
      }
    }
    out << '"'; 
  }

  template <typename T>
  inline static std::ostream& operator<<(std::ostream& out, const T& v) {
     using std::operator<<;
     static bool recursion = false;
     if (recursion == false) { recursion = true; out << v; recursion = false; }
     return out;
  }

  inline static std::ostream &operator<<(std::ostream &out, const unsigned char &v) {
    out << (unsigned int)v;
    return out;
  }

  inline static std::ostream &operator<<(std::ostream &out, const char &v) {
    out << (unsigned char)v;
    return out;
  }

  template <size_t N>
  inline static std::ostream &operator<<(std::ostream &out, const char (&v)[N]) {
    print_escaped_string(out, v, N);
    return out;
  }

  inline static std::ostream &operator<<(std::ostream &out, const char *v) {
    print_escaped_string(out, v, strlen(v));
    return out;
  }
// End of basic ostream ops

inline static std::ostream& operator<<(std::ostream& out, const __locale_struct& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("__locale_struct::__names").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "__names=");
      roctracer::hip_support::detail::operator<<(out, v.__names);
      std::operator<<(out, ", ");
    }
    if (std::string("__locale_struct::__ctype_toupper").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "__ctype_toupper=");
      roctracer::hip_support::detail::operator<<(out, v.__ctype_toupper);
      std::operator<<(out, ", ");
    }
    if (std::string("__locale_struct::__ctype_tolower").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "__ctype_tolower=");
      roctracer::hip_support::detail::operator<<(out, v.__ctype_tolower);
      std::operator<<(out, ", ");
    }
    if (std::string("__locale_struct::__ctype_b").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "__ctype_b=");
      roctracer::hip_support::detail::operator<<(out, v.__ctype_b);
      std::operator<<(out, ", ");
    }
    if (std::string("__locale_struct::__locales").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "__locales=");
      roctracer::hip_support::detail::operator<<(out, v.__locales);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipDeviceArch_t& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipDeviceArch_t::hasDynamicParallelism").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasDynamicParallelism=");
      roctracer::hip_support::detail::operator<<(out, v.hasDynamicParallelism);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::has3dGrid").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "has3dGrid=");
      roctracer::hip_support::detail::operator<<(out, v.has3dGrid);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasSurfaceFuncs").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasSurfaceFuncs=");
      roctracer::hip_support::detail::operator<<(out, v.hasSurfaceFuncs);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasSyncThreadsExt").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasSyncThreadsExt=");
      roctracer::hip_support::detail::operator<<(out, v.hasSyncThreadsExt);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasThreadFenceSystem").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasThreadFenceSystem=");
      roctracer::hip_support::detail::operator<<(out, v.hasThreadFenceSystem);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasFunnelShift").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasFunnelShift=");
      roctracer::hip_support::detail::operator<<(out, v.hasFunnelShift);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasWarpShuffle").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasWarpShuffle=");
      roctracer::hip_support::detail::operator<<(out, v.hasWarpShuffle);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasWarpBallot").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasWarpBallot=");
      roctracer::hip_support::detail::operator<<(out, v.hasWarpBallot);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasWarpVote").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasWarpVote=");
      roctracer::hip_support::detail::operator<<(out, v.hasWarpVote);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasDoubles").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasDoubles=");
      roctracer::hip_support::detail::operator<<(out, v.hasDoubles);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasSharedInt64Atomics").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasSharedInt64Atomics=");
      roctracer::hip_support::detail::operator<<(out, v.hasSharedInt64Atomics);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasGlobalInt64Atomics").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasGlobalInt64Atomics=");
      roctracer::hip_support::detail::operator<<(out, v.hasGlobalInt64Atomics);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasFloatAtomicAdd").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasFloatAtomicAdd=");
      roctracer::hip_support::detail::operator<<(out, v.hasFloatAtomicAdd);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasSharedFloatAtomicExch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasSharedFloatAtomicExch=");
      roctracer::hip_support::detail::operator<<(out, v.hasSharedFloatAtomicExch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasSharedInt32Atomics").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasSharedInt32Atomics=");
      roctracer::hip_support::detail::operator<<(out, v.hasSharedInt32Atomics);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasGlobalFloatAtomicExch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasGlobalFloatAtomicExch=");
      roctracer::hip_support::detail::operator<<(out, v.hasGlobalFloatAtomicExch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceArch_t::hasGlobalInt32Atomics").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hasGlobalInt32Atomics=");
      roctracer::hip_support::detail::operator<<(out, v.hasGlobalInt32Atomics);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipUUID& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipUUID::bytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "bytes=");
      roctracer::hip_support::detail::operator<<(out, v.bytes);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipDeviceProp_tR0600& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipDeviceProp_tR0600::asicRevision").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "asicRevision=");
      roctracer::hip_support::detail::operator<<(out, v.asicRevision);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::isLargeBar").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "isLargeBar=");
      roctracer::hip_support::detail::operator<<(out, v.isLargeBar);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::cooperativeMultiDeviceUnmatchedSharedMem").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeMultiDeviceUnmatchedSharedMem=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedSharedMem);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::cooperativeMultiDeviceUnmatchedBlockDim").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeMultiDeviceUnmatchedBlockDim=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedBlockDim);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::cooperativeMultiDeviceUnmatchedGridDim").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeMultiDeviceUnmatchedGridDim=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedGridDim);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::cooperativeMultiDeviceUnmatchedFunc").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeMultiDeviceUnmatchedFunc=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedFunc);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::hdpRegFlushCntl").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hdpRegFlushCntl=");
      roctracer::hip_support::detail::operator<<(out, v.hdpRegFlushCntl);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::hdpMemFlushCntl").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hdpMemFlushCntl=");
      roctracer::hip_support::detail::operator<<(out, v.hdpMemFlushCntl);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::arch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "arch=");
      roctracer::hip_support::detail::operator<<(out, v.arch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::clockInstructionRate").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "clockInstructionRate=");
      roctracer::hip_support::detail::operator<<(out, v.clockInstructionRate);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxSharedMemoryPerMultiProcessor").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxSharedMemoryPerMultiProcessor=");
      roctracer::hip_support::detail::operator<<(out, v.maxSharedMemoryPerMultiProcessor);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::gcnArchName").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "gcnArchName=");
      roctracer::hip_support::detail::operator<<(out, v.gcnArchName);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::hipReserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hipReserved=");
      roctracer::hip_support::detail::operator<<(out, v.hipReserved);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::unifiedFunctionPointers").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "unifiedFunctionPointers=");
      roctracer::hip_support::detail::operator<<(out, v.unifiedFunctionPointers);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::clusterLaunch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "clusterLaunch=");
      roctracer::hip_support::detail::operator<<(out, v.clusterLaunch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::ipcEventSupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "ipcEventSupported=");
      roctracer::hip_support::detail::operator<<(out, v.ipcEventSupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::deferredMappingHipArraySupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "deferredMappingHipArraySupported=");
      roctracer::hip_support::detail::operator<<(out, v.deferredMappingHipArraySupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::memoryPoolSupportedHandleTypes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "memoryPoolSupportedHandleTypes=");
      roctracer::hip_support::detail::operator<<(out, v.memoryPoolSupportedHandleTypes);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::gpuDirectRDMAWritesOrdering").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "gpuDirectRDMAWritesOrdering=");
      roctracer::hip_support::detail::operator<<(out, v.gpuDirectRDMAWritesOrdering);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::gpuDirectRDMAFlushWritesOptions").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "gpuDirectRDMAFlushWritesOptions=");
      roctracer::hip_support::detail::operator<<(out, v.gpuDirectRDMAFlushWritesOptions);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::gpuDirectRDMASupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "gpuDirectRDMASupported=");
      roctracer::hip_support::detail::operator<<(out, v.gpuDirectRDMASupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::memoryPoolsSupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "memoryPoolsSupported=");
      roctracer::hip_support::detail::operator<<(out, v.memoryPoolsSupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::timelineSemaphoreInteropSupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "timelineSemaphoreInteropSupported=");
      roctracer::hip_support::detail::operator<<(out, v.timelineSemaphoreInteropSupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::hostRegisterReadOnlySupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hostRegisterReadOnlySupported=");
      roctracer::hip_support::detail::operator<<(out, v.hostRegisterReadOnlySupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::sparseHipArraySupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sparseHipArraySupported=");
      roctracer::hip_support::detail::operator<<(out, v.sparseHipArraySupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::hostRegisterSupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hostRegisterSupported=");
      roctracer::hip_support::detail::operator<<(out, v.hostRegisterSupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::reservedSharedMemPerBlock").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reservedSharedMemPerBlock=");
      roctracer::hip_support::detail::operator<<(out, v.reservedSharedMemPerBlock);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::accessPolicyMaxWindowSize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "accessPolicyMaxWindowSize=");
      roctracer::hip_support::detail::operator<<(out, v.accessPolicyMaxWindowSize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxBlocksPerMultiProcessor").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxBlocksPerMultiProcessor=");
      roctracer::hip_support::detail::operator<<(out, v.maxBlocksPerMultiProcessor);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::directManagedMemAccessFromHost").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "directManagedMemAccessFromHost=");
      roctracer::hip_support::detail::operator<<(out, v.directManagedMemAccessFromHost);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::pageableMemoryAccessUsesHostPageTables").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pageableMemoryAccessUsesHostPageTables=");
      roctracer::hip_support::detail::operator<<(out, v.pageableMemoryAccessUsesHostPageTables);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::sharedMemPerBlockOptin").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sharedMemPerBlockOptin=");
      roctracer::hip_support::detail::operator<<(out, v.sharedMemPerBlockOptin);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::cooperativeMultiDeviceLaunch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeMultiDeviceLaunch=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeMultiDeviceLaunch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::cooperativeLaunch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeLaunch=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeLaunch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::canUseHostPointerForRegisteredMem").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "canUseHostPointerForRegisteredMem=");
      roctracer::hip_support::detail::operator<<(out, v.canUseHostPointerForRegisteredMem);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::computePreemptionSupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "computePreemptionSupported=");
      roctracer::hip_support::detail::operator<<(out, v.computePreemptionSupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::concurrentManagedAccess").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "concurrentManagedAccess=");
      roctracer::hip_support::detail::operator<<(out, v.concurrentManagedAccess);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::pageableMemoryAccess").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pageableMemoryAccess=");
      roctracer::hip_support::detail::operator<<(out, v.pageableMemoryAccess);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::singleToDoublePrecisionPerfRatio").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "singleToDoublePrecisionPerfRatio=");
      roctracer::hip_support::detail::operator<<(out, v.singleToDoublePrecisionPerfRatio);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::hostNativeAtomicSupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hostNativeAtomicSupported=");
      roctracer::hip_support::detail::operator<<(out, v.hostNativeAtomicSupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::multiGpuBoardGroupID").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "multiGpuBoardGroupID=");
      roctracer::hip_support::detail::operator<<(out, v.multiGpuBoardGroupID);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::isMultiGpuBoard").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "isMultiGpuBoard=");
      roctracer::hip_support::detail::operator<<(out, v.isMultiGpuBoard);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::managedMemory").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "managedMemory=");
      roctracer::hip_support::detail::operator<<(out, v.managedMemory);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::regsPerMultiprocessor").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "regsPerMultiprocessor=");
      roctracer::hip_support::detail::operator<<(out, v.regsPerMultiprocessor);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::sharedMemPerMultiprocessor").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sharedMemPerMultiprocessor=");
      roctracer::hip_support::detail::operator<<(out, v.sharedMemPerMultiprocessor);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::localL1CacheSupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "localL1CacheSupported=");
      roctracer::hip_support::detail::operator<<(out, v.localL1CacheSupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::globalL1CacheSupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "globalL1CacheSupported=");
      roctracer::hip_support::detail::operator<<(out, v.globalL1CacheSupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::streamPrioritiesSupported").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "streamPrioritiesSupported=");
      roctracer::hip_support::detail::operator<<(out, v.streamPrioritiesSupported);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxThreadsPerMultiProcessor").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxThreadsPerMultiProcessor=");
      roctracer::hip_support::detail::operator<<(out, v.maxThreadsPerMultiProcessor);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::persistingL2CacheMaxSize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "persistingL2CacheMaxSize=");
      roctracer::hip_support::detail::operator<<(out, v.persistingL2CacheMaxSize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::l2CacheSize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "l2CacheSize=");
      roctracer::hip_support::detail::operator<<(out, v.l2CacheSize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::memoryBusWidth").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "memoryBusWidth=");
      roctracer::hip_support::detail::operator<<(out, v.memoryBusWidth);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::memoryClockRate").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "memoryClockRate=");
      roctracer::hip_support::detail::operator<<(out, v.memoryClockRate);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::unifiedAddressing").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "unifiedAddressing=");
      roctracer::hip_support::detail::operator<<(out, v.unifiedAddressing);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::asyncEngineCount").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "asyncEngineCount=");
      roctracer::hip_support::detail::operator<<(out, v.asyncEngineCount);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::tccDriver").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "tccDriver=");
      roctracer::hip_support::detail::operator<<(out, v.tccDriver);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::pciDomainID").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pciDomainID=");
      roctracer::hip_support::detail::operator<<(out, v.pciDomainID);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::pciDeviceID").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pciDeviceID=");
      roctracer::hip_support::detail::operator<<(out, v.pciDeviceID);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::pciBusID").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pciBusID=");
      roctracer::hip_support::detail::operator<<(out, v.pciBusID);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::ECCEnabled").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "ECCEnabled=");
      roctracer::hip_support::detail::operator<<(out, v.ECCEnabled);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::concurrentKernels").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "concurrentKernels=");
      roctracer::hip_support::detail::operator<<(out, v.concurrentKernels);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::surfaceAlignment").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "surfaceAlignment=");
      roctracer::hip_support::detail::operator<<(out, v.surfaceAlignment);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxSurfaceCubemapLayered").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxSurfaceCubemapLayered=");
      roctracer::hip_support::detail::operator<<(out, v.maxSurfaceCubemapLayered);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxSurfaceCubemap").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxSurfaceCubemap=");
      roctracer::hip_support::detail::operator<<(out, v.maxSurfaceCubemap);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxSurface2DLayered").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxSurface2DLayered=");
      roctracer::hip_support::detail::operator<<(out, v.maxSurface2DLayered);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxSurface1DLayered").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxSurface1DLayered=");
      roctracer::hip_support::detail::operator<<(out, v.maxSurface1DLayered);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxSurface3D").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxSurface3D=");
      roctracer::hip_support::detail::operator<<(out, v.maxSurface3D);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxSurface2D").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxSurface2D=");
      roctracer::hip_support::detail::operator<<(out, v.maxSurface2D);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxSurface1D").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxSurface1D=");
      roctracer::hip_support::detail::operator<<(out, v.maxSurface1D);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTextureCubemapLayered").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTextureCubemapLayered=");
      roctracer::hip_support::detail::operator<<(out, v.maxTextureCubemapLayered);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTexture2DLayered").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture2DLayered=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture2DLayered);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTexture1DLayered").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture1DLayered=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture1DLayered);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTextureCubemap").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTextureCubemap=");
      roctracer::hip_support::detail::operator<<(out, v.maxTextureCubemap);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTexture3DAlt").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture3DAlt=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture3DAlt);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTexture3D").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture3D=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture3D);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTexture2DGather").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture2DGather=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture2DGather);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTexture2DLinear").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture2DLinear=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture2DLinear);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTexture2DMipmap").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture2DMipmap=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture2DMipmap);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTexture2D").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture2D=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture2D);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTexture1DLinear").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture1DLinear=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture1DLinear);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTexture1DMipmap").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture1DMipmap=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture1DMipmap);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxTexture1D").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture1D=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture1D);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::computeMode").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "computeMode=");
      roctracer::hip_support::detail::operator<<(out, v.computeMode);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::canMapHostMemory").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "canMapHostMemory=");
      roctracer::hip_support::detail::operator<<(out, v.canMapHostMemory);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::integrated").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "integrated=");
      roctracer::hip_support::detail::operator<<(out, v.integrated);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::kernelExecTimeoutEnabled").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "kernelExecTimeoutEnabled=");
      roctracer::hip_support::detail::operator<<(out, v.kernelExecTimeoutEnabled);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::multiProcessorCount").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "multiProcessorCount=");
      roctracer::hip_support::detail::operator<<(out, v.multiProcessorCount);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::deviceOverlap").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "deviceOverlap=");
      roctracer::hip_support::detail::operator<<(out, v.deviceOverlap);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::texturePitchAlignment").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "texturePitchAlignment=");
      roctracer::hip_support::detail::operator<<(out, v.texturePitchAlignment);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::textureAlignment").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "textureAlignment=");
      roctracer::hip_support::detail::operator<<(out, v.textureAlignment);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::minor").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "minor=");
      roctracer::hip_support::detail::operator<<(out, v.minor);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::major").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "major=");
      roctracer::hip_support::detail::operator<<(out, v.major);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::totalConstMem").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "totalConstMem=");
      roctracer::hip_support::detail::operator<<(out, v.totalConstMem);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::clockRate").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "clockRate=");
      roctracer::hip_support::detail::operator<<(out, v.clockRate);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxGridSize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxGridSize=");
      roctracer::hip_support::detail::operator<<(out, v.maxGridSize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxThreadsDim").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxThreadsDim=");
      roctracer::hip_support::detail::operator<<(out, v.maxThreadsDim);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::maxThreadsPerBlock").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxThreadsPerBlock=");
      roctracer::hip_support::detail::operator<<(out, v.maxThreadsPerBlock);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::memPitch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "memPitch=");
      roctracer::hip_support::detail::operator<<(out, v.memPitch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::warpSize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "warpSize=");
      roctracer::hip_support::detail::operator<<(out, v.warpSize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::regsPerBlock").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "regsPerBlock=");
      roctracer::hip_support::detail::operator<<(out, v.regsPerBlock);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::sharedMemPerBlock").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sharedMemPerBlock=");
      roctracer::hip_support::detail::operator<<(out, v.sharedMemPerBlock);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::totalGlobalMem").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "totalGlobalMem=");
      roctracer::hip_support::detail::operator<<(out, v.totalGlobalMem);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::luidDeviceNodeMask").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "luidDeviceNodeMask=");
      roctracer::hip_support::detail::operator<<(out, v.luidDeviceNodeMask);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::luid").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "luid=");
      roctracer::hip_support::detail::operator<<(out, v.luid);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::uuid").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "uuid=");
      roctracer::hip_support::detail::operator<<(out, v.uuid);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0600::name").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "name=");
      roctracer::hip_support::detail::operator<<(out, v.name);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipPointerAttribute_t& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipPointerAttribute_t::allocationFlags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "allocationFlags=");
      roctracer::hip_support::detail::operator<<(out, v.allocationFlags);
      std::operator<<(out, ", ");
    }
    if (std::string("hipPointerAttribute_t::isManaged").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "isManaged=");
      roctracer::hip_support::detail::operator<<(out, v.isManaged);
      std::operator<<(out, ", ");
    }
    if (std::string("hipPointerAttribute_t::device").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "device=");
      roctracer::hip_support::detail::operator<<(out, v.device);
      std::operator<<(out, ", ");
    }
    if (std::string("hipPointerAttribute_t::type").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "type=");
      roctracer::hip_support::detail::operator<<(out, v.type);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipChannelFormatDesc& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipChannelFormatDesc::f").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "f=");
      roctracer::hip_support::detail::operator<<(out, v.f);
      std::operator<<(out, ", ");
    }
    if (std::string("hipChannelFormatDesc::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("hipChannelFormatDesc::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("hipChannelFormatDesc::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("hipChannelFormatDesc::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const HIP_ARRAY_DESCRIPTOR& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("HIP_ARRAY_DESCRIPTOR::NumChannels").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "NumChannels=");
      roctracer::hip_support::detail::operator<<(out, v.NumChannels);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_ARRAY_DESCRIPTOR::Format").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "Format=");
      roctracer::hip_support::detail::operator<<(out, v.Format);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_ARRAY_DESCRIPTOR::Height").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "Height=");
      roctracer::hip_support::detail::operator<<(out, v.Height);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_ARRAY_DESCRIPTOR::Width").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "Width=");
      roctracer::hip_support::detail::operator<<(out, v.Width);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const HIP_ARRAY3D_DESCRIPTOR& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("HIP_ARRAY3D_DESCRIPTOR::Flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "Flags=");
      roctracer::hip_support::detail::operator<<(out, v.Flags);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_ARRAY3D_DESCRIPTOR::NumChannels").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "NumChannels=");
      roctracer::hip_support::detail::operator<<(out, v.NumChannels);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_ARRAY3D_DESCRIPTOR::Format").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "Format=");
      roctracer::hip_support::detail::operator<<(out, v.Format);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_ARRAY3D_DESCRIPTOR::Depth").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "Depth=");
      roctracer::hip_support::detail::operator<<(out, v.Depth);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_ARRAY3D_DESCRIPTOR::Height").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "Height=");
      roctracer::hip_support::detail::operator<<(out, v.Height);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_ARRAY3D_DESCRIPTOR::Width").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "Width=");
      roctracer::hip_support::detail::operator<<(out, v.Width);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hip_Memcpy2D& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hip_Memcpy2D::Height").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "Height=");
      roctracer::hip_support::detail::operator<<(out, v.Height);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::WidthInBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "WidthInBytes=");
      roctracer::hip_support::detail::operator<<(out, v.WidthInBytes);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::dstPitch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstPitch=");
      roctracer::hip_support::detail::operator<<(out, v.dstPitch);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::dstArray").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstArray=");
      roctracer::hip_support::detail::operator<<(out, v.dstArray);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::dstDevice").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstDevice=");
      roctracer::hip_support::detail::operator<<(out, v.dstDevice);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::dstMemoryType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstMemoryType=");
      roctracer::hip_support::detail::operator<<(out, v.dstMemoryType);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::dstY").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstY=");
      roctracer::hip_support::detail::operator<<(out, v.dstY);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::dstXInBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstXInBytes=");
      roctracer::hip_support::detail::operator<<(out, v.dstXInBytes);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::srcPitch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcPitch=");
      roctracer::hip_support::detail::operator<<(out, v.srcPitch);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::srcArray").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcArray=");
      roctracer::hip_support::detail::operator<<(out, v.srcArray);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::srcDevice").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcDevice=");
      roctracer::hip_support::detail::operator<<(out, v.srcDevice);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::srcMemoryType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcMemoryType=");
      roctracer::hip_support::detail::operator<<(out, v.srcMemoryType);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::srcY").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcY=");
      roctracer::hip_support::detail::operator<<(out, v.srcY);
      std::operator<<(out, ", ");
    }
    if (std::string("hip_Memcpy2D::srcXInBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcXInBytes=");
      roctracer::hip_support::detail::operator<<(out, v.srcXInBytes);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipMipmappedArray& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipMipmappedArray::num_channels").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "num_channels=");
      roctracer::hip_support::detail::operator<<(out, v.num_channels);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMipmappedArray::format").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "format=");
      roctracer::hip_support::detail::operator<<(out, v.format);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMipmappedArray::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMipmappedArray::max_mipmap_level").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "max_mipmap_level=");
      roctracer::hip_support::detail::operator<<(out, v.max_mipmap_level);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMipmappedArray::min_mipmap_level").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "min_mipmap_level=");
      roctracer::hip_support::detail::operator<<(out, v.min_mipmap_level);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMipmappedArray::depth").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "depth=");
      roctracer::hip_support::detail::operator<<(out, v.depth);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMipmappedArray::height").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "height=");
      roctracer::hip_support::detail::operator<<(out, v.height);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMipmappedArray::width").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "width=");
      roctracer::hip_support::detail::operator<<(out, v.width);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMipmappedArray::type").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "type=");
      roctracer::hip_support::detail::operator<<(out, v.type);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMipmappedArray::desc").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "desc=");
      roctracer::hip_support::detail::operator<<(out, v.desc);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const HIP_TEXTURE_DESC& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("HIP_TEXTURE_DESC::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_TEXTURE_DESC::borderColor").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "borderColor=");
      roctracer::hip_support::detail::operator<<(out, v.borderColor);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_TEXTURE_DESC::maxMipmapLevelClamp").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxMipmapLevelClamp=");
      roctracer::hip_support::detail::operator<<(out, v.maxMipmapLevelClamp);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_TEXTURE_DESC::minMipmapLevelClamp").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "minMipmapLevelClamp=");
      roctracer::hip_support::detail::operator<<(out, v.minMipmapLevelClamp);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_TEXTURE_DESC::mipmapLevelBias").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "mipmapLevelBias=");
      roctracer::hip_support::detail::operator<<(out, v.mipmapLevelBias);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_TEXTURE_DESC::mipmapFilterMode").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "mipmapFilterMode=");
      roctracer::hip_support::detail::operator<<(out, v.mipmapFilterMode);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_TEXTURE_DESC::maxAnisotropy").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxAnisotropy=");
      roctracer::hip_support::detail::operator<<(out, v.maxAnisotropy);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_TEXTURE_DESC::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_TEXTURE_DESC::filterMode").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "filterMode=");
      roctracer::hip_support::detail::operator<<(out, v.filterMode);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_TEXTURE_DESC::addressMode").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "addressMode=");
      roctracer::hip_support::detail::operator<<(out, v.addressMode);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipResourceDesc& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipResourceDesc::resType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "resType=");
      roctracer::hip_support::detail::operator<<(out, v.resType);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const HIP_RESOURCE_DESC& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("HIP_RESOURCE_DESC::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_RESOURCE_DESC::resType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "resType=");
      roctracer::hip_support::detail::operator<<(out, v.resType);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipResourceViewDesc& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipResourceViewDesc::lastLayer").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "lastLayer=");
      roctracer::hip_support::detail::operator<<(out, v.lastLayer);
      std::operator<<(out, ", ");
    }
    if (std::string("hipResourceViewDesc::firstLayer").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "firstLayer=");
      roctracer::hip_support::detail::operator<<(out, v.firstLayer);
      std::operator<<(out, ", ");
    }
    if (std::string("hipResourceViewDesc::lastMipmapLevel").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "lastMipmapLevel=");
      roctracer::hip_support::detail::operator<<(out, v.lastMipmapLevel);
      std::operator<<(out, ", ");
    }
    if (std::string("hipResourceViewDesc::firstMipmapLevel").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "firstMipmapLevel=");
      roctracer::hip_support::detail::operator<<(out, v.firstMipmapLevel);
      std::operator<<(out, ", ");
    }
    if (std::string("hipResourceViewDesc::depth").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "depth=");
      roctracer::hip_support::detail::operator<<(out, v.depth);
      std::operator<<(out, ", ");
    }
    if (std::string("hipResourceViewDesc::height").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "height=");
      roctracer::hip_support::detail::operator<<(out, v.height);
      std::operator<<(out, ", ");
    }
    if (std::string("hipResourceViewDesc::width").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "width=");
      roctracer::hip_support::detail::operator<<(out, v.width);
      std::operator<<(out, ", ");
    }
    if (std::string("hipResourceViewDesc::format").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "format=");
      roctracer::hip_support::detail::operator<<(out, v.format);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const HIP_RESOURCE_VIEW_DESC& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("HIP_RESOURCE_VIEW_DESC::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_RESOURCE_VIEW_DESC::lastLayer").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "lastLayer=");
      roctracer::hip_support::detail::operator<<(out, v.lastLayer);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_RESOURCE_VIEW_DESC::firstLayer").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "firstLayer=");
      roctracer::hip_support::detail::operator<<(out, v.firstLayer);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_RESOURCE_VIEW_DESC::lastMipmapLevel").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "lastMipmapLevel=");
      roctracer::hip_support::detail::operator<<(out, v.lastMipmapLevel);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_RESOURCE_VIEW_DESC::firstMipmapLevel").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "firstMipmapLevel=");
      roctracer::hip_support::detail::operator<<(out, v.firstMipmapLevel);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_RESOURCE_VIEW_DESC::depth").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "depth=");
      roctracer::hip_support::detail::operator<<(out, v.depth);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_RESOURCE_VIEW_DESC::height").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "height=");
      roctracer::hip_support::detail::operator<<(out, v.height);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_RESOURCE_VIEW_DESC::width").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "width=");
      roctracer::hip_support::detail::operator<<(out, v.width);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_RESOURCE_VIEW_DESC::format").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "format=");
      roctracer::hip_support::detail::operator<<(out, v.format);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipPitchedPtr& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipPitchedPtr::ysize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "ysize=");
      roctracer::hip_support::detail::operator<<(out, v.ysize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipPitchedPtr::xsize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "xsize=");
      roctracer::hip_support::detail::operator<<(out, v.xsize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipPitchedPtr::pitch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pitch=");
      roctracer::hip_support::detail::operator<<(out, v.pitch);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipExtent& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipExtent::depth").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "depth=");
      roctracer::hip_support::detail::operator<<(out, v.depth);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExtent::height").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "height=");
      roctracer::hip_support::detail::operator<<(out, v.height);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExtent::width").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "width=");
      roctracer::hip_support::detail::operator<<(out, v.width);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipPos& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipPos::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("hipPos::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("hipPos::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipMemcpy3DParms& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipMemcpy3DParms::kind").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "kind=");
      roctracer::hip_support::detail::operator<<(out, v.kind);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemcpy3DParms::extent").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "extent=");
      roctracer::hip_support::detail::operator<<(out, v.extent);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemcpy3DParms::dstPtr").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstPtr=");
      roctracer::hip_support::detail::operator<<(out, v.dstPtr);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemcpy3DParms::dstPos").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstPos=");
      roctracer::hip_support::detail::operator<<(out, v.dstPos);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemcpy3DParms::dstArray").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstArray=");
      roctracer::hip_support::detail::operator<<(out, v.dstArray);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemcpy3DParms::srcPtr").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcPtr=");
      roctracer::hip_support::detail::operator<<(out, v.srcPtr);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemcpy3DParms::srcPos").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcPos=");
      roctracer::hip_support::detail::operator<<(out, v.srcPos);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemcpy3DParms::srcArray").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcArray=");
      roctracer::hip_support::detail::operator<<(out, v.srcArray);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const HIP_MEMCPY3D& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("HIP_MEMCPY3D::Depth").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "Depth=");
      roctracer::hip_support::detail::operator<<(out, v.Depth);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::Height").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "Height=");
      roctracer::hip_support::detail::operator<<(out, v.Height);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::WidthInBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "WidthInBytes=");
      roctracer::hip_support::detail::operator<<(out, v.WidthInBytes);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::dstHeight").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstHeight=");
      roctracer::hip_support::detail::operator<<(out, v.dstHeight);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::dstPitch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstPitch=");
      roctracer::hip_support::detail::operator<<(out, v.dstPitch);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::dstArray").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstArray=");
      roctracer::hip_support::detail::operator<<(out, v.dstArray);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::dstDevice").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstDevice=");
      roctracer::hip_support::detail::operator<<(out, v.dstDevice);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::dstMemoryType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstMemoryType=");
      roctracer::hip_support::detail::operator<<(out, v.dstMemoryType);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::dstLOD").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstLOD=");
      roctracer::hip_support::detail::operator<<(out, v.dstLOD);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::dstZ").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstZ=");
      roctracer::hip_support::detail::operator<<(out, v.dstZ);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::dstY").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstY=");
      roctracer::hip_support::detail::operator<<(out, v.dstY);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::dstXInBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dstXInBytes=");
      roctracer::hip_support::detail::operator<<(out, v.dstXInBytes);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::srcHeight").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcHeight=");
      roctracer::hip_support::detail::operator<<(out, v.srcHeight);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::srcPitch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcPitch=");
      roctracer::hip_support::detail::operator<<(out, v.srcPitch);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::srcArray").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcArray=");
      roctracer::hip_support::detail::operator<<(out, v.srcArray);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::srcDevice").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcDevice=");
      roctracer::hip_support::detail::operator<<(out, v.srcDevice);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::srcMemoryType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcMemoryType=");
      roctracer::hip_support::detail::operator<<(out, v.srcMemoryType);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::srcLOD").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcLOD=");
      roctracer::hip_support::detail::operator<<(out, v.srcLOD);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::srcZ").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcZ=");
      roctracer::hip_support::detail::operator<<(out, v.srcZ);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::srcY").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcY=");
      roctracer::hip_support::detail::operator<<(out, v.srcY);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMCPY3D::srcXInBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "srcXInBytes=");
      roctracer::hip_support::detail::operator<<(out, v.srcXInBytes);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const uchar1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("uchar1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const uchar2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("uchar2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("uchar2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const uchar3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("uchar3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("uchar3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("uchar3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const uchar4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("uchar4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("uchar4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("uchar4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("uchar4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const char1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("char1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const char2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("char2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("char2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const char3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("char3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("char3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("char3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const char4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("char4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("char4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("char4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("char4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ushort1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ushort1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ushort2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ushort2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("ushort2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ushort3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ushort3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("ushort3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("ushort3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ushort4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ushort4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("ushort4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("ushort4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("ushort4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const short1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("short1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const short2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("short2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("short2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const short3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("short3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("short3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("short3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const short4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("short4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("short4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("short4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("short4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const uint1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("uint1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const uint2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("uint2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("uint2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const uint3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("uint3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("uint3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("uint3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const uint4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("uint4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("uint4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("uint4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("uint4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const int1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("int1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const int2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("int2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("int2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const int3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("int3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("int3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("int3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const int4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("int4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("int4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("int4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("int4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ulong1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ulong1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ulong2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ulong2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("ulong2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ulong3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ulong3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("ulong3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("ulong3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ulong4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ulong4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("ulong4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("ulong4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("ulong4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const long1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("long1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const long2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("long2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("long2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const long3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("long3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("long3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("long3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const long4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("long4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("long4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("long4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("long4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ulonglong1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ulonglong1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ulonglong2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ulonglong2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("ulonglong2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ulonglong3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ulonglong3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("ulonglong3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("ulonglong3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const ulonglong4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("ulonglong4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("ulonglong4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("ulonglong4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("ulonglong4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const longlong1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("longlong1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const longlong2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("longlong2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("longlong2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const longlong3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("longlong3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("longlong3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("longlong3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const longlong4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("longlong4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("longlong4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("longlong4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("longlong4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const float1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("float1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const float2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("float2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("float2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const float3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("float3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("float3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("float3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const float4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("float4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("float4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("float4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("float4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const double1& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("double1::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const double2& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("double2::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("double2::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const double3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("double3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("double3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("double3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const double4& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("double4::w").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "w=");
      roctracer::hip_support::detail::operator<<(out, v.w);
      std::operator<<(out, ", ");
    }
    if (std::string("double4::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("double4::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("double4::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const textureReference& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("textureReference::format").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "format=");
      roctracer::hip_support::detail::operator<<(out, v.format);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::numChannels").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "numChannels=");
      roctracer::hip_support::detail::operator<<(out, v.numChannels);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::textureObject").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "textureObject=");
      roctracer::hip_support::detail::operator<<(out, v.textureObject);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::maxMipmapLevelClamp").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxMipmapLevelClamp=");
      roctracer::hip_support::detail::operator<<(out, v.maxMipmapLevelClamp);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::minMipmapLevelClamp").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "minMipmapLevelClamp=");
      roctracer::hip_support::detail::operator<<(out, v.minMipmapLevelClamp);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::mipmapLevelBias").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "mipmapLevelBias=");
      roctracer::hip_support::detail::operator<<(out, v.mipmapLevelBias);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::mipmapFilterMode").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "mipmapFilterMode=");
      roctracer::hip_support::detail::operator<<(out, v.mipmapFilterMode);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::maxAnisotropy").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxAnisotropy=");
      roctracer::hip_support::detail::operator<<(out, v.maxAnisotropy);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::sRGB").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sRGB=");
      roctracer::hip_support::detail::operator<<(out, v.sRGB);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::channelDesc").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "channelDesc=");
      roctracer::hip_support::detail::operator<<(out, v.channelDesc);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::filterMode").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "filterMode=");
      roctracer::hip_support::detail::operator<<(out, v.filterMode);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::readMode").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "readMode=");
      roctracer::hip_support::detail::operator<<(out, v.readMode);
      std::operator<<(out, ", ");
    }
    if (std::string("textureReference::normalized").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "normalized=");
      roctracer::hip_support::detail::operator<<(out, v.normalized);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipTextureDesc& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipTextureDesc::maxMipmapLevelClamp").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxMipmapLevelClamp=");
      roctracer::hip_support::detail::operator<<(out, v.maxMipmapLevelClamp);
      std::operator<<(out, ", ");
    }
    if (std::string("hipTextureDesc::minMipmapLevelClamp").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "minMipmapLevelClamp=");
      roctracer::hip_support::detail::operator<<(out, v.minMipmapLevelClamp);
      std::operator<<(out, ", ");
    }
    if (std::string("hipTextureDesc::mipmapLevelBias").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "mipmapLevelBias=");
      roctracer::hip_support::detail::operator<<(out, v.mipmapLevelBias);
      std::operator<<(out, ", ");
    }
    if (std::string("hipTextureDesc::mipmapFilterMode").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "mipmapFilterMode=");
      roctracer::hip_support::detail::operator<<(out, v.mipmapFilterMode);
      std::operator<<(out, ", ");
    }
    if (std::string("hipTextureDesc::maxAnisotropy").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxAnisotropy=");
      roctracer::hip_support::detail::operator<<(out, v.maxAnisotropy);
      std::operator<<(out, ", ");
    }
    if (std::string("hipTextureDesc::normalizedCoords").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "normalizedCoords=");
      roctracer::hip_support::detail::operator<<(out, v.normalizedCoords);
      std::operator<<(out, ", ");
    }
    if (std::string("hipTextureDesc::borderColor").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "borderColor=");
      roctracer::hip_support::detail::operator<<(out, v.borderColor);
      std::operator<<(out, ", ");
    }
    if (std::string("hipTextureDesc::sRGB").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sRGB=");
      roctracer::hip_support::detail::operator<<(out, v.sRGB);
      std::operator<<(out, ", ");
    }
    if (std::string("hipTextureDesc::readMode").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "readMode=");
      roctracer::hip_support::detail::operator<<(out, v.readMode);
      std::operator<<(out, ", ");
    }
    if (std::string("hipTextureDesc::filterMode").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "filterMode=");
      roctracer::hip_support::detail::operator<<(out, v.filterMode);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const surfaceReference& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("surfaceReference::surfaceObject").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "surfaceObject=");
      roctracer::hip_support::detail::operator<<(out, v.surfaceObject);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipIpcMemHandle_t& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipIpcMemHandle_t::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipIpcEventHandle_t& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipIpcEventHandle_t::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipFuncAttributes& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipFuncAttributes::sharedSizeBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sharedSizeBytes=");
      roctracer::hip_support::detail::operator<<(out, v.sharedSizeBytes);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFuncAttributes::ptxVersion").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "ptxVersion=");
      roctracer::hip_support::detail::operator<<(out, v.ptxVersion);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFuncAttributes::preferredShmemCarveout").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "preferredShmemCarveout=");
      roctracer::hip_support::detail::operator<<(out, v.preferredShmemCarveout);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFuncAttributes::numRegs").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "numRegs=");
      roctracer::hip_support::detail::operator<<(out, v.numRegs);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFuncAttributes::maxThreadsPerBlock").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxThreadsPerBlock=");
      roctracer::hip_support::detail::operator<<(out, v.maxThreadsPerBlock);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFuncAttributes::maxDynamicSharedSizeBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxDynamicSharedSizeBytes=");
      roctracer::hip_support::detail::operator<<(out, v.maxDynamicSharedSizeBytes);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFuncAttributes::localSizeBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "localSizeBytes=");
      roctracer::hip_support::detail::operator<<(out, v.localSizeBytes);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFuncAttributes::constSizeBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "constSizeBytes=");
      roctracer::hip_support::detail::operator<<(out, v.constSizeBytes);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFuncAttributes::cacheModeCA").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cacheModeCA=");
      roctracer::hip_support::detail::operator<<(out, v.cacheModeCA);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFuncAttributes::binaryVersion").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "binaryVersion=");
      roctracer::hip_support::detail::operator<<(out, v.binaryVersion);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipMemLocation& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipMemLocation::id").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "id=");
      roctracer::hip_support::detail::operator<<(out, v.id);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemLocation::type").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "type=");
      roctracer::hip_support::detail::operator<<(out, v.type);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipMemAccessDesc& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipMemAccessDesc::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemAccessDesc::location").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "location=");
      roctracer::hip_support::detail::operator<<(out, v.location);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipMemPoolProps& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipMemPoolProps::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemPoolProps::maxSize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxSize=");
      roctracer::hip_support::detail::operator<<(out, v.maxSize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemPoolProps::location").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "location=");
      roctracer::hip_support::detail::operator<<(out, v.location);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemPoolProps::handleTypes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handleTypes=");
      roctracer::hip_support::detail::operator<<(out, v.handleTypes);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemPoolProps::allocType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "allocType=");
      roctracer::hip_support::detail::operator<<(out, v.allocType);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipMemPoolPtrExportData& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipMemPoolPtrExportData::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const dim3& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("dim3::z").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "z=");
      roctracer::hip_support::detail::operator<<(out, v.z);
      std::operator<<(out, ", ");
    }
    if (std::string("dim3::y").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "y=");
      roctracer::hip_support::detail::operator<<(out, v.y);
      std::operator<<(out, ", ");
    }
    if (std::string("dim3::x").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "x=");
      roctracer::hip_support::detail::operator<<(out, v.x);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipLaunchParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipLaunchParams::stream").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "stream=");
      roctracer::hip_support::detail::operator<<(out, v.stream);
      std::operator<<(out, ", ");
    }
    if (std::string("hipLaunchParams::sharedMem").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sharedMem=");
      roctracer::hip_support::detail::operator<<(out, v.sharedMem);
      std::operator<<(out, ", ");
    }
    if (std::string("hipLaunchParams::blockDim").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "blockDim=");
      roctracer::hip_support::detail::operator<<(out, v.blockDim);
      std::operator<<(out, ", ");
    }
    if (std::string("hipLaunchParams::gridDim").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "gridDim=");
      roctracer::hip_support::detail::operator<<(out, v.gridDim);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipFunctionLaunchParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipFunctionLaunchParams::hStream").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hStream=");
      roctracer::hip_support::detail::operator<<(out, v.hStream);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFunctionLaunchParams::sharedMemBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sharedMemBytes=");
      roctracer::hip_support::detail::operator<<(out, v.sharedMemBytes);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFunctionLaunchParams::blockDimZ").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "blockDimZ=");
      roctracer::hip_support::detail::operator<<(out, v.blockDimZ);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFunctionLaunchParams::blockDimY").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "blockDimY=");
      roctracer::hip_support::detail::operator<<(out, v.blockDimY);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFunctionLaunchParams::blockDimX").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "blockDimX=");
      roctracer::hip_support::detail::operator<<(out, v.blockDimX);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFunctionLaunchParams::gridDimZ").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "gridDimZ=");
      roctracer::hip_support::detail::operator<<(out, v.gridDimZ);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFunctionLaunchParams::gridDimY").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "gridDimY=");
      roctracer::hip_support::detail::operator<<(out, v.gridDimY);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFunctionLaunchParams::gridDimX").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "gridDimX=");
      roctracer::hip_support::detail::operator<<(out, v.gridDimX);
      std::operator<<(out, ", ");
    }
    if (std::string("hipFunctionLaunchParams::function").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "function=");
      roctracer::hip_support::detail::operator<<(out, v.function);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipExternalMemoryHandleDesc& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipExternalMemoryHandleDesc::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalMemoryHandleDesc::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalMemoryHandleDesc::size").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "size=");
      roctracer::hip_support::detail::operator<<(out, v.size);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalMemoryHandleDesc_st::union ::handle.fd").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle.fd=");
      roctracer::hip_support::detail::operator<<(out, v.handle.fd);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalMemoryHandleDesc::type").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "type=");
      roctracer::hip_support::detail::operator<<(out, v.type);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipExternalMemoryBufferDesc& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipExternalMemoryBufferDesc::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalMemoryBufferDesc::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalMemoryBufferDesc::size").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "size=");
      roctracer::hip_support::detail::operator<<(out, v.size);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalMemoryBufferDesc::offset").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "offset=");
      roctracer::hip_support::detail::operator<<(out, v.offset);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipExternalMemoryMipmappedArrayDesc& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipExternalMemoryMipmappedArrayDesc::numLevels").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "numLevels=");
      roctracer::hip_support::detail::operator<<(out, v.numLevels);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalMemoryMipmappedArrayDesc::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalMemoryMipmappedArrayDesc::extent").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "extent=");
      roctracer::hip_support::detail::operator<<(out, v.extent);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalMemoryMipmappedArrayDesc::formatDesc").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "formatDesc=");
      roctracer::hip_support::detail::operator<<(out, v.formatDesc);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalMemoryMipmappedArrayDesc::offset").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "offset=");
      roctracer::hip_support::detail::operator<<(out, v.offset);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipExternalSemaphoreHandleDesc& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipExternalSemaphoreHandleDesc::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalSemaphoreHandleDesc::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalSemaphoreHandleDesc_st::union ::handle.fd").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "handle.fd=");
      roctracer::hip_support::detail::operator<<(out, v.handle.fd);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalSemaphoreHandleDesc::type").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "type=");
      roctracer::hip_support::detail::operator<<(out, v.type);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipExternalSemaphoreSignalParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipExternalSemaphoreSignalParams::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalSemaphoreSignalParams::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipExternalSemaphoreWaitParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipExternalSemaphoreWaitParams::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalSemaphoreWaitParams::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipHostNodeParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipHostNodeParams::fn").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "fn=");
      roctracer::hip_support::detail::operator<<(out, v.fn);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipKernelNodeParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipKernelNodeParams::sharedMemBytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sharedMemBytes=");
      roctracer::hip_support::detail::operator<<(out, v.sharedMemBytes);
      std::operator<<(out, ", ");
    }
    if (std::string("hipKernelNodeParams::gridDim").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "gridDim=");
      roctracer::hip_support::detail::operator<<(out, v.gridDim);
      std::operator<<(out, ", ");
    }
    if (std::string("hipKernelNodeParams::blockDim").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "blockDim=");
      roctracer::hip_support::detail::operator<<(out, v.blockDim);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipMemsetParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipMemsetParams::width").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "width=");
      roctracer::hip_support::detail::operator<<(out, v.width);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemsetParams::value").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "value=");
      roctracer::hip_support::detail::operator<<(out, v.value);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemsetParams::pitch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pitch=");
      roctracer::hip_support::detail::operator<<(out, v.pitch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemsetParams::height").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "height=");
      roctracer::hip_support::detail::operator<<(out, v.height);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemsetParams::elementSize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "elementSize=");
      roctracer::hip_support::detail::operator<<(out, v.elementSize);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipMemAllocNodeParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipMemAllocNodeParams::bytesize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "bytesize=");
      roctracer::hip_support::detail::operator<<(out, v.bytesize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemAllocNodeParams::accessDescCount").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "accessDescCount=");
      roctracer::hip_support::detail::operator<<(out, v.accessDescCount);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemAllocNodeParams::accessDescs").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "accessDescs=");
      roctracer::hip_support::detail::operator<<(out, v.accessDescs);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemAllocNodeParams::poolProps").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "poolProps=");
      roctracer::hip_support::detail::operator<<(out, v.poolProps);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipAccessPolicyWindow& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipAccessPolicyWindow::num_bytes").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "num_bytes=");
      roctracer::hip_support::detail::operator<<(out, v.num_bytes);
      std::operator<<(out, ", ");
    }
    if (std::string("hipAccessPolicyWindow::missProp").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "missProp=");
      roctracer::hip_support::detail::operator<<(out, v.missProp);
      std::operator<<(out, ", ");
    }
    if (std::string("hipAccessPolicyWindow::hitRatio").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hitRatio=");
      roctracer::hip_support::detail::operator<<(out, v.hitRatio);
      std::operator<<(out, ", ");
    }
    if (std::string("hipAccessPolicyWindow::hitProp").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hitProp=");
      roctracer::hip_support::detail::operator<<(out, v.hitProp);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipLaunchAttributeValue& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipLaunchAttributeValue::priority").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "priority=");
      roctracer::hip_support::detail::operator<<(out, v.priority);
      std::operator<<(out, ", ");
    }
    if (std::string("hipLaunchAttributeValue::cooperative").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperative=");
      roctracer::hip_support::detail::operator<<(out, v.cooperative);
      std::operator<<(out, ", ");
    }
    if (std::string("hipLaunchAttributeValue::accessPolicyWindow").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "accessPolicyWindow=");
      roctracer::hip_support::detail::operator<<(out, v.accessPolicyWindow);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const HIP_MEMSET_NODE_PARAMS& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("HIP_MEMSET_NODE_PARAMS::height").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "height=");
      roctracer::hip_support::detail::operator<<(out, v.height);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMSET_NODE_PARAMS::width").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "width=");
      roctracer::hip_support::detail::operator<<(out, v.width);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMSET_NODE_PARAMS::elementSize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "elementSize=");
      roctracer::hip_support::detail::operator<<(out, v.elementSize);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMSET_NODE_PARAMS::value").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "value=");
      roctracer::hip_support::detail::operator<<(out, v.value);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMSET_NODE_PARAMS::pitch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pitch=");
      roctracer::hip_support::detail::operator<<(out, v.pitch);
      std::operator<<(out, ", ");
    }
    if (std::string("HIP_MEMSET_NODE_PARAMS::dst").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "dst=");
      roctracer::hip_support::detail::operator<<(out, v.dst);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipGraphInstantiateParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipGraphInstantiateParams::uploadStream").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "uploadStream=");
      roctracer::hip_support::detail::operator<<(out, v.uploadStream);
      std::operator<<(out, ", ");
    }
    if (std::string("hipGraphInstantiateParams::result_out").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "result_out=");
      roctracer::hip_support::detail::operator<<(out, v.result_out);
      std::operator<<(out, ", ");
    }
    if (std::string("hipGraphInstantiateParams::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
      std::operator<<(out, ", ");
    }
    if (std::string("hipGraphInstantiateParams::errNode_out").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "errNode_out=");
      roctracer::hip_support::detail::operator<<(out, v.errNode_out);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipMemAllocationProp& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipMemAllocationProp::location").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "location=");
      roctracer::hip_support::detail::operator<<(out, v.location);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemAllocationProp::requestedHandleType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "requestedHandleType=");
      roctracer::hip_support::detail::operator<<(out, v.requestedHandleType);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemAllocationProp::type").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "type=");
      roctracer::hip_support::detail::operator<<(out, v.type);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipExternalSemaphoreSignalNodeParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipExternalSemaphoreSignalNodeParams::numExtSems").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "numExtSems=");
      roctracer::hip_support::detail::operator<<(out, v.numExtSems);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalSemaphoreSignalNodeParams::paramsArray").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "paramsArray=");
      roctracer::hip_support::detail::operator<<(out, v.paramsArray);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalSemaphoreSignalNodeParams::extSemArray").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "extSemArray=");
      roctracer::hip_support::detail::operator<<(out, v.extSemArray);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipExternalSemaphoreWaitNodeParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipExternalSemaphoreWaitNodeParams::numExtSems").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "numExtSems=");
      roctracer::hip_support::detail::operator<<(out, v.numExtSems);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalSemaphoreWaitNodeParams::paramsArray").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "paramsArray=");
      roctracer::hip_support::detail::operator<<(out, v.paramsArray);
      std::operator<<(out, ", ");
    }
    if (std::string("hipExternalSemaphoreWaitNodeParams::extSemArray").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "extSemArray=");
      roctracer::hip_support::detail::operator<<(out, v.extSemArray);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipArrayMapInfo& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipArrayMapInfo::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("hipArrayMapInfo::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
      std::operator<<(out, ", ");
    }
    if (std::string("hipArrayMapInfo::deviceBitMask").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "deviceBitMask=");
      roctracer::hip_support::detail::operator<<(out, v.deviceBitMask);
      std::operator<<(out, ", ");
    }
    if (std::string("hipArrayMapInfo::offset").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "offset=");
      roctracer::hip_support::detail::operator<<(out, v.offset);
      std::operator<<(out, ", ");
    }
    if (std::string("hipArrayMapInfo::union ::memHandle.memHandle").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "memHandle.memHandle=");
      roctracer::hip_support::detail::operator<<(out, v.memHandle.memHandle);
      std::operator<<(out, ", ");
    }
    if (std::string("hipArrayMapInfo::memHandleType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "memHandleType=");
      roctracer::hip_support::detail::operator<<(out, v.memHandleType);
      std::operator<<(out, ", ");
    }
    if (std::string("hipArrayMapInfo::memOperationType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "memOperationType=");
      roctracer::hip_support::detail::operator<<(out, v.memOperationType);
      std::operator<<(out, ", ");
    }
    if (std::string("hipArrayMapInfo::subresourceType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "subresourceType=");
      roctracer::hip_support::detail::operator<<(out, v.subresourceType);
      std::operator<<(out, ", ");
    }
    if (std::string("hipArrayMapInfo::resourceType").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "resourceType=");
      roctracer::hip_support::detail::operator<<(out, v.resourceType);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipMemcpyNodeParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipMemcpyNodeParams::copyParams").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "copyParams=");
      roctracer::hip_support::detail::operator<<(out, v.copyParams);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemcpyNodeParams::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("hipMemcpyNodeParams::flags").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "flags=");
      roctracer::hip_support::detail::operator<<(out, v.flags);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipChildGraphNodeParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipChildGraphNodeParams::graph").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "graph=");
      roctracer::hip_support::detail::operator<<(out, v.graph);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipEventWaitNodeParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipEventWaitNodeParams::event").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "event=");
      roctracer::hip_support::detail::operator<<(out, v.event);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipEventRecordNodeParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipEventRecordNodeParams::event").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "event=");
      roctracer::hip_support::detail::operator<<(out, v.event);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipMemFreeNodeParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipGraphNodeParams& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipGraphNodeParams::reserved2").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved2=");
      roctracer::hip_support::detail::operator<<(out, v.reserved2);
      std::operator<<(out, ", ");
    }
    if (std::string("hipGraphNodeParams::reserved0").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved0=");
      roctracer::hip_support::detail::operator<<(out, v.reserved0);
      std::operator<<(out, ", ");
    }
    if (std::string("hipGraphNodeParams::type").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "type=");
      roctracer::hip_support::detail::operator<<(out, v.type);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipGraphEdgeData& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipGraphEdgeData::type").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "type=");
      roctracer::hip_support::detail::operator<<(out, v.type);
      std::operator<<(out, ", ");
    }
    if (std::string("hipGraphEdgeData::to_port").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "to_port=");
      roctracer::hip_support::detail::operator<<(out, v.to_port);
      std::operator<<(out, ", ");
    }
    if (std::string("hipGraphEdgeData::reserved").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "reserved=");
      roctracer::hip_support::detail::operator<<(out, 0);
      std::operator<<(out, ", ");
    }
    if (std::string("hipGraphEdgeData::from_port").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "from_port=");
      roctracer::hip_support::detail::operator<<(out, v.from_port);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
inline static std::ostream& operator<<(std::ostream& out, const hipDeviceProp_tR0000& v)
{
  std::operator<<(out, '{');
  HIP_depth_max_cnt++;
  if (HIP_depth_max == -1 || HIP_depth_max_cnt <= HIP_depth_max) {
    if (std::string("hipDeviceProp_tR0000::pageableMemoryAccessUsesHostPageTables").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pageableMemoryAccessUsesHostPageTables=");
      roctracer::hip_support::detail::operator<<(out, v.pageableMemoryAccessUsesHostPageTables);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::pageableMemoryAccess").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pageableMemoryAccess=");
      roctracer::hip_support::detail::operator<<(out, v.pageableMemoryAccess);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::concurrentManagedAccess").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "concurrentManagedAccess=");
      roctracer::hip_support::detail::operator<<(out, v.concurrentManagedAccess);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::directManagedMemAccessFromHost").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "directManagedMemAccessFromHost=");
      roctracer::hip_support::detail::operator<<(out, v.directManagedMemAccessFromHost);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::managedMemory").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "managedMemory=");
      roctracer::hip_support::detail::operator<<(out, v.managedMemory);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::asicRevision").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "asicRevision=");
      roctracer::hip_support::detail::operator<<(out, v.asicRevision);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::isLargeBar").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "isLargeBar=");
      roctracer::hip_support::detail::operator<<(out, v.isLargeBar);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::cooperativeMultiDeviceUnmatchedSharedMem").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeMultiDeviceUnmatchedSharedMem=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedSharedMem);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::cooperativeMultiDeviceUnmatchedBlockDim").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeMultiDeviceUnmatchedBlockDim=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedBlockDim);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::cooperativeMultiDeviceUnmatchedGridDim").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeMultiDeviceUnmatchedGridDim=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedGridDim);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::cooperativeMultiDeviceUnmatchedFunc").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeMultiDeviceUnmatchedFunc=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeMultiDeviceUnmatchedFunc);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::tccDriver").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "tccDriver=");
      roctracer::hip_support::detail::operator<<(out, v.tccDriver);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::ECCEnabled").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "ECCEnabled=");
      roctracer::hip_support::detail::operator<<(out, v.ECCEnabled);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::kernelExecTimeoutEnabled").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "kernelExecTimeoutEnabled=");
      roctracer::hip_support::detail::operator<<(out, v.kernelExecTimeoutEnabled);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::texturePitchAlignment").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "texturePitchAlignment=");
      roctracer::hip_support::detail::operator<<(out, v.texturePitchAlignment);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::textureAlignment").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "textureAlignment=");
      roctracer::hip_support::detail::operator<<(out, v.textureAlignment);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::memPitch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "memPitch=");
      roctracer::hip_support::detail::operator<<(out, v.memPitch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::hdpRegFlushCntl").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hdpRegFlushCntl=");
      roctracer::hip_support::detail::operator<<(out, v.hdpRegFlushCntl);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::hdpMemFlushCntl").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "hdpMemFlushCntl=");
      roctracer::hip_support::detail::operator<<(out, v.hdpMemFlushCntl);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::maxTexture3D").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture3D=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture3D);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::maxTexture2D").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture2D=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture2D);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::maxTexture1D").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture1D=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture1D);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::maxTexture1DLinear").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxTexture1DLinear=");
      roctracer::hip_support::detail::operator<<(out, v.maxTexture1DLinear);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::cooperativeMultiDeviceLaunch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeMultiDeviceLaunch=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeMultiDeviceLaunch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::cooperativeLaunch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "cooperativeLaunch=");
      roctracer::hip_support::detail::operator<<(out, v.cooperativeLaunch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::integrated").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "integrated=");
      roctracer::hip_support::detail::operator<<(out, v.integrated);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::gcnArchName").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "gcnArchName=");
      roctracer::hip_support::detail::operator<<(out, v.gcnArchName);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::gcnArch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "gcnArch=");
      roctracer::hip_support::detail::operator<<(out, v.gcnArch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::canMapHostMemory").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "canMapHostMemory=");
      roctracer::hip_support::detail::operator<<(out, v.canMapHostMemory);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::isMultiGpuBoard").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "isMultiGpuBoard=");
      roctracer::hip_support::detail::operator<<(out, v.isMultiGpuBoard);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::maxSharedMemoryPerMultiProcessor").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxSharedMemoryPerMultiProcessor=");
      roctracer::hip_support::detail::operator<<(out, v.maxSharedMemoryPerMultiProcessor);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::pciDeviceID").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pciDeviceID=");
      roctracer::hip_support::detail::operator<<(out, v.pciDeviceID);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::pciBusID").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pciBusID=");
      roctracer::hip_support::detail::operator<<(out, v.pciBusID);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::pciDomainID").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "pciDomainID=");
      roctracer::hip_support::detail::operator<<(out, v.pciDomainID);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::concurrentKernels").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "concurrentKernels=");
      roctracer::hip_support::detail::operator<<(out, v.concurrentKernels);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::arch").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "arch=");
      roctracer::hip_support::detail::operator<<(out, v.arch);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::clockInstructionRate").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "clockInstructionRate=");
      roctracer::hip_support::detail::operator<<(out, v.clockInstructionRate);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::computeMode").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "computeMode=");
      roctracer::hip_support::detail::operator<<(out, v.computeMode);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::maxThreadsPerMultiProcessor").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxThreadsPerMultiProcessor=");
      roctracer::hip_support::detail::operator<<(out, v.maxThreadsPerMultiProcessor);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::l2CacheSize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "l2CacheSize=");
      roctracer::hip_support::detail::operator<<(out, v.l2CacheSize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::multiProcessorCount").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "multiProcessorCount=");
      roctracer::hip_support::detail::operator<<(out, v.multiProcessorCount);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::minor").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "minor=");
      roctracer::hip_support::detail::operator<<(out, v.minor);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::major").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "major=");
      roctracer::hip_support::detail::operator<<(out, v.major);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::totalConstMem").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "totalConstMem=");
      roctracer::hip_support::detail::operator<<(out, v.totalConstMem);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::memoryBusWidth").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "memoryBusWidth=");
      roctracer::hip_support::detail::operator<<(out, v.memoryBusWidth);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::memoryClockRate").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "memoryClockRate=");
      roctracer::hip_support::detail::operator<<(out, v.memoryClockRate);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::clockRate").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "clockRate=");
      roctracer::hip_support::detail::operator<<(out, v.clockRate);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::maxGridSize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxGridSize=");
      roctracer::hip_support::detail::operator<<(out, v.maxGridSize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::maxThreadsDim").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxThreadsDim=");
      roctracer::hip_support::detail::operator<<(out, v.maxThreadsDim);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::maxThreadsPerBlock").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "maxThreadsPerBlock=");
      roctracer::hip_support::detail::operator<<(out, v.maxThreadsPerBlock);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::warpSize").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "warpSize=");
      roctracer::hip_support::detail::operator<<(out, v.warpSize);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::regsPerBlock").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "regsPerBlock=");
      roctracer::hip_support::detail::operator<<(out, v.regsPerBlock);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::sharedMemPerBlock").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "sharedMemPerBlock=");
      roctracer::hip_support::detail::operator<<(out, v.sharedMemPerBlock);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::totalGlobalMem").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "totalGlobalMem=");
      roctracer::hip_support::detail::operator<<(out, v.totalGlobalMem);
      std::operator<<(out, ", ");
    }
    if (std::string("hipDeviceProp_tR0000::name").find(HIP_structs_regex) != std::string::npos)   {
      std::operator<<(out, "name=");
      roctracer::hip_support::detail::operator<<(out, v.name);
    }
  };
  HIP_depth_max_cnt--;
  std::operator<<(out, '}');
  return out;
}
// end ostream ops for HIP 
};};};

inline static std::ostream& operator<<(std::ostream& out, const __locale_struct& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipDeviceArch_t& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipUUID& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipDeviceProp_tR0600& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipPointerAttribute_t& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipChannelFormatDesc& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const HIP_ARRAY_DESCRIPTOR& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const HIP_ARRAY3D_DESCRIPTOR& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hip_Memcpy2D& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipMipmappedArray& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const HIP_TEXTURE_DESC& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipResourceDesc& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const HIP_RESOURCE_DESC& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipResourceViewDesc& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const HIP_RESOURCE_VIEW_DESC& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipPitchedPtr& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipExtent& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipPos& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipMemcpy3DParms& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const HIP_MEMCPY3D& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const uchar1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const uchar2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const uchar3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const uchar4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const char1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const char2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const char3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const char4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ushort1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ushort2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ushort3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ushort4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const short1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const short2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const short3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const short4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const uint1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const uint2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const uint3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const uint4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const int1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const int2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const int3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const int4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ulong1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ulong2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ulong3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ulong4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const long1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const long2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const long3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const long4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ulonglong1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ulonglong2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ulonglong3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const ulonglong4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const longlong1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const longlong2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const longlong3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const longlong4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const float1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const float2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const float3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const float4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const double1& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const double2& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const double3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const double4& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const textureReference& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipTextureDesc& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const surfaceReference& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipIpcMemHandle_t& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipIpcEventHandle_t& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipFuncAttributes& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipMemLocation& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipMemAccessDesc& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipMemPoolProps& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipMemPoolPtrExportData& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const dim3& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipLaunchParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipFunctionLaunchParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipExternalMemoryHandleDesc& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipExternalMemoryBufferDesc& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipExternalMemoryMipmappedArrayDesc& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipExternalSemaphoreHandleDesc& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipExternalSemaphoreSignalParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipExternalSemaphoreWaitParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipHostNodeParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipKernelNodeParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipMemsetParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipMemAllocNodeParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipAccessPolicyWindow& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipLaunchAttributeValue& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const HIP_MEMSET_NODE_PARAMS& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipGraphInstantiateParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipMemAllocationProp& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipExternalSemaphoreSignalNodeParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipExternalSemaphoreWaitNodeParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipArrayMapInfo& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipMemcpyNodeParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipChildGraphNodeParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipEventWaitNodeParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipEventRecordNodeParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipMemFreeNodeParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipGraphNodeParams& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipGraphEdgeData& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

inline static std::ostream& operator<<(std::ostream& out, const hipDeviceProp_tR0000& v)
{
  roctracer::hip_support::detail::operator<<(out, v);
  return out;
}

#endif //__cplusplus
#endif // INC_HIP_OSTREAM_OPS_H_
 
