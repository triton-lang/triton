/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_PROF_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_PROF_H

// HIP ROCclr Op IDs enumeration
enum HipVdiOpId {
  kHipVdiOpIdDispatch = 0,
  kHipVdiOpIdCopy     = 1,
  kHipVdiOpIdBarrier  = 2,
  kHipVdiOpIdNumber   = 3
};

// Types of ROCclr commands
enum HipVdiCommandKind {
  kHipVdiCommandKernel            = 0x11F0,
  kHipVdiCommandTask              = 0x11F1,
  kHipVdiMemcpyDeviceToHost       = 0x11F3,
  kHipHipVdiMemcpyHostToDevice    = 0x11F4,
  kHipVdiMemcpyDeviceToDevice     = 0x11F5,
  kHipVidMemcpyDeviceToHostRect   = 0x1201,
  kHipVdiMemcpyHostToDeviceRect   = 0x1202,
  kHipVdiMemcpyDeviceToDeviceRect = 0x1203,
  kHipVdiFillMemory               = 0x1207,
};

/**
 * @brief Initializes activity callback
 *
 * @param [input] id_callback Event ID callback function
 * @param [input] op_callback Event operation callback function
 * @param [input] arg         Arguments passed into callback
 *
 * @returns None
 */
void hipInitActivityCallback(void* id_callback, void* op_callback, void* arg);

/**
 * @brief Enables activity callback
 *
 * @param [input] op      Operation, which will trigger a callback (@see HipVdiOpId)
 * @param [input] enable  Enable state for the callback
 *
 * @returns True if successful
 */
bool hipEnableActivityCallback(uint32_t op, bool enable);

/**
 * @brief Returns the description string for the operation kind
 *
 * @param [input] id      Command kind id (@see HipVdiCommandKind)
 *
 * @returns A pointer to a const string with the command description
 */
const char* hipGetCmdName(uint32_t id);

#endif // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_PROF_H

