////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2024, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_INC_HSA_API_TRACE_VERSION_H
#define HSA_RUNTIME_INC_HSA_API_TRACE_VERSION_H

// CODE IN THIS FILE **MUST** BE C-COMPATIBLE

// Major Ids of the Api tables exported by Hsa Core Runtime
#define HSA_API_TABLE_MAJOR_VERSION                 0x03
#define HSA_CORE_API_TABLE_MAJOR_VERSION            0x02
#define HSA_AMD_EXT_API_TABLE_MAJOR_VERSION         0x02
#define HSA_FINALIZER_API_TABLE_MAJOR_VERSION       0x02
#define HSA_IMAGE_API_TABLE_MAJOR_VERSION           0x02
#define HSA_AQLPROFILE_API_TABLE_MAJOR_VERSION      0x01
#define HSA_TOOLS_API_TABLE_MAJOR_VERSION           0x01
#define HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION     0x01

// Step Ids of the Api tables exported by Hsa Core Runtime
#define HSA_API_TABLE_STEP_VERSION                  0x01
#define HSA_CORE_API_TABLE_STEP_VERSION             0x00
#define HSA_AMD_EXT_API_TABLE_STEP_VERSION          0x03
#define HSA_FINALIZER_API_TABLE_STEP_VERSION        0x00
#define HSA_IMAGE_API_TABLE_STEP_VERSION            0x00
#define HSA_AQLPROFILE_API_TABLE_STEP_VERSION       0x00
#define HSA_TOOLS_API_TABLE_STEP_VERSION            0x00
#define HSA_PC_SAMPLING_API_TABLE_STEP_VERSION      0x00

#endif  // HSA_RUNTIME_INC_HSA_API_TRACE_VERSION_H
