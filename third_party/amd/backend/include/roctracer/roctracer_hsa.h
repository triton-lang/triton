/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef ROCTRACER_HSA_H_
#define ROCTRACER_HSA_H_

#include "roctracer.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include "hsa_ostream_ops.h"
#include "hsa_prof_str.h"

// HSA OP ID enumeration
enum hsa_op_id_t {
  HSA_OP_ID_DISPATCH = 0,
  HSA_OP_ID_COPY = 1,
  HSA_OP_ID_BARRIER = 2,
  HSA_OP_ID_RESERVED1 = 3,
  HSA_OP_ID_NUMBER
};

// HSA EVT ID enumeration
enum hsa_evt_id_t {
  HSA_EVT_ID_ALLOCATE = 0,  // Memory allocate callback
  HSA_EVT_ID_DEVICE = 1,    // Device assign callback
  HSA_EVT_ID_MEMCOPY = 2,   // Memcopy callback
  HSA_EVT_ID_SUBMIT = 3,    // Packet submission callback
  HSA_EVT_ID_KSYMBOL = 4,   // Loading/unloading of kernel symbol
  HSA_EVT_ID_CODEOBJ = 5,   // Loading/unloading of device code object
  HSA_EVT_ID_NUMBER
};

struct hsa_ops_properties_t {
  void* reserved1[4];
};

// HSA EVT data type
typedef struct {
  union {
    struct {
      const void* ptr;  // allocated area ptr
      size_t size;      // allocated area size, zero size means 'free' callback
      hsa_amd_segment_t segment;  // allocated area's memory segment type
      hsa_amd_memory_pool_global_flag_t
          global_flag;  // allocated area's memory global flag
      int is_code;      // equal to 1 if code is allocated
    } allocate;

    struct {
      hsa_device_type_t type;  // type of assigned device
      uint32_t id;             // id of assigned device
      hsa_agent_t agent;       // device HSA agent handle
      const void* ptr;         // ptr the device is assigned to
    } device;

    struct {
      const void* dst;  // memcopy dst ptr
      const void* src;  // memcopy src ptr
      size_t size;      // memcopy size bytes
    } memcopy;

    struct {
      const void* packet;  // submitted to GPU packet
      const char*
          kernel_name;     // kernel name, NULL if not a kernel dispatch packet
      hsa_queue_t* queue;  // HSA queue the packet was submitted to
      uint32_t device_type;  // type of device the packet is submitted to
      uint32_t device_id;    // id of device the packet is submitted to
    } submit;

    struct {
      uint64_t object;       // kernel symbol object
      const char* name;      // kernel symbol name
      uint32_t name_length;  // kernel symbol name length
      int unload;            // symbol executable destroy
    } ksymbol;

    struct {
      uint32_t storage_type;  // code object storage type
      int storage_file;       // origin file descriptor
      uint64_t memory_base;   // origin memory base
      uint64_t memory_size;   // origin memory size
      uint64_t load_base;     // code object load base
      uint64_t load_size;     // code object load size
      uint64_t load_delta;    // code object load size
      uint32_t uri_length;  // URI string length (not including the terminating
                            // NUL character)
      const char* uri;      // URI string
      int unload;           // unload flag
    } codeobj;
  };
} hsa_evt_data_t;

#endif  // ROCTRACER_HSA_H_
