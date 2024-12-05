////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

// Undefine the macro in case it is defined in the system elf.h.
#undef EM_AMDGPU

#ifndef AMD_HSA_ELF_H
#define AMD_HSA_ELF_H

// AMD GPU Specific ELF Header Enumeration Values.
//
// Values are copied from LLVM BinaryFormat/ELF.h . This file also contains
// code object V1 defintions which are not part of the LLVM header. Code object
// V1 was only supported by the Finalizer which is now deprecated and removed.
//
// TODO: Deprecate and remove V1 support and replace this header with using the
// LLVM header.
namespace ELF {

// Machine architectures
// See current registered ELF machine architectures at:
//    http://www.uxsglobal.com/developers/gabi/latest/ch4.eheader.html
enum {
  EM_AMDGPU = 224,        // AMD GPU architecture
};

// OS ABI identification.
enum {
  ELFOSABI_AMDGPU_HSA = 64,    // AMD HSA runtime
};

// AMDGPU OS ABI Version identification.
enum {
  // ELFABIVERSION_AMDGPU_HSA_V1 does not exist because OS ABI identification
  // was never defined for V1.
  ELFABIVERSION_AMDGPU_HSA_V2 = 0,
  ELFABIVERSION_AMDGPU_HSA_V3 = 1,
  ELFABIVERSION_AMDGPU_HSA_V4 = 2,
  ELFABIVERSION_AMDGPU_HSA_V5 = 3
};

// AMDGPU specific e_flags.
enum : unsigned {
  // Processor selection mask for EF_AMDGPU_MACH_* values.
  EF_AMDGPU_MACH = 0x0ff,

  // Not specified processor.
  EF_AMDGPU_MACH_NONE = 0x000,

  // AMDGCN-based processors.
  EF_AMDGPU_MACH_AMDGCN_GFX600        = 0x020,
  EF_AMDGPU_MACH_AMDGCN_GFX601        = 0x021,
  EF_AMDGPU_MACH_AMDGCN_GFX700        = 0x022,
  EF_AMDGPU_MACH_AMDGCN_GFX701        = 0x023,
  EF_AMDGPU_MACH_AMDGCN_GFX702        = 0x024,
  EF_AMDGPU_MACH_AMDGCN_GFX703        = 0x025,
  EF_AMDGPU_MACH_AMDGCN_GFX704        = 0x026,
  EF_AMDGPU_MACH_AMDGCN_RESERVED_0X27 = 0x027,
  EF_AMDGPU_MACH_AMDGCN_GFX801        = 0x028,
  EF_AMDGPU_MACH_AMDGCN_GFX802        = 0x029,
  EF_AMDGPU_MACH_AMDGCN_GFX803        = 0x02a,
  EF_AMDGPU_MACH_AMDGCN_GFX810        = 0x02b,
  EF_AMDGPU_MACH_AMDGCN_GFX900        = 0x02c,
  EF_AMDGPU_MACH_AMDGCN_GFX902        = 0x02d,
  EF_AMDGPU_MACH_AMDGCN_GFX904        = 0x02e,
  EF_AMDGPU_MACH_AMDGCN_GFX906        = 0x02f,
  EF_AMDGPU_MACH_AMDGCN_GFX908        = 0x030,
  EF_AMDGPU_MACH_AMDGCN_GFX909        = 0x031,
  EF_AMDGPU_MACH_AMDGCN_GFX90C        = 0x032,
  EF_AMDGPU_MACH_AMDGCN_GFX1010       = 0x033,
  EF_AMDGPU_MACH_AMDGCN_GFX1011       = 0x034,
  EF_AMDGPU_MACH_AMDGCN_GFX1012       = 0x035,
  EF_AMDGPU_MACH_AMDGCN_GFX1030       = 0x036,
  EF_AMDGPU_MACH_AMDGCN_GFX1031       = 0x037,
  EF_AMDGPU_MACH_AMDGCN_GFX1032       = 0x038,
  EF_AMDGPU_MACH_AMDGCN_GFX1033       = 0x039,
  EF_AMDGPU_MACH_AMDGCN_GFX602        = 0x03a,
  EF_AMDGPU_MACH_AMDGCN_GFX705        = 0x03b,
  EF_AMDGPU_MACH_AMDGCN_GFX805        = 0x03c,
  EF_AMDGPU_MACH_AMDGCN_GFX1035       = 0x03d,
  EF_AMDGPU_MACH_AMDGCN_GFX1034       = 0x03e,
  EF_AMDGPU_MACH_AMDGCN_GFX90A        = 0x03f,
  EF_AMDGPU_MACH_AMDGCN_GFX940        = 0x040,
  EF_AMDGPU_MACH_AMDGCN_GFX1100       = 0x041,
  EF_AMDGPU_MACH_AMDGCN_GFX1013       = 0x042,
  EF_AMDGPU_MACH_AMDGCN_GFX1150       = 0x043,
  EF_AMDGPU_MACH_AMDGCN_GFX1103       = 0x044,
  EF_AMDGPU_MACH_AMDGCN_GFX1036       = 0x045,
  EF_AMDGPU_MACH_AMDGCN_GFX1101       = 0x046,
  EF_AMDGPU_MACH_AMDGCN_GFX1102       = 0x047,
  EF_AMDGPU_MACH_AMDGCN_GFX1151       = 0x04a,
  EF_AMDGPU_MACH_AMDGCN_GFX941        = 0x04b,
  EF_AMDGPU_MACH_AMDGCN_GFX942        = 0x04c,

  // First/last AMDGCN-based processors.
  EF_AMDGPU_MACH_AMDGCN_FIRST = EF_AMDGPU_MACH_AMDGCN_GFX600,
  EF_AMDGPU_MACH_AMDGCN_LAST = EF_AMDGPU_MACH_AMDGCN_GFX942,

  // Indicates if the "xnack" target feature is enabled for all code contained
  // in the object.
  //
  // Only valid for ELFOSABI_AMDGPU_HSA and ELFABIVERSION_AMDGPU_HSA_V2.
  EF_AMDGPU_FEATURE_XNACK_V2 = 0x01,
  // Indicates if the trap handler is enabled for all code contained
  // in the object.
  //
  // Only valid for ELFOSABI_AMDGPU_HSA and ELFABIVERSION_AMDGPU_HSA_V2.
  EF_AMDGPU_FEATURE_TRAP_HANDLER_V2 = 0x02,

  // Indicates if the "xnack" target feature is enabled for all code contained
  // in the object.
  //
  // Only valid for ELFOSABI_AMDGPU_HSA and ELFABIVERSION_AMDGPU_HSA_V3.
  EF_AMDGPU_FEATURE_XNACK_V3 = 0x100,
  // Indicates if the "sramecc" target feature is enabled for all code
  // contained in the object.
  //
  // Only valid for ELFOSABI_AMDGPU_HSA and ELFABIVERSION_AMDGPU_HSA_V3.
  EF_AMDGPU_FEATURE_SRAMECC_V3 = 0x200,

  // XNACK selection mask for EF_AMDGPU_FEATURE_XNACK_* values.
  //
  // Only valid for ELFOSABI_AMDGPU_HSA and ELFABIVERSION_AMDGPU_HSA_V4,
  // ELFABIVERSION_AMDGPU_HSA_V5.
  EF_AMDGPU_FEATURE_XNACK_V4 = 0x300,
  // XNACK is not supported.
  EF_AMDGPU_FEATURE_XNACK_UNSUPPORTED_V4 = 0x000,
  // XNACK is any/default/unspecified.
  EF_AMDGPU_FEATURE_XNACK_ANY_V4 = 0x100,
  // XNACK is off.
  EF_AMDGPU_FEATURE_XNACK_OFF_V4 = 0x200,
  // XNACK is on.
  EF_AMDGPU_FEATURE_XNACK_ON_V4 = 0x300,

  // SRAMECC selection mask for EF_AMDGPU_FEATURE_SRAMECC_* values.
  //
  // Only valid for ELFOSABI_AMDGPU_HSA and ELFABIVERSION_AMDGPU_HSA_V4,
  // ELFABIVERSION_AMDGPU_HSA_V5.
  EF_AMDGPU_FEATURE_SRAMECC_V4 = 0xc00,
  // SRAMECC is not supported.
  EF_AMDGPU_FEATURE_SRAMECC_UNSUPPORTED_V4 = 0x000,
  // SRAMECC is any/default/unspecified.
  EF_AMDGPU_FEATURE_SRAMECC_ANY_V4 = 0x400,
  // SRAMECC is off.
  EF_AMDGPU_FEATURE_SRAMECC_OFF_V4 = 0x800,
  // SRAMECC is on.
  EF_AMDGPU_FEATURE_SRAMECC_ON_V4 = 0xc00,
};

} // end namespace ELF

// ELF Section Header Flag Enumeration Values.
#define SHF_AMDGPU_HSA_GLOBAL   (0x00100000 & SHF_MASKOS)
#define SHF_AMDGPU_HSA_READONLY (0x00200000 & SHF_MASKOS)
#define SHF_AMDGPU_HSA_CODE     (0x00400000 & SHF_MASKOS)
#define SHF_AMDGPU_HSA_AGENT    (0x00800000 & SHF_MASKOS)

//
typedef enum {
  AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM = 0,
  AMDGPU_HSA_SEGMENT_GLOBAL_AGENT = 1,
  AMDGPU_HSA_SEGMENT_READONLY_AGENT = 2,
  AMDGPU_HSA_SEGMENT_CODE_AGENT = 3,
  AMDGPU_HSA_SEGMENT_LAST,
} amdgpu_hsa_elf_segment_t;

// ELF Program Header Type Enumeration Values.
#define PT_AMDGPU_HSA_LOAD_GLOBAL_PROGRAM (PT_LOOS + AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM)
#define PT_AMDGPU_HSA_LOAD_GLOBAL_AGENT   (PT_LOOS + AMDGPU_HSA_SEGMENT_GLOBAL_AGENT)
#define PT_AMDGPU_HSA_LOAD_READONLY_AGENT (PT_LOOS + AMDGPU_HSA_SEGMENT_READONLY_AGENT)
#define PT_AMDGPU_HSA_LOAD_CODE_AGENT     (PT_LOOS + AMDGPU_HSA_SEGMENT_CODE_AGENT)

// ELF Symbol Type Enumeration Values.
#define STT_AMDGPU_HSA_KERNEL            (STT_LOOS + 0)
#define STT_AMDGPU_HSA_INDIRECT_FUNCTION (STT_LOOS + 1)
#define STT_AMDGPU_HSA_METADATA          (STT_LOOS + 2)

// ELF Symbol Binding Enumeration Values.
#define STB_AMDGPU_HSA_EXTERNAL (STB_LOOS + 0)

// ELF Symbol Other Information Creation/Retrieval.
#define ELF64_ST_AMDGPU_ALLOCATION(o)  (((o) >> 2) & 0x3)
#define ELF64_ST_AMDGPU_FLAGS(o)       ((o) >> 4)
#define ELF64_ST_AMDGPU_OTHER(f, a, v) (((f) << 4) + (((a) & 0x3) << 2) + ((v) & 0x3))

typedef enum {
  AMDGPU_HSA_SYMBOL_ALLOCATION_DEFAULT = 0,
  AMDGPU_HSA_SYMBOL_ALLOCATION_GLOBAL_PROGRAM = 1,
  AMDGPU_HSA_SYMBOL_ALLOCATION_GLOBAL_AGENT = 2,
  AMDGPU_HSA_SYMBOL_ALLOCATION_READONLY_AGENT = 3,
  AMDGPU_HSA_SYMBOL_ALLOCATION_LAST,
} amdgpu_hsa_symbol_allocation_t;

// ELF Symbol Allocation Enumeration Values.
#define STA_AMDGPU_HSA_DEFAULT        AMDGPU_HSA_SYMBOL_ALLOCATION_DEFAULT
#define STA_AMDGPU_HSA_GLOBAL_PROGRAM AMDGPU_HSA_SYMBOL_ALLOCATION_GLOBAL_PROGRAM
#define STA_AMDGPU_HSA_GLOBAL_AGENT   AMDGPU_HSA_SYMBOL_ALLOCATION_GLOBAL_AGENT
#define STA_AMDGPU_HSA_READONLY_AGENT AMDGPU_HSA_SYMBOL_ALLOCATION_READONLY_AGENT

typedef enum {
  AMDGPU_HSA_SYMBOL_FLAG_DEFAULT = 0,
  AMDGPU_HSA_SYMBOL_FLAG_CONST = 1,
  AMDGPU_HSA_SYMBOL_FLAG_LAST,
} amdgpu_hsa_symbol_flag_t;

// ELF Symbol Flag Enumeration Values.
#define STF_AMDGPU_HSA_CONST AMDGPU_HSA_SYMBOL_FLAG_CONST

// AMD GPU Relocation Type Enumeration Values.
#define R_AMDGPU_NONE         0
#define R_AMDGPU_32_LOW       1
#define R_AMDGPU_32_HIGH      2
#define R_AMDGPU_64           3
#define R_AMDGPU_INIT_SAMPLER 4
#define R_AMDGPU_INIT_IMAGE   5
#define R_AMDGPU_RELATIVE64   13

// AMD GPU Note Type Enumeration Values.
#define NT_AMD_HSA_CODE_OBJECT_VERSION 1
#define NT_AMD_HSA_HSAIL               2
#define NT_AMD_HSA_ISA_VERSION         3
#define NT_AMD_HSA_PRODUCER            4
#define NT_AMD_HSA_PRODUCER_OPTIONS    5
#define NT_AMD_HSA_EXTENSION           6
#define NT_AMD_HSA_ISA_NAME            11
/* AMDGPU snapshots of runtime, agent and queues state for use in core dump */
#define NT_AMDGPU_CORE_STATE           33
#define NT_AMD_HSA_HLDEBUG_DEBUG       101
#define NT_AMD_HSA_HLDEBUG_TARGET      102

// AMD GPU Metadata Kind Enumeration Values.
typedef uint16_t amdgpu_hsa_metadata_kind16_t;
typedef enum {
  AMDGPU_HSA_METADATA_KIND_NONE = 0,
  AMDGPU_HSA_METADATA_KIND_INIT_SAMP = 1,
  AMDGPU_HSA_METADATA_KIND_INIT_ROIMG = 2,
  AMDGPU_HSA_METADATA_KIND_INIT_WOIMG = 3,
  AMDGPU_HSA_METADATA_KIND_INIT_RWIMG = 4
} amdgpu_hsa_metadata_kind_t;

// AMD GPU Sampler Coordinate Normalization Enumeration Values.
typedef uint8_t amdgpu_hsa_sampler_coord8_t;
typedef enum {
  AMDGPU_HSA_SAMPLER_COORD_UNNORMALIZED = 0,
  AMDGPU_HSA_SAMPLER_COORD_NORMALIZED = 1
} amdgpu_hsa_sampler_coord_t;

// AMD GPU Sampler Filter Enumeration Values.
typedef uint8_t amdgpu_hsa_sampler_filter8_t;
typedef enum {
  AMDGPU_HSA_SAMPLER_FILTER_NEAREST = 0,
  AMDGPU_HSA_SAMPLER_FILTER_LINEAR = 1
} amdgpu_hsa_sampler_filter_t;

// AMD GPU Sampler Addressing Enumeration Values.
typedef uint8_t amdgpu_hsa_sampler_addressing8_t;
typedef enum {
  AMDGPU_HSA_SAMPLER_ADDRESSING_UNDEFINED = 0,
  AMDGPU_HSA_SAMPLER_ADDRESSING_CLAMP_TO_EDGE = 1,
  AMDGPU_HSA_SAMPLER_ADDRESSING_CLAMP_TO_BORDER = 2,
  AMDGPU_HSA_SAMPLER_ADDRESSING_REPEAT = 3,
  AMDGPU_HSA_SAMPLER_ADDRESSING_MIRRORED_REPEAT = 4
} amdgpu_hsa_sampler_addressing_t;

// AMD GPU Sampler Descriptor.
typedef struct amdgpu_hsa_sampler_descriptor_s {
  uint16_t size;
  amdgpu_hsa_metadata_kind16_t kind;
  amdgpu_hsa_sampler_coord8_t coord;
  amdgpu_hsa_sampler_filter8_t filter;
  amdgpu_hsa_sampler_addressing8_t addressing;
  uint8_t reserved1;
} amdgpu_hsa_sampler_descriptor_t;

// AMD GPU Image Geometry Enumeration Values.
typedef uint8_t amdgpu_hsa_image_geometry8_t;
typedef enum {
  AMDGPU_HSA_IMAGE_GEOMETRY_1D = 0,
  AMDGPU_HSA_IMAGE_GEOMETRY_2D = 1,
  AMDGPU_HSA_IMAGE_GEOMETRY_3D = 2,
  AMDGPU_HSA_IMAGE_GEOMETRY_1DA = 3,
  AMDGPU_HSA_IMAGE_GEOMETRY_2DA = 4,
  AMDGPU_HSA_IMAGE_GEOMETRY_1DB = 5,
  AMDGPU_HSA_IMAGE_GEOMETRY_2DDEPTH = 6,
  AMDGPU_HSA_IMAGE_GEOMETRY_2DADEPTH = 7
} amdgpu_hsa_image_geometry_t;

// AMD GPU Image Channel Order Enumeration Values.
typedef uint8_t amdgpu_hsa_image_channel_order8_t;
typedef enum {
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_A = 0,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_R = 1,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_RX = 2,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_RG = 3,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_RGX = 4,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_RA = 5,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_RGB = 6,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_RGBX = 7,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_RGBA = 8,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_BGRA = 9,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_ARGB = 10,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_ABGR = 11,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_SRGB = 12,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_SRGBX = 13,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_SRGBA = 14,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_SBGRA = 15,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_INTENSITY = 16,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_LUMINANCE = 17,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_DEPTH = 18,
  AMDGPU_HSA_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL = 19
} amdgpu_hsa_image_channel_order_t;

// AMD GPU Image Channel Type Enumeration Values.
typedef uint8_t amdgpu_hsa_image_channel_type8_t;
typedef enum {
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_SNORM_INT8 = 0,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_SNORM_INT16 = 1,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_UNORM_INT8 = 2,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_UNORM_INT16 = 3,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_UNORM_INT24 = 4,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_SHORT_555 = 5,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_SHORT_565 = 6,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_INT_101010 = 7,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_SIGNED_INT8 = 8,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_SIGNED_INT16 = 9,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_SIGNED_INT32 = 10,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8 = 11,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16 = 12,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32 = 13,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_HALF_FLOAT = 14,
  AMDGPU_HSA_IMAGE_CHANNEL_TYPE_FLOAT = 15
} amdgpu_hsa_image_channel_type_t;

// AMD GPU Image Descriptor.
typedef struct amdgpu_hsa_image_descriptor_s {
  uint16_t size;
  amdgpu_hsa_metadata_kind16_t kind;
  amdgpu_hsa_image_geometry8_t geometry;
  amdgpu_hsa_image_channel_order8_t channel_order;
  amdgpu_hsa_image_channel_type8_t channel_type;
  uint8_t reserved1;
  uint64_t width;
  uint64_t height;
  uint64_t depth;
  uint64_t array;
} amdgpu_hsa_image_descriptor_t;

typedef struct amdgpu_hsa_note_code_object_version_s {
  uint32_t major_version;
  uint32_t minor_version;
} amdgpu_hsa_note_code_object_version_t;

typedef struct amdgpu_hsa_note_hsail_s {
  uint32_t hsail_major_version;
  uint32_t hsail_minor_version;
  uint8_t profile;
  uint8_t machine_model;
  uint8_t default_float_round;
} amdgpu_hsa_note_hsail_t;

typedef struct amdgpu_hsa_note_isa_s {
  uint16_t vendor_name_size;
  uint16_t architecture_name_size;
  uint32_t major;
  uint32_t minor;
  uint32_t stepping;
  char vendor_and_architecture_name[1];
} amdgpu_hsa_note_isa_t;

typedef struct amdgpu_hsa_note_producer_s {
  uint16_t producer_name_size;
  uint16_t reserved;
  uint32_t producer_major_version;
  uint32_t producer_minor_version;
  char producer_name[1];
} amdgpu_hsa_note_producer_t;

typedef struct amdgpu_hsa_note_producer_options_s {
  uint16_t producer_options_size;
  char producer_options[1];
} amdgpu_hsa_note_producer_options_t;

typedef enum {
  AMDGPU_HSA_RODATA_GLOBAL_PROGRAM = 0,
  AMDGPU_HSA_RODATA_GLOBAL_AGENT,
  AMDGPU_HSA_RODATA_READONLY_AGENT,
  AMDGPU_HSA_DATA_GLOBAL_PROGRAM,
  AMDGPU_HSA_DATA_GLOBAL_AGENT,
  AMDGPU_HSA_DATA_READONLY_AGENT,
  AMDGPU_HSA_BSS_GLOBAL_PROGRAM,
  AMDGPU_HSA_BSS_GLOBAL_AGENT,
  AMDGPU_HSA_BSS_READONLY_AGENT,
  AMDGPU_HSA_SECTION_LAST,
} amdgpu_hsa_elf_section_t;

#endif // AMD_HSA_ELF_H
