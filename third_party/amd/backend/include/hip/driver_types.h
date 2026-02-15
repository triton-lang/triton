/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_INCLUDE_HIP_DRIVER_TYPES_H
#define HIP_INCLUDE_HIP_DRIVER_TYPES_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#if __cplusplus
#include <cstdlib>
#else
#include <stdlib.h>  // size_t
#endif
#endif

#if !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "driver_types.h"
#elif defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

/**
 *  @defgroup DriverTypes Driver Types
 *  @{
 *  This section describes the driver data types.
 *
 */

typedef void* hipDeviceptr_t;
/**
 * HIP channel format kinds
 */
typedef enum hipChannelFormatKind {
  hipChannelFormatKindSigned = 0,    ///< Signed channel format
  hipChannelFormatKindUnsigned = 1,  ///< Unsigned channel format
  hipChannelFormatKindFloat = 2,     ///< Float channel format
  hipChannelFormatKindNone = 3       ///< No channel format
} hipChannelFormatKind;
/**
 * HIP channel format descriptor
 */
typedef struct hipChannelFormatDesc {
  int x;
  int y;
  int z;
  int w;
  enum hipChannelFormatKind f;  ///< Channel format kind
} hipChannelFormatDesc;
/** @brief The hipTexRefSetArray function flags parameter override format value*/
#define HIP_TRSA_OVERRIDE_FORMAT 0x01
/** @brief The hipTexRefSetFlags function flags parameter read as integer value*/
#define HIP_TRSF_READ_AS_INTEGER 0x01
/** @brief The hipTexRefSetFlags function flags parameter normalized coordinate value*/
#define HIP_TRSF_NORMALIZED_COORDINATES 0x02
/** @brief The hipTexRefSetFlags function flags parameter srgb value*/
#define HIP_TRSF_SRGB 0x10

typedef struct hipArray* hipArray_t;
typedef const struct hipArray* hipArray_const_t;
/**
 * HIP array format
 */
typedef enum hipArray_Format {
  HIP_AD_FORMAT_UNSIGNED_INT8 = 0x01,   ///< Unsigned 8-bit array format
  HIP_AD_FORMAT_UNSIGNED_INT16 = 0x02,  ///< Unsigned 16-bit array format
  HIP_AD_FORMAT_UNSIGNED_INT32 = 0x03,  ///< Unsigned 32-bit array format
  HIP_AD_FORMAT_SIGNED_INT8 = 0x08,     ///< Signed 8-bit array format
  HIP_AD_FORMAT_SIGNED_INT16 = 0x09,    ///< Signed 16-bit array format
  HIP_AD_FORMAT_SIGNED_INT32 = 0x0a,    ///< Signed 32-bit array format
  HIP_AD_FORMAT_HALF = 0x10,            ///< Half array format
  HIP_AD_FORMAT_FLOAT = 0x20            ///< Float array format
} hipArray_Format;
/**
 * HIP array descriptor
 */
typedef struct HIP_ARRAY_DESCRIPTOR {
  size_t Width;                 ///< Width of the array
  size_t Height;                ///< Height of the array
  enum hipArray_Format Format;  ///< Format of the array
  unsigned int NumChannels;     ///< Number of channels of the array
} HIP_ARRAY_DESCRIPTOR;

/**
 * HIP 3D array descriptor
 */
typedef struct HIP_ARRAY3D_DESCRIPTOR {
  size_t Width;                 ///< Width of the array
  size_t Height;                ///< Height of the array
  size_t Depth;                 ///< Depth of the array
  enum hipArray_Format Format;  ///< Format of the array
  unsigned int NumChannels;     ///< Number of channels of the array
  unsigned int Flags;           ///< Flags of the array
} HIP_ARRAY3D_DESCRIPTOR;
#if !defined(__HIPCC_RTC__)
/**
 * HIP 2D memory copy parameters
 */
typedef struct hip_Memcpy2D {
  size_t srcXInBytes;           ///< Source width in bytes
  size_t srcY;                  ///< Source height
  hipMemoryType srcMemoryType;  ///< Source memory type
  const void* srcHost;          ///< Source pointer
  hipDeviceptr_t srcDevice;     ///< Source device
  hipArray_t srcArray;          ///< Source array
  size_t srcPitch;              ///< Source pitch
  size_t dstXInBytes;           ///< Destination width in bytes
  size_t dstY;                  ///< Destination height
  hipMemoryType dstMemoryType;  ///< Destination memory type
  void* dstHost;                ///< Destination pointer
  hipDeviceptr_t dstDevice;     ///< Destination device
  hipArray_t dstArray;          ///< Destination array
  size_t dstPitch;              ///< Destination pitch
  size_t WidthInBytes;          ///< Width in bytes of the 2D memory copy
  size_t Height;                ///< Height of the 2D memory copy
} hip_Memcpy2D;
#endif  // !defined(__HIPCC_RTC__)
/**
 * HIP mipmapped array
 */
typedef struct hipMipmappedArray {
  void* data;                        ///< Data pointer of the mipmapped array
  struct hipChannelFormatDesc desc;  ///< Description of the mipmapped array
  unsigned int type;                 ///< Type of the mipmapped array
  unsigned int width;                ///< Width of the mipmapped array
  unsigned int height;               ///< Height of the mipmapped array
  unsigned int depth;                ///< Depth of the mipmapped array
  unsigned int min_mipmap_level;     ///< Minimum level of the mipmapped array
  unsigned int max_mipmap_level;     ///< Maximum level of the mipmapped array
  unsigned int flags;                ///< Flags of the mipmapped array
  enum hipArray_Format format;       ///< Format of the mipmapped array
  unsigned int num_channels;         ///< Number of channels of the mipmapped array
} hipMipmappedArray;
/**
 * HIP mipmapped array pointer
 */
typedef struct hipMipmappedArray* hipMipmappedArray_t;
typedef hipMipmappedArray_t hipmipmappedArray;
typedef const struct hipMipmappedArray* hipMipmappedArray_const_t;
/**
 * HIP resource types
 */
typedef enum hipResourceType {
  hipResourceTypeArray = 0x00,           ///< Array resource
  hipResourceTypeMipmappedArray = 0x01,  ///< Mipmapped array resource
  hipResourceTypeLinear = 0x02,          ///< Linear resource
  hipResourceTypePitch2D = 0x03          ///< Pitch 2D resource
} hipResourceType;
typedef enum HIPresourcetype_enum {
  HIP_RESOURCE_TYPE_ARRAY = 0x00,            ///< Array resource
  HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,  ///< Mipmapped array resource
  HIP_RESOURCE_TYPE_LINEAR = 0x02,           ///< Linear resource
  HIP_RESOURCE_TYPE_PITCH2D = 0x03           ///< Pitch 2D resource
} HIPresourcetype,
    hipResourcetype;
/**
 * HIP texture address modes
 */
typedef enum HIPaddress_mode_enum {
  HIP_TR_ADDRESS_MODE_WRAP = 0,    ///< Wrap address mode
  HIP_TR_ADDRESS_MODE_CLAMP = 1,   ///< Clamp address mode
  HIP_TR_ADDRESS_MODE_MIRROR = 2,  ///< Mirror address mode
  HIP_TR_ADDRESS_MODE_BORDER = 3   ///< Border address mode
} HIPaddress_mode;
/**
 * HIP filter modes
 */
typedef enum HIPfilter_mode_enum {
  HIP_TR_FILTER_MODE_POINT = 0,  ///< Filter mode point
  HIP_TR_FILTER_MODE_LINEAR = 1  ///< Filter mode linear
} HIPfilter_mode;
/**
 * HIP texture descriptor
 */
typedef struct HIP_TEXTURE_DESC_st {
  HIPaddress_mode addressMode[3];   ///< Address modes
  HIPfilter_mode filterMode;        ///< Filter mode
  unsigned int flags;               ///< Flags
  unsigned int maxAnisotropy;       ///< Maximum anisotropy ratio
  HIPfilter_mode mipmapFilterMode;  ///< Mipmap filter mode
  float mipmapLevelBias;            ///< Mipmap level bias
  float minMipmapLevelClamp;        ///< Mipmap minimum level clamp
  float maxMipmapLevelClamp;        ///< Mipmap maximum level clamp
  float borderColor[4];             ///< Border Color
  int reserved[12];
} HIP_TEXTURE_DESC;
/**
 * HIP texture resource view formats
 */
typedef enum hipResourceViewFormat {
  hipResViewFormatNone = 0x00,  ///< No resource view format (use underlying resource format)
  hipResViewFormatUnsignedChar1 = 0x01,              ///< 1 channel, unsigned 8-bit integers
  hipResViewFormatUnsignedChar2 = 0x02,              ///< 2 channels, unsigned 8-bit integers
  hipResViewFormatUnsignedChar4 = 0x03,              ///< 4 channels, unsigned 8-bit integers
  hipResViewFormatSignedChar1 = 0x04,                ///< 1 channel, signed 8-bit integers
  hipResViewFormatSignedChar2 = 0x05,                ///< 2 channels, signed 8-bit integers
  hipResViewFormatSignedChar4 = 0x06,                ///< 4 channels, signed 8-bit integers
  hipResViewFormatUnsignedShort1 = 0x07,             ///< 1 channel, unsigned 16-bit integers
  hipResViewFormatUnsignedShort2 = 0x08,             ///< 2 channels, unsigned 16-bit integers
  hipResViewFormatUnsignedShort4 = 0x09,             ///< 4 channels, unsigned 16-bit integers
  hipResViewFormatSignedShort1 = 0x0a,               ///< 1 channel, signed 16-bit integers
  hipResViewFormatSignedShort2 = 0x0b,               ///< 2 channels, signed 16-bit integers
  hipResViewFormatSignedShort4 = 0x0c,               ///< 4 channels, signed 16-bit integers
  hipResViewFormatUnsignedInt1 = 0x0d,               ///< 1 channel, unsigned 32-bit integers
  hipResViewFormatUnsignedInt2 = 0x0e,               ///< 2 channels, unsigned 32-bit integers
  hipResViewFormatUnsignedInt4 = 0x0f,               ///< 4 channels, unsigned 32-bit integers
  hipResViewFormatSignedInt1 = 0x10,                 ///< 1 channel, signed 32-bit integers
  hipResViewFormatSignedInt2 = 0x11,                 ///< 2 channels, signed 32-bit integers
  hipResViewFormatSignedInt4 = 0x12,                 ///< 4 channels, signed 32-bit integers
  hipResViewFormatHalf1 = 0x13,                      ///< 1 channel, 16-bit floating point
  hipResViewFormatHalf2 = 0x14,                      ///< 2 channels, 16-bit floating point
  hipResViewFormatHalf4 = 0x15,                      ///< 4 channels, 16-bit floating point
  hipResViewFormatFloat1 = 0x16,                     ///< 1 channel, 32-bit floating point
  hipResViewFormatFloat2 = 0x17,                     ///< 2 channels, 32-bit floating point
  hipResViewFormatFloat4 = 0x18,                     ///< 4 channels, 32-bit floating point
  hipResViewFormatUnsignedBlockCompressed1 = 0x19,   ///< Block-compressed 1
  hipResViewFormatUnsignedBlockCompressed2 = 0x1a,   ///< Block-compressed 2
  hipResViewFormatUnsignedBlockCompressed3 = 0x1b,   ///< Block-compressed 3
  hipResViewFormatUnsignedBlockCompressed4 = 0x1c,   ///< Block-compressed 4 unsigned
  hipResViewFormatSignedBlockCompressed4 = 0x1d,     ///< Block-compressed 4 signed
  hipResViewFormatUnsignedBlockCompressed5 = 0x1e,   ///< Block-compressed 5 unsigned
  hipResViewFormatSignedBlockCompressed5 = 0x1f,     ///< Block-compressed 5 signed
  hipResViewFormatUnsignedBlockCompressed6H = 0x20,  ///< Block-compressed 6 unsigned half-float
  hipResViewFormatSignedBlockCompressed6H = 0x21,    ///< Block-compressed 6 signed half-float
  hipResViewFormatUnsignedBlockCompressed7 = 0x22    ///< Block-compressed 7
} hipResourceViewFormat;
/**
 * HIP texture resource view formats
 */
typedef enum HIPresourceViewFormat_enum {
  HIP_RES_VIEW_FORMAT_NONE = 0x00,  ///< No resource view format (use underlying resource format)
  HIP_RES_VIEW_FORMAT_UINT_1X8 = 0x01,       ///< 1 channel, unsigned 8-bit integers
  HIP_RES_VIEW_FORMAT_UINT_2X8 = 0x02,       ///< 2 channels, unsigned 8-bit integers
  HIP_RES_VIEW_FORMAT_UINT_4X8 = 0x03,       ///< 4 channels, unsigned 8-bit integers
  HIP_RES_VIEW_FORMAT_SINT_1X8 = 0x04,       ///< 1 channel, signed 8-bit integers
  HIP_RES_VIEW_FORMAT_SINT_2X8 = 0x05,       ///< 2 channels, signed 8-bit integers
  HIP_RES_VIEW_FORMAT_SINT_4X8 = 0x06,       ///< 4 channels, signed 8-bit integers
  HIP_RES_VIEW_FORMAT_UINT_1X16 = 0x07,      ///< 1 channel, unsigned 16-bit integers
  HIP_RES_VIEW_FORMAT_UINT_2X16 = 0x08,      ///< 2 channels, unsigned 16-bit integers
  HIP_RES_VIEW_FORMAT_UINT_4X16 = 0x09,      ///< 4 channels, unsigned 16-bit integers
  HIP_RES_VIEW_FORMAT_SINT_1X16 = 0x0a,      ///< 1 channel, signed 16-bit integers
  HIP_RES_VIEW_FORMAT_SINT_2X16 = 0x0b,      ///< 2 channels, signed 16-bit integers
  HIP_RES_VIEW_FORMAT_SINT_4X16 = 0x0c,      ///< 4 channels, signed 16-bit integers
  HIP_RES_VIEW_FORMAT_UINT_1X32 = 0x0d,      ///< 1 channel, unsigned 32-bit integers
  HIP_RES_VIEW_FORMAT_UINT_2X32 = 0x0e,      ///< 2 channels, unsigned 32-bit integers
  HIP_RES_VIEW_FORMAT_UINT_4X32 = 0x0f,      ///< 4 channels, unsigned 32-bit integers
  HIP_RES_VIEW_FORMAT_SINT_1X32 = 0x10,      ///< 1 channel, signed 32-bit integers
  HIP_RES_VIEW_FORMAT_SINT_2X32 = 0x11,      ///< 2 channels, signed 32-bit integers
  HIP_RES_VIEW_FORMAT_SINT_4X32 = 0x12,      ///< 4 channels, signed 32-bit integers
  HIP_RES_VIEW_FORMAT_FLOAT_1X16 = 0x13,     ///< 1 channel, 16-bit floating point
  HIP_RES_VIEW_FORMAT_FLOAT_2X16 = 0x14,     ///< 2 channels, 16-bit floating point
  HIP_RES_VIEW_FORMAT_FLOAT_4X16 = 0x15,     ///< 4 channels, 16-bit floating point
  HIP_RES_VIEW_FORMAT_FLOAT_1X32 = 0x16,     ///< 1 channel, 32-bit floating point
  HIP_RES_VIEW_FORMAT_FLOAT_2X32 = 0x17,     ///< 2 channels, 32-bit floating point
  HIP_RES_VIEW_FORMAT_FLOAT_4X32 = 0x18,     ///< 4 channels, 32-bit floating point
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = 0x19,   ///< Block-compressed 1
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = 0x1a,   ///< Block-compressed 2
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = 0x1b,   ///< Block-compressed 3
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = 0x1c,   ///< Block-compressed 4 unsigned
  HIP_RES_VIEW_FORMAT_SIGNED_BC4 = 0x1d,     ///< Block-compressed 4 signed
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = 0x1e,   ///< Block-compressed 5 unsigned
  HIP_RES_VIEW_FORMAT_SIGNED_BC5 = 0x1f,     ///< Block-compressed 5 signed
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20,  ///< Block-compressed 6 unsigned half-float
  HIP_RES_VIEW_FORMAT_SIGNED_BC6H = 0x21,    ///< Block-compressed 6 signed half-float
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = 0x22    ///< Block-compressed 7
} HIPresourceViewFormat;
/**
 * HIP resource descriptor
 */
typedef struct hipResourceDesc {
  enum hipResourceType resType;  ///< Resource type
  union {
    struct {
      hipArray_t array;  ///< HIP array
    } array;
    struct {
      hipMipmappedArray_t mipmap;  ///< HIP mipmapped array
    } mipmap;
    struct {
      void* devPtr;                      ///< Device pointer
      struct hipChannelFormatDesc desc;  ///< Channel format description
      size_t sizeInBytes;                ///< Size in bytes
    } linear;
    struct {
      void* devPtr;                      ///< Device pointer
      struct hipChannelFormatDesc desc;  ///< Channel format description
      size_t width;                      ///< Width of the array in elements
      size_t height;                     ///< Height of the array in elements
      size_t pitchInBytes;               ///< Pitch between two rows in bytes
    } pitch2D;
  } res;
} hipResourceDesc;

/**
 * HIP resource view descriptor struct
 */
typedef struct HIP_RESOURCE_DESC_st {
  HIPresourcetype resType;  ///< Resource type
  union {
    struct {
      hipArray_t hArray;  ///< HIP array
    } array;
    struct {
      hipMipmappedArray_t hMipmappedArray;  ///< HIP mipmapped array
    } mipmap;
    struct {
      hipDeviceptr_t devPtr;     ///< Device pointer
      hipArray_Format format;    ///< Array format
      unsigned int numChannels;  ///< Channels per array element
      size_t sizeInBytes;        ///< Size in bytes
    } linear;
    struct {
      hipDeviceptr_t devPtr;     ///< Device pointer
      hipArray_Format format;    ///< Array format
      unsigned int numChannels;  ///< Channels per array element
      size_t width;              ///< Width of the array in elements
      size_t height;             ///< Height of the array in elements
      size_t pitchInBytes;       ///< Pitch between two rows in bytes
    } pitch2D;
    struct {
      int reserved[32];
    } reserved;
  } res;
  unsigned int flags;  ///< Flags (must be zero)
} HIP_RESOURCE_DESC;
/**
 * HIP resource view descriptor
 */
struct hipResourceViewDesc {
  enum hipResourceViewFormat format;  ///< Resource view format
  size_t width;                       ///< Width of the resource view
  size_t height;                      ///< Height of the resource view
  size_t depth;                       ///< Depth of the resource view
  unsigned int firstMipmapLevel;      ///< First defined mipmap level
  unsigned int lastMipmapLevel;       ///< Last defined mipmap level
  unsigned int firstLayer;            ///< First layer index
  unsigned int lastLayer;             ///< Last layer index
};
/**
 * Resource view descriptor
 */
typedef struct HIP_RESOURCE_VIEW_DESC_st {
  HIPresourceViewFormat format;   ///< Resource view format
  size_t width;                   ///< Width of the resource view
  size_t height;                  ///< Height of the resource view
  size_t depth;                   ///< Depth of the resource view
  unsigned int firstMipmapLevel;  ///< First defined mipmap level
  unsigned int lastMipmapLevel;   ///< Last defined mipmap level
  unsigned int firstLayer;        ///< First layer index
  unsigned int lastLayer;         ///< Last layer index
  unsigned int reserved[16];
} HIP_RESOURCE_VIEW_DESC;
/**
 * Memory copy types
 */
#if !defined(__HIPCC_RTC__)
typedef enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,            ///< Host-to-Host Copy
  hipMemcpyHostToDevice = 1,          ///< Host-to-Device Copy
  hipMemcpyDeviceToHost = 2,          ///< Device-to-Host Copy
  hipMemcpyDeviceToDevice = 3,        ///< Device-to-Device Copy
  hipMemcpyDefault = 4,               ///< Runtime will automatically determine
                                      ///< copy-kind based on virtual addresses.
  hipMemcpyDeviceToDeviceNoCU = 1024  ///< Device-to-Device Copy without using compute units
} hipMemcpyKind;
/**
 * HIP pithed pointer
 */
typedef struct hipPitchedPtr {
  void* ptr;     ///< Pointer to the allocated memory
  size_t pitch;  ///< Pitch in bytes
  size_t xsize;  ///< Logical size of the first dimension of allocation in elements
  size_t ysize;  ///< Logical size of the second dimension of allocation in elements
} hipPitchedPtr;
/**
 * HIP extent
 */
typedef struct hipExtent {
  size_t width;  // Width in elements when referring to array memory, in bytes when referring to
                 // linear memory
  size_t height;
  size_t depth;
} hipExtent;
/**
 *  HIP position
 */
typedef struct hipPos {
  size_t x;  ///< X coordinate
  size_t y;  ///< Y coordinate
  size_t z;  ///< Z coordinate
} hipPos;
/**
 * HIP 3D memory copy parameters
 */
typedef struct hipMemcpy3DParms {
  hipArray_t srcArray;          ///< Source array
  struct hipPos srcPos;         ///< Source position
  struct hipPitchedPtr srcPtr;  ///< Source pointer
  hipArray_t dstArray;          ///< Destination array
  struct hipPos dstPos;         ///< Destination position
  struct hipPitchedPtr dstPtr;  ///< Destination pointer
  struct hipExtent extent;      ///< Extent of 3D memory copy
  enum hipMemcpyKind kind;      ///< Kind of 3D memory copy
} hipMemcpy3DParms;
/**
 * HIP 3D memory copy
 */
typedef struct HIP_MEMCPY3D {
  size_t srcXInBytes;           ///< Source X in bytes
  size_t srcY;                  ///< Source Y
  size_t srcZ;                  ///< Source Z
  size_t srcLOD;                ///< Source LOD
  hipMemoryType srcMemoryType;  ///< Source memory type
  const void* srcHost;          ///< Source host pointer
  hipDeviceptr_t srcDevice;     ///< Source device
  hipArray_t srcArray;          ///< Source array
  size_t srcPitch;              ///< Source pitch
  size_t srcHeight;             ///< Source height
  size_t dstXInBytes;           ///< Destination X in bytes
  size_t dstY;                  ///< Destination Y
  size_t dstZ;                  ///< Destination Z
  size_t dstLOD;                ///< Destination LOD
  hipMemoryType dstMemoryType;  ///< Destination memory type
  void* dstHost;                ///< Destination host pointer
  hipDeviceptr_t dstDevice;     ///< Destination device
  hipArray_t dstArray;          ///< Destination array
  size_t dstPitch;              ///< Destination pitch
  size_t dstHeight;             ///< Destination height
  size_t WidthInBytes;          ///< Width in bytes of 3D memory copy
  size_t Height;                ///< Height in bytes of 3D memory copy
  size_t Depth;                 ///< Depth in bytes of 3D memory copy
} HIP_MEMCPY3D;
/**
 * Specifies the type of location
 */
typedef enum hipMemLocationType {
  hipMemLocationTypeInvalid = 0,
  hipMemLocationTypeNone = 0,
  hipMemLocationTypeDevice = 1,    ///< Device location, thus it's HIP device ID
  hipMemLocationTypeHost = 2,      ///< Host location, id is ignored
  hipMemLocationTypeHostNuma = 3,  ///< Host NUMA node location, id is host NUMA node id
  hipMemLocationTypeHostNumaCurrent =
      4  ///< Host NUMA node closest to current thread’s CPU, id is ignored
} hipMemLocationType;
/**
 * Specifies a memory location.
 *
 * To specify a gpu, set type = @p hipMemLocationTypeDevice and set id = the gpu's device ID
 */
typedef struct hipMemLocation {
  hipMemLocationType type;  ///< Specifies the location type, which describes the meaning of id
  int id;                   ///< Identifier for the provided location type @p hipMemLocationType
} hipMemLocation;

/**
 * Flags to specify for copies within a batch. Used with hipMemcpyBatchAsync
 */
typedef enum hipMemcpyFlags {
  hipMemcpyFlagDefault = 0x0,                  ///< Default flag
  hipMemcpyFlagPreferOverlapWithCompute = 0x1  ///< Tries to overlap copy with compute work.
} hipMemcpyFlags;

/**
 * Flags to specify order in which source pointer is accessed by Batch memcpy
 */
typedef enum hipMemcpySrcAccessOrder {
  hipMemcpySrcAccessOrderInvalid = 0x0,  ///< Default Invalid.
  hipMemcpySrcAccessOrderStream = 0x1,   ///< Access to source pointer must be in stream order.
  hipMemcpySrcAccessOrderDuringApiCall =
      0x2,  ///< Access to source pointer can be out of stream order and all accesses must be
            ///< complete before API call returns.
  hipMemcpySrcAccessOrderAny =
      0x3,  ///< Access to the source pointer can be out of stream order and the accesses can happen
            ///< even after the API call return.
  hipMemcpySrcAccessOrderMax = 0x7FFFFFFF
} hipMemcpySrcAccessOrder;

/**
 * Attributes for copies within a batch.
 */
typedef struct hipMemcpyAttributes {
  hipMemcpySrcAccessOrder
      srcAccessOrder;  ///< Source access ordering to be observed for copies with this attribute.
  hipMemLocation srcLocHint;  ///< Location hint for src operand.
  hipMemLocation dstLocHint;  ///< Location hint for destination operand.
  unsigned int flags;         ///< Additional Flags for copies. See hipMemcpyFlags.
} hipMemcpyAttributes;
/**
 * Operand types for individual copies within a batch
 */
typedef enum hipMemcpy3DOperandType {
  hipMemcpyOperandTypePointer = 0x1,  ///< Mempcy operand is a valid pointer.
  hipMemcpyOperandTypeArray = 0x2,    ///< Memcpy operand is a valid hipArray.
  hipMemcpyOperandTypeMax = 0x7FFFFFFF
} hipMemcpy3DOperandType;

/**
 * Struct representing offset into a hipArray_t in elements.
 */
typedef struct hipOffset3D {
  size_t x;
  size_t y;
  size_t z;
} hipOffset3D;
/**
 *  Struct representing an operand for copy with hipMemcpy3DBatchAsync.
 */
typedef struct hipMemcpy3DOperand {
  hipMemcpy3DOperandType type;
  union {
    struct {
      void* ptr;
      size_t rowLength;        ///< Length of each row in elements.
      size_t layerHeight;      ///< Height of each layer in elements.
      hipMemLocation locHint;  ///< Location Hint for the operand.
    } ptr;
    struct {
      hipArray_t array;    ///< Array struct for hipMemcpyOperandTypeArray.
      hipOffset3D offset;  ///< Offset into array in elements.
    } array;
  } op;
} hipMemcpy3DOperand;

/**
 * HIP 3D Batch Op
 */
typedef struct hipMemcpy3DBatchOp {
  hipMemcpy3DOperand src;
  hipMemcpy3DOperand dst;
  hipExtent extent;
  hipMemcpySrcAccessOrder srcAccessOrder;
  unsigned int flags;
} hipMemcpy3DBatchOp;

typedef struct hipMemcpy3DPeerParms {
  hipArray_t srcArray;   ///< Source memory address
  hipPos srcPos;         ///< Source position offset
  hipPitchedPtr srcPtr;  ///< Pitched source memory address
  int srcDevice;         ///< Source device
  hipArray_t dstArray;   ///< Destination memory address
  hipPos dstPos;         ///< Destination position offset
  hipPitchedPtr dstPtr;  ///< Pitched destination memory address
  int dstDevice;         ///< Destination device
  hipExtent extent;      ///< Requested memory copy size
} hipMemcpy3DPeerParms;

/**
 * @brief Make hipPitchedPtr
 *
 * @param [in] d Pointer to the allocated memory
 * @param [in] p Pitch in bytes
 * @param [in] xsz Logical size of the first dimension of allocation in elements
 * @param [in] ysz Logical size of the second dimension of allocation in elements
 *
 * @returns The created hipPitchedPtr
 */
static inline struct hipPitchedPtr make_hipPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) {
  struct hipPitchedPtr s;
  s.ptr = d;
  s.pitch = p;
  s.xsize = xsz;
  s.ysize = ysz;
  return s;
}
/**
 * @brief Make hipPos struct
 *
 * @param [in] x X coordinate of the new hipPos
 * @param [in] y Y coordinate of the new hipPos
 * @param [in] z Z coordinate of the new hipPos
 *
 * @returns The created hipPos struct
 */
static inline struct hipPos make_hipPos(size_t x, size_t y, size_t z) {
  struct hipPos p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}
/**
 * @brief Make hipExtent struct
 *
 * @param [in] w Width of the new hipExtent
 * @param [in] h Height of the new hipExtent
 * @param [in] d Depth of the new hipExtent
 *
 * @returns The created hipExtent struct
 */
static inline struct hipExtent make_hipExtent(size_t w, size_t h, size_t d) {
  struct hipExtent e;
  e.width = w;
  e.height = h;
  e.depth = d;
  return e;
}
typedef enum hipFunction_attribute {
  HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,  ///< The maximum number of threads per block. Depends
                                             ///< on function and device.
  HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,  ///< The statically allocated shared memory size in bytes
                                         ///< per block required by the function.
  HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,   ///< The user-allocated constant memory by the function in
                                         ///< bytes.
  HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,   ///< The local memory usage of each thread by this function
                                         ///< in bytes.
  HIP_FUNC_ATTRIBUTE_NUM_REGS,  ///< The number of registers used by each thread of this function.
  HIP_FUNC_ATTRIBUTE_PTX_VERSION,                       ///< PTX version
  HIP_FUNC_ATTRIBUTE_BINARY_VERSION,                    ///< Binary version
  HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA,                     ///< Cache mode
  HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,     ///< The maximum dynamic shared memory per
                                                        ///< block for this function in bytes.
  HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,  ///< The shared memory carveout preference
                                                        ///< in percent of the maximum shared
                                                        ///< memory.
  HIP_FUNC_ATTRIBUTE_MAX
} hipFunction_attribute;

typedef enum hipPointer_attribute {
  HIP_POINTER_ATTRIBUTE_CONTEXT = 1,     ///< The context on which a pointer was allocated
                                         ///< @warning This attribute is not supported in HIP
  HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,     ///< memory type describing the location of a pointer
  HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,  ///< address at which the pointer is allocated on the
                                         ///< device
  HIP_POINTER_ATTRIBUTE_HOST_POINTER,    ///< address at which the pointer is allocated on the host
  HIP_POINTER_ATTRIBUTE_P2P_TOKENS,      ///< A pair of tokens for use with Linux kernel interface
                                         ///< @warning This attribute is not supported in HIP
  HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,     ///< Synchronize every synchronous memory operation
                                         ///< initiated on this region
  HIP_POINTER_ATTRIBUTE_BUFFER_ID,       ///< Unique ID for an allocated memory region
  HIP_POINTER_ATTRIBUTE_IS_MANAGED,      ///< Indicates if the pointer points to managed memory
  HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,  ///< device ordinal of a device on which a pointer
                                         ///< was allocated or registered
  HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE,  ///< if this pointer maps to an allocation
                                                    ///< that is suitable for hipIpcGetMemHandle
                                                    ///< @warning This attribute is not supported in
                                                    ///< HIP
  HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,           ///< Starting address for this requested pointer
  HIP_POINTER_ATTRIBUTE_RANGE_SIZE,  ///< Size of the address range for this requested pointer
  HIP_POINTER_ATTRIBUTE_MAPPED,      ///< tells if this pointer is in a valid address range
                                     ///< that is mapped to a backing allocation
  HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,  ///< Bitmask of allowed hipmemAllocationHandleType
                                               ///< for this allocation @warning This attribute is
                                               ///< not supported in HIP
  HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,  ///< returns if the memory referenced by
                                                     ///< this pointer can be used with the
                                                     ///< GPUDirect RDMA API
                                                     ///< @warning This attribute is not supported
                                                     ///< in HIP
  HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS,   ///< Returns the access flags the device associated with
                                        ///< for the corresponding memory referenced by the ptr
  HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE  ///< Returns the mempool handle for the allocation if
                                        ///< it was allocated from a mempool
                                        ///< @warning This attribute is not supported in HIP
} hipPointer_attribute;

// doxygen end DriverTypes
/**
 * @}
 */

#endif  // !defined(__HIPCC_RTC__)
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
#endif
