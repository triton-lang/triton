/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#endif

#if !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "driver_types.h"
#elif defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

#if !defined(__HIPCC_RTC__)
#ifndef __cplusplus
#include <stdbool.h>
#endif
#endif // !defined(__HIPCC_RTC__)
typedef void* hipDeviceptr_t;
typedef enum hipChannelFormatKind {
    hipChannelFormatKindSigned = 0,
    hipChannelFormatKindUnsigned = 1,
    hipChannelFormatKindFloat = 2,
    hipChannelFormatKindNone = 3
}hipChannelFormatKind;
typedef struct hipChannelFormatDesc {
    int x;
    int y;
    int z;
    int w;
    enum hipChannelFormatKind f;
}hipChannelFormatDesc;
#define HIP_TRSA_OVERRIDE_FORMAT 0x01
#define HIP_TRSF_READ_AS_INTEGER 0x01
#define HIP_TRSF_NORMALIZED_COORDINATES 0x02
#define HIP_TRSF_SRGB 0x10

typedef struct hipArray* hipArray_t;
typedef const struct hipArray* hipArray_const_t;
typedef enum hipArray_Format {
    HIP_AD_FORMAT_UNSIGNED_INT8 = 0x01,
    HIP_AD_FORMAT_UNSIGNED_INT16 = 0x02,
    HIP_AD_FORMAT_UNSIGNED_INT32 = 0x03,
    HIP_AD_FORMAT_SIGNED_INT8 = 0x08,
    HIP_AD_FORMAT_SIGNED_INT16 = 0x09,
    HIP_AD_FORMAT_SIGNED_INT32 = 0x0a,
    HIP_AD_FORMAT_HALF = 0x10,
    HIP_AD_FORMAT_FLOAT = 0x20
}hipArray_Format;
typedef struct HIP_ARRAY_DESCRIPTOR {
  size_t Width;
  size_t Height;
  enum hipArray_Format Format;
  unsigned int NumChannels;
}HIP_ARRAY_DESCRIPTOR;
typedef struct HIP_ARRAY3D_DESCRIPTOR {
  size_t Width;
  size_t Height;
  size_t Depth;
  enum hipArray_Format Format;
  unsigned int NumChannels;
  unsigned int Flags;
}HIP_ARRAY3D_DESCRIPTOR;
#if !defined(__HIPCC_RTC__)
typedef struct hip_Memcpy2D {
    size_t srcXInBytes;
    size_t srcY;
    hipMemoryType srcMemoryType;
    const void* srcHost;
    hipDeviceptr_t srcDevice;
    hipArray_t srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    hipMemoryType dstMemoryType;
    void* dstHost;
    hipDeviceptr_t dstDevice;
    hipArray_t dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
} hip_Memcpy2D;
#endif // !defined(__HIPCC_RTC__)
typedef struct hipMipmappedArray {
  void* data;
  struct hipChannelFormatDesc desc;
  unsigned int type;
  unsigned int width;
  unsigned int height;
  unsigned int depth;
  unsigned int min_mipmap_level;
  unsigned int max_mipmap_level;
  unsigned int flags;
  enum hipArray_Format format;
  unsigned int num_channels;
} hipMipmappedArray;
typedef struct hipMipmappedArray* hipMipmappedArray_t;
typedef hipMipmappedArray_t hipmipmappedArray;
typedef const struct hipMipmappedArray* hipMipmappedArray_const_t;
/**
 * hip resource types
 */
typedef enum hipResourceType {
    hipResourceTypeArray = 0x00,
    hipResourceTypeMipmappedArray = 0x01,
    hipResourceTypeLinear = 0x02,
    hipResourceTypePitch2D = 0x03
}hipResourceType;
typedef enum HIPresourcetype_enum {
    HIP_RESOURCE_TYPE_ARRAY           = 0x00, /**< Array resoure */
    HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01, /**< Mipmapped array resource */
    HIP_RESOURCE_TYPE_LINEAR          = 0x02, /**< Linear resource */
    HIP_RESOURCE_TYPE_PITCH2D         = 0x03  /**< Pitch 2D resource */
} HIPresourcetype, hipResourcetype;
/**
 * hip address modes
 */
typedef enum HIPaddress_mode_enum {
    HIP_TR_ADDRESS_MODE_WRAP   = 0,
    HIP_TR_ADDRESS_MODE_CLAMP  = 1,
    HIP_TR_ADDRESS_MODE_MIRROR = 2,
    HIP_TR_ADDRESS_MODE_BORDER = 3
} HIPaddress_mode;
/**
 * hip filter modes
 */
typedef enum HIPfilter_mode_enum {
    HIP_TR_FILTER_MODE_POINT  = 0,
    HIP_TR_FILTER_MODE_LINEAR = 1
} HIPfilter_mode;
/**
 * Texture descriptor
 */
typedef struct HIP_TEXTURE_DESC_st {
    HIPaddress_mode addressMode[3];  /**< Address modes */
    HIPfilter_mode filterMode;       /**< Filter mode */
    unsigned int flags;              /**< Flags */
    unsigned int maxAnisotropy;      /**< Maximum anisotropy ratio */
    HIPfilter_mode mipmapFilterMode; /**< Mipmap filter mode */
    float mipmapLevelBias;           /**< Mipmap level bias */
    float minMipmapLevelClamp;       /**< Mipmap minimum level clamp */
    float maxMipmapLevelClamp;       /**< Mipmap maximum level clamp */
    float borderColor[4];            /**< Border Color */
    int reserved[12];
} HIP_TEXTURE_DESC;
/**
 * hip texture resource view formats
 */
typedef enum hipResourceViewFormat {
    hipResViewFormatNone = 0x00,
    hipResViewFormatUnsignedChar1 = 0x01,
    hipResViewFormatUnsignedChar2 = 0x02,
    hipResViewFormatUnsignedChar4 = 0x03,
    hipResViewFormatSignedChar1 = 0x04,
    hipResViewFormatSignedChar2 = 0x05,
    hipResViewFormatSignedChar4 = 0x06,
    hipResViewFormatUnsignedShort1 = 0x07,
    hipResViewFormatUnsignedShort2 = 0x08,
    hipResViewFormatUnsignedShort4 = 0x09,
    hipResViewFormatSignedShort1 = 0x0a,
    hipResViewFormatSignedShort2 = 0x0b,
    hipResViewFormatSignedShort4 = 0x0c,
    hipResViewFormatUnsignedInt1 = 0x0d,
    hipResViewFormatUnsignedInt2 = 0x0e,
    hipResViewFormatUnsignedInt4 = 0x0f,
    hipResViewFormatSignedInt1 = 0x10,
    hipResViewFormatSignedInt2 = 0x11,
    hipResViewFormatSignedInt4 = 0x12,
    hipResViewFormatHalf1 = 0x13,
    hipResViewFormatHalf2 = 0x14,
    hipResViewFormatHalf4 = 0x15,
    hipResViewFormatFloat1 = 0x16,
    hipResViewFormatFloat2 = 0x17,
    hipResViewFormatFloat4 = 0x18,
    hipResViewFormatUnsignedBlockCompressed1 = 0x19,
    hipResViewFormatUnsignedBlockCompressed2 = 0x1a,
    hipResViewFormatUnsignedBlockCompressed3 = 0x1b,
    hipResViewFormatUnsignedBlockCompressed4 = 0x1c,
    hipResViewFormatSignedBlockCompressed4 = 0x1d,
    hipResViewFormatUnsignedBlockCompressed5 = 0x1e,
    hipResViewFormatSignedBlockCompressed5 = 0x1f,
    hipResViewFormatUnsignedBlockCompressed6H = 0x20,
    hipResViewFormatSignedBlockCompressed6H = 0x21,
    hipResViewFormatUnsignedBlockCompressed7 = 0x22
}hipResourceViewFormat;
typedef enum HIPresourceViewFormat_enum
{
    HIP_RES_VIEW_FORMAT_NONE          = 0x00, /**< No resource view format (use underlying resource format) */
    HIP_RES_VIEW_FORMAT_UINT_1X8      = 0x01, /**< 1 channel unsigned 8-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_2X8      = 0x02, /**< 2 channel unsigned 8-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_4X8      = 0x03, /**< 4 channel unsigned 8-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_1X8      = 0x04, /**< 1 channel signed 8-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_2X8      = 0x05, /**< 2 channel signed 8-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_4X8      = 0x06, /**< 4 channel signed 8-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_1X16     = 0x07, /**< 1 channel unsigned 16-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_2X16     = 0x08, /**< 2 channel unsigned 16-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_4X16     = 0x09, /**< 4 channel unsigned 16-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_1X16     = 0x0a, /**< 1 channel signed 16-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_2X16     = 0x0b, /**< 2 channel signed 16-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_4X16     = 0x0c, /**< 4 channel signed 16-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_1X32     = 0x0d, /**< 1 channel unsigned 32-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_2X32     = 0x0e, /**< 2 channel unsigned 32-bit integers */
    HIP_RES_VIEW_FORMAT_UINT_4X32     = 0x0f, /**< 4 channel unsigned 32-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_1X32     = 0x10, /**< 1 channel signed 32-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_2X32     = 0x11, /**< 2 channel signed 32-bit integers */
    HIP_RES_VIEW_FORMAT_SINT_4X32     = 0x12, /**< 4 channel signed 32-bit integers */
    HIP_RES_VIEW_FORMAT_FLOAT_1X16    = 0x13, /**< 1 channel 16-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_2X16    = 0x14, /**< 2 channel 16-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_4X16    = 0x15, /**< 4 channel 16-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_1X32    = 0x16, /**< 1 channel 32-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_2X32    = 0x17, /**< 2 channel 32-bit floating point */
    HIP_RES_VIEW_FORMAT_FLOAT_4X32    = 0x18, /**< 4 channel 32-bit floating point */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC1  = 0x19, /**< Block compressed 1 */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC2  = 0x1a, /**< Block compressed 2 */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC3  = 0x1b, /**< Block compressed 3 */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC4  = 0x1c, /**< Block compressed 4 unsigned */
    HIP_RES_VIEW_FORMAT_SIGNED_BC4    = 0x1d, /**< Block compressed 4 signed */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC5  = 0x1e, /**< Block compressed 5 unsigned */
    HIP_RES_VIEW_FORMAT_SIGNED_BC5    = 0x1f, /**< Block compressed 5 signed */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20, /**< Block compressed 6 unsigned half-float */
    HIP_RES_VIEW_FORMAT_SIGNED_BC6H   = 0x21, /**< Block compressed 6 signed half-float */
    HIP_RES_VIEW_FORMAT_UNSIGNED_BC7  = 0x22  /**< Block compressed 7 */
} HIPresourceViewFormat;
/**
 * HIP resource descriptor
 */
typedef struct hipResourceDesc {
    enum hipResourceType resType;
    union {
        struct {
            hipArray_t array;
        } array;
        struct {
            hipMipmappedArray_t mipmap;
        } mipmap;
        struct {
            void* devPtr;
            struct hipChannelFormatDesc desc;
            size_t sizeInBytes;
        } linear;
        struct {
            void* devPtr;
            struct hipChannelFormatDesc desc;
            size_t width;
            size_t height;
            size_t pitchInBytes;
        } pitch2D;
    } res;
}hipResourceDesc;
typedef struct HIP_RESOURCE_DESC_st
{
    HIPresourcetype resType;                     /**< Resource type */
    union {
        struct {
            hipArray_t hArray;                   /**< HIP array */
        } array;
        struct {
            hipMipmappedArray_t hMipmappedArray; /**< HIP mipmapped array */
        } mipmap;
        struct {
            hipDeviceptr_t devPtr;               /**< Device pointer */
            hipArray_Format format;              /**< Array format */
            unsigned int numChannels;            /**< Channels per array element */
            size_t sizeInBytes;                  /**< Size in bytes */
        } linear;
        struct {
            hipDeviceptr_t devPtr;               /**< Device pointer */
            hipArray_Format format;              /**< Array format */
            unsigned int numChannels;            /**< Channels per array element */
            size_t width;                        /**< Width of the array in elements */
            size_t height;                       /**< Height of the array in elements */
            size_t pitchInBytes;                 /**< Pitch between two rows in bytes */
        } pitch2D;
        struct {
            int reserved[32];
        } reserved;
    } res;
    unsigned int flags;                          /**< Flags (must be zero) */
} HIP_RESOURCE_DESC;
/**
 * hip resource view descriptor
 */
struct hipResourceViewDesc {
    enum hipResourceViewFormat format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
};
/**
 * Resource view descriptor
 */
typedef struct HIP_RESOURCE_VIEW_DESC_st
{
    HIPresourceViewFormat format;   /**< Resource view format */
    size_t width;                   /**< Width of the resource view */
    size_t height;                  /**< Height of the resource view */
    size_t depth;                   /**< Depth of the resource view */
    unsigned int firstMipmapLevel;  /**< First defined mipmap level */
    unsigned int lastMipmapLevel;   /**< Last defined mipmap level */
    unsigned int firstLayer;        /**< First layer index */
    unsigned int lastLayer;         /**< Last layer index */
    unsigned int reserved[16];
} HIP_RESOURCE_VIEW_DESC;
/**
 * Memory copy types
 *
 */
#if !defined(__HIPCC_RTC__)
typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,            ///< Host-to-Host Copy
    hipMemcpyHostToDevice = 1,          ///< Host-to-Device Copy
    hipMemcpyDeviceToHost = 2,          ///< Device-to-Host Copy
    hipMemcpyDeviceToDevice = 3,        ///< Device-to-Device Copy
    hipMemcpyDefault = 4,               ///< Runtime will automatically determine
                                        ///<copy-kind based on virtual addresses.
    hipMemcpyDeviceToDeviceNoCU = 1024  ///< Device-to-Device Copy without using compute units
} hipMemcpyKind;
typedef struct hipPitchedPtr {
    void* ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
}hipPitchedPtr;
typedef struct hipExtent {
    size_t width;  // Width in elements when referring to array memory, in bytes when referring to
                   // linear memory
    size_t height;
    size_t depth;
}hipExtent;
typedef struct hipPos {
    size_t x;
    size_t y;
    size_t z;
}hipPos;
typedef struct hipMemcpy3DParms {
    hipArray_t srcArray;
    struct hipPos srcPos;
    struct hipPitchedPtr srcPtr;
    hipArray_t dstArray;
    struct hipPos dstPos;
    struct hipPitchedPtr dstPtr;
    struct hipExtent extent;
    enum hipMemcpyKind kind;
} hipMemcpy3DParms;
typedef struct HIP_MEMCPY3D {
  size_t srcXInBytes;
  size_t srcY;
  size_t srcZ;
  size_t srcLOD;
  hipMemoryType srcMemoryType;
  const void* srcHost;
  hipDeviceptr_t srcDevice;
  hipArray_t srcArray;
  size_t srcPitch;
  size_t srcHeight;
  size_t dstXInBytes;
  size_t dstY;
  size_t dstZ;
  size_t dstLOD;
  hipMemoryType dstMemoryType;
  void* dstHost;
  hipDeviceptr_t dstDevice;
  hipArray_t dstArray;
  size_t dstPitch;
  size_t dstHeight;
  size_t WidthInBytes;
  size_t Height;
  size_t Depth;
} HIP_MEMCPY3D;
static inline struct hipPitchedPtr make_hipPitchedPtr(void* d, size_t p, size_t xsz,
                                                          size_t ysz) {
    struct hipPitchedPtr s;
    s.ptr = d;
    s.pitch = p;
    s.xsize = xsz;
    s.ysize = ysz;
    return s;
}
static inline struct hipPos make_hipPos(size_t x, size_t y, size_t z) {
    struct hipPos p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
}
static inline struct hipExtent make_hipExtent(size_t w, size_t h, size_t d) {
    struct hipExtent e;
    e.width = w;
    e.height = h;
    e.depth = d;
    return e;
}
typedef enum hipFunction_attribute {
    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_NUM_REGS,
    HIP_FUNC_ATTRIBUTE_PTX_VERSION,
    HIP_FUNC_ATTRIBUTE_BINARY_VERSION,
    HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA,
    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
    HIP_FUNC_ATTRIBUTE_MAX
} hipFunction_attribute;

typedef enum hipPointer_attribute {
    HIP_POINTER_ATTRIBUTE_CONTEXT = 1,   ///< The context on which a pointer was allocated
                                         ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,   ///< memory type describing location of a pointer
    HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,///< address at which the pointer is allocated on device
    HIP_POINTER_ATTRIBUTE_HOST_POINTER,  ///< address at which the pointer is allocated on host
    HIP_POINTER_ATTRIBUTE_P2P_TOKENS,    ///< A pair of tokens for use with linux kernel interface
                                         ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,   ///< Synchronize every synchronous memory operation
                                         ///< initiated on this region
    HIP_POINTER_ATTRIBUTE_BUFFER_ID,     ///< Unique ID for an allocated memory region
    HIP_POINTER_ATTRIBUTE_IS_MANAGED,    ///< Indicates if the pointer points to managed memory
    HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,///< device ordinal of a device on which a pointer
                                         ///< was allocated or registered
    HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE, ///< if this pointer maps to an allocation
                                                     ///< that is suitable for hipIpcGetMemHandle
                                                     ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,///< Starting address for this requested pointer
    HIP_POINTER_ATTRIBUTE_RANGE_SIZE,      ///< Size of the address range for this requested pointer
    HIP_POINTER_ATTRIBUTE_MAPPED,          ///< tells if this pointer is in a valid address range
                                           ///< that is mapped to a backing allocation
    HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,///< Bitmask of allowed hipmemAllocationHandleType
                                           ///< for this allocation @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE, ///< returns if the memory referenced by
                                           ///< this pointer can be used with the GPUDirect RDMA API
                                           ///< @warning - not supported in HIP
    HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS,    ///< Returns the access flags the device associated with
                                           ///< for the corresponding memory referenced by the ptr
    HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE   ///< Returns the mempool handle for the allocation if
                                           ///< it was allocated from a mempool
                                           ///< @warning - not supported in HIP
} hipPointer_attribute;

#endif // !defined(__HIPCC_RTC__)
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
#endif
