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

#ifndef HIP_INCLUDE_HIP_TEXTURE_TYPES_H
#define HIP_INCLUDE_HIP_TEXTURE_TYPES_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#pragma clang diagnostic ignored "-Wc++98-compat"
#endif

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#endif

#if !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "texture_types.h"
#elif defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
#if !defined(__HIPCC_RTC__)
#include <limits.h>
#include <hip/channel_descriptor.h>
#include <hip/driver_types.h>
#endif // !defined(__HIPCC_RTC__)

#define hipTextureType1D 0x01
#define hipTextureType2D 0x02
#define hipTextureType3D 0x03
#define hipTextureTypeCubemap 0x0C
#define hipTextureType1DLayered 0xF1
#define hipTextureType2DLayered 0xF2
#define hipTextureTypeCubemapLayered 0xFC

/**
 * Should be same as HSA_IMAGE_OBJECT_SIZE_DWORD/HSA_SAMPLER_OBJECT_SIZE_DWORD
 */
#define HIP_IMAGE_OBJECT_SIZE_DWORD 12
#define HIP_SAMPLER_OBJECT_SIZE_DWORD 8
#define HIP_SAMPLER_OBJECT_OFFSET_DWORD HIP_IMAGE_OBJECT_SIZE_DWORD
#define HIP_TEXTURE_OBJECT_SIZE_DWORD (HIP_IMAGE_OBJECT_SIZE_DWORD + HIP_SAMPLER_OBJECT_SIZE_DWORD)

/**
 * An opaque value that represents a hip texture object
 */
struct __hip_texture;
typedef struct __hip_texture* hipTextureObject_t;

/**
 * hip texture address modes
 */
enum hipTextureAddressMode {
    hipAddressModeWrap = 0,
    hipAddressModeClamp = 1,
    hipAddressModeMirror = 2,
    hipAddressModeBorder = 3
};

/**
 * hip texture filter modes
 */
enum hipTextureFilterMode { hipFilterModePoint = 0, hipFilterModeLinear = 1 };

/**
 * hip texture read modes
 */
enum hipTextureReadMode { hipReadModeElementType = 0, hipReadModeNormalizedFloat = 1 };

/**
 * hip texture reference
 */
typedef struct textureReference {
    int normalized;
    enum hipTextureReadMode readMode;// used only for driver API's
    enum hipTextureFilterMode filterMode;
    enum hipTextureAddressMode addressMode[3];  // Texture address mode for up to 3 dimensions
    struct hipChannelFormatDesc channelDesc;
    int sRGB;                    // Perform sRGB->linear conversion during texture read
    unsigned int maxAnisotropy;  // Limit to the anisotropy ratio
    enum hipTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;

    hipTextureObject_t textureObject;
    int numChannels;
    enum hipArray_Format format;
}textureReference;

/**
 * hip texture descriptor
 */
typedef struct hipTextureDesc {
    enum hipTextureAddressMode addressMode[3];  // Texture address mode for up to 3 dimensions
    enum hipTextureFilterMode filterMode;
    enum hipTextureReadMode readMode;
    int sRGB;  // Perform sRGB->linear conversion during texture read
    float borderColor[4];
    int normalizedCoords;
    unsigned int maxAnisotropy;
    enum hipTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
}hipTextureDesc;

#if __cplusplus

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
#if __HIP__
#define __HIP_TEXTURE_ATTRIB __attribute__((device_builtin_texture_type))
#else
#define __HIP_TEXTURE_ATTRIB
#endif

typedef textureReference* hipTexRef;

template <class T, int texType = hipTextureType1D,
          enum hipTextureReadMode mode = hipReadModeElementType>
struct __HIP_TEXTURE_ATTRIB texture : public textureReference {
    texture(int norm = 0, enum hipTextureFilterMode fMode = hipFilterModePoint,
            enum hipTextureAddressMode aMode = hipAddressModeClamp) {
        normalized = norm;
        readMode = mode;
        filterMode = fMode;
        addressMode[0] = aMode;
        addressMode[1] = aMode;
        addressMode[2] = aMode;
        channelDesc = hipCreateChannelDesc<T>();
        sRGB = 0;
        textureObject = nullptr;
        maxAnisotropy = 0;
        mipmapLevelBias = 0;
        minMipmapLevelClamp = 0;
        maxMipmapLevelClamp = 0;
    }

    texture(int norm, enum hipTextureFilterMode fMode, enum hipTextureAddressMode aMode,
            struct hipChannelFormatDesc desc) {
        normalized = norm;
        readMode = mode;
        filterMode = fMode;
        addressMode[0] = aMode;
        addressMode[1] = aMode;
        addressMode[2] = aMode;
        channelDesc = desc;
        sRGB = 0;
        textureObject = nullptr;
        maxAnisotropy = 0;
        mipmapLevelBias = 0;
        minMipmapLevelClamp = 0;
        maxMipmapLevelClamp = 0;
    }
};

#endif /* __cplusplus */

#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif
