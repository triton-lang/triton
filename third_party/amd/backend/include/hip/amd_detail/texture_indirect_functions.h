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

#pragma once

#if defined(__cplusplus)

#if !defined(__HIPCC_RTC__)
#include <hip/hip_vector_types.h>
#include <hip/hip_texture_types.h>
#include <hip/amd_detail/texture_fetch_functions.h>
#include <hip/amd_detail/ockl_image.h>
#include <type_traits>
#endif // !defined(__HIPCC_RTC__)

#define TEXTURE_OBJECT_PARAMETERS_INIT                                                            \
    unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)textureObject; \
    unsigned int ADDRESS_SPACE_CONSTANT* s = i + HIP_SAMPLER_OBJECT_OFFSET_DWORD;

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1Dfetch(hipTextureObject_t textureObject, int x)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_load_1Db(i, x);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1Dfetch(T *ptr, hipTextureObject_t textureObject, int x)
{
    *ptr = tex1Dfetch<T>(textureObject, x);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1D(hipTextureObject_t textureObject, float x)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_1D(i, s, x);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1D(T *ptr, hipTextureObject_t textureObject, float x)
{
    *ptr = tex1D<T>(textureObject, x);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2D(hipTextureObject_t textureObject, float x, float y)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_2D(i, s, float2(x, y).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2D(T *ptr, hipTextureObject_t textureObject, float x, float y)
{
    *ptr = tex2D<T>(textureObject, x, y);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex3D(hipTextureObject_t textureObject, float x, float y, float z)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_3D(i, s, float4(x, y, z, 0.0f).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex3D(T *ptr, hipTextureObject_t textureObject, float x, float y, float z)
{
    *ptr = tex3D<T>(textureObject, x, y, z);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLayered(hipTextureObject_t textureObject, float x, int layer)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLayered(T *ptr, hipTextureObject_t textureObject, float x, int layer)
{
    *ptr = tex1DLayered<T>(textureObject, x, layer);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DLayered(hipTextureObject_t textureObject, float x, float y, int layer)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLayered(T *ptr, hipTextureObject_t textureObject, float x, float y, int layer)
{
    *ptr = tex1DLayered<T>(textureObject, x, y, layer);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__  T texCubemap(hipTextureObject_t textureObject, float x, float y, float z)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_CM(i, s, float4(x, y, z, 0.0f).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemap(T *ptr, hipTextureObject_t textureObject, float x, float y, float z)
{
    *ptr = texCubemap<T>(textureObject, x, y, z);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapLayered(hipTextureObject_t textureObject, float x, float y, float z, int layer)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_CMa(i, s, float4(x, y, z, layer).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLayered(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, int layer)
{
    *ptr = texCubemapLayered<T>(textureObject, x, y, z, layer);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2Dgather(hipTextureObject_t textureObject, float x, float y, int comp = 0)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    switch (comp) {
    case 1: {
        auto tmp = __ockl_image_gather4r_2D(i, s, float2(x, y).data);
        return __hipMapFrom<T>(tmp);
        break;
    }
    case 2: {
        auto tmp = __ockl_image_gather4g_2D(i, s, float2(x, y).data);
        return __hipMapFrom<T>(tmp);
        break;
    }
    case 3: {
        auto tmp = __ockl_image_gather4b_2D(i, s, float2(x, y).data);
        return __hipMapFrom<T>(tmp);
        break;
    }
    default: {
        auto tmp = __ockl_image_gather4a_2D(i, s, float2(x, y).data);
        return __hipMapFrom<T>(tmp);
        break;
    }
    }
    return {};
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2Dgather(T *ptr, hipTextureObject_t textureObject, float x, float y, int comp = 0)
{
    *ptr = texCubemapLayered<T>(textureObject, x, y, comp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLod(hipTextureObject_t textureObject, float x, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_lod_1D(i, s, x, level);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLod(T *ptr, hipTextureObject_t textureObject, float x, float level)
{
    *ptr = tex1DLod<T>(textureObject, x, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DLod(hipTextureObject_t textureObject, float x, float y, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_lod_2D(i, s, float2(x, y).data, level);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLod(T *ptr, hipTextureObject_t textureObject, float x, float y, float level)
{
    *ptr = tex2DLod<T>(textureObject, x, y, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex3DLod(hipTextureObject_t textureObject, float x, float y, float z, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_lod_3D(i, s, float4(x, y, z, 0.0f).data, level);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex3DLod(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, float level)
{
    *ptr = tex3DLod<T>(textureObject, x, y, z, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLayeredLod(hipTextureObject_t textureObject, float x, int layer, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_1Da(i, s, float2(x, layer).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLayeredLod(T *ptr, hipTextureObject_t textureObject, float x, int layer, float level)
{
    *ptr = tex1DLayeredLod<T>(textureObject, x, layer, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__  T tex2DLayeredLod(hipTextureObject_t textureObject, float x, float y, int layer, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_2Da(i, s, float4(x, y, layer, 0.0f).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLayeredLod(T *ptr, hipTextureObject_t textureObject, float x, float y, int layer, float level)
{
    *ptr = tex2DLayeredLod<T>(textureObject, x, y, layer, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapLod(hipTextureObject_t textureObject, float x, float y, float z, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_lod_CM(i, s, float4(x, y, z, 0.0f).data, level);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLod(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, float level)
{
    *ptr = texCubemapLod<T>(textureObject, x, y, z, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapGrad(hipTextureObject_t textureObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    // TODO missing in device libs.
    // auto tmp = __ockl_image_sample_grad_CM(i, s, float4(x, y, z, 0.0f).data, float4(dPdx.x, dPdx.y, dPdx.z, 0.0f).data, float4(dPdy.x, dPdy.y, dPdy.z, 0.0f).data);
    // return __hipMapFrom<T>(tmp);
    return {};
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapGrad(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
    *ptr = texCubemapGrad<T>(textureObject, x, y, z, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapLayeredLod(hipTextureObject_t textureObject, float x, float y, float z, int layer, float level)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_lod_CMa(i, s, float4(x, y, z, layer).data, level);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLayeredLod(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, int layer, float level)
{
    *ptr = texCubemapLayeredLod<T>(textureObject, x, y, z, layer, level);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DGrad(hipTextureObject_t textureObject, float x, float dPdx, float dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_grad_1D(i, s, x, dPdx, dPdy);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DGrad(T *ptr, hipTextureObject_t textureObject, float x, float dPdx, float dPdy)
{
    *ptr = tex1DGrad<T>(textureObject, x, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DGrad(hipTextureObject_t textureObject, float x, float y, float2 dPdx, float2 dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_grad_2D(i, s, float2(x, y).data, float2(dPdx.x, dPdx.y).data,  float2(dPdy.x, dPdy.y).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DGrad(T *ptr, hipTextureObject_t textureObject, float x, float y, float2 dPdx, float2 dPdy)
{
    *ptr = tex2DGrad<T>(textureObject, x, y, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex3DGrad(hipTextureObject_t textureObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_grad_3D(i, s, float4(x, y, z, 0.0f).data, float4(dPdx.x, dPdx.y, dPdx.z, 0.0f).data, float4(dPdy.x, dPdy.y, dPdy.z, 0.0f).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex3DGrad(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, float4 dPdx, float4 dPdy)
{
    *ptr = tex3DGrad<T>(textureObject, x, y, z, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLayeredGrad(hipTextureObject_t textureObject, float x, int layer, float dPdx, float dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_grad_1Da(i, s, float2(x, layer).data, dPdx, dPdy);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLayeredGrad(T *ptr, hipTextureObject_t textureObject, float x, int layer, float dPdx, float dPdy)
{
    *ptr = tex1DLayeredGrad<T>(textureObject, x, layer, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DLayeredGrad(hipTextureObject_t textureObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    auto tmp = __ockl_image_sample_grad_2Da(i, s, float4(x, y, layer, 0.0f).data, float2(dPdx.x, dPdx.y).data, float2(dPdy.x, dPdy.y).data);
    return __hipMapFrom<T>(tmp);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLayeredGrad(T *ptr, hipTextureObject_t textureObject, float x, float y, int layer, float2 dPdx, float2 dPdy)
{
    *ptr = tex2DLayeredGrad<T>(textureObject, x, y, layer, dPdx, dPdy);
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__  T texCubemapLayeredGrad(hipTextureObject_t textureObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{
    TEXTURE_OBJECT_PARAMETERS_INIT
    // TODO missing in device libs.
    // auto tmp = __ockl_image_sample_grad_CMa(i, s, float4(x, y, z, layer).data, float4(dPdx.x, dPdx.y, dPdx.z, 0.0f).data, float4(dPdy.x, dPdy.y, dPdy.z, 0.0f).data);
    // return __hipMapFrom<T>(tmp);
    return {};
}

template <
    typename T,
    typename std::enable_if<__hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLayeredGrad(T *ptr, hipTextureObject_t textureObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{
    *ptr = texCubemapLayeredGrad<T>(textureObject, x, y, z, layer, dPdx, dPdy);
}

#endif
