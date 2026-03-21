/*
Copyright (c) 2015 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip/amd_detail/ockl_image.h>
#include <type_traits>
#endif  // !defined(__HIPCC_RTC__)

#define TEXTURE_PARAMETERS_INIT                                                                    \
  unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)t.textureObject;  \
  unsigned int ADDRESS_SPACE_CONSTANT* s = i + HIP_SAMPLER_OBJECT_OFFSET_DWORD;                    \
  (void)s;

template <typename T> struct __hip_is_tex_surf_scalar_channel_type {
  static constexpr bool value =
      __hip_internal::is_same<T, char>::value || __hip_internal::is_same<T, unsigned char>::value ||
      __hip_internal::is_same<T, short>::value ||
      __hip_internal::is_same<T, unsigned short>::value || __hip_internal::is_same<T, int>::value ||
      __hip_internal::is_same<T, unsigned int>::value || __hip_internal::is_same<T, float>::value;
};

template <typename T> struct __hip_is_tex_surf_channel_type {
  static constexpr bool value = __hip_is_tex_surf_scalar_channel_type<T>::value;
};

template <typename T, unsigned int rank>
struct __hip_is_tex_surf_channel_type<HIP_vector_type<T, rank>> {
  static constexpr bool value = __hip_is_tex_surf_scalar_channel_type<T>::value &&
                                ((rank == 1) || (rank == 2) || (rank == 4));
};

template <typename T> struct __hip_is_tex_normalized_channel_type {
  static constexpr bool value =
      __hip_internal::is_same<T, char>::value || __hip_internal::is_same<T, unsigned char>::value ||
      __hip_internal::is_same<T, short>::value || __hip_internal::is_same<T, unsigned short>::value;
};

template <typename T, unsigned int rank>
struct __hip_is_tex_normalized_channel_type<HIP_vector_type<T, rank>> {
  static constexpr bool value =
      __hip_is_tex_normalized_channel_type<T>::value && ((rank == 1) || (rank == 2) || (rank == 4));
};

template <typename T, hipTextureReadMode readMode, typename Enable = void> struct __hip_tex_ret {
  static_assert(__hip_internal::is_same<Enable, void>::value, "Invalid channel type!");
};

/*
 * Map from device function return U to scalar texture type T
 */
template <typename T, typename U> __forceinline__ __device__
    typename __hip_internal::enable_if<__hip_is_tex_surf_scalar_channel_type<T>::value,
                                       const T>::type
    __hipMapFrom(const U& u) {
  if constexpr (sizeof(T) < sizeof(float)) {
    union {
      U u;
      int i;
    } d = {u};
    return static_cast<T>(d.i);
  } else {  // sizeof(T) == sizeof(float)
    union {
      U u;
      T t;
    } d = {u};
    return d.t;
  }
}

/*
 * Map from device function return U to vector texture type T
 */
template <typename T, typename U> __forceinline__ __device__ typename __hip_internal::enable_if<
    __hip_is_tex_surf_scalar_channel_type<typename T::value_type>::value, const T>::type
__hipMapFrom(const U& u) {
  if constexpr (sizeof(typename T::value_type) < sizeof(float)) {
    union {
      U u;
      int4 i4;
    } d = {u};
    return __hipMapVector<typename T::value_type, sizeof(T) / sizeof(typename T::value_type)>(d.i4);
  } else {  // sizeof(typename T::value_type) == sizeof(float)
    union {
      U u;
      T t;
    } d = {u};
    return d.t;
  }
}

/*
 * Map from scalar texture type T to device function input U
 */
template <typename U, typename T> __forceinline__ __device__
    typename __hip_internal::enable_if<__hip_is_tex_surf_scalar_channel_type<T>::value,
                                       const U>::type
    __hipMapTo(const T& t) {
  if constexpr (sizeof(T) < sizeof(float)) {
    union {
      U u;
      int i;
    } d = {0};
    d.i = static_cast<int>(t);
    return d.u;
  } else {  // sizeof(T) == sizeof(float)
    union {
      U u;
      T t;
    } d = {0};
    d.t = t;
    return d.u;
  }
}

/*
 * Map from vector texture type T to device function input U
 */
template <typename U, typename T> __forceinline__ __device__ typename __hip_internal::enable_if<
    __hip_is_tex_surf_scalar_channel_type<typename T::value_type>::value, const U>::type
__hipMapTo(const T& t) {
  if constexpr (sizeof(typename T::value_type) < sizeof(float)) {
    union {
      U u;
      int4 i4;
    } d = {0};
    d.i4 = __hipMapVector<int, 4>(t);
    return d.u;
  } else {  // sizeof(typename T::value_type) == sizeof(float)
    union {
      U u;
      T t;
    } d = {0};
    d.t = t;
    return d.u;
  }
}

template <typename T, hipTextureReadMode readMode> using __hip_tex_ret_t =
    typename __hip_tex_ret<T, readMode, bool>::type;

template <typename T> struct __hip_tex_ret<
    T, hipReadModeElementType,
    typename __hip_internal::enable_if<__hip_is_tex_surf_channel_type<T>::value, bool>::type> {
  using type = T;
};

template <typename T, unsigned int rank> struct __hip_tex_ret<
    HIP_vector_type<T, rank>, hipReadModeElementType,
    typename __hip_internal::enable_if<
        __hip_is_tex_surf_channel_type<HIP_vector_type<T, rank>>::value, bool>::type> {
  using type = HIP_vector_type<__hip_tex_ret_t<T, hipReadModeElementType>, rank>;
};

template <typename T>
struct __hip_tex_ret<T, hipReadModeNormalizedFloat,
                     typename __hip_internal::enable_if<
                         __hip_is_tex_normalized_channel_type<T>::value, bool>::type> {
  using type = float;
};

template <typename T, unsigned int rank> struct __hip_tex_ret<
    HIP_vector_type<T, rank>, hipReadModeNormalizedFloat,
    typename __hip_internal::enable_if<
        __hip_is_tex_normalized_channel_type<HIP_vector_type<T, rank>>::value, bool>::type> {
  using type = HIP_vector_type<__hip_tex_ret_t<T, hipReadModeNormalizedFloat>, rank>;
};


template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1Dfetch(
    texture<T, hipTextureType1D, readMode> t, int x) {
  TEXTURE_PARAMETERS_INIT;
  auto tmp = __ockl_image_load_1Db(i, x);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1D(
    texture<T, hipTextureType1D, readMode> t, float x) {
  TEXTURE_PARAMETERS_INIT;
  auto tmp = __ockl_image_sample_1D(i, s, x);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2D(
    texture<T, hipTextureType2D, readMode> t, float x, float y) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, y};
  auto tmp = __ockl_image_sample_2D(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLayered(
    texture<T, hipTextureType1DLayered, readMode> t, float x, int layer) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, layer};
  auto tmp = __ockl_image_sample_1Da(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLayered(
    texture<T, hipTextureType2DLayered, readMode> t, float x, float y, int layer) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, layer, 0.0f};
  auto tmp = __ockl_image_sample_2Da(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex3D(
    texture<T, hipTextureType3D, readMode> t, float x, float y, float z) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_3D(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemap(
    texture<T, hipTextureTypeCubemap, readMode> t, float x, float y, float z) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_CM(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLod(
    texture<T, hipTextureType1D, readMode> t, float x, float level) {
  TEXTURE_PARAMETERS_INIT;
  auto tmp = __ockl_image_sample_lod_1D(i, s, x, level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLod(
    texture<T, hipTextureType2D, readMode> t, float x, float y, float level) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, y};
  auto tmp = __ockl_image_sample_lod_2D(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLayeredLod(
    texture<T, hipTextureType1DLayered, readMode> t, float x, int layer, float level) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, layer};
  auto tmp = __ockl_image_sample_lod_1Da(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLayeredLod(
    texture<T, hipTextureType2DLayered, readMode> t, float x, float y, int layer, float level) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, layer, 0.0f};
  auto tmp = __ockl_image_sample_lod_2Da(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex3DLod(
    texture<T, hipTextureType3D, readMode> t, float x, float y, float z, float level) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_lod_3D(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapLod(
    texture<T, hipTextureTypeCubemap, readMode> t, float x, float y, float z, float level) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_lod_CM(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapLayered(
    texture<T, hipTextureTypeCubemapLayered, readMode> t, float x, float y, float z, int layer) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, layer};
  auto tmp = __ockl_image_sample_CMa(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapLayeredLod(
    texture<T, hipTextureTypeCubemapLayered, readMode> t, float x, float y, float z, int layer,
    float level) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, layer};
  auto tmp = __ockl_image_sample_lod_CMa(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapGrad(
    texture<T, hipTextureTypeCubemap, readMode> t, float x, float y, float z, float4 dPdx,
    float4 dPdy) {
  TEXTURE_PARAMETERS_INIT;
  (void)x;
  (void)y;
  (void)z;
  (void)dPdx;
  (void)dPdy;
  // TODO missing in device libs.
  // auto tmp = __ockl_image_sample_grad_CM(i, s, get_native_vector(float4(x, y, z, 0.0f)),
  // get_native_vector(float4(dPdx.x, dPdx.y, dPdx.z, 0.0f)), get_native_vector(float4(dPdy.x,
  // dPdy.y, dPdy.z, 0.0f))); return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
  return {};
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode>
texCubemapLayeredGrad(texture<T, hipTextureTypeCubemapLayered, readMode> t, float x, float y,
                      float z, int layer, float4 dPdx, float4 dPdy) {
  TEXTURE_PARAMETERS_INIT;
  (void)x;
  (void)y;
  (void)z;
  (void)layer;
  (void)dPdx;
  (void)dPdy;
  // TODO missing in device libs.
  // auto tmp = __ockl_image_sample_grad_CMa(i, s, get_native_vector(float4(x, y, z, layer)),
  // get_native_vector(float4(dPdx.x, dPdx.y, dPdx.z, 0.0f)), get_native_vector(float4(dPdy.x,
  // dPdy.y, dPdy.z, 0.0f))); return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
  return {};
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DGrad(
    texture<T, hipTextureType1D, readMode> t, float x, float dPdx, float dPdy) {
  TEXTURE_PARAMETERS_INIT;
  auto tmp = __ockl_image_sample_grad_1D(i, s, x, dPdx, dPdy);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DGrad(
    texture<T, hipTextureType2D, readMode> t, float x, float y, float2 dPdx, float2 dPdy) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, y};
  auto tmp = __ockl_image_sample_grad_2D(i, s, get_native_vector(coords), get_native_vector(dPdx),
                                         get_native_vector(dPdy));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLayeredGrad(
    texture<T, hipTextureType1DLayered, readMode> t, float x, int layer, float dPdx, float dPdy) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, layer};
  auto tmp = __ockl_image_sample_grad_1Da(i, s, get_native_vector(coords), dPdx, dPdy);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLayeredGrad(
    texture<T, hipTextureType2DLayered, readMode> t, float x, float y, int layer, float2 dPdx,
    float2 dPdy) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, layer, 0.0f};
  auto tmp = __ockl_image_sample_grad_2Da(i, s, get_native_vector(coords), get_native_vector(dPdx),
                                          get_native_vector(dPdy));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex3DGrad(
    texture<T, hipTextureType3D, readMode> t, float x, float y, float z, float4 dPdx, float4 dPdy) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, 0.0f};
  float4 gradx{dPdx.x, dPdx.y, dPdx.z, 0.0f};
  float4 grady{dPdy.x, dPdy.y, dPdy.z, 0.0f};
  auto tmp = __ockl_image_sample_grad_3D(i, s, get_native_vector(coords), get_native_vector(gradx),
                                         get_native_vector(grady));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode, typename Enable = void>
struct __hip_tex2dgather_ret {
  static_assert(__hip_internal::is_same<Enable, void>::value, "Invalid channel type!");
};

template <typename T, hipTextureReadMode readMode> using __hip_tex2dgather_ret_t =
    typename __hip_tex2dgather_ret<T, readMode, bool>::type;

template <typename T> struct __hip_tex2dgather_ret<
    T, hipReadModeElementType,
    typename __hip_internal::enable_if<__hip_is_tex_surf_channel_type<T>::value, bool>::type> {
  using type = HIP_vector_type<T, 4>;
};

template <typename T, unsigned int rank> struct __hip_tex2dgather_ret<
    HIP_vector_type<T, rank>, hipReadModeElementType,
    typename __hip_internal::enable_if<
        __hip_is_tex_surf_channel_type<HIP_vector_type<T, rank>>::value, bool>::type> {
  using type = HIP_vector_type<T, 4>;
};

template <typename T>
struct __hip_tex2dgather_ret<T, hipReadModeNormalizedFloat,
                             typename __hip_internal::enable_if<
                                 __hip_is_tex_normalized_channel_type<T>::value, bool>::type> {
  using type = float4;
};

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex2dgather_ret_t<T, readMode> tex2Dgather(
    texture<T, hipTextureType2D, readMode> t, float x, float y, int comp = 0) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, y};
  switch (comp) {
    case 1: {
      auto tmp = __ockl_image_gather4g_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
    case 2: {
      auto tmp = __ockl_image_gather4b_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
    case 3: {
      auto tmp = __ockl_image_gather4a_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
    default: {
      auto tmp = __ockl_image_gather4r_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
  }
  return {};
}

#endif
