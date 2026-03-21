/*
Copyright (c) 2018 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_SURFACE_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_SURFACE_FUNCTIONS_H

#if defined(__cplusplus)

#if !defined(__HIPCC_RTC__)
#include <hip/surface_types.h>
#include <hip/hip_vector_types.h>
#include <hip/amd_detail/texture_fetch_functions.h>
#include <hip/amd_detail/ockl_image.h>
#endif

#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__
#else
#define __HOST_DEVICE__ __host__ __device__
#endif

#define __HIP_SURFACE_OBJECT_PARAMETERS_INIT                                                       \
  unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)surfObj;

/**
 *  @defgroup SurfaceAPI Surface API
 *  @{
 */

// CUDA is using byte address, need map to pixel address for HIP
static __HOST_DEVICE__ __forceinline__ int __hipGetPixelAddr(int x, int format, int order) {
  /*
  * use below format index to generate format LUT
    typedef enum {
      HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8 = 0,
      HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16 = 1,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8 = 2,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16 = 3,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24 = 4,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555 = 5,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565 = 6,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010 = 7,
      HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8 = 8,
      HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16 = 9,
      HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32 = 10,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8 = 11,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16 = 12,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32 = 13,
      HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT = 14,
      HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT = 15
    } hsa_ext_image_channel_type_t;
  */
  static const int FormatLUT[] = {0, 1, 0, 1, 3, 1, 1, 1, 0, 1, 2, 0, 1, 2, 1, 2};
  x = FormatLUT[format] == 3 ? x / FormatLUT[format] : x >> FormatLUT[format];

  /*
  * use below order index to generate order LUT
    typedef enum {
      HSA_EXT_IMAGE_CHANNEL_ORDER_A = 0,
      HSA_EXT_IMAGE_CHANNEL_ORDER_R = 1,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RX = 2,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RG = 3,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGX = 4,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RA = 5,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGB = 6,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX = 7,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA = 8,
      HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA = 9,
      HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB = 10,
      HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR = 11,
      HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB = 12,
      HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX = 13,
      HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA = 14,
      HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA = 15,
      HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY = 16,
      HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE = 17,
      HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH = 18,
      HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL = 19
    } hsa_ext_image_channel_order_t;
  */
  static const int OrderLUT[] = {0, 0, 1, 1, 3, 1, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 0, 0, 0, 0};
  return x = OrderLUT[order] == 3 ? x / OrderLUT[order] : x >> OrderLUT[order];
}

/** \brief Reads the value at coordinate x from the one-dimensional surface.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The coordinate where the value will be read out.
 *  \param boundaryMode [in] The boundary mode is currently ignored.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1Dread(T* data, hipSurfaceObject_t surfObj, int x,
                                                  int boundaryMode = hipBoundaryModeZero) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT;
  (void)boundaryMode;
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
  auto tmp = __ockl_image_load_1D(i, x);
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the one-dimensional surface at coordinate x.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The coordinate where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1Dwrite(T data, hipSurfaceObject_t surfObj, int x) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_1D(i, x, tmp);
}


/** \brief Reads the value from the two-dimensional surface at coordinate x, y.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the value will be read out.
 *  \param y [in] The y coordinate where the value will be read out.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2Dread(T* data, hipSurfaceObject_t surfObj, int x,
                                                  int y) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __ockl_image_load_2D(i, get_native_vector(coords));
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the two-dimensional surface at coordinate
 *         x, y.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param y [in] The y coordinate where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2Dwrite(T data, hipSurfaceObject_t surfObj, int x,
                                                   int y) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_2D(i, get_native_vector(coords), tmp);
}

/** \brief Reads the value from the three-dimensional surface at coordinate
 *         x, y, z.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the value will be read out.
 *  \param y [in] The y coordinate where the value will be read out.
 *  \param z [in] The z coordinate where the value will be read out.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf3Dread(T* data, hipSurfaceObject_t surfObj, int x, int y,
                                                  int z) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_3D(i), __ockl_image_channel_order_3D(i));
  int4 coords{x, y, z, 0};
  auto tmp = __ockl_image_load_3D(i, get_native_vector(coords));
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the three-dimensional surface at coordinate
 *         x, y, z.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param y [in] The y coordinate where the data will be written.
 *  \param z [in] The z coordinate where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf3Dwrite(T data, hipSurfaceObject_t surfObj, int x, int y,
                                                   int z) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_3D(i), __ockl_image_channel_order_3D(i));
  int4 coords{x, y, z, 0};
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_3D(i, get_native_vector(coords), tmp);
}

/** \brief Reads the value from the one-dimensional layered surface at
 *         coordinate x and layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The coordinate where the value will be read out.
 *  \param layer [in] The layer index where the value will be read out.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1DLayeredread(T* data, hipSurfaceObject_t surfObj, int x,
                                                         int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
  auto tmp = __ockl_image_load_lod_1D(i, x, layer);
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the one-dimensional layered surface at
 *         coordinate x and layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param layer [in] The layer index where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1DLayeredwrite(T data, hipSurfaceObject_t surfObj, int x,
                                                          int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_lod_1D(i, x, layer, tmp);
}

/** \brief Reads the value from the two-dimensional layered surface at
 *         coordinate x, y and layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the value will be read out.
 *  \param y [in] The y coordinate where the value will be read out.
 *  \param layer [in] The layer index where the value will be read out.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2DLayeredread(T* data, hipSurfaceObject_t surfObj, int x,
                                                         int y, int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __ockl_image_load_lod_2D(i, get_native_vector(coords), layer);
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the two-dimensional layered surface at
 *         coordinate x, y and layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param y [in] The y coordinate where the data will be written.
 *  \param layer [in] The layer index where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2DLayeredwrite(T data, hipSurfaceObject_t surfObj, int x,
                                                          int y, int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_lod_2D(i, get_native_vector(coords), layer, tmp);
}

/** \brief Reads the value from the cubemap surface at coordinate x, y and
 *         face index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the value will be read out.
 *  \param y [in] The y coordinate where the value will be read out.
 *  \param face [in] The face index where the value will be read out.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapread(T* data, hipSurfaceObject_t surfObj, int x,
                                                       int y, int face) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __ockl_image_load_CM(i, get_native_vector(coords), face);
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the cubemap surface at coordinate x, y and
 *         face index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param y [in] The y coordinate where the data will be written.
 *  \param face [in] The face index where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapwrite(T data, hipSurfaceObject_t surfObj, int x,
                                                        int y, int face) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_CM(i, get_native_vector(coords), face, tmp);
}

/** \brief Reads the value from the layered cubemap surface at coordinate x, y
 *         and face, layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the value will be read out.
 *  \param y [in] The y coordinate where the value will be read out.
 *  \param face [in] The face index where the value will be read out.
 *  \param layer [in] The layer index where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapLayeredread(T* data, hipSurfaceObject_t surfObj,
                                                              int x, int y, int face, int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __ockl_image_load_lod_CM(i, get_native_vector(coords), face, layer);
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the layered cubemap surface at coordinate
 *         x, y and face, layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value to write to the surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param y [in] The y coordinate where the data will be written.
 *  \param face [in] The face index where the data will be written.
 *  \param layer [in] The layer index where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapLayeredwrite(T* data, hipSurfaceObject_t surfObj,
                                                               int x, int y, int face, int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_lod_CM(i, get_native_vector(coords), face, layer, tmp);
}

// Doxygen end group SurfaceAPI
/**
 * @}
 */

#endif

#endif
