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

#ifndef HSA_EXT_IMAGE_H
#define HSA_EXT_IMAGE_H

#include "hsa.h"

#undef HSA_API
#ifdef HSA_EXPORT_IMAGES
#define HSA_API HSA_API_EXPORT
#else
#define HSA_API HSA_API_IMPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/ 

/** \defgroup ext-images Images and Samplers
 *  @{
 */

/**
 * @brief Enumeration constants added to ::hsa_status_t by this extension.
 *
 * @remark Additions to hsa_status_t
 */
enum {
    /**
     * Image format is not supported.
     */
    HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED = 0x3000,
    /**
     * Image size is not supported.
     */
    HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED = 0x3001,
    /**
     * Image pitch is not supported or invalid.
     */
    HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED = 0x3002,
    /**
     * Sampler descriptor is not supported or invalid.
     */
    HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED = 0x3003
};

/**
 * @brief Enumeration constants added to ::hsa_agent_info_t by this
 * extension.
 *
 * @remark Additions to hsa_agent_info_t
 */
enum {
  /**
   * Maximum number of elements in 1D images. Must be at least 16384. The type
   * of this attribute is size_t.
   */
  HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS = 0x3000,
  /**
   * Maximum number of elements in 1DA images. Must be at least 16384. The type
   * of this attribute is size_t.
   */
  HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS = 0x3001,
  /**
   * Maximum number of elements in 1DB images. Must be at least 65536. The type
   * of this attribute is size_t.
   */
  HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS = 0x3002,
  /**
   * Maximum dimensions (width, height) of 2D images, in image elements. The X
   * and Y maximums must be at least 16384. The type of this attribute is
   * size_t[2].
   */
  HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS = 0x3003,
  /**
   * Maximum dimensions (width, height) of 2DA images, in image elements. The X
   * and Y maximums must be at least 16384. The type of this attribute is
   * size_t[2].
   */
  HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS = 0x3004,
  /**
   * Maximum dimensions (width, height) of 2DDEPTH images, in image
   * elements. The X and Y maximums must be at least 16384. The type of this
   * attribute is size_t[2].
   */
  HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS = 0x3005,
  /**
   * Maximum dimensions (width, height) of 2DADEPTH images, in image
   * elements. The X and Y maximums must be at least 16384. The type of this
   * attribute is size_t[2].
   */
  HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS = 0x3006,
  /**
   * Maximum dimensions (width, height, depth) of 3D images, in image
   * elements. The maximum along any dimension must be at least 2048. The type
   * of this attribute is size_t[3].
   */
  HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS = 0x3007,
  /**
   * Maximum number of image layers in a image array. Must be at least 2048. The
   * type of this attribute is size_t.
   */
  HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS = 0x3008,
  /**
   * Maximum number of read-only image handles that can be created for an agent at any one
   * time. Must be at least 128. The type of this attribute is size_t.
   */
  HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES = 0x3009,
  /**
   * Maximum number of write-only and read-write image handles (combined) that
   * can be created for an agent at any one time. Must be at least 64. The type of this
   * attribute is size_t.
   */
  HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES = 0x300A,
  /**
   * Maximum number of sampler handlers that can be created for an agent at any one
   * time. Must be at least 16. The type of this attribute is size_t.
   */
  HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS = 0x300B,
  /**
   * Image pitch alignment. The agent only supports linear image data
   * layouts with a row pitch that is a multiple of this value. Must be
   * a power of 2. The type of this attribute is size_t.
   */
  HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT = 0x300C
};

/**
 * @brief Image handle, populated by ::hsa_ext_image_create or
 * ::hsa_ext_image_create_with_layout. Image
 * handles are only unique within an agent, not across agents.
 *
 */
typedef struct hsa_ext_image_s {
  /**
   *  Opaque handle. For a given agent, two handles reference the same object of
   *  the enclosing type if and only if they are equal.
   */
    uint64_t handle;

} hsa_ext_image_t;

/**
 * @brief Geometry associated with the image. This specifies the
 * number of image dimensions and whether the image is an image
 * array. See the <em>Image Geometry</em> section in the <em>HSA
 * Programming Reference Manual</em> for definitions on each
 * geometry. The enumeration values match the BRIG type @p
 * hsa_ext_brig_image_geometry_t.
 */
typedef enum {
/**
   * One-dimensional image addressed by width coordinate.
   */
  HSA_EXT_IMAGE_GEOMETRY_1D = 0,

  /**
   * Two-dimensional image addressed by width and height coordinates.
   */
  HSA_EXT_IMAGE_GEOMETRY_2D = 1,

  /**
   * Three-dimensional image addressed by width, height, and depth coordinates.
   */
  HSA_EXT_IMAGE_GEOMETRY_3D = 2,

  /**
   * Array of one-dimensional images with the same size and format. 1D arrays
   * are addressed by width and index coordinate.
   */
  HSA_EXT_IMAGE_GEOMETRY_1DA = 3,

  /**
   * Array of two-dimensional images with the same size and format. 2D arrays
   * are addressed by width,  height, and index coordinates.
   */
  HSA_EXT_IMAGE_GEOMETRY_2DA = 4,

  /**
   * One-dimensional image addressed by width coordinate. It has
   * specific restrictions compared to ::HSA_EXT_IMAGE_GEOMETRY_1D. An
   * image with an opaque image data layout will always use a linear
   * image data layout, and one with an explicit image data layout
   * must specify ::HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR.
   */
  HSA_EXT_IMAGE_GEOMETRY_1DB = 5,

  /**
   * Two-dimensional depth image addressed by width and height coordinates.
   */
  HSA_EXT_IMAGE_GEOMETRY_2DDEPTH = 6,

  /**
   * Array of two-dimensional depth images with the same size and format. 2D
   * arrays are addressed by width, height, and index coordinates.
   */
  HSA_EXT_IMAGE_GEOMETRY_2DADEPTH = 7
} hsa_ext_image_geometry_t;

/**
 * @brief Channel type associated with the elements of an image. See
 * the <em>Channel Type</em> section in the <em>HSA Programming Reference
 * Manual</em> for definitions on each channel type. The
 * enumeration values and definition match the BRIG type @p
 * hsa_ext_brig_image_channel_type_t.
 */
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

/**
 * @brief A fixed-size type used to represent ::hsa_ext_image_channel_type_t constants.
 */
typedef uint32_t hsa_ext_image_channel_type32_t;
    
/**
 *
 * @brief Channel order associated with the elements of an image. See
 * the <em>Channel Order</em> section in the <em>HSA Programming Reference
 * Manual</em> for definitions on each channel order. The
 * enumeration values match the BRIG type @p
 * hsa_ext_brig_image_channel_order_t.
 */
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

/**
 * @brief A fixed-size type used to represent ::hsa_ext_image_channel_order_t constants.
 */
typedef uint32_t hsa_ext_image_channel_order32_t;
    

/**
 * @brief Image format.
 */
typedef struct hsa_ext_image_format_s {
  /**
    * Channel type.
    */
    hsa_ext_image_channel_type32_t channel_type;

   /**
    * Channel order.
    */
    hsa_ext_image_channel_order32_t channel_order;
} hsa_ext_image_format_t;

/**
 * @brief Implementation independent image descriptor.
 */
typedef struct hsa_ext_image_descriptor_s {
    /**
     * Image geometry.
     */
    hsa_ext_image_geometry_t geometry;
    /**
     * Width of the image, in components.
     */
    size_t width;
    /**
     * Height of the image, in components. Only used if the geometry is
     * ::HSA_EXT_IMAGE_GEOMETRY_2D, ::HSA_EXT_IMAGE_GEOMETRY_3D,
     * HSA_EXT_IMAGE_GEOMETRY_2DA, HSA_EXT_IMAGE_GEOMETRY_2DDEPTH, or
     * HSA_EXT_IMAGE_GEOMETRY_2DADEPTH, otherwise must be 0.
     */
    size_t height;
    /**
     * Depth of the image, in components. Only used if the geometry is
     * ::HSA_EXT_IMAGE_GEOMETRY_3D, otherwise must be 0.
     */
    size_t depth;
    /**
     * Number of image layers in the image array. Only used if the geometry is
     * ::HSA_EXT_IMAGE_GEOMETRY_1DA, ::HSA_EXT_IMAGE_GEOMETRY_2DA, or
     * HSA_EXT_IMAGE_GEOMETRY_2DADEPTH, otherwise must be 0.
     */
    size_t array_size;
    /**
     * Image format.
     */
    hsa_ext_image_format_t format;
} hsa_ext_image_descriptor_t;

/**
 * @brief Image capability.
 */
typedef enum  {
   /**
    * Images of this geometry, format, and layout are not supported by
    * the agent.
    */
    HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED = 0x0,
   /**
    * Read-only images of this geometry, format, and layout are
    * supported by the agent.
    */
    HSA_EXT_IMAGE_CAPABILITY_READ_ONLY = 0x1,
   /**
    * Write-only images of this geometry, format, and layout are
    * supported by the agent.
    */
    HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY = 0x2,
   /**
    * Read-write images of this geometry, format, and layout are
    * supported by the agent.
    */
    HSA_EXT_IMAGE_CAPABILITY_READ_WRITE = 0x4,
   /**
    * @deprecated Images of this geometry, format, and layout can be accessed from
    * read-modify-write atomic operations in the agent.
    */
    HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE = 0x8,
    /**
    * Images of this geometry, format, and layout are guaranteed to
    * have a consistent data layout regardless of how they are
    * accessed by the associated agent.
    */
    HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT = 0x10
} hsa_ext_image_capability_t;

/**
 * @brief Image data layout.
 *
 * @details An image data layout denotes such aspects of image data
 * layout as tiling and organization of channels in memory. Some image
 * data layouts may only apply to specific image geometries, formats,
 * and access permissions. Different agents may support different
 * image layout identifiers, including vendor specific layouts. Note
 * that an agent may not support the same image data layout for
 * different access permissions to images with the same image
 * geometry, size, and format. If multiple agents support the same
 * image data layout then it is possible to use separate image handles
 * for each agent that references the same image data.
 */

typedef enum  {
   /**
    * An implementation specific opaque image data layout which can
    * vary depending on the agent, geometry, image format, image size,
    * and access permissions.
    */
    HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE = 0x0,
   /**
    * The image data layout is specified by the following rules in
    * ascending byte address order. For a 3D image, 2DA image array,
    * or 1DA image array, the image data is stored as a linear sequence
    * of adjacent 2D image slices, 2D images, or 1D images
    * respectively, spaced according to the slice pitch. Each 2D image
    * is stored as a linear sequence of adjacent image rows, spaced
    * according to the row pitch. Each 1D or 1DB image is stored as a
    * single image row. Each image row is stored as a linear sequence
    * of image elements. Each image element is stored as a linear
    * sequence of image components specified by the left to right
    * channel order definition. Each image component is stored using
    * the memory type specified by the channel type.
    *
    * The 1DB image geometry always uses the linear image data layout.
    */
    HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR = 0x1
} hsa_ext_image_data_layout_t;

/**
 * @brief Retrieve the supported image capabilities for a given combination of
 * agent, geometry, and image format for an image created with an opaque image
 * data layout.
 *
 * @param[in] agent Agent to be associated with the image handle.
 *
 * @param[in] geometry Geometry.
 *
 * @param[in] image_format Pointer to an image format. Must not be NULL.
 *
 * @param[out] capability_mask Pointer to a memory location where the HSA
 * runtime stores a bit-mask of supported image capability
 * (::hsa_ext_image_capability_t) values. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p image_format is
 * NULL, or @p capability_mask is NULL.
 */
hsa_status_t HSA_API hsa_ext_image_get_capability(
    hsa_agent_t agent,
    hsa_ext_image_geometry_t geometry,
    const hsa_ext_image_format_t *image_format,
    uint32_t *capability_mask);

/**
 * @brief Retrieve the supported image capabilities for a given combination of
 * agent, geometry, image format, and image layout for an image created with
 * an explicit image data layout.
 *
 * @param[in] agent Agent to be associated with the image handle.
 *
 * @param[in] geometry Geometry.
 *
 * @param[in] image_format Pointer to an image format. Must not be NULL.
 *
 * @param[in] image_data_layout The image data layout.
 * It is invalid to use ::HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE; use
 * ::hsa_ext_image_get_capability instead.
 *
 * @param[out] capability_mask Pointer to a memory location where the HSA
 * runtime stores a bit-mask of supported image capability
 * (::hsa_ext_image_capability_t) values. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p image_format is
 * NULL, @p image_data_layout is ::HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE,
 * or @p capability_mask is NULL.
 */
hsa_status_t HSA_API hsa_ext_image_get_capability_with_layout(
    hsa_agent_t agent,
    hsa_ext_image_geometry_t geometry,
    const hsa_ext_image_format_t *image_format,
    hsa_ext_image_data_layout_t image_data_layout,
    uint32_t *capability_mask);

/**
 * @brief Agent specific image size and alignment requirements, populated by
 * ::hsa_ext_image_data_get_info and ::hsa_ext_image_data_get_info_with_layout.
 */
typedef struct hsa_ext_image_data_info_s {
  /**
   * Image data size, in bytes.
   */
  size_t size;

  /**
   * Image data alignment, in bytes. Must always be a power of 2.
   */
  size_t alignment;

} hsa_ext_image_data_info_t;

/**
 * @brief Retrieve the image data requirements for a given combination of agent, image
 * descriptor, and access permission for an image created with an opaque image
 * data layout.
 *
 * @details The optimal image data size and alignment requirements may
 * vary depending on the image attributes specified in @p
 * image_descriptor, the @p access_permission, and the @p agent. Also,
 * different implementations of the HSA runtime may return different
 * requirements for the same input values.
 *
 * The implementation must return the same image data requirements for
 * different access permissions with matching image descriptors as long
 * as ::hsa_ext_image_get_capability reports
 * ::HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT. Image
 * descriptors match if they have the same values, with the exception
 * that s-form channel orders match the corresponding non-s-form
 * channel order and vice versa.
 *
 * @param[in] agent Agent to be associated with the image handle.
 *
 * @param[in] image_descriptor Pointer to an image descriptor. Must not be NULL.
 *
 * @param[in] access_permission Access permission of the image when
 * accessed by @p agent. The access permission defines how the agent
 * is allowed to access the image and must match the corresponding
 * HSAIL image handle type. The @p agent must support the image format
 * specified in @p image_descriptor for the given @p
 * access_permission.
 *
 * @param[out] image_data_info Memory location where the runtime stores the
 * size and alignment requirements. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED The @p
 * agent does not support the image format specified by @p
 * image_descriptor with the specified @p access_permission.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED The agent
 * does not support the image dimensions specified by @p
 * image_descriptor with the specified @p access_permission.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p image_descriptor is NULL, @p
 * access_permission is not a valid access permission value, or @p
 * image_data_info is NULL.
 */
hsa_status_t HSA_API hsa_ext_image_data_get_info(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t *image_descriptor,
    hsa_access_permission_t access_permission,
    hsa_ext_image_data_info_t *image_data_info);

/**
 * @brief Retrieve the image data requirements for a given combination of
 * image descriptor, access permission, image data layout, image data row pitch,
 * and image data slice pitch for an image created with an explicit image
 * data layout.
 *
 * @details The image data size and alignment requirements may vary
 * depending on the image attributes specified in @p image_descriptor,
 * the @p access_permission, and the image layout. However, different
 * implementations of the HSA runtime will return the same
 * requirements for the same input values.
 *
 * The implementation must return the same image data requirements for
 * different access permissions with matching image descriptors and
 * matching image layouts as long as ::hsa_ext_image_get_capability
 * reports
 * ::HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT. Image
 * descriptors match if they have the same values, with the exception
 * that s-form channel orders match the corresponding non-s-form
 * channel order and vice versa. Image layouts match if they are the
 * same image data layout and use the same image row and slice pitch
 * values.
 *
 * @param[in] image_descriptor Pointer to an image descriptor. Must not be NULL.
 *
 * @param[in] access_permission Access permission of the image when
 * accessed by an agent. The access permission defines how the agent
 * is allowed to access the image and must match the corresponding
 * HSAIL image handle type.
 *
 * @param[in] image_data_layout The image data layout to use.
 * It is invalid to use ::HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE; use
 * ::hsa_ext_image_data_get_info instead.
 *
 * @param[in] image_data_row_pitch The size in bytes for a single row
 * of the image in the image data. If 0 is specified then the default
 * row pitch value is used: image width * image element byte size.
 * The value used must be greater than or equal to the default row
 * pitch, and be a multiple of the image element byte size. For the
 * linear image layout it must also be a multiple of the image linear
 * row pitch alignment for the agents that will access the image data
 * using image instructions.
 *
 * @param[in] image_data_slice_pitch The size in bytes of a single
 * slice of a 3D image, or the size in bytes of each image layer in an
 * image array in the image data. If 0 is specified then the default
 * slice pitch value is used: row pitch * height if geometry is
 * ::HSA_EXT_IMAGE_GEOMETRY_3D, ::HSA_EXT_IMAGE_GEOMETRY_2DA, or
 * ::HSA_EXT_IMAGE_GEOMETRY_2DADEPTH; row pitch if geometry is
 * ::HSA_EXT_IMAGE_GEOMETRY_1DA; and 0 otherwise. The value used must
 * be 0 if the default slice pitch is 0, be greater than or equal to
 * the default slice pitch, and be a multiple of the row pitch.
 *
 * @param[out] image_data_info Memory location where the runtime stores the
 * size and alignment requirements. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED The image
 * format specified by @p image_descriptor is not supported for the
 * @p access_permission and @p image_data_layout specified.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED The image
 * dimensions specified by @p image_descriptor are not supported for
 * the @p access_permission and @p image_data_layout specified.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED The row and
 * slice pitch specified by @p image_data_row_pitch and @p
 * image_data_slice_pitch are invalid or not supported.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p image_descriptor is
 * NULL, @p image_data_layout is ::HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE,
 * or @p image_data_info is NULL.
 */
hsa_status_t HSA_API hsa_ext_image_data_get_info_with_layout(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t *image_descriptor,
    hsa_access_permission_t access_permission,
    hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    hsa_ext_image_data_info_t *image_data_info);

/**
 * @brief Creates an agent specific image handle to an image with an
 * opaque image data layout.
 *
 * @details Images with an opaque image data layout created with
 * different access permissions but matching image descriptors and
 * same agent can share the same image data if
 * ::HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT is reported
 * by ::hsa_ext_image_get_capability for the image format specified in
 * the image descriptor. Image descriptors match if they have the same
 * values, with the exception that s-form channel orders match the
 * corresponding non-s-form channel order and vice versa.
 *
 * If necessary, an application can use image operations (import,
 * export, copy, clear) to prepare the image for the intended use
 * regardless of the access permissions.
 *
 * @param[in] agent agent to be associated with the image handle created.
 *
 * @param[in] image_descriptor Pointer to an image descriptor. Must not be NULL.
 *
 * @param[in] image_data Image data buffer that must have been allocated
 * according to the size and alignment requirements dictated by
 * ::hsa_ext_image_data_get_info. Must not be NULL.
 *
 * Any previous memory contents are preserved upon creation. The application is
 * responsible for ensuring that the lifetime of the image data exceeds that of
 * all the associated images.
 *
 * @param[in] access_permission Access permission of the image when
 * accessed by agent. The access permission defines how the agent
 * is allowed to access the image using the image handle created and
 * must match the corresponding HSAIL image handle type. The agent
 * must support the image format specified in @p image_descriptor for
 * the given @p access_permission.
 *
 * @param[out] image Pointer to a memory location where the HSA runtime stores
 * the newly created image handle. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED The agent
 * does not have the capability to support the image format contained
 * in @p image_descriptor using the specified @p access_permission.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED The agent
 * does not support the image dimensions specified by @p
 * image_descriptor using the specified @p access_permission.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * support the creation of more image handles with the given @p access_permission).
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p image_descriptor is NULL, @p
 * image_data is NULL, @p image_data does not have a valid alignment,
 * @p access_permission is not a valid access permission
 * value, or @p image is NULL.
 */
hsa_status_t HSA_API hsa_ext_image_create(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t *image_descriptor,
    const void *image_data,
    hsa_access_permission_t access_permission,
    hsa_ext_image_t *image);

/**
 * @brief Creates an agent specific image handle to an image with an explicit
 * image data layout.
 *
 * @details Images with an explicit image data layout created with
 * different access permissions but matching image descriptors and
 * matching image layout can share the same image data if
 * ::HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT is reported
 * by ::hsa_ext_image_get_capability_with_layout for the image format
 * specified in the image descriptor and specified image data
 * layout. Image descriptors match if they have the same values, with
 * the exception that s-form channel orders match the corresponding
 * non-s-form channel order and vice versa. Image layouts match if
 * they are the same image data layout and use the same image row and
 * slice values.
 *
 * If necessary, an application can use image operations (import, export, copy,
 * clear) to prepare the image for the intended use regardless of the access
 * permissions.
 *
 * @param[in] agent agent to be associated with the image handle created.
 *
 * @param[in] image_descriptor Pointer to an image descriptor. Must not be NULL.
 *
 * @param[in] image_data Image data buffer that must have been allocated
 * according to the size and alignment requirements dictated by
 * ::hsa_ext_image_data_get_info_with_layout. Must not be NULL.
 *
 * Any previous memory contents are preserved upon creation. The application is
 * responsible for ensuring that the lifetime of the image data exceeds that of
 * all the associated images.
 *
 * @param[in] access_permission Access permission of the image when
 * accessed by the agent. The access permission defines how the agent
 * is allowed to access the image and must match the corresponding
 * HSAIL image handle type. The agent must support the image format
 * specified in @p image_descriptor for the given @p access_permission
 * and @p image_data_layout.
 *
 * @param[in] image_data_layout The image data layout to use for the
 * @p image_data. It is invalid to use
 * ::HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE; use ::hsa_ext_image_create
 * instead.
 *
 * @param[in] image_data_row_pitch The size in bytes for a single row
 * of the image in the image data. If 0 is specified then the default
 * row pitch value is used: image width * image element byte size.
 * The value used must be greater than or equal to the default row
 * pitch, and be a multiple of the image element byte size. For the
 * linear image layout it must also be a multiple of the image linear
 * row pitch alignment for the agents that will access the image data
 * using image instructions.
 *
 * @param[in] image_data_slice_pitch The size in bytes of a single
 * slice of a 3D image, or the size in bytes of each image layer in an
 * image array in the image data. If 0 is specified then the default
 * slice pitch value is used: row pitch * height if geometry is
 * ::HSA_EXT_IMAGE_GEOMETRY_3D, ::HSA_EXT_IMAGE_GEOMETRY_2DA, or
 * ::HSA_EXT_IMAGE_GEOMETRY_2DADEPTH; row pitch if geometry is
 * ::HSA_EXT_IMAGE_GEOMETRY_1DA; and 0 otherwise. The value used must
 * be 0 if the default slice pitch is 0, be greater than or equal to
 * the default slice pitch, and be a multiple of the row pitch.
 *
 * @param[out] image Pointer to a memory location where the HSA runtime stores
 * the newly created image handle. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED The agent does
 * not have the capability to support the image format contained in the image
 * descriptor using the specified @p access_permission and @p image_data_layout.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED The agent
 * does not support the image dimensions specified by @p
 * image_descriptor using the specified @p access_permission and @p
 * image_data_layout.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED The agent does
 * not support the row and slice pitch specified by @p image_data_row_pitch
 * and @p image_data_slice_pitch, or the values are invalid.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * support the creation of more image handles with the given @p access_permission).
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p image_descriptor is NULL, @p
 * image_data is NULL, @p image_data does not have a valid alignment,
 * @p image_data_layout is ::HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE,
 * or @p image is NULL.
 */
hsa_status_t HSA_API hsa_ext_image_create_with_layout(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t *image_descriptor,
    const void *image_data,
    hsa_access_permission_t access_permission,
    hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    hsa_ext_image_t *image);

/**
 * @brief Destroy an image handle previously created using ::hsa_ext_image_create or
 * ::hsa_ext_image_create_with_layout.
 *
 * @details Destroying the image handle does not free the associated image data,
 * or modify its contents. The application should not destroy an image handle while
 * there are references to it queued for execution or currently being used in a
 * kernel dispatch.
 *
 * @param[in] agent Agent associated with the image handle.
 *
 * @param[in] image Image handle to destroy.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 */
hsa_status_t HSA_API hsa_ext_image_destroy(
    hsa_agent_t agent,
    hsa_ext_image_t image);

/**
 * @brief Copies a portion of one image (the source) to another image (the
 * destination).
 *
 * @details The source and destination image formats should be the
 * same, with the exception that s-form channel orders match the
 * corresponding non-s-form channel order and vice versa. For example,
 * it is allowed to copy a source image with a channel order of
 * HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB to a destination image with a
 * channel order of HSA_EXT_IMAGE_CHANNEL_ORDER_RGB.
 *
 * The source and destination images do not have to be of the same geometry and
 * appropriate scaling is performed by the HSA runtime. It is possible to copy
 * subregions between any combinations of source and destination geometries, provided
 * that the dimensions of the subregions are the same. For example, it is
 * allowed to copy a rectangular region from a 2D image to a slice of a 3D
 * image.
 *
 * If the source and destination image data overlap, or the combination of
 * offset and range references an out-out-bounds element in any of the images,
 * the behavior is undefined.
 *
 * @param[in] agent Agent associated with both the source and destination image handles.
 *
 * @param[in] src_image Image handle of source image. The agent associated with the source
 * image handle must be identical to that of the destination image.
 *
 * @param[in] src_offset Pointer to the offset within the source image where to
 * copy the data from. Must not be NULL.
 *
 * @param[in] dst_image Image handle of destination image.
 *
 * @param[in] dst_offset Pointer to the offset within the destination
 * image where to copy the data. Must not be NULL.
 *
 * @param[in] range Dimensions of the image portion to be copied. The HSA
 * runtime computes the size of the image data to be copied using this
 * argument. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p src_offset is
 * NULL, @p dst_offset is NULL, or @p range is NULL.
 */
hsa_status_t HSA_API hsa_ext_image_copy(
    hsa_agent_t agent,
    hsa_ext_image_t src_image,
    const hsa_dim3_t* src_offset,
    hsa_ext_image_t dst_image,
    const hsa_dim3_t* dst_offset,
    const hsa_dim3_t* range);

/**
 * @brief Image region.
 */
typedef struct hsa_ext_image_region_s {
   /**
    * Offset within an image (in coordinates).
    */
    hsa_dim3_t offset;

   /**
    * Dimension size of the image range (in coordinates). The x, y, and z dimensions
    * correspond to width, height, and depth or index respectively.
    */
    hsa_dim3_t range;
} hsa_ext_image_region_t;

/**
 * @brief Import a linearly organized image data from memory directly to an
 * image handle.
 *
 * @details This operation updates the image data referenced by the image handle
 * from the source memory. The size of the data imported from memory is
 * implicitly derived from the image region.
 *
 * It is the application's responsibility to avoid out of bounds memory access.
 *
 * None of the source memory or destination image data memory can
 * overlap. Overlapping of any of the source and destination image
 * data memory within the import operation produces undefined results.
 *
 * @param[in] agent Agent associated with the image handle.
 *
 * @param[in] src_memory Source memory. Must not be NULL.
 *
 * @param[in] src_row_pitch The size in bytes of a single row of the image in the
 * source memory. If the value is smaller than the destination image region
 * width * image element byte size, then region width * image element byte
 * size is used.
 *
 * @param[in] src_slice_pitch The size in bytes of a single 2D slice of a 3D image,
 * or the size in bytes of each image layer in an image array in the source memory.
 * If the geometry is ::HSA_EXT_IMAGE_GEOMETRY_1DA and the value is smaller than the
 * value used for @p src_row_pitch, then the value used for @p src_row_pitch is used.
 * If the geometry is ::HSA_EXT_IMAGE_GEOMETRY_3D, ::HSA_EXT_IMAGE_GEOMETRY_2DA, or
 * HSA_EXT_IMAGE_GEOMETRY_2DADEPTH and the value is smaller than the value used for
 * @p src_row_pitch * destination image region height, then the value used for
 * @p src_row_pitch * destination image region height is used.
 * Otherwise, the value is not used.
 *
 * @param[in] dst_image Image handle of destination image.
 *
 * @param[in] image_region Pointer to the image region to be updated. Must not
 * be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p src_memory is NULL, or @p
 * image_region is NULL.
 *
 */
hsa_status_t HSA_API hsa_ext_image_import(
    hsa_agent_t agent,
    const void *src_memory,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    hsa_ext_image_t dst_image,
    const hsa_ext_image_region_t *image_region);

/**
 * @brief Export the image data to linearly organized memory.
 *
 * @details The operation updates the destination memory with the image data of
 * @p src_image. The size of the data exported to memory is implicitly derived
 * from the image region.
 *
 * It is the application's responsibility to avoid out of bounds memory access.
 *
 * None of the destination memory or source image data memory can
 * overlap. Overlapping of any of the source and destination image
 * data memory within the export operation produces undefined results.
 *
 * @param[in] agent Agent associated with the image handle.
 *
 * @param[in] src_image Image handle of source image.
 *
 * @param[in] dst_memory Destination memory. Must not be NULL.
 *
 * @param[in] dst_row_pitch The size in bytes of a single row of the image in the
 * destination memory. If the value is smaller than the source image region
 * width * image element byte size, then region width * image element byte
 * size is used.
 *
 * @param[in] dst_slice_pitch The size in bytes of a single 2D slice of a 3D image,
 * or the size in bytes of each image in an image array in the destination memory.
 * If the geometry is ::HSA_EXT_IMAGE_GEOMETRY_1DA and the value is smaller than the
 * value used for @p dst_row_pitch, then the value used for @p dst_row_pitch is used.
 * If the geometry is ::HSA_EXT_IMAGE_GEOMETRY_3D, ::HSA_EXT_IMAGE_GEOMETRY_2DA, or
 * HSA_EXT_IMAGE_GEOMETRY_2DADEPTH and the value is smaller than the value used for
 * @p dst_row_pitch * source image region height, then the value used for
 * @p dst_row_pitch * source image region height is used.
 * Otherwise, the value is not used.
 *
 * @param[in] image_region Pointer to the image region to be exported. Must not
 * be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p dst_memory is NULL, or @p
 * image_region is NULL.
 */
hsa_status_t HSA_API hsa_ext_image_export(
    hsa_agent_t agent,
    hsa_ext_image_t src_image,
    void *dst_memory,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    const hsa_ext_image_region_t *image_region);

/**
 * @brief Clear a region of an image so that every image element has
 * the specified value.
 *
 * @param[in] agent Agent associated with the image handle.
 *
 * @param[in] image Image handle for image to be cleared.
 *
 * @param[in] data The value to which to set each image element being
 * cleared. It is specified as an array of image component values. The
 * number of array elements must match the number of access components
 * for the image channel order. The type of each array element must
 * match the image access type of the image channel type. When the
 * value is used to set the value of an image element, the conversion
 * method corresponding to the image channel type is used. See the
 * <em>Channel Order</em> section and <em>Channel Type</em> section in
 * the <em>HSA Programming Reference Manual</em> for more
 * information. Must not be NULL.
 *
 * @param[in] image_region Pointer to the image region to clear. Must not be
 * NULL. If the region references an out-out-bounds element, the behavior is
 * undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p data is NULL, or @p
 * image_region is NULL.
 */
hsa_status_t HSA_API hsa_ext_image_clear(
    hsa_agent_t agent,
    hsa_ext_image_t image,
    const void* data,
    const hsa_ext_image_region_t *image_region);

/**
 * @brief Sampler handle. Samplers are populated by
 * ::hsa_ext_sampler_create. Sampler handles are only unique within an
 * agent, not across agents.
 */
typedef struct hsa_ext_sampler_s {
  /**
   *  Opaque handle. For a given agent, two handles reference the same object of
   *  the enclosing type if and only if they are equal.
   */
    uint64_t handle;
} hsa_ext_sampler_t;

/**
 * @brief Sampler address modes. The sampler address mode describes
 * the processing of out-of-range image coordinates. See the
 * <em>Addressing Mode</em> section in the <em>HSA Programming Reference
 * Manual</em> for definitions on each address mode. The values
 * match the BRIG type @p hsa_ext_brig_sampler_addressing_t.
 */
typedef enum {
  /**
   * Out-of-range coordinates are not handled.
   */
  HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED = 0,

  /**
   * Clamp out-of-range coordinates to the image edge.
   */
  HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE = 1,

  /**
   * Clamp out-of-range coordinates to the image border color.
   */
  HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER = 2,

  /**
   * Wrap out-of-range coordinates back into the valid coordinate
   * range so the image appears as repeated tiles.
   */
  HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT = 3,

  /**
   * Mirror out-of-range coordinates back into the valid coordinate
   * range so the image appears as repeated tiles with every other
   * tile a reflection.
   */
  HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT = 4

} hsa_ext_sampler_addressing_mode_t;

/**
 * @brief A fixed-size type used to represent ::hsa_ext_sampler_addressing_mode_t constants.
 */
typedef uint32_t hsa_ext_sampler_addressing_mode32_t;

/**
 * @brief Sampler coordinate normalization modes. See the
 * <em>Coordinate Normalization Mode</em> section in the <em>HSA
 * Programming Reference Manual</em> for definitions on each
 * coordinate normalization mode. The values match the BRIG type @p
 * hsa_ext_brig_sampler_coord_normalization_t.
 */
typedef enum {

  /**
   * Coordinates are used to directly address an image element.
   */
  HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED = 0,

  /**
   * Coordinates are scaled by the image dimension size before being
   * used to address an image element.
   */
  HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED = 1

} hsa_ext_sampler_coordinate_mode_t;

/**
 * @brief A fixed-size type used to represent ::hsa_ext_sampler_coordinate_mode_t constants.
 */
typedef uint32_t hsa_ext_sampler_coordinate_mode32_t;
    

/**
 * @brief Sampler filter modes. See the <em>Filter Mode</em> section
 * in the <em>HSA Programming Reference Manual</em> for definitions
 * on each address mode. The enumeration values match the BRIG type @p
 * hsa_ext_brig_sampler_filter_t.
 */
typedef enum {
  /**
   * Filter to the image element nearest (in Manhattan distance) to the
   * specified coordinate.
   */
  HSA_EXT_SAMPLER_FILTER_MODE_NEAREST = 0,

  /**
   * Filter to the image element calculated by combining the elements in a 2x2
   * square block or 2x2x2 cube block around the specified coordinate. The
   * elements are combined using linear interpolation.
   */
  HSA_EXT_SAMPLER_FILTER_MODE_LINEAR = 1

} hsa_ext_sampler_filter_mode_t;

/**
 * @brief A fixed-size type used to represent ::hsa_ext_sampler_filter_mode_t constants.
 */
typedef uint32_t hsa_ext_sampler_filter_mode32_t;

/**
 * @brief Implementation independent sampler descriptor.
 */
typedef struct hsa_ext_sampler_descriptor_s {
  /**
   * Sampler coordinate mode describes the normalization of image coordinates.
   */
  hsa_ext_sampler_coordinate_mode32_t coordinate_mode;

  /**
   * Sampler filter type describes the type of sampling performed.
   */
  hsa_ext_sampler_filter_mode32_t filter_mode;

  /**
   * Sampler address mode describes the processing of out-of-range image
   * coordinates.
   */
  hsa_ext_sampler_addressing_mode32_t address_mode;

} hsa_ext_sampler_descriptor_t;

/**
 * @brief Create an agent specific sampler handle for a given agent
 * independent sampler descriptor and agent.
 *
 * @param[in] agent Agent to be associated with the sampler handle created.
 *
 * @param[in] sampler_descriptor Pointer to a sampler descriptor. Must not be
 * NULL.
 *
 * @param[out] sampler Memory location where the HSA runtime stores the newly
 * created sampler handle. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED The
 * @p agent does not have the capability to support the properties
 * specified by @p sampler_descriptor or it is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p sampler_descriptor is NULL, or
 * @p sampler is NULL.
 */
hsa_status_t HSA_API hsa_ext_sampler_create(
    hsa_agent_t agent,
    const hsa_ext_sampler_descriptor_t *sampler_descriptor,
    hsa_ext_sampler_t *sampler);

/**
 * @brief Destroy a sampler handle previously created using ::hsa_ext_sampler_create.
 *
 * @details The sampler handle should not be destroyed while there are
 * references to it queued for execution or currently being used in a
 * kernel dispatch.
 *
 * @param[in] agent Agent associated with the sampler handle.
 *
 * @param[in] sampler Sampler handle to destroy.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 */
hsa_status_t HSA_API hsa_ext_sampler_destroy(
    hsa_agent_t agent,
    hsa_ext_sampler_t sampler);


#define hsa_ext_images_1_00

/**
 * @brief The function pointer table for the images v1.00 extension. Can be returned by ::hsa_system_get_extension_table or ::hsa_system_get_major_extension_table.
 */
typedef struct hsa_ext_images_1_00_pfn_s {

  hsa_status_t (*hsa_ext_image_get_capability)(
    hsa_agent_t agent,
    hsa_ext_image_geometry_t geometry,
    const hsa_ext_image_format_t *image_format,
    uint32_t *capability_mask);

  hsa_status_t (*hsa_ext_image_data_get_info)(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t *image_descriptor,
    hsa_access_permission_t access_permission,
    hsa_ext_image_data_info_t *image_data_info);

  hsa_status_t (*hsa_ext_image_create)(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t *image_descriptor,
    const void *image_data,
    hsa_access_permission_t access_permission,
    hsa_ext_image_t *image);

  hsa_status_t (*hsa_ext_image_destroy)(
    hsa_agent_t agent,
    hsa_ext_image_t image);

  hsa_status_t (*hsa_ext_image_copy)(
    hsa_agent_t agent,
    hsa_ext_image_t src_image,
    const hsa_dim3_t* src_offset,
    hsa_ext_image_t dst_image,
    const hsa_dim3_t* dst_offset,
    const hsa_dim3_t* range);

  hsa_status_t (*hsa_ext_image_import)(
    hsa_agent_t agent,
    const void *src_memory,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    hsa_ext_image_t dst_image,
    const hsa_ext_image_region_t *image_region);

  hsa_status_t (*hsa_ext_image_export)(
    hsa_agent_t agent,
    hsa_ext_image_t src_image,
    void *dst_memory,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    const hsa_ext_image_region_t *image_region);

  hsa_status_t (*hsa_ext_image_clear)(
    hsa_agent_t agent,
    hsa_ext_image_t image,
    const void* data,
    const hsa_ext_image_region_t *image_region);

  hsa_status_t (*hsa_ext_sampler_create)(
    hsa_agent_t agent,
    const hsa_ext_sampler_descriptor_t *sampler_descriptor,
    hsa_ext_sampler_t *sampler);

  hsa_status_t (*hsa_ext_sampler_destroy)(
    hsa_agent_t agent,
    hsa_ext_sampler_t sampler);

} hsa_ext_images_1_00_pfn_t;

#define hsa_ext_images_1

/**
 * @brief The function pointer table for the images v1 extension. Can be returned by ::hsa_system_get_extension_table or ::hsa_system_get_major_extension_table.
 */
typedef struct hsa_ext_images_1_pfn_s {

  hsa_status_t (*hsa_ext_image_get_capability)(
    hsa_agent_t agent,
    hsa_ext_image_geometry_t geometry,
    const hsa_ext_image_format_t *image_format,
    uint32_t *capability_mask);

  hsa_status_t (*hsa_ext_image_data_get_info)(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t *image_descriptor,
    hsa_access_permission_t access_permission,
    hsa_ext_image_data_info_t *image_data_info);

  hsa_status_t (*hsa_ext_image_create)(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t *image_descriptor,
    const void *image_data,
    hsa_access_permission_t access_permission,
    hsa_ext_image_t *image);

  hsa_status_t (*hsa_ext_image_destroy)(
    hsa_agent_t agent,
    hsa_ext_image_t image);

  hsa_status_t (*hsa_ext_image_copy)(
    hsa_agent_t agent,
    hsa_ext_image_t src_image,
    const hsa_dim3_t* src_offset,
    hsa_ext_image_t dst_image,
    const hsa_dim3_t* dst_offset,
    const hsa_dim3_t* range);

  hsa_status_t (*hsa_ext_image_import)(
    hsa_agent_t agent,
    const void *src_memory,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    hsa_ext_image_t dst_image,
    const hsa_ext_image_region_t *image_region);

  hsa_status_t (*hsa_ext_image_export)(
    hsa_agent_t agent,
    hsa_ext_image_t src_image,
    void *dst_memory,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    const hsa_ext_image_region_t *image_region);

  hsa_status_t (*hsa_ext_image_clear)(
    hsa_agent_t agent,
    hsa_ext_image_t image,
    const void* data,
    const hsa_ext_image_region_t *image_region);

  hsa_status_t (*hsa_ext_sampler_create)(
    hsa_agent_t agent,
    const hsa_ext_sampler_descriptor_t *sampler_descriptor,
    hsa_ext_sampler_t *sampler);

  hsa_status_t (*hsa_ext_sampler_destroy)(
    hsa_agent_t agent,
    hsa_ext_sampler_t sampler);

  hsa_status_t (*hsa_ext_image_get_capability_with_layout)(
    hsa_agent_t agent,
    hsa_ext_image_geometry_t geometry,
    const hsa_ext_image_format_t *image_format,
    hsa_ext_image_data_layout_t image_data_layout,
    uint32_t *capability_mask);

  hsa_status_t (*hsa_ext_image_data_get_info_with_layout)(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t *image_descriptor,
    hsa_access_permission_t access_permission,
    hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    hsa_ext_image_data_info_t *image_data_info);

  hsa_status_t (*hsa_ext_image_create_with_layout)(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t *image_descriptor,
    const void *image_data,
    hsa_access_permission_t access_permission,
    hsa_ext_image_data_layout_t image_data_layout,
    size_t image_data_row_pitch,
    size_t image_data_slice_pitch,
    hsa_ext_image_t *image);

} hsa_ext_images_1_pfn_t;
/** @} */
    
#ifdef __cplusplus
}  // end extern "C" block
#endif /*__cplusplus*/ 

#endif
