/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_INCLUDE_AMD_HIP_GL_INTEROP_H
#define HIP_INCLUDE_AMD_HIP_GL_INTEROP_H

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *
 * @addtogroup GlobalDefs
 * @{
 *
 */

/**
 * HIP Devices used by current OpenGL Context.
 */
typedef enum hipGLDeviceList {
    hipGLDeviceListAll = 1,           ///< All hip devices used by current OpenGL context.
    hipGLDeviceListCurrentFrame = 2,  ///< Hip devices used by current OpenGL context in current
                                    ///< frame
    hipGLDeviceListNextFrame = 3      ///< Hip devices used by current OpenGL context in next
                                    ///< frame.
} hipGLDeviceList;


/** GLuint as uint.*/
typedef unsigned int GLuint;
/** GLenum as uint.*/
typedef unsigned int GLenum;
/*
* @}
*/

/**
 *  @ingroup GL
 *  @{
 *
 */
/**
 * @brief Queries devices associated with the current OpenGL context.
 *
 * @param [out] pHipDeviceCount - Pointer of number of devices on the current GL context.
 * @param [out] pHipDevices - Pointer of devices on the current OpenGL context.
 * @param [in] hipDeviceCount - Size of device.
 * @param [in] deviceList - The setting of devices. It could be either hipGLDeviceListCurrentFrame
 * for the devices used to render the current frame, or hipGLDeviceListAll for all devices.
 * The default setting is Invalid deviceList value.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorNotSupported
 *
 */
hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices,
                           unsigned int hipDeviceCount, hipGLDeviceList deviceList);
/**
 * @brief Registers a GL Buffer for interop and returns corresponding graphics resource.
 *
 * @param [out] resource - Returns pointer of graphics resource.
 * @param [in] buffer - Buffer to be registered.
 * @param [in] flags - Register flags.
 * 
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorUnknown, #hipErrorInvalidResourceHandle
 *
 */
hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer,
                                       unsigned int flags);
/**
 * @brief Register a GL Image for interop and returns the corresponding graphic resource.
 *
 * @param [out] resource - Returns pointer of graphics resource.
 * @param [in] image - Image to be registered.
 * @param [in] target - Valid target value Id.
 * @param [in] flags - Register flags.
 *
 * @returns #hipSuccess, #hipErrorInvalidValue, #hipErrorUnknown, #hipErrorInvalidResourceHandle
 *
 */
hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image,
                                      GLenum target, unsigned int flags);
/*
* @}
*/
#if defined(__cplusplus)
}
#endif /* __cplusplus */
#endif /* HIP_INCLUDE_AMD_HIP_GL_INTEROP_H */
