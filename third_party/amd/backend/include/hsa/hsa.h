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

#ifndef HSA_RUNTIME_INC_HSA_H_
#define HSA_RUNTIME_INC_HSA_H_

#include <stddef.h>   /* size_t */
#include <stdint.h>   /* uintXX_t */

#ifndef __cplusplus
#include <stdbool.h>  /* bool */
#endif /* __cplusplus */

// Placeholder for calling convention and import/export macros
#ifndef HSA_CALL
#define HSA_CALL
#endif

#ifndef HSA_EXPORT_DECORATOR
#ifdef __GNUC__
#define HSA_EXPORT_DECORATOR __attribute__ ((visibility ("default")))
#else
#define HSA_EXPORT_DECORATOR
#endif
#endif
#define HSA_API_EXPORT HSA_EXPORT_DECORATOR HSA_CALL
#define HSA_API_IMPORT HSA_CALL

#if !defined(HSA_API) && defined(HSA_EXPORT)
#define HSA_API HSA_API_EXPORT
#else
#define HSA_API HSA_API_IMPORT
#endif

// Detect and set large model builds.
#undef HSA_LARGE_MODEL
#if defined(__LP64__) || defined(_M_X64)
#define HSA_LARGE_MODEL
#endif

// Try to detect CPU endianness
#if !defined(LITTLEENDIAN_CPU) && !defined(BIGENDIAN_CPU)
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define LITTLEENDIAN_CPU
#elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define BIGENDIAN_CPU
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || \
      defined(_M_X64) || defined(__loongarch64) || defined(__riscv)
#define LITTLEENDIAN_CPU
#endif
#endif

#undef HSA_LITTLE_ENDIAN
#if defined(LITTLEENDIAN_CPU)
#define HSA_LITTLE_ENDIAN
#elif defined(BIGENDIAN_CPU)
#else
#error "BIGENDIAN_CPU or LITTLEENDIAN_CPU must be defined"
#endif

#ifndef HSA_DEPRECATED
#define HSA_DEPRECATED
//#ifdef __GNUC__
//#define HSA_DEPRECATED __attribute__((deprecated))
//#else
//#define HSA_DEPRECATED __declspec(deprecated)
//#endif
#endif

#define HSA_VERSION_1_0                              1

#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

/** \defgroup status Runtime Notifications
 *  @{
 */

/**
 * @brief Status codes.
 */
typedef enum {
  /**
   * The function has been executed successfully.
   */
  HSA_STATUS_SUCCESS = 0x0,
  /**
   * A traversal over a list of elements has been interrupted by the
   * application before completing.
   */
  HSA_STATUS_INFO_BREAK = 0x1,
  /**
   * A generic error has occurred.
   */
  HSA_STATUS_ERROR = 0x1000,
  /**
   * One of the actual arguments does not meet a precondition stated in the
   * documentation of the corresponding formal argument.
   */
  HSA_STATUS_ERROR_INVALID_ARGUMENT = 0x1001,
  /**
   * The requested queue creation is not valid.
   */
  HSA_STATUS_ERROR_INVALID_QUEUE_CREATION = 0x1002,
  /**
   * The requested allocation is not valid.
   */
  HSA_STATUS_ERROR_INVALID_ALLOCATION = 0x1003,
  /**
   * The agent is invalid.
   */
  HSA_STATUS_ERROR_INVALID_AGENT = 0x1004,
  /**
   * The memory region is invalid.
   */
  HSA_STATUS_ERROR_INVALID_REGION = 0x1005,
  /**
   * The signal is invalid.
   */
  HSA_STATUS_ERROR_INVALID_SIGNAL = 0x1006,
  /**
   * The queue is invalid.
   */
  HSA_STATUS_ERROR_INVALID_QUEUE = 0x1007,
  /**
   * The HSA runtime failed to allocate the necessary resources. This error
   * may also occur when the HSA runtime needs to spawn threads or create
   * internal OS-specific events.
   */
  HSA_STATUS_ERROR_OUT_OF_RESOURCES = 0x1008,
  /**
   * The AQL packet is malformed.
   */
  HSA_STATUS_ERROR_INVALID_PACKET_FORMAT = 0x1009,
  /**
   * An error has been detected while releasing a resource.
   */
  HSA_STATUS_ERROR_RESOURCE_FREE = 0x100A,
  /**
   * An API other than ::hsa_init has been invoked while the reference count
   * of the HSA runtime is 0.
   */
  HSA_STATUS_ERROR_NOT_INITIALIZED = 0x100B,
  /**
   * The maximum reference count for the object has been reached.
   */
  HSA_STATUS_ERROR_REFCOUNT_OVERFLOW = 0x100C,
  /**
   * The arguments passed to a functions are not compatible.
   */
  HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS = 0x100D,
  /**
   * The index is invalid.
   */
  HSA_STATUS_ERROR_INVALID_INDEX = 0x100E,
  /**
   * The instruction set architecture is invalid.
   */
  HSA_STATUS_ERROR_INVALID_ISA = 0x100F,
  /**
   * The instruction set architecture name is invalid.
   */
  HSA_STATUS_ERROR_INVALID_ISA_NAME = 0x1017,
  /**
   * The code object is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CODE_OBJECT = 0x1010,
  /**
   * The executable is invalid.
   */
  HSA_STATUS_ERROR_INVALID_EXECUTABLE = 0x1011,
  /**
   * The executable is frozen.
   */
  HSA_STATUS_ERROR_FROZEN_EXECUTABLE = 0x1012,
  /**
   * There is no symbol with the given name.
   */
  HSA_STATUS_ERROR_INVALID_SYMBOL_NAME = 0x1013,
  /**
   * The variable is already defined.
   */
  HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED = 0x1014,
  /**
   * The variable is undefined.
   */
  HSA_STATUS_ERROR_VARIABLE_UNDEFINED = 0x1015,
  /**
   * An HSAIL operation resulted in a hardware exception.
   */
  HSA_STATUS_ERROR_EXCEPTION = 0x1016,
  /**
   * The code object symbol is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CODE_SYMBOL = 0x1018,
  /**
   * The executable symbol is invalid.
   */
  HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL = 0x1019,
  /**
   * The file descriptor is invalid.
   */
  HSA_STATUS_ERROR_INVALID_FILE = 0x1020,
  /**
   * The code object reader is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER = 0x1021,
  /**
   * The cache is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CACHE = 0x1022,
  /**
   * The wavefront is invalid.
   */
  HSA_STATUS_ERROR_INVALID_WAVEFRONT = 0x1023,
  /**
   * The signal group is invalid.
   */
  HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP = 0x1024,
  /**
   * The HSA runtime is not in the configuration state.
   */
  HSA_STATUS_ERROR_INVALID_RUNTIME_STATE = 0x1025,
  /**
  * The queue received an error that may require process termination.
  */
  HSA_STATUS_ERROR_FATAL = 0x1026
} hsa_status_t;

/**
 * @brief Query additional information about a status code.
 *
 * @param[in] status Status code.
 *
 * @param[out] status_string A NUL-terminated string that describes the error
 * status.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p status is an invalid
 * status code, or @p status_string is NULL.
 */
hsa_status_t HSA_API hsa_status_string(
    hsa_status_t status,
    const char ** status_string);

/** @} */

/** \defgroup common Common Definitions
 *  @{
 */

/**
 * @brief Three-dimensional coordinate.
 */
typedef struct hsa_dim3_s {
  /**
   * X dimension.
   */
   uint32_t x;

  /**
   * Y dimension.
   */
   uint32_t y;

   /**
    * Z dimension.
    */
   uint32_t z;
} hsa_dim3_t;

/**
 * @brief Access permissions.
 */
typedef enum {
  /**
   * Used to remove existing access
   */
  HSA_ACCESS_PERMISSION_NONE = 0,
  /**
   * Read-only access.
   */
  HSA_ACCESS_PERMISSION_RO = 1,
  /**
   * Write-only access.
   */
  HSA_ACCESS_PERMISSION_WO = 2,
  /**
   * Read and write access.
   */
  HSA_ACCESS_PERMISSION_RW = 3
} hsa_access_permission_t;

/**
 * @brief POSIX file descriptor.
 */
typedef int hsa_file_t;

/** @} **/


/** \defgroup initshutdown Initialization and Shut Down
 *  @{
 */

/**
 * @brief Initialize the HSA runtime.
 *
 * @details Initializes the HSA runtime if it is not already initialized, and
 * increases the reference counter associated with the HSA runtime for the
 * current process. Invocation of any HSA function other than ::hsa_init results
 * in undefined behavior if the current HSA runtime reference counter is less
 * than one.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_REFCOUNT_OVERFLOW The HSA runtime reference
 * count reaches INT32_MAX.
 */
hsa_status_t HSA_API hsa_init();

/**
 * @brief Shut down the HSA runtime.
 *
 * @details Decreases the reference count of the HSA runtime instance. When the
 * reference count reaches 0, the HSA runtime is no longer considered valid
 * but the application might call ::hsa_init to initialize the HSA runtime
 * again.
 *
 * Once the reference count of the HSA runtime reaches 0, all the resources
 * associated with it (queues, signals, agent information, etc.) are
 * considered invalid and any attempt to reference them in subsequent API calls
 * results in undefined behavior. When the reference count reaches 0, the HSA
 * runtime may release resources associated with it.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 */
hsa_status_t HSA_API hsa_shut_down();

/** @} **/

/** \defgroup agentinfo System and Agent Information
 *  @{
 */

/**
 * @brief Endianness. A convention used to interpret the bytes making up a data
 * word.
 */
typedef enum {
    /**
     * The least significant byte is stored in the smallest address.
     */
    HSA_ENDIANNESS_LITTLE = 0,
    /**
     * The most significant byte is stored in the smallest address.
     */
    HSA_ENDIANNESS_BIG = 1
} hsa_endianness_t;

/**
 * @brief Machine model. A machine model determines the size of certain data
 * types in HSA runtime and an agent.
 */
typedef enum {
    /**
     * Small machine model. Addresses use 32 bits.
     */
    HSA_MACHINE_MODEL_SMALL = 0,
    /**
     * Large machine model. Addresses use 64 bits.
     */
    HSA_MACHINE_MODEL_LARGE = 1
} hsa_machine_model_t;

/**
 * @brief Profile. A profile indicates a particular level of feature
 * support. For example, in the base profile the application must use the HSA
 * runtime allocator to reserve shared virtual memory, while in the full profile
 * any host pointer can be shared across all the agents.
 */
typedef enum {
    /**
     * Base profile.
     */
    HSA_PROFILE_BASE = 0,
    /**
     * Full profile.
     */
    HSA_PROFILE_FULL = 1
} hsa_profile_t;

/**
 * @brief System attributes.
 */
typedef enum {
  /**
   * Major version of the HSA runtime specification supported by the
   * implementation. The type of this attribute is uint16_t.
   */
  HSA_SYSTEM_INFO_VERSION_MAJOR = 0,
  /**
   * Minor version of the HSA runtime specification supported by the
   * implementation. The type of this attribute is uint16_t.
   */
  HSA_SYSTEM_INFO_VERSION_MINOR = 1,
  /**
   * Current timestamp. The value of this attribute monotonically increases at a
   * constant rate. The type of this attribute is uint64_t.
   */
  HSA_SYSTEM_INFO_TIMESTAMP = 2,
  /**
   * Timestamp value increase rate, in Hz. The timestamp (clock) frequency is
   * in the range 1-400MHz. The type of this attribute is uint64_t.
   */
  HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY = 3,
  /**
   * Maximum duration of a signal wait operation. Expressed as a count based on
   * the timestamp frequency. The type of this attribute is uint64_t.
   */
  HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT = 4,
  /**
   * Endianness of the system. The type of this attribute is ::hsa_endianness_t.
   */
  HSA_SYSTEM_INFO_ENDIANNESS = 5,
  /**
   * Machine model supported by the HSA runtime. The type of this attribute is
   * ::hsa_machine_model_t.
   */
  HSA_SYSTEM_INFO_MACHINE_MODEL = 6,
  /**
   * Bit-mask indicating which extensions are supported by the
   * implementation. An extension with an ID of @p i is supported if the bit at
   * position @p i is set. The type of this attribute is uint8_t[128].
   */
  HSA_SYSTEM_INFO_EXTENSIONS = 7,
  /**
  * String containing the ROCr build identifier.
  */
  HSA_AMD_SYSTEM_INFO_BUILD_VERSION = 0x200,
  /**
   * Returns true if hsa_amd_svm_* APIs are supported by the driver.  The type of
   * this attribute is bool.
   */
  HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED = 0x201,
  // TODO: Should this be per Agent?
  /**
   * Returns true if all Agents have access to system allocated memory (such as
   * that allocated by mmap, malloc, or new) by default.
   * If false then system allocated memory may only be made SVM accessible to
   * an Agent by declaration of accessibility with hsa_amd_svm_set_attributes.
   * The type of this attribute is bool.
   */
  HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT = 0x202,
  /**
   * Returns true if mwaitx is enabled on this system
   * The type of this attribute is bool.
   */
  HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED = 0x203,
  /**
   * Returns true if DMABUF APIs are supported by the driver.  The type of
   * this attribute is bool.
   */
  HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED = 0x204,
  /**
   * Returns true if Virtual Memory APIs are supported by the driver.  The type of
   * this attribute is bool.
   */
  HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED = 0x205,
  /**
   * Returns true if XNACK is enabled on this system.  The type of
   * this attribute is bool.
   */
  HSA_AMD_SYSTEM_INFO_XNACK_ENABLED = 0x206,
  /**
   * Major version of the HSA runtime extension specification supported by the
   * implementation. The type of this attribute is uint16_t.
   */
  HSA_AMD_SYSTEM_INFO_EXT_VERSION_MAJOR = 0x207,
  /**
   * Minor version of the HSA runtime extension specification supported by the
   * implementation. The type of this attribute is uint16_t.
   */
  HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR = 0x208,
} hsa_system_info_t;

/**
 * @brief Get the current value of a system attribute.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * system attribute, or @p value is NULL.
 */
hsa_status_t HSA_API hsa_system_get_info(
    hsa_system_info_t attribute,
    void* value);

/**
 * @brief HSA extensions.
 */
typedef enum {
  /**
   * Finalizer extension.
   */
  HSA_EXTENSION_FINALIZER = 0,
  /**
   * Images extension.
   */
  HSA_EXTENSION_IMAGES = 1,

  /**
   * Performance counter extension.
   */
  HSA_EXTENSION_PERFORMANCE_COUNTERS = 2,

  /**
   * Profiling events extension.
   */
  HSA_EXTENSION_PROFILING_EVENTS = 3,
  /**
   * Extension count.
   */
  HSA_EXTENSION_STD_LAST = 3,
  /**
   * First AMD extension number.
   */
  HSA_AMD_FIRST_EXTENSION = 0x200,
  /**
   * Profiler extension.
   */
  HSA_EXTENSION_AMD_PROFILER = 0x200,
  /**
   * Loader extension.
   */
  HSA_EXTENSION_AMD_LOADER = 0x201,
  /**
   * AqlProfile extension.
   */
  HSA_EXTENSION_AMD_AQLPROFILE = 0x202,
  /**
   * Last AMD extension.
   */
  HSA_AMD_LAST_EXTENSION = 0x202
} hsa_extension_t;

/**
 * @brief Query the name of a given extension.
 *
 * @param[in] extension Extension identifier. If the extension is not supported
 * by the implementation (see ::HSA_SYSTEM_INFO_EXTENSIONS), the behavior
 * is undefined.
 *
 * @param[out] name Pointer to a memory location where the HSA runtime stores
 * the extension name. The extension name is a NUL-terminated string.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p extension is not a valid
 * extension, or @p name is NULL.
 */
hsa_status_t HSA_API hsa_extension_get_name(
    uint16_t extension,
    const char **name);

/**
 * @deprecated
 *
 * @brief Query if a given version of an extension is supported by the HSA
 * implementation.
 *
 * @param[in] extension Extension identifier.
 *
 * @param[in] version_major Major version number.
 *
 * @param[in] version_minor Minor version number.
 *
 * @param[out] result Pointer to a memory location where the HSA runtime stores
 * the result of the check. The result is true if the specified version of the
 * extension is supported, and false otherwise.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p extension is not a valid
 * extension, or @p result is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_system_extension_supported(
    uint16_t extension,
    uint16_t version_major,
    uint16_t version_minor,
    bool* result);

/**
 * @brief Query if a given version of an extension is supported by the HSA
 * implementation. All minor versions from 0 up to the returned @p version_minor
 * must be supported by the implementation.
 *
 * @param[in] extension Extension identifier.
 *
 * @param[in] version_major Major version number.
 *
 * @param[out] version_minor Minor version number.
 *
 * @param[out] result Pointer to a memory location where the HSA runtime stores
 * the result of the check. The result is true if the specified version of the
 * extension is supported, and false otherwise.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p extension is not a valid
 * extension, or @p version_minor is NULL, or @p result is NULL.
 */
hsa_status_t HSA_API hsa_system_major_extension_supported(
    uint16_t extension,
    uint16_t version_major,
    uint16_t *version_minor,
    bool* result);


/**
 * @deprecated
 *
 * @brief Retrieve the function pointers corresponding to a given version of an
 * extension. Portable applications are expected to invoke the extension API
 * using the returned function pointers
 *
 * @details The application is responsible for verifying that the given version
 * of the extension is supported by the HSA implementation (see
 * ::hsa_system_extension_supported). If the given combination of extension,
 * major version, and minor version is not supported by the implementation, the
 * behavior is undefined.
 *
 * @param[in] extension Extension identifier.
 *
 * @param[in] version_major Major version number for which to retrieve the
 * function pointer table.
 *
 * @param[in] version_minor Minor version number for which to retrieve the
 * function pointer table.
 *
 * @param[out] table Pointer to an application-allocated function pointer table
 * that is populated by the HSA runtime. Must not be NULL. The memory associated
 * with table can be reused or freed after the function returns.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p extension is not a valid
 * extension, or @p table is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_system_get_extension_table(
    uint16_t extension,
    uint16_t version_major,
    uint16_t version_minor,
    void *table);

/**
 * @brief Retrieve the function pointers corresponding to a given major version
 * of an extension. Portable applications are expected to invoke the extension
 * API using the returned function pointers.
 *
 * @details The application is responsible for verifying that the given major
 * version of the extension is supported by the HSA implementation (see
 * ::hsa_system_major_extension_supported). If the given combination of extension
 * and major version is not supported by the implementation, the behavior is
 * undefined. Additionally if the length doesn't allow space for a full minor
 * version, it is implementation defined if only some of the function pointers for
 * that minor version get written.
 *
 * @param[in] extension Extension identifier.
 *
 * @param[in] version_major Major version number for which to retrieve the
 * function pointer table.
 *
 * @param[in] table_length Size in bytes of the function pointer table to be
 * populated. The implementation will not write more than this many bytes to the
 * table.
 *
 * @param[out] table Pointer to an application-allocated function pointer table
 * that is populated by the HSA runtime. Must not be NULL. The memory associated
 * with table can be reused or freed after the function returns.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p extension is not a valid
 * extension, or @p table is NULL.
 */
hsa_status_t HSA_API hsa_system_get_major_extension_table(
    uint16_t extension,
    uint16_t version_major,
    size_t table_length,
    void *table);

/**
 * @brief Struct containing an opaque handle to an agent, a device that participates in
 * the HSA memory model. An agent can submit AQL packets for execution, and
 * may also accept AQL packets for execution (agent dispatch packets or kernel
 * dispatch packets launching HSAIL-derived binaries).
 */
typedef struct hsa_agent_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_agent_t;

/**
 * @brief Agent features.
 */
typedef enum {
    /**
     * The agent supports AQL packets of kernel dispatch type. If this
     * feature is enabled, the agent is also a kernel agent.
     */
    HSA_AGENT_FEATURE_KERNEL_DISPATCH = 1,
    /**
     * The agent supports AQL packets of agent dispatch type.
     */
    HSA_AGENT_FEATURE_AGENT_DISPATCH = 2
} hsa_agent_feature_t;

/**
 * @brief Hardware device type.
 */
typedef enum {
    /**
     * CPU device.
     */
    HSA_DEVICE_TYPE_CPU = 0,
    /**
     * GPU device.
     */
    HSA_DEVICE_TYPE_GPU = 1,
    /**
     * DSP device.
     */
    HSA_DEVICE_TYPE_DSP = 2
} hsa_device_type_t;

/**
 * @brief Default floating-point rounding mode.
 */
typedef enum {
  /**
   * Use a default floating-point rounding mode specified elsewhere.
   */
  HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT = 0,
  /**
   * Operations that specify the default floating-point mode are rounded to zero
   * by default.
   */
  HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO = 1,
  /**
   * Operations that specify the default floating-point mode are rounded to the
   * nearest representable number and that ties should be broken by selecting
   * the value with an even least significant bit.
   */
  HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR = 2
} hsa_default_float_rounding_mode_t;

/**
 * @brief Agent attributes.
 */
typedef enum {
  /**
   * Agent name. The type of this attribute is a NUL-terminated char[64]. The
   * name must be at most 63 characters long (not including the NUL terminator)
   * and all array elements not used for the name must be NUL.
   */
  HSA_AGENT_INFO_NAME = 0,
  /**
   * Name of vendor. The type of this attribute is a NUL-terminated char[64].
   * The name must be at most 63 characters long (not including the NUL
   * terminator) and all array elements not used for the name must be NUL.
   */
  HSA_AGENT_INFO_VENDOR_NAME = 1,
  /**
   * Agent capability. The type of this attribute is ::hsa_agent_feature_t.
   */
  HSA_AGENT_INFO_FEATURE = 2,
  /**
   * @deprecated Query ::HSA_ISA_INFO_MACHINE_MODELS for a given intruction set
   * architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Machine model supported by the agent. The type of this attribute is
   * ::hsa_machine_model_t.
   */
  HSA_AGENT_INFO_MACHINE_MODEL = 3,
  /**
   * @deprecated Query ::HSA_ISA_INFO_PROFILES for a given intruction set
   * architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Profile supported by the agent. The type of this attribute is
   * ::hsa_profile_t.
   */
  HSA_AGENT_INFO_PROFILE = 4,
  /**
   * @deprecated Query ::HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES for a given
   * intruction set architecture supported by the agent instead.  If more than
   * one ISA is supported by the agent, the returned value corresponds to the
   * first ISA enumerated by ::hsa_agent_iterate_isas.
   *
   * Default floating-point rounding mode. The type of this attribute is
   * ::hsa_default_float_rounding_mode_t, but the value
   * ::HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT is not allowed.
   */
  HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 5,
  /**
   * @deprecated Query ::HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES
   * for a given intruction set architecture supported by the agent instead.  If
   * more than one ISA is supported by the agent, the returned value corresponds
   * to the first ISA enumerated by ::hsa_agent_iterate_isas.
   *
   * A bit-mask of ::hsa_default_float_rounding_mode_t values, representing the
   * default floating-point rounding modes supported by the agent in the Base
   * profile. The type of this attribute is uint32_t. The default floating-point
   * rounding mode (::HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE) bit must not
   * be set.
   */
  HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 23,
  /**
   * @deprecated Query ::HSA_ISA_INFO_FAST_F16_OPERATION for a given intruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Flag indicating that the f16 HSAIL operation is at least as fast as the
   * f32 operation in the current agent. The value of this attribute is
   * undefined if the agent is not a kernel agent. The type of this
   * attribute is bool.
   */
  HSA_AGENT_INFO_FAST_F16_OPERATION = 24,
  /**
   * @deprecated Query ::HSA_WAVEFRONT_INFO_SIZE for a given wavefront and
   * intruction set architecture supported by the agent instead.  If more than
   * one ISA is supported by the agent, the returned value corresponds to the
   * first ISA enumerated by ::hsa_agent_iterate_isas and the first wavefront
   * enumerated by ::hsa_isa_iterate_wavefronts for that ISA.
   *
   * Number of work-items in a wavefront. Must be a power of 2 in the range
   * [1,256]. The value of this attribute is undefined if the agent is not
   * a kernel agent. The type of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_WAVEFRONT_SIZE = 6,
  /**
   * @deprecated Query ::HSA_ISA_INFO_WORKGROUP_MAX_DIM for a given intruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum number of work-items of each dimension of a work-group.  Each
   * maximum must be greater than 0. No maximum can exceed the value of
   * ::HSA_AGENT_INFO_WORKGROUP_MAX_SIZE. The value of this attribute is
   * undefined if the agent is not a kernel agent. The type of this
   * attribute is uint16_t[3].
   */
  HSA_AGENT_INFO_WORKGROUP_MAX_DIM = 7,
  /**
   * @deprecated Query ::HSA_ISA_INFO_WORKGROUP_MAX_SIZE for a given intruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum total number of work-items in a work-group. The value of this
   * attribute is undefined if the agent is not a kernel agent. The type
   * of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_WORKGROUP_MAX_SIZE = 8,
  /**
   * @deprecated Query ::HSA_ISA_INFO_GRID_MAX_DIM for a given intruction set
   * architecture supported by the agent instead.
   *
   * Maximum number of work-items of each dimension of a grid. Each maximum must
   * be greater than 0, and must not be smaller than the corresponding value in
   * ::HSA_AGENT_INFO_WORKGROUP_MAX_DIM. No maximum can exceed the value of
   * ::HSA_AGENT_INFO_GRID_MAX_SIZE. The value of this attribute is undefined
   * if the agent is not a kernel agent. The type of this attribute is
   * ::hsa_dim3_t.
   */
  HSA_AGENT_INFO_GRID_MAX_DIM = 9,
  /**
   * @deprecated Query ::HSA_ISA_INFO_GRID_MAX_SIZE for a given intruction set
   * architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum total number of work-items in a grid. The value of this attribute
   * is undefined if the agent is not a kernel agent. The type of this
   * attribute is uint32_t.
   */
  HSA_AGENT_INFO_GRID_MAX_SIZE = 10,
  /**
   * @deprecated Query ::HSA_ISA_INFO_FBARRIER_MAX_SIZE for a given intruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum number of fbarriers per work-group. Must be at least 32. The value
   * of this attribute is undefined if the agent is not a kernel agent. The
   * type of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_FBARRIER_MAX_SIZE = 11,
  /**
   * @deprecated The maximum number of queues is not statically determined.
   *
   * Maximum number of queues that can be active (created but not destroyed) at
   * one time in the agent. The type of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_QUEUES_MAX = 12,
  /**
   * Minimum number of packets that a queue created in the agent
   * can hold. Must be a power of 2 greater than 0. Must not exceed
   * the value of ::HSA_AGENT_INFO_QUEUE_MAX_SIZE. The type of this
   * attribute is uint32_t.
   */
  HSA_AGENT_INFO_QUEUE_MIN_SIZE = 13,
  /**
   * Maximum number of packets that a queue created in the agent can
   * hold. Must be a power of 2 greater than 0. The type of this attribute
   * is uint32_t.
   */
  HSA_AGENT_INFO_QUEUE_MAX_SIZE = 14,
  /**
   * Type of a queue created in the agent. The type of this attribute is
   * ::hsa_queue_type32_t.
   */
  HSA_AGENT_INFO_QUEUE_TYPE = 15,
  /**
   * @deprecated NUMA information is not exposed anywhere else in the API.
   *
   * Identifier of the NUMA node associated with the agent. The type of this
   * attribute is uint32_t.
   */
  HSA_AGENT_INFO_NODE = 16,
  /**
   * Type of hardware device associated with the agent. The type of this
   * attribute is ::hsa_device_type_t.
   */
  HSA_AGENT_INFO_DEVICE = 17,
  /**
   * @deprecated Query ::hsa_agent_iterate_caches to retrieve information about
   * the caches present in a given agent.
   *
   * Array of data cache sizes (L1..L4). Each size is expressed in bytes. A size
   * of 0 for a particular level indicates that there is no cache information
   * for that level. The type of this attribute is uint32_t[4].
   */
  HSA_AGENT_INFO_CACHE_SIZE = 18,
  /**
   * @deprecated An agent may support multiple instruction set
   * architectures. See ::hsa_agent_iterate_isas.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Instruction set architecture of the agent. The type of this attribute
   * is ::hsa_isa_t.
   */
  HSA_AGENT_INFO_ISA = 19,
  /**
   * Bit-mask indicating which extensions are supported by the agent. An
   * extension with an ID of @p i is supported if the bit at position @p i is
   * set. The type of this attribute is uint8_t[128].
   */
  HSA_AGENT_INFO_EXTENSIONS = 20,
  /**
   * Major version of the HSA runtime specification supported by the
   * agent. The type of this attribute is uint16_t.
   */
  HSA_AGENT_INFO_VERSION_MAJOR = 21,
  /**
   * Minor version of the HSA runtime specification supported by the
   * agent. The type of this attribute is uint16_t.
   */
  HSA_AGENT_INFO_VERSION_MINOR = 22,
  /**
   * This enum does not have a fixed underlying type, thus in C++ post D2338:
   * If the enumeration type does not have a fixed underlying type, the value is
   * unchanged if the original value is within the range of the enumeration
   * values (9.7.1 [dcl.enum]), and otherwise, the behavior is
   * undefined.
   * Thus increase the range of this enum to encompass vendor extensions.
   */
  HSA_AGENT_INFO_LAST = INT32_MAX
} hsa_agent_info_t;

/**
 * @brief Get the current value of an attribute for a given agent.
 *
 * @param[in] agent A valid agent.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * agent attribute, or @p value is NULL.
 */
hsa_status_t HSA_API hsa_agent_get_info(
    hsa_agent_t agent,
    hsa_agent_info_t attribute,
    void* value);

/**
 * @brief Iterate over the available agents, and invoke an
 * application-defined callback on every iteration.
 *
 * @param[in] callback Callback to be invoked once per agent. The HSA
 * runtime passes two arguments to the callback: the agent and the
 * application data.  If @p callback returns a status other than
 * ::HSA_STATUS_SUCCESS for a particular iteration, the traversal stops and
 * ::hsa_iterate_agents returns that status value.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL.
*/
hsa_status_t HSA_API hsa_iterate_agents(
    hsa_status_t (*callback)(hsa_agent_t agent, void* data),
    void* data);

/*

// If we do not know the size of an attribute, we need to query it first
// Note: this API will not be in the spec unless needed
hsa_status_t HSA_API hsa_agent_get_info_size(
    hsa_agent_t agent,
    hsa_agent_info_t attribute,
    size_t* size);

// Set the value of an agents attribute
// Note: this API will not be in the spec unless needed
hsa_status_t HSA_API hsa_agent_set_info(
    hsa_agent_t agent,
    hsa_agent_info_t attribute,
    void* value);

*/

/**
 * @brief Exception policies applied in the presence of hardware exceptions.
 */
typedef enum {
    /**
     * If a hardware exception is detected, a work-item signals an exception.
     */
    HSA_EXCEPTION_POLICY_BREAK = 1,
    /**
     * If a hardware exception is detected, a hardware status bit is set.
     */
    HSA_EXCEPTION_POLICY_DETECT = 2
} hsa_exception_policy_t;

/**
 * @deprecated Use ::hsa_isa_get_exception_policies for a given intruction set
 * architecture supported by the agent instead. If more than one ISA is
 * supported by the agent, this function uses the first value returned by
 * ::hsa_agent_iterate_isas.
 *
 * @brief Retrieve the exception policy support for a given combination of
 * agent and profile
 *
 * @param[in] agent Agent.
 *
 * @param[in] profile Profile.
 *
 * @param[out] mask Pointer to a memory location where the HSA runtime stores a
 * mask of ::hsa_exception_policy_t values. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p profile is not a valid
 * profile, or @p mask is NULL.
 *
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_agent_get_exception_policies(
    hsa_agent_t agent,
    hsa_profile_t profile,
    uint16_t *mask);

/**
 * @brief Cache handle.
 */
typedef struct hsa_cache_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_cache_t;

/**
 * @brief Cache attributes.
 */
typedef enum {
  /**
   * The length of the cache name in bytes, not including the NUL terminator.
   * The type of this attribute is uint32_t.
   */
  HSA_CACHE_INFO_NAME_LENGTH = 0,
  /**
   * Human-readable description.  The type of this attribute is a NUL-terminated
   * character array with the length equal to the value of
   * ::HSA_CACHE_INFO_NAME_LENGTH attribute.
   */
  HSA_CACHE_INFO_NAME = 1,
  /**
   * Cache level. A L1 cache must return a value of 1, a L2 must return a value
   * of 2, and so on.  The type of this attribute is uint8_t.
   */
  HSA_CACHE_INFO_LEVEL = 2,
  /**
   * Cache size, in bytes. A value of 0 indicates that there is no size
   * information available. The type of this attribute is uint32_t.
   */
  HSA_CACHE_INFO_SIZE = 3
} hsa_cache_info_t;

/**
 * @brief Get the current value of an attribute for a given cache object.
 *
 * @param[in] cache Cache.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CACHE The cache is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * instruction set architecture attribute, or @p value is
 * NULL.
 */
hsa_status_t HSA_API hsa_cache_get_info(
    hsa_cache_t cache,
    hsa_cache_info_t attribute,
    void* value);

/**
 * @brief Iterate over the memory caches of a given agent, and
 * invoke an application-defined callback on every iteration.
 *
 * @details Caches are visited in ascending order according to the value of the
 * ::HSA_CACHE_INFO_LEVEL attribute.
 *
 * @param[in] agent A valid agent.
 *
 * @param[in] callback Callback to be invoked once per cache that is present in
 * the agent.  The HSA runtime passes two arguments to the callback: the cache
 * and the application data.  If @p callback returns a status other than
 * ::HSA_STATUS_SUCCESS for a particular iteration, the traversal stops and
 * that value is returned.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL.
 */
hsa_status_t HSA_API hsa_agent_iterate_caches(
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_cache_t cache, void* data),
    void* data);

/**
 * @deprecated
 *
 * @brief Query if a given version of an extension is supported by an agent
 *
 * @param[in] extension Extension identifier.
 *
 * @param[in] agent Agent.
 *
 * @param[in] version_major Major version number.
 *
 * @param[in] version_minor Minor version number.
 *
 * @param[out] result Pointer to a memory location where the HSA runtime stores
 * the result of the check. The result is true if the specified version of the
 * extension is supported, and false otherwise. The result must be false if
 * ::hsa_system_extension_supported returns false for the same extension
 * version.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p extension is not a valid
 * extension, or @p result is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_agent_extension_supported(
    uint16_t extension,
    hsa_agent_t agent,
    uint16_t version_major,
    uint16_t version_minor,
    bool* result);

/**
 * @brief Query if a given version of an extension is supported by an agent. All
 * minor versions from 0 up to the returned @p version_minor must be supported.
 *
 * @param[in] extension Extension identifier.
 *
 * @param[in] agent Agent.
 *
 * @param[in] version_major Major version number.
 *
 * @param[out] version_minor Minor version number.
 *
 * @param[out] result Pointer to a memory location where the HSA runtime stores
 * the result of the check. The result is true if the specified version of the
 * extension is supported, and false otherwise. The result must be false if
 * ::hsa_system_extension_supported returns false for the same extension
 * version.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p extension is not a valid
 * extension, or @p version_minor is NULL, or @p result is NULL.
 */
hsa_status_t HSA_API hsa_agent_major_extension_supported(
    uint16_t extension,
    hsa_agent_t agent,
    uint16_t version_major,
    uint16_t *version_minor,
    bool* result);


/** @} */


/** \defgroup signals Signals
 *  @{
 */

/**
 * @brief Signal handle.
 */
typedef struct hsa_signal_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal. The value 0 is reserved.
   */
  uint64_t handle;
} hsa_signal_t;

/**
 * @brief Signal value. The value occupies 32 bits in small machine mode, and 64
 * bits in large machine mode.
 */
#ifdef HSA_LARGE_MODEL
  typedef int64_t hsa_signal_value_t;
#else
  typedef int32_t hsa_signal_value_t;
#endif

/**
 * @brief Create a signal.
 *
 * @param[in] initial_value Initial value of the signal.
 *
 * @param[in] num_consumers Size of @p consumers. A value of 0 indicates that
 * any agent might wait on the signal.
 *
 * @param[in] consumers List of agents that might consume (wait on) the
 * signal. If @p num_consumers is 0, this argument is ignored; otherwise, the
 * HSA runtime might use the list to optimize the handling of the signal
 * object. If an agent not listed in @p consumers waits on the returned
 * signal, the behavior is undefined. The memory associated with @p consumers
 * can be reused or freed after the function returns.
 *
 * @param[out] signal Pointer to a memory location where the HSA runtime will
 * store the newly created signal handle. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p signal is NULL, @p
 * num_consumers is greater than 0 but @p consumers is NULL, or @p consumers
 * contains duplicates.
 */
hsa_status_t HSA_API hsa_signal_create(
    hsa_signal_value_t initial_value,
    uint32_t num_consumers,
    const hsa_agent_t *consumers,
    hsa_signal_t *signal);

/**
 * @brief Destroy a signal previous created by ::hsa_signal_create.
 *
 * @param[in] signal Signal.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SIGNAL @p signal is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT The handle in @p signal is 0.
 */
hsa_status_t HSA_API hsa_signal_destroy(
    hsa_signal_t signal);

/**
 * @brief Atomically read the current value of a signal.
 *
 * @param[in] signal Signal.
 *
 * @return Value of the signal.
*/
hsa_signal_value_t HSA_API hsa_signal_load_scacquire(
    hsa_signal_t signal);

/**
 * @copydoc hsa_signal_load_scacquire
 */
hsa_signal_value_t HSA_API hsa_signal_load_relaxed(
    hsa_signal_t signal);

/**
 * @deprecated Renamed as ::hsa_signal_load_scacquire.
 *
 * @copydoc hsa_signal_load_scacquire
*/
hsa_signal_value_t HSA_API HSA_DEPRECATED hsa_signal_load_acquire(
    hsa_signal_t signal);

/**
 * @brief Atomically set the value of a signal.
 *
 * @details If the value of the signal is changed, all the agents waiting
 * on @p signal for which @p value satisfies their wait condition are awakened.
 *
 * @param[in] signal Signal.
 *
 * @param[in] value New signal value.
 */
void HSA_API hsa_signal_store_relaxed(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_store_relaxed
 */
void HSA_API hsa_signal_store_screlease(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_store_screlease.
 *
 * @copydoc hsa_signal_store_screlease
 */
void HSA_API HSA_DEPRECATED hsa_signal_store_release(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @brief Atomically set the value of a signal without necessarily notifying the
 * the agents waiting on it.
 *
 * @details The agents waiting on @p signal may not wake up even when the new
 * value satisfies their wait condition. If the application wants to update the
 * signal and there is no need to notify any agent, invoking this function can
 * be more efficient than calling the non-silent counterpart.
 *
 * @param[in] signal Signal.
 *
 * @param[in] value New signal value.
 */
void HSA_API hsa_signal_silent_store_relaxed(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_silent_store_relaxed
 */
void HSA_API hsa_signal_silent_store_screlease(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @brief Atomically set the value of a signal and return its previous value.
 *
 * @details If the value of the signal is changed, all the agents waiting
 * on @p signal for which @p value satisfies their wait condition are awakened.
 *
 * @param[in] signal Signal. If @p signal is a queue doorbell signal, the
 * behavior is undefined.
 *
 * @param[in] value New value.
 *
 * @return Value of the signal prior to the exchange.
 *
 */
hsa_signal_value_t HSA_API hsa_signal_exchange_scacq_screl(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_exchange_scacq_screl.
 *
 * @copydoc hsa_signal_exchange_scacq_screl
 */
hsa_signal_value_t HSA_API HSA_DEPRECATED hsa_signal_exchange_acq_rel(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_exchange_scacq_screl
 */
hsa_signal_value_t HSA_API hsa_signal_exchange_scacquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_exchange_scacquire.
 *
 * @copydoc hsa_signal_exchange_scacquire
 */
hsa_signal_value_t HSA_API HSA_DEPRECATED hsa_signal_exchange_acquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_exchange_scacq_screl
 */
hsa_signal_value_t HSA_API hsa_signal_exchange_relaxed(
    hsa_signal_t signal,
    hsa_signal_value_t value);
/**
 * @copydoc hsa_signal_exchange_scacq_screl
 */
hsa_signal_value_t HSA_API hsa_signal_exchange_screlease(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_exchange_screlease.
 *
 * @copydoc hsa_signal_exchange_screlease
 */
hsa_signal_value_t HSA_API HSA_DEPRECATED hsa_signal_exchange_release(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @brief Atomically set the value of a signal if the observed value is equal to
 * the expected value. The observed value is returned regardless of whether the
 * replacement was done.
 *
 * @details If the value of the signal is changed, all the agents waiting
 * on @p signal for which @p value satisfies their wait condition are awakened.
 *
 * @param[in] signal Signal. If @p signal is a queue
 * doorbell signal, the behavior is undefined.
 *
 * @param[in] expected Value to compare with.
 *
 * @param[in] value New value.
 *
 * @return Observed value of the signal.
 *
 */
hsa_signal_value_t HSA_API hsa_signal_cas_scacq_screl(
    hsa_signal_t signal,
    hsa_signal_value_t expected,
    hsa_signal_value_t value);


/**
 * @deprecated Renamed as ::hsa_signal_cas_scacq_screl.
 *
 * @copydoc hsa_signal_cas_scacq_screl
 */
hsa_signal_value_t HSA_API HSA_DEPRECATED hsa_signal_cas_acq_rel(
    hsa_signal_t signal,
    hsa_signal_value_t expected,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_cas_scacq_screl
 */
hsa_signal_value_t HSA_API hsa_signal_cas_scacquire(
    hsa_signal_t signal,
    hsa_signal_value_t expected,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_cas_scacquire.
 *
 * @copydoc hsa_signal_cas_scacquire
 */
hsa_signal_value_t HSA_API HSA_DEPRECATED hsa_signal_cas_acquire(
    hsa_signal_t signal,
    hsa_signal_value_t expected,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_cas_scacq_screl
 */
hsa_signal_value_t HSA_API hsa_signal_cas_relaxed(
    hsa_signal_t signal,
    hsa_signal_value_t expected,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_cas_scacq_screl
 */
hsa_signal_value_t HSA_API hsa_signal_cas_screlease(
    hsa_signal_t signal,
    hsa_signal_value_t expected,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_cas_screlease.
 *
 * @copydoc hsa_signal_cas_screlease
 */
hsa_signal_value_t HSA_API HSA_DEPRECATED hsa_signal_cas_release(
    hsa_signal_t signal,
    hsa_signal_value_t expected,
    hsa_signal_value_t value);

/**
 * @brief Atomically increment the value of a signal by a given amount.
 *
 * @details If the value of the signal is changed, all the agents waiting on
 * @p signal for which @p value satisfies their wait condition are awakened.
 *
 * @param[in] signal Signal. If @p signal is a queue doorbell signal, the
 * behavior is undefined.
 *
 * @param[in] value Value to add to the value of the signal.
 *
 */
void HSA_API hsa_signal_add_scacq_screl(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_add_scacq_screl.
 *
 * @copydoc hsa_signal_add_scacq_screl
 */
void HSA_API HSA_DEPRECATED hsa_signal_add_acq_rel(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_add_scacq_screl
 */
void HSA_API hsa_signal_add_scacquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_add_scacquire.
 *
 * @copydoc hsa_signal_add_scacquire
 */
void HSA_API HSA_DEPRECATED hsa_signal_add_acquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_add_scacq_screl
 */
void HSA_API hsa_signal_add_relaxed(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_add_scacq_screl
 */
void HSA_API hsa_signal_add_screlease(
    hsa_signal_t signal,
    hsa_signal_value_t value);


/**
 * @deprecated Renamed as ::hsa_signal_add_screlease.
 *
 * @copydoc hsa_signal_add_screlease
 */
void HSA_API HSA_DEPRECATED hsa_signal_add_release(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @brief Atomically decrement the value of a signal by a given amount.
 *
 * @details If the value of the signal is changed, all the agents waiting on
 * @p signal for which @p value satisfies their wait condition are awakened.
 *
 * @param[in] signal Signal. If @p signal is a queue doorbell signal, the
 * behavior is undefined.
 *
 * @param[in] value Value to subtract from the value of the signal.
 *
 */
void HSA_API hsa_signal_subtract_scacq_screl(
    hsa_signal_t signal,
    hsa_signal_value_t value);


/**
 * @deprecated Renamed as ::hsa_signal_subtract_scacq_screl.
 *
 * @copydoc hsa_signal_subtract_scacq_screl
 */
void HSA_API HSA_DEPRECATED hsa_signal_subtract_acq_rel(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_subtract_scacq_screl
 */
void HSA_API hsa_signal_subtract_scacquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_subtract_scacquire.
 *
 * @copydoc hsa_signal_subtract_scacquire
 */
void HSA_API HSA_DEPRECATED hsa_signal_subtract_acquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_subtract_scacq_screl
 */
void HSA_API hsa_signal_subtract_relaxed(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_subtract_scacq_screl
 */
void HSA_API hsa_signal_subtract_screlease(
    hsa_signal_t signal,
    hsa_signal_value_t value);


/**
 * @deprecated Renamed as ::hsa_signal_subtract_screlease.
 *
 * @copydoc hsa_signal_subtract_screlease
 */
void HSA_API HSA_DEPRECATED hsa_signal_subtract_release(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @brief Atomically perform a bitwise AND operation between the value of a
 * signal and a given value.
 *
 * @details If the value of the signal is changed, all the agents waiting on
 * @p signal for which @p value satisfies their wait condition are awakened.
 *
 * @param[in] signal Signal. If @p signal is a queue doorbell signal, the
 * behavior is undefined.
 *
 * @param[in] value Value to AND with the value of the signal.
 *
 */
void HSA_API hsa_signal_and_scacq_screl(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_and_scacq_screl.
 *
 * @copydoc hsa_signal_and_scacq_screl
 */
void HSA_API HSA_DEPRECATED hsa_signal_and_acq_rel(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_and_scacq_screl
 */
void HSA_API hsa_signal_and_scacquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_and_scacquire.
 *
 * @copydoc hsa_signal_and_scacquire
 */
void HSA_API HSA_DEPRECATED hsa_signal_and_acquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_and_scacq_screl
 */
void HSA_API hsa_signal_and_relaxed(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_and_scacq_screl
 */
void HSA_API hsa_signal_and_screlease(
    hsa_signal_t signal,
    hsa_signal_value_t value);


/**
 * @deprecated Renamed as ::hsa_signal_and_screlease.
 *
 * @copydoc hsa_signal_and_screlease
 */
void HSA_API HSA_DEPRECATED hsa_signal_and_release(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @brief Atomically perform a bitwise OR operation between the value of a
 * signal and a given value.
 *
 * @details If the value of the signal is changed, all the agents waiting on
 * @p signal for which @p value satisfies their wait condition are awakened.
 *
 * @param[in] signal Signal. If @p signal is a queue doorbell signal, the
 * behavior is undefined.
 *
 * @param[in] value Value to OR with the value of the signal.
 */
void HSA_API hsa_signal_or_scacq_screl(
    hsa_signal_t signal,
    hsa_signal_value_t value);


/**
 * @deprecated Renamed as ::hsa_signal_or_scacq_screl.
 *
 * @copydoc hsa_signal_or_scacq_screl
 */
void HSA_API HSA_DEPRECATED hsa_signal_or_acq_rel(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_or_scacq_screl
 */
void HSA_API hsa_signal_or_scacquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_or_scacquire.
 *
 * @copydoc hsa_signal_or_scacquire
 */
void HSA_API HSA_DEPRECATED hsa_signal_or_acquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_or_scacq_screl
 */
void HSA_API hsa_signal_or_relaxed(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_or_scacq_screl
 */
void HSA_API hsa_signal_or_screlease(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_or_screlease.
 *
 * @copydoc hsa_signal_or_screlease
 */
void HSA_API HSA_DEPRECATED hsa_signal_or_release(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @brief Atomically perform a bitwise XOR operation between the value of a
 * signal and a given value.
 *
 * @details If the value of the signal is changed, all the agents waiting on
 * @p signal for which @p value satisfies their wait condition are awakened.
 *
 * @param[in] signal Signal. If @p signal is a queue doorbell signal, the
 * behavior is undefined.
 *
 * @param[in] value Value to XOR with the value of the signal.
 *
 */
void HSA_API hsa_signal_xor_scacq_screl(
    hsa_signal_t signal,
    hsa_signal_value_t value);


/**
 * @deprecated Renamed as ::hsa_signal_xor_scacq_screl.
 *
 * @copydoc hsa_signal_xor_scacq_screl
 */
void HSA_API HSA_DEPRECATED hsa_signal_xor_acq_rel(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_xor_scacq_screl
 */
void HSA_API hsa_signal_xor_scacquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_xor_scacquire.
 *
 * @copydoc hsa_signal_xor_scacquire
 */
void HSA_API HSA_DEPRECATED hsa_signal_xor_acquire(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_xor_scacq_screl
 */
void HSA_API hsa_signal_xor_relaxed(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @copydoc hsa_signal_xor_scacq_screl
 */
void HSA_API hsa_signal_xor_screlease(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @deprecated Renamed as ::hsa_signal_xor_screlease.
 *
 * @copydoc hsa_signal_xor_screlease
 */
void HSA_API HSA_DEPRECATED hsa_signal_xor_release(
    hsa_signal_t signal,
    hsa_signal_value_t value);

/**
 * @brief Wait condition operator.
 */
typedef enum {
    /**
     * The two operands are equal.
     */
    HSA_SIGNAL_CONDITION_EQ = 0,
    /**
     * The two operands are not equal.
     */
    HSA_SIGNAL_CONDITION_NE = 1,
    /**
     * The first operand is less than the second operand.
     */
    HSA_SIGNAL_CONDITION_LT = 2,
    /**
     * The first operand is greater than or equal to the second operand.
     */
    HSA_SIGNAL_CONDITION_GTE = 3
} hsa_signal_condition_t;

/**
 * @brief State of the application thread during a signal wait.
 */
typedef enum {
    /**
     * The application thread may be rescheduled while waiting on the signal.
     */
    HSA_WAIT_STATE_BLOCKED = 0,
    /**
     * The application thread stays active while waiting on a signal.
     */
    HSA_WAIT_STATE_ACTIVE = 1
} hsa_wait_state_t;


/**
 * @brief Wait until a signal value satisfies a specified condition, or a
 * certain amount of time has elapsed.
 *
 * @details A wait operation can spuriously resume at any time sooner than the
 * timeout (for example, due to system or other external factors) even when the
 * condition has not been met.
 *
 * The function is guaranteed to return if the signal value satisfies the
 * condition at some point in time during the wait, but the value returned to
 * the application might not satisfy the condition. The application must ensure
 * that signals are used in such way that wait wakeup conditions are not
 * invalidated before dependent threads have woken up.
 *
 * When the wait operation internally loads the value of the passed signal, it
 * uses the memory order indicated in the function name.
 *
 * @param[in] signal Signal.
 *
 * @param[in] condition Condition used to compare the signal value with @p
 * compare_value.
 *
 * @param[in] compare_value Value to compare with.
 *
 * @param[in] timeout_hint Maximum duration of the wait.  Specified in the same
 * unit as the system timestamp. The operation might block for a shorter or
 * longer time even if the condition is not met. A value of UINT64_MAX indicates
 * no maximum.
 *
 * @param[in] wait_state_hint Hint used by the application to indicate the
 * preferred waiting state. The actual waiting state is ultimately decided by
 * HSA runtime and may not match the provided hint. A value of
 * ::HSA_WAIT_STATE_ACTIVE may improve the latency of response to a signal
 * update by avoiding rescheduling overhead.
 *
 * @return Observed value of the signal, which might not satisfy the specified
 * condition.
 *
*/
hsa_signal_value_t HSA_API hsa_signal_wait_scacquire(
    hsa_signal_t signal,
    hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value,
    uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint);

/**
 * @copydoc hsa_signal_wait_scacquire
 */
hsa_signal_value_t HSA_API hsa_signal_wait_relaxed(
    hsa_signal_t signal,
    hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value,
    uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint);

/**
 * @deprecated Renamed as ::hsa_signal_wait_scacquire.
 *
 * @copydoc hsa_signal_wait_scacquire
 */
hsa_signal_value_t HSA_API HSA_DEPRECATED hsa_signal_wait_acquire(
    hsa_signal_t signal,
    hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value,
    uint64_t timeout_hint,
    hsa_wait_state_t wait_state_hint);

/**
 * @brief Group of signals.
 */
typedef struct hsa_signal_group_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_signal_group_t;

/**
 * @brief Create a signal group.
 *
 * @param[in] num_signals Number of elements in @p signals. Must not be 0.
 *
 * @param[in] signals List of signals in the group. The list must not contain
 * any repeated elements. Must not be NULL.
 *
 * @param[in] num_consumers Number of elements in @p consumers. Must not be 0.
 *
 * @param[in] consumers List of agents that might consume (wait on) the signal
 * group. The list must not contain repeated elements, and must be a subset of
 * the set of agents that are allowed to wait on all the signals in the
 * group. If an agent not listed in @p consumers waits on the returned group,
 * the behavior is undefined. The memory associated with @p consumers can be
 * reused or freed after the function returns. Must not be NULL.
 *
 * @param[out] signal_group Pointer to newly created signal group. Must not be
 * NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p num_signals is 0, @p signals
 * is NULL, @p num_consumers is 0, @p consumers is NULL, or @p signal_group is
 * NULL.
 */
hsa_status_t HSA_API hsa_signal_group_create(
    uint32_t num_signals,
    const hsa_signal_t *signals,
    uint32_t num_consumers,
    const hsa_agent_t *consumers,
    hsa_signal_group_t *signal_group);

/**
 * @brief Destroy a signal group previous created by ::hsa_signal_group_create.
 *
 * @param[in] signal_group Signal group.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP @p signal_group is invalid.
 */
hsa_status_t HSA_API hsa_signal_group_destroy(
    hsa_signal_group_t signal_group);

/**
 * @brief Wait until the value of at least one of the signals in a signal group
 * satisfies its associated condition.
 *
 * @details The function is guaranteed to return if the value of at least one of
 * the signals in the group satisfies its associated condition at some point in
 * time during the wait, but the signal value returned to the application may no
 * longer satisfy the condition. The application must ensure that signals in the
 * group are used in such way that wait wakeup conditions are not invalidated
 * before dependent threads have woken up.
 *
 * When this operation internally loads the value of the passed signal, it uses
 * the memory order indicated in the function name.
 *
 * @param[in] signal_group Signal group.
 *
 * @param[in] conditions List of conditions. Each condition, and the value at
 * the same index in @p compare_values, is used to compare the value of the
 * signal at that index in @p signal_group (the signal passed by the application
 * to ::hsa_signal_group_create at that particular index). The size of @p
 * conditions must not be smaller than the number of signals in @p signal_group;
 * any extra elements are ignored. Must not be NULL.
 *
 * @param[in] compare_values List of comparison values.  The size of @p
 * compare_values must not be smaller than the number of signals in @p
 * signal_group; any extra elements are ignored. Must not be NULL.
 *
 * @param[in] wait_state_hint Hint used by the application to indicate the
 * preferred waiting state. The actual waiting state is decided by the HSA runtime
 * and may not match the provided hint. A value of ::HSA_WAIT_STATE_ACTIVE may
 * improve the latency of response to a signal update by avoiding rescheduling
 * overhead.
 *
 * @param[out] signal Signal in the group that satisfied the associated
 * condition. If several signals satisfied their condition, the function can
 * return any of those signals. Must not be NULL.
 *
 * @param[out] value Observed value for @p signal, which might no longer satisfy
 * the specified condition. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP @p signal_group is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p conditions is NULL, @p
 * compare_values is NULL, @p signal is NULL, or @p value is NULL.
 */
hsa_status_t HSA_API hsa_signal_group_wait_any_scacquire(
    hsa_signal_group_t signal_group,
    const hsa_signal_condition_t *conditions,
    const hsa_signal_value_t *compare_values,
    hsa_wait_state_t wait_state_hint,
    hsa_signal_t *signal,
    hsa_signal_value_t *value);

/**
 * @copydoc hsa_signal_group_wait_any_scacquire
 */
hsa_status_t HSA_API hsa_signal_group_wait_any_relaxed(
    hsa_signal_group_t signal_group,
    const hsa_signal_condition_t *conditions,
    const hsa_signal_value_t *compare_values,
    hsa_wait_state_t wait_state_hint,
    hsa_signal_t *signal,
    hsa_signal_value_t *value);

/** @} */

/** \defgroup memory Memory
 *  @{
 */

/**
 * @brief A memory region represents a block of virtual memory with certain
 * properties. For example, the HSA runtime represents fine-grained memory in
 * the global segment using a region. A region might be associated with more
 * than one agent.
 */
typedef struct hsa_region_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_region_t;

/** @} */


/** \defgroup queue Queues
 *  @{
 */

/**
 * @brief Queue type. Intended to be used for dynamic queue protocol
 * determination.
 */
typedef enum {
  /**
   * Queue supports multiple producers. Use of multiproducer queue mechanics is
   * required.
   */
  HSA_QUEUE_TYPE_MULTI = 0,
  /**
   * Queue only supports a single producer. In some scenarios, the application
   * may want to limit the submission of AQL packets to a single agent. Queues
   * that support a single producer may be more efficient than queues supporting
   * multiple producers. Use of multiproducer queue mechanics is not supported.
   */
  HSA_QUEUE_TYPE_SINGLE = 1,
  /**
   * Queue supports multiple producers and cooperative dispatches. Cooperative
   * dispatches are able to use GWS synchronization. Queues of this type may be
   * limited in number. The runtime may return the same queue to serve multiple
   * ::hsa_queue_create calls when this type is given. Callers must inspect the
   * returned queue to discover queue size. Queues of this type are reference
   * counted and require a matching number of ::hsa_queue_destroy calls to
   * release. Use of multiproducer queue mechanics is required. See
   * ::HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES to query agent support for this
   * type.
   */
  HSA_QUEUE_TYPE_COOPERATIVE = 2
} hsa_queue_type_t;

/**
 * @brief A fixed-size type used to represent ::hsa_queue_type_t constants.
 */
typedef uint32_t hsa_queue_type32_t;

/**
 * @brief Queue features.
 */
typedef enum {
  /**
   * Queue supports kernel dispatch packets.
   */
  HSA_QUEUE_FEATURE_KERNEL_DISPATCH = 1,

  /**
   * Queue supports agent dispatch packets.
   */
  HSA_QUEUE_FEATURE_AGENT_DISPATCH = 2
} hsa_queue_feature_t;

/**
 * @brief User mode queue.
 *
 * @details The queue structure is read-only and allocated by the HSA runtime,
 * but agents can directly modify the contents of the buffer pointed by @a
 * base_address, or use HSA runtime APIs to access the doorbell signal.
 *
 */
typedef struct hsa_queue_s {
  /**
   * Queue type.
   */
  hsa_queue_type32_t type;

  /**
   * Queue features mask. This is a bit-field of ::hsa_queue_feature_t
   * values. Applications should ignore any unknown set bits.
   */
  uint32_t features;

#ifdef HSA_LARGE_MODEL
  void* base_address;
#elif defined HSA_LITTLE_ENDIAN
  /**
   * Starting address of the HSA runtime-allocated buffer used to store the AQL
   * packets. Must be aligned to the size of an AQL packet.
   */
  void* base_address;
  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved0;
#else
  uint32_t reserved0;
  void* base_address;
#endif

  /**
   * Signal object used by the application to indicate the ID of a packet that
   * is ready to be processed. The HSA runtime manages the doorbell signal. If
   * the application tries to replace or destroy this signal, the behavior is
   * undefined.
   *
   * If @a type is ::HSA_QUEUE_TYPE_SINGLE, the doorbell signal value must be
   * updated in a monotonically increasing fashion. If @a type is
   * ::HSA_QUEUE_TYPE_MULTI, the doorbell signal value can be updated with any
   * value.
   */
  hsa_signal_t doorbell_signal;

  /**
   * Maximum number of packets the queue can hold. Must be a power of 2.
   */
  uint32_t size;
  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;
  /**
   * Queue identifier, which is unique over the lifetime of the application.
   */
  uint64_t id;

} hsa_queue_t;

/**
 * @brief Create a user mode queue.
 *
 * @details The HSA runtime creates the queue structure, the underlying packet
 * buffer, the completion signal, and the write and read indexes. The initial
 * value of the write and read indexes is 0. The type of every packet in the
 * buffer is initialized to ::HSA_PACKET_TYPE_INVALID.
 *
 * The application should only rely on the error code returned to determine if
 * the queue is valid.
 *
 * @param[in] agent Agent where to create the queue.
 *
 * @param[in] size Number of packets the queue is expected to
 * hold. Must be a power of 2 between 1 and the value of
 * ::HSA_AGENT_INFO_QUEUE_MAX_SIZE in @p agent. The size of the newly
 * created queue is the maximum of @p size and the value of
 * ::HSA_AGENT_INFO_QUEUE_MIN_SIZE in @p agent.
 *
 * @param[in] type Type of the queue, a bitwise OR of hsa_queue_type_t values.
 * If the value of ::HSA_AGENT_INFO_QUEUE_TYPE in @p agent is ::HSA_QUEUE_TYPE_SINGLE,
 * then @p type must also be ::HSA_QUEUE_TYPE_SINGLE.
 *
 * @param[in] callback Callback invoked by the HSA runtime for every
 * asynchronous event related to the newly created queue. May be NULL. The HSA
 * runtime passes three arguments to the callback: a code identifying the event
 * that triggered the invocation, a pointer to the queue where the event
 * originated, and the application data.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @param[in] private_segment_size Hint indicating the maximum
 * expected private segment usage per work-item, in bytes. There may
 * be performance degradation if the application places a kernel
 * dispatch packet in the queue and the corresponding private segment
 * usage exceeds @p private_segment_size. If the application does not
 * want to specify any particular value for this argument, @p
 * private_segment_size must be UINT32_MAX. If the queue does not
 * support kernel dispatch packets, this argument is ignored.
 *
 * @param[in] group_segment_size Hint indicating the maximum expected
 * group segment usage per work-group, in bytes. There may be
 * performance degradation if the application places a kernel dispatch
 * packet in the queue and the corresponding group segment usage
 * exceeds @p group_segment_size. If the application does not want to
 * specify any particular value for this argument, @p
 * group_segment_size must be UINT32_MAX. If the queue does not
 * support kernel dispatch packets, this argument is ignored.
 *
 * @param[out] queue Memory location where the HSA runtime stores a pointer to
 * the newly created queue.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_QUEUE_CREATION @p agent does not
 * support queues of the given type.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p size is not a power of two,
 * @p size is 0, @p type is an invalid queue type, or @p queue is NULL.
 *
 */
hsa_status_t HSA_API hsa_queue_create(
    hsa_agent_t agent,
    uint32_t size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t *source, void *data),
    void *data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t **queue);

/**
 * @brief Create a queue for which the application or a kernel is responsible
 * for processing the AQL packets.
 *
 * @details The application can use this function to create queues where AQL
 * packets are not parsed by the packet processor associated with an agent,
 * but rather by a unit of execution running on that agent (for example, a
 * thread in the host application).
 *
 * The application is responsible for ensuring that all the producers and
 * consumers of the resulting queue can access the provided doorbell signal
 * and memory region. The application is also responsible for ensuring that the
 * unit of execution processing the queue packets supports the indicated
 * features (AQL packet types).
 *
 * When the queue is created, the HSA runtime allocates the packet buffer using
 * @p region, and the write and read indexes. The initial value of the write and
 * read indexes is 0, and the type of every packet in the buffer is initialized
 * to ::HSA_PACKET_TYPE_INVALID. The value of the @e size, @e type, @e features,
 * and @e doorbell_signal fields in the returned queue match the values passed
 * by the application.
 *
 * @param[in] region Memory region that the HSA runtime should use to allocate
 * the AQL packet buffer and any other queue metadata.
 *
 * @param[in] size Number of packets the queue is expected to hold. Must be a
 * power of 2 greater than 0.
 *
 * @param[in] type Queue type.
 *
 * @param[in] features Supported queue features. This is a bit-field of
 * ::hsa_queue_feature_t values.
 *
 * @param[in] doorbell_signal Doorbell signal that the HSA runtime must
 * associate with the returned queue. The signal handle must not be 0.
 *
 * @param[out] queue Memory location where the HSA runtime stores a pointer to
 * the newly created queue. The application should not rely on the value
 * returned for this argument but only in the status code to determine if the
 * queue is valid. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p size is not a power of two, @p
 * size is 0, @p type is an invalid queue type, the doorbell signal handle is
 * 0, or @p queue is NULL.
 *
 */
hsa_status_t HSA_API hsa_soft_queue_create(
    hsa_region_t region,
    uint32_t size,
    hsa_queue_type32_t type,
    uint32_t features,
    hsa_signal_t doorbell_signal,
    hsa_queue_t **queue);

/**
 * @brief Destroy a user mode queue.
 *
 * @details When a queue is destroyed, the state of the AQL packets that have
 * not been yet fully processed (their completion phase has not finished)
 * becomes undefined. It is the responsibility of the application to ensure that
 * all pending queue operations are finished if their results are required.
 *
 * The resources allocated by the HSA runtime during queue creation (queue
 * structure, ring buffer, doorbell signal) are released.  The queue should not
 * be accessed after being destroyed.
 *
 * @param[in] queue Pointer to a queue created using ::hsa_queue_create.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_QUEUE The queue is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p queue is NULL.
 */
hsa_status_t HSA_API hsa_queue_destroy(
    hsa_queue_t *queue);

/**
 * @brief Inactivate a queue.
 *
 * @details Inactivating the queue aborts any pending executions and prevent any
 * new packets from being processed. Any more packets written to the queue once
 * it is inactivated will be ignored by the packet processor.
 *
 * @param[in] queue Pointer to a queue.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_QUEUE The queue is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p queue is NULL.
 */
hsa_status_t HSA_API hsa_queue_inactivate(
    hsa_queue_t *queue);

/**
 * @deprecated Renamed as ::hsa_queue_load_read_index_scacquire.
 *
 * @copydoc hsa_queue_load_read_index_scacquire
 */
uint64_t HSA_API HSA_DEPRECATED hsa_queue_load_read_index_acquire(
    const hsa_queue_t *queue);

/**
 * @brief Atomically load the read index of a queue.
 *
 * @param[in] queue Pointer to a queue.
 *
 * @return Read index of the queue pointed by @p queue.
 */
uint64_t HSA_API hsa_queue_load_read_index_scacquire(
    const hsa_queue_t *queue);

/**
 * @copydoc hsa_queue_load_read_index_scacquire
 */
uint64_t HSA_API hsa_queue_load_read_index_relaxed(
    const hsa_queue_t *queue);

/**
 * @deprecated Renamed as ::hsa_queue_load_write_index_scacquire.
 *
 * @copydoc hsa_queue_load_write_index_scacquire
 */
uint64_t HSA_API HSA_DEPRECATED hsa_queue_load_write_index_acquire(
    const hsa_queue_t *queue);

/**
 * @brief Atomically load the write index of a queue.
 *
 * @param[in] queue Pointer to a queue.
 *
 * @return Write index of the queue pointed by @p queue.
 */
uint64_t HSA_API hsa_queue_load_write_index_scacquire(
    const hsa_queue_t *queue);

/**
 * @copydoc hsa_queue_load_write_index_scacquire
 */
uint64_t HSA_API hsa_queue_load_write_index_relaxed(
    const hsa_queue_t *queue);

/**
 * @brief Atomically set the write index of a queue.
 *
 * @details It is recommended that the application uses this function to update
 * the write index when there is a single agent submitting work to the queue
 * (the queue type is ::HSA_QUEUE_TYPE_SINGLE).
 *
 * @param[in] queue Pointer to a queue.
 *
 * @param[in] value Value to assign to the write index.
 *
 */
void HSA_API hsa_queue_store_write_index_relaxed(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @deprecated Renamed as ::hsa_queue_store_write_index_screlease.
 *
 * @copydoc hsa_queue_store_write_index_screlease
 */
void HSA_API HSA_DEPRECATED hsa_queue_store_write_index_release(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @copydoc hsa_queue_store_write_index_relaxed
 */
void HSA_API hsa_queue_store_write_index_screlease(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @deprecated Renamed as ::hsa_queue_cas_write_index_scacq_screl.
 *
 * @copydoc hsa_queue_cas_write_index_scacq_screl
 */
uint64_t HSA_API HSA_DEPRECATED hsa_queue_cas_write_index_acq_rel(
    const hsa_queue_t *queue,
    uint64_t expected,
    uint64_t value);

/**
 * @brief Atomically set the write index of a queue if the observed value is
 * equal to the expected value. The application can inspect the returned value
 * to determine if the replacement was done.
 *
 * @param[in] queue Pointer to a queue.
 *
 * @param[in] expected Expected value.
 *
 * @param[in] value Value to assign to the write index if @p expected matches
 * the observed write index. Must be greater than @p expected.
 *
 * @return Previous value of the write index.
 */
uint64_t HSA_API hsa_queue_cas_write_index_scacq_screl(
    const hsa_queue_t *queue,
    uint64_t expected,
    uint64_t value);

/**
 * @deprecated Renamed as ::hsa_queue_cas_write_index_scacquire.
 *
 * @copydoc hsa_queue_cas_write_index_scacquire
 */
uint64_t HSA_API HSA_DEPRECATED hsa_queue_cas_write_index_acquire(
    const hsa_queue_t *queue,
    uint64_t expected,
    uint64_t value);

/**
 * @copydoc hsa_queue_cas_write_index_scacq_screl
 */
uint64_t HSA_API hsa_queue_cas_write_index_scacquire(
    const hsa_queue_t *queue,
    uint64_t expected,
    uint64_t value);

/**
 * @copydoc hsa_queue_cas_write_index_scacq_screl
 */
uint64_t HSA_API hsa_queue_cas_write_index_relaxed(
    const hsa_queue_t *queue,
    uint64_t expected,
    uint64_t value);

/**
 * @deprecated Renamed as ::hsa_queue_cas_write_index_screlease.
 *
 * @copydoc hsa_queue_cas_write_index_screlease
 */
uint64_t HSA_API HSA_DEPRECATED hsa_queue_cas_write_index_release(
    const hsa_queue_t *queue,
    uint64_t expected,
    uint64_t value);

/**
 * @copydoc hsa_queue_cas_write_index_scacq_screl
 */
uint64_t HSA_API hsa_queue_cas_write_index_screlease(
    const hsa_queue_t *queue,
    uint64_t expected,
    uint64_t value);

/**
 * @deprecated Renamed as ::hsa_queue_add_write_index_scacq_screl.
 *
 * @copydoc hsa_queue_add_write_index_scacq_screl
 */
uint64_t HSA_API HSA_DEPRECATED hsa_queue_add_write_index_acq_rel(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @brief Atomically increment the write index of a queue by an offset.
 *
 * @param[in] queue Pointer to a queue.
 *
 * @param[in] value Value to add to the write index.
 *
 * @return Previous value of the write index.
 */
uint64_t HSA_API hsa_queue_add_write_index_scacq_screl(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @deprecated Renamed as ::hsa_queue_add_write_index_scacquire.
 *
 * @copydoc hsa_queue_add_write_index_scacquire
 */
uint64_t HSA_API HSA_DEPRECATED hsa_queue_add_write_index_acquire(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @copydoc hsa_queue_add_write_index_scacq_screl
 */
uint64_t HSA_API hsa_queue_add_write_index_scacquire(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @copydoc hsa_queue_add_write_index_scacq_screl
 */
uint64_t HSA_API hsa_queue_add_write_index_relaxed(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @deprecated Renamed as ::hsa_queue_add_write_index_screlease.
 *
 * @copydoc hsa_queue_add_write_index_screlease
 */
uint64_t HSA_API HSA_DEPRECATED hsa_queue_add_write_index_release(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @copydoc hsa_queue_add_write_index_scacq_screl
 */
uint64_t HSA_API hsa_queue_add_write_index_screlease(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @brief Atomically set the read index of a queue.
 *
 * @details Modifications of the read index are not allowed and result in
 * undefined behavior if the queue is associated with an agent for which
 * only the corresponding packet processor is permitted to update the read
 * index.
 *
 * @param[in] queue Pointer to a queue.
 *
 * @param[in] value Value to assign to the read index.
 *
 */
void HSA_API hsa_queue_store_read_index_relaxed(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @deprecated Renamed as ::hsa_queue_store_read_index_screlease.
 *
 * @copydoc hsa_queue_store_read_index_screlease
 */
void HSA_API HSA_DEPRECATED hsa_queue_store_read_index_release(
    const hsa_queue_t *queue,
    uint64_t value);

/**
 * @copydoc hsa_queue_store_read_index_relaxed
 */
void HSA_API hsa_queue_store_read_index_screlease(
   const hsa_queue_t *queue,
   uint64_t value);
/** @} */


/** \defgroup aql Architected Queuing Language
 *  @{
 */

/**
 * @brief Packet type.
 */
typedef enum {
  /**
   * Vendor-specific packet.
   */
  HSA_PACKET_TYPE_VENDOR_SPECIFIC = 0,
  /**
   * The packet has been processed in the past, but has not been reassigned to
   * the packet processor. A packet processor must not process a packet of this
   * type. All queues support this packet type.
   */
  HSA_PACKET_TYPE_INVALID = 1,
  /**
   * Packet used by agents for dispatching jobs to kernel agents. Not all
   * queues support packets of this type (see ::hsa_queue_feature_t).
   */
  HSA_PACKET_TYPE_KERNEL_DISPATCH = 2,
  /**
   * Packet used by agents to delay processing of subsequent packets, and to
   * express complex dependencies between multiple packets. All queues support
   * this packet type.
   */
  HSA_PACKET_TYPE_BARRIER_AND = 3,
  /**
   * Packet used by agents for dispatching jobs to agents.  Not all
   * queues support packets of this type (see ::hsa_queue_feature_t).
   */
  HSA_PACKET_TYPE_AGENT_DISPATCH = 4,
  /**
   * Packet used by agents to delay processing of subsequent packets, and to
   * express complex dependencies between multiple packets. All queues support
   * this packet type.
   */
  HSA_PACKET_TYPE_BARRIER_OR = 5
} hsa_packet_type_t;

/**
 * @brief Scope of the memory fence operation associated with a packet.
 */
typedef enum {
  /**
   * No scope (no fence is applied). The packet relies on external fences to
   * ensure visibility of memory updates.
   */
  HSA_FENCE_SCOPE_NONE = 0,
  /**
   * The fence is applied with agent scope for the global segment.
   */
  HSA_FENCE_SCOPE_AGENT = 1,
  /**
   * The fence is applied across both agent and system scope for the global
   * segment.
   */
  HSA_FENCE_SCOPE_SYSTEM = 2
} hsa_fence_scope_t;

/**
 * @brief Sub-fields of the @a header field that is present in any AQL
 * packet. The offset (with respect to the address of @a header) of a sub-field
 * is identical to its enumeration constant. The width of each sub-field is
 * determined by the corresponding value in ::hsa_packet_header_width_t. The
 * offset and the width are expressed in bits.
 */
 typedef enum {
  /**
   * Packet type. The value of this sub-field must be one of
   * ::hsa_packet_type_t. If the type is ::HSA_PACKET_TYPE_VENDOR_SPECIFIC, the
   * packet layout is vendor-specific.
   */
   HSA_PACKET_HEADER_TYPE = 0,
  /**
   * Barrier bit. If the barrier bit is set, the processing of the current
   * packet only launches when all preceding packets (within the same queue) are
   * complete.
   */
   HSA_PACKET_HEADER_BARRIER = 8,
  /**
   * Acquire fence scope. The value of this sub-field determines the scope and
   * type of the memory fence operation applied before the packet enters the
   * active phase. An acquire fence ensures that any subsequent global segment
   * or image loads by any unit of execution that belongs to a dispatch that has
   * not yet entered the active phase on any queue of the same kernel agent,
   * sees any data previously released at the scopes specified by the acquire
   * fence. The value of this sub-field must be one of ::hsa_fence_scope_t.
   */
   HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = 9,
   /**
    * @deprecated Renamed as ::HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE.
    */
   HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE = 9,
  /**
   * Release fence scope, The value of this sub-field determines the scope and
   * type of the memory fence operation applied after kernel completion but
   * before the packet is completed. A release fence makes any global segment or
   * image data that was stored by any unit of execution that belonged to a
   * dispatch that has completed the active phase on any queue of the same
   * kernel agent visible in all the scopes specified by the release fence. The
   * value of this sub-field must be one of ::hsa_fence_scope_t.
   */
   HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = 11,
   /**
    * @deprecated Renamed as ::HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE.
    */
   HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE = 11
 } hsa_packet_header_t;

/**
 * @brief Width (in bits) of the sub-fields in ::hsa_packet_header_t.
 */
 typedef enum {
   HSA_PACKET_HEADER_WIDTH_TYPE = 8,
   HSA_PACKET_HEADER_WIDTH_BARRIER = 1,
   HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE = 2,
   /**
    * @deprecated Use HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE.
    */
   HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE = 2,
   HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE = 2,
   /**
    * @deprecated Use HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE.
    */
   HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE = 2
 } hsa_packet_header_width_t;

/**
 * @brief Sub-fields of the kernel dispatch packet @a setup field. The offset
 * (with respect to the address of @a setup) of a sub-field is identical to its
 * enumeration constant. The width of each sub-field is determined by the
 * corresponding value in ::hsa_kernel_dispatch_packet_setup_width_t. The
 * offset and the width are expressed in bits.
 */
 typedef enum {
  /**
   * Number of dimensions of the grid. Valid values are 1, 2, or 3.
   *
   */
   HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS = 0
 } hsa_kernel_dispatch_packet_setup_t;

/**
 * @brief Width (in bits) of the sub-fields in
 * ::hsa_kernel_dispatch_packet_setup_t.
 */
 typedef enum {
   HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS = 2
 } hsa_kernel_dispatch_packet_setup_width_t;

/**
 * @brief AQL kernel dispatch packet
 */
typedef struct hsa_kernel_dispatch_packet_s {
  /**
   * Packet header. Used to configure multiple packet parameters such as the
   * packet type. The parameters are described by ::hsa_packet_header_t.
   */
  uint16_t header;

  /**
   * Dispatch setup parameters. Used to configure kernel dispatch parameters
   * such as the number of dimensions in the grid. The parameters are described
   * by ::hsa_kernel_dispatch_packet_setup_t.
   */
  uint16_t setup;

  /**
   * X dimension of work-group, in work-items. Must be greater than 0.
   */
  uint16_t workgroup_size_x;

  /**
   * Y dimension of work-group, in work-items. Must be greater than
   * 0. If the grid has 1 dimension, the only valid value is 1.
   */
  uint16_t workgroup_size_y;

  /**
   * Z dimension of work-group, in work-items. Must be greater than
   * 0. If the grid has 1 or 2 dimensions, the only valid value is 1.
   */
  uint16_t workgroup_size_z;

  /**
   * Reserved. Must be 0.
   */
  uint16_t reserved0;

  /**
   * X dimension of grid, in work-items. Must be greater than 0. Must
   * not be smaller than @a workgroup_size_x.
   */
  uint32_t grid_size_x;

  /**
   * Y dimension of grid, in work-items. Must be greater than 0. If the grid has
   * 1 dimension, the only valid value is 1. Must not be smaller than @a
   * workgroup_size_y.
   */
  uint32_t grid_size_y;

  /**
   * Z dimension of grid, in work-items. Must be greater than 0. If the grid has
   * 1 or 2 dimensions, the only valid value is 1. Must not be smaller than @a
   * workgroup_size_z.
   */
  uint32_t grid_size_z;

  /**
   * Size in bytes of private memory allocation request (per work-item).
   */
  uint32_t private_segment_size;

  /**
   * Size in bytes of group memory allocation request (per work-group). Must not
   * be less than the sum of the group memory used by the kernel (and the
   * functions it calls directly or indirectly) and the dynamically allocated
   * group segment variables.
   */
  uint32_t group_segment_size;

  /**
   * Opaque handle to a code object that includes an implementation-defined
   * executable code for the kernel.
   */
  uint64_t kernel_object;

#ifdef HSA_LARGE_MODEL
  void* kernarg_address;
#elif defined HSA_LITTLE_ENDIAN
  /**
   * Pointer to a buffer containing the kernel arguments. May be NULL.
   *
   * The buffer must be allocated using ::hsa_memory_allocate, and must not be
   * modified once the kernel dispatch packet is enqueued until the dispatch has
   * completed execution.
   */
  void* kernarg_address;
  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;
#else
  uint32_t reserved1;
  void* kernarg_address;
#endif

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved2;

  /**
   * Signal used to indicate completion of the job. The application can use the
   * special signal handle 0 to indicate that no signal is used.
   */
  hsa_signal_t completion_signal;

} hsa_kernel_dispatch_packet_t;

/**
 * @brief Agent dispatch packet.
 */
typedef struct hsa_agent_dispatch_packet_s {
  /**
   * Packet header. Used to configure multiple packet parameters such as the
   * packet type. The parameters are described by ::hsa_packet_header_t.
   */
  uint16_t header;

  /**
   * Application-defined function to be performed by the destination agent.
   */
  uint16_t type;

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved0;

#ifdef HSA_LARGE_MODEL
  void* return_address;
#elif defined HSA_LITTLE_ENDIAN
  /**
   * Address where to store the function return values, if any.
   */
  void* return_address;
  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;
#else
  uint32_t reserved1;
  void* return_address;
#endif

  /**
   * Function arguments.
   */
  uint64_t arg[4];

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved2;

  /**
   * Signal used to indicate completion of the job. The application can use the
   * special signal handle 0 to indicate that no signal is used.
   */
  hsa_signal_t completion_signal;

} hsa_agent_dispatch_packet_t;

/**
 * @brief Barrier-AND packet.
 */
typedef struct hsa_barrier_and_packet_s {
  /**
   * Packet header. Used to configure multiple packet parameters such as the
   * packet type. The parameters are described by ::hsa_packet_header_t.
   */
  uint16_t header;

  /**
   * Reserved. Must be 0.
   */
  uint16_t reserved0;

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;

  /**
   * Array of dependent signal objects. Signals with a handle value of 0 are
   * allowed and are interpreted by the packet processor as satisfied
   * dependencies.
   */
  hsa_signal_t dep_signal[5];

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved2;

  /**
   * Signal used to indicate completion of the job. The application can use the
   * special signal handle 0 to indicate that no signal is used.
   */
  hsa_signal_t completion_signal;

} hsa_barrier_and_packet_t;

/**
 * @brief Barrier-OR packet.
 */
typedef struct hsa_barrier_or_packet_s {
  /**
   * Packet header. Used to configure multiple packet parameters such as the
   * packet type. The parameters are described by ::hsa_packet_header_t.
   */
  uint16_t header;

  /**
   * Reserved. Must be 0.
   */
  uint16_t reserved0;

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;

  /**
   * Array of dependent signal objects. Signals with a handle value of 0 are
   * allowed and are interpreted by the packet processor as dependencies not
   * satisfied.
   */
  hsa_signal_t dep_signal[5];

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved2;

  /**
   * Signal used to indicate completion of the job. The application can use the
   * special signal handle 0 to indicate that no signal is used.
   */
  hsa_signal_t completion_signal;

} hsa_barrier_or_packet_t;

/** @} */

/** \addtogroup memory Memory
 *  @{
 */

/**
 * @brief Memory segments associated with a region.
 */
typedef enum {
  /**
   * Global segment. Used to hold data that is shared by all agents.
   */
  HSA_REGION_SEGMENT_GLOBAL = 0,
  /**
   * Read-only segment. Used to hold data that remains constant during the
   * execution of a kernel.
   */
  HSA_REGION_SEGMENT_READONLY = 1,
  /**
   * Private segment. Used to hold data that is local to a single work-item.
   */
  HSA_REGION_SEGMENT_PRIVATE = 2,
  /**
   * Group segment. Used to hold data that is shared by the work-items of a
   * work-group.
  */
  HSA_REGION_SEGMENT_GROUP = 3,
  /**
   * Kernarg segment. Used to store kernel arguments.
  */
  HSA_REGION_SEGMENT_KERNARG = 4
} hsa_region_segment_t;

/**
 * @brief Global region flags.
 */
typedef enum {
  /**
   * The application can use memory in the region to store kernel arguments, and
   * provide the values for the kernarg segment of a kernel dispatch. If this
   * flag is set, then ::HSA_REGION_GLOBAL_FLAG_FINE_GRAINED must be set.
   */
  HSA_REGION_GLOBAL_FLAG_KERNARG = 1,
  /**
   * Updates to memory in this region are immediately visible to all the
   * agents under the terms of the HSA memory model. If this
   * flag is set, then ::HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED must not be set.
   */
  HSA_REGION_GLOBAL_FLAG_FINE_GRAINED = 2,
  /**
   * Updates to memory in this region can be performed by a single agent at
   * a time. If a different agent in the system is allowed to access the
   * region, the application must explicitely invoke ::hsa_memory_assign_agent
   * in order to transfer ownership to that agent for a particular buffer.
   */
  HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED = 4,

  /**
   * Updates to memory in this region have extended scope, where the device-scope atomics
   * to this memory type act as system-scope with respect to all variables located in
   * memory regions of this type.
   * Note: On non-compliant systems, the application may still be responsible for performing
   * device-specific actions necessary to achieve system-scope coherence.
   */
  HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED = 8
} hsa_region_global_flag_t;

/**
 * @brief Attributes of a memory region.
 */
typedef enum {
  /**
   * Segment where memory in the region can be used. The type of this
   * attribute is ::hsa_region_segment_t.
   */
  HSA_REGION_INFO_SEGMENT = 0,
  /**
   * Flag mask. The value of this attribute is undefined if the value of
   * ::HSA_REGION_INFO_SEGMENT is not ::HSA_REGION_SEGMENT_GLOBAL. The type of
   * this attribute is uint32_t, a bit-field of ::hsa_region_global_flag_t
   * values.
   */
  HSA_REGION_INFO_GLOBAL_FLAGS = 1,
  /**
   * Size of this region, in bytes. The type of this attribute is size_t.
   */
  HSA_REGION_INFO_SIZE = 2,
  /**
   * Maximum allocation size in this region, in bytes. Must not exceed the value
   * of ::HSA_REGION_INFO_SIZE. The type of this attribute is size_t.
   *
   * If the region is in the global or readonly segments, this is the maximum
   * size that the application can pass to ::hsa_memory_allocate.
   *
   * If the region is in the group segment, this is the maximum size (per
   * work-group) that can be requested for a given kernel dispatch. If the
   * region is in the private segment, this is the maximum size (per work-item)
   * that can be requested for a specific kernel dispatch, and must be at least
   * 256 bytes.
   */
  HSA_REGION_INFO_ALLOC_MAX_SIZE = 4,
  /**
   * Maximum size (per work-group) of private memory that can be requested for a
   * specific kernel dispatch. Must be at least 65536 bytes. The type of this
   * attribute is uint32_t. The value of this attribute is undefined if the
   * region is not in the private segment.
   */
  HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE = 8,
  /**
   * Indicates whether memory in this region can be allocated using
   * ::hsa_memory_allocate. The type of this attribute is bool.
   *
   * The value of this flag is always false for regions in the group and private
   * segments.
   */
  HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED = 5,
  /**
   * Allocation granularity of buffers allocated by ::hsa_memory_allocate in
   * this region. The size of a buffer allocated in this region is a multiple of
   * the value of this attribute. The value of this attribute is only defined if
   * ::HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED is true for this region. The type
   * of this attribute is size_t.
   */
  HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE = 6,
  /**
   * Alignment of buffers allocated by ::hsa_memory_allocate in this region. The
   * value of this attribute is only defined if
   * ::HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED is true for this region, and must be
   * a power of 2. The type of this attribute is size_t.
   */
  HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT = 7
} hsa_region_info_t;

/**
 * @brief Get the current value of an attribute of a region.
 *
 * @param[in] region A valid region.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to a application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_REGION The region is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * region attribute, or @p value is NULL.
 */
hsa_status_t HSA_API hsa_region_get_info(
    hsa_region_t region,
    hsa_region_info_t attribute,
    void* value);

/**
 * @brief Iterate over the memory regions associated with a given agent, and
 * invoke an application-defined callback on every iteration.
 *
 * @param[in] agent A valid agent.
 *
 * @param[in] callback Callback to be invoked once per region that is
 * accessible from the agent.  The HSA runtime passes two arguments to the
 * callback, the region and the application data.  If @p callback returns a
 * status other than ::HSA_STATUS_SUCCESS for a particular iteration, the
 * traversal stops and ::hsa_agent_iterate_regions returns that status value.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL.
 */
hsa_status_t HSA_API hsa_agent_iterate_regions(
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_region_t region, void* data),
    void* data);

/**
 * @brief Allocate a block of memory in a given region.
 *
 * @param[in] region Region where to allocate memory from. The region must have
 * the ::HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED flag set.
 *
 * @param[in] size Allocation size, in bytes. Must not be zero. This value is
 * rounded up to the nearest multiple of ::HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE
 * in @p region.
 *
 * @param[out] ptr Pointer to the location where to store the base address of
 * the allocated block. The returned base address is aligned to the value of
 * ::HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT in @p region. If the allocation
 * fails, the returned value is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_REGION The region is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION The host is not allowed to
 * allocate memory in @p region, or @p size is greater than the value of
 * HSA_REGION_INFO_ALLOC_MAX_SIZE in @p region.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p ptr is NULL, or @p size is 0.
 */
hsa_status_t HSA_API hsa_memory_allocate(hsa_region_t region,
    size_t size,
    void** ptr);

/**
 * @brief Deallocate a block of memory previously allocated using
 * ::hsa_memory_allocate.
 *
 * @param[in] ptr Pointer to a memory block. If @p ptr does not match a value
 * previously returned by ::hsa_memory_allocate, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 */
hsa_status_t HSA_API hsa_memory_free(void* ptr);

/**
 * @brief Copy a block of memory from the location pointed to by @p src to the
 * memory block pointed to by @p dst.
 *
 * @param[out] dst Buffer where the content is to be copied. If @p dst is in
 * coarse-grained memory, the copied data is only visible to the agent currently
 * assigned (::hsa_memory_assign_agent) to @p dst.
 *
 * @param[in] src A valid pointer to the source of data to be copied. The source
 * buffer must not overlap with the destination buffer. If the source buffer is
 * in coarse-grained memory then it must be assigned to an agent, from which the
 * data will be retrieved.
 *
 * @param[in] size Number of bytes to copy. If @p size is 0, no copy is
 * performed and the function returns success. Copying a number of bytes larger
 * than the size of the buffers pointed by @p dst or @p src results in undefined
 * behavior.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT The source or destination
 * pointers are NULL.
 */
hsa_status_t HSA_API hsa_memory_copy(
    void *dst,
    const void *src,
    size_t size);

/**
 * @brief Change the ownership of a global, coarse-grained buffer.
 *
 * @details The contents of a coarse-grained buffer are visible to an agent
 * only after ownership has been explicitely transferred to that agent. Once the
 * operation completes, the previous owner cannot longer access the data in the
 * buffer.
 *
 * An implementation of the HSA runtime is allowed, but not required, to change
 * the physical location of the buffer when ownership is transferred to a
 * different agent. In general the application must not assume this
 * behavior. The virtual location (address) of the passed buffer is never
 * modified.
 *
 * @param[in] ptr Base address of a global buffer. The pointer must match an
 * address previously returned by ::hsa_memory_allocate. The size of the buffer
 * affected by the ownership change is identical to the size of that previous
 * allocation. If @p ptr points to a fine-grained global buffer, no operation is
 * performed and the function returns success. If @p ptr does not point to
 * global memory, the behavior is undefined.
 *
 * @param[in] agent Agent that becomes the owner of the buffer. The
 * application is responsible for ensuring that @p agent has access to the
 * region that contains the buffer. It is allowed to change ownership to an
 * agent that is already the owner of the buffer, with the same or different
 * access permissions.
 *
 * @param[in] access Access permissions requested for the new owner.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p ptr is NULL, or @p access is
 * not a valid access value.
 */
hsa_status_t HSA_API hsa_memory_assign_agent(
    void *ptr,
    hsa_agent_t agent,
    hsa_access_permission_t access);

/**
 *
 * @brief Register a global, fine-grained buffer.
 *
 * @details Registering a buffer serves as an indication to the HSA runtime that
 * the memory might be accessed from a kernel agent other than the
 * host. Registration is a performance hint that allows the HSA runtime
 * implementation to know which buffers will be accessed by some of the kernel
 * agents ahead of time.
 *
 * Registration is only recommended for buffers in the global segment that have
 * not been allocated using the HSA allocator (::hsa_memory_allocate), but an OS
 * allocator instead. Registering an OS-allocated buffer in the base profile is
 * equivalent to a no-op.
 *
 * Registrations should not overlap.
 *
 * @param[in] ptr A buffer in global, fine-grained memory. If a NULL pointer is
 * passed, no operation is performed. If the buffer has been allocated using
 * ::hsa_memory_allocate, or has already been registered, no operation is
 * performed.
 *
 * @param[in] size Requested registration size in bytes. A size of 0 is
 * only allowed if @p ptr is NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p size is 0 but @p ptr
 * is not NULL.
 */
hsa_status_t HSA_API hsa_memory_register(
    void *ptr,
    size_t size);

/**
 *
 * @brief Deregister memory previously registered using ::hsa_memory_register.
 *
 * @details If the memory interval being deregistered does not match a previous
 * registration (start and end addresses), the behavior is undefined.
 *
 * @param[in] ptr A pointer to the base of the buffer to be deregistered. If
 * a NULL pointer is passed, no operation is performed.
 *
 * @param[in] size Size of the buffer to be deregistered.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 */
hsa_status_t HSA_API hsa_memory_deregister(
    void *ptr,
    size_t size);

/** @} */


/** \defgroup instruction-set-architecture Instruction Set Architecture.
 *  @{
 */

/**
 * @brief Instruction set architecture.
 */
typedef struct hsa_isa_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_isa_t;

/**
 * @brief Retrieve a reference to an instruction set architecture handle out of
 * a symbolic name.
 *
 * @param[in] name Vendor-specific name associated with a a particular
 * instruction set architecture. @p name must start with the vendor name and a
 * colon (for example, "AMD:"). The rest of the name is vendor-specific. Must be
 * a NUL-terminated string.
 *
 * @param[out] isa Memory location where the HSA runtime stores the ISA handle
 * corresponding to the given name. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ISA_NAME The given name does not
 * correspond to any instruction set architecture.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p name is NULL, or @p isa is
 * NULL.
 */
hsa_status_t HSA_API hsa_isa_from_name(
    const char *name,
    hsa_isa_t *isa);

/**
 * @brief Iterate over the instruction sets supported by the given agent, and
 * invoke an application-defined callback on every iteration. The iterator is
 * deterministic: if an agent supports several instruction set architectures,
 * they are traversed in the same order in every invocation of this function.
 *
 * @param[in] agent A valid agent.
 *
 * @param[in] callback Callback to be invoked once per instruction set
 * architecture.  The HSA runtime passes two arguments to the callback: the
 * ISA and the application data.  If @p callback returns a status other than
 * ::HSA_STATUS_SUCCESS for a particular iteration, the traversal stops and
 * that status value is returned.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL.
 */
hsa_status_t HSA_API hsa_agent_iterate_isas(
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_isa_t isa, void *data),
    void *data);

/**
 * @brief Instruction set architecture attributes.
 */
typedef enum {
  /**
   * The length of the ISA name in bytes, not including the NUL terminator. The
   * type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_NAME_LENGTH = 0,
  /**
   * Human-readable description.  The type of this attribute is character array
   * with the length equal to the value of ::HSA_ISA_INFO_NAME_LENGTH attribute.
   */
  HSA_ISA_INFO_NAME = 1,
  /**
   * @deprecated
   *
   * Number of call conventions supported by the instruction set architecture.
   * Must be greater than zero. The type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_CALL_CONVENTION_COUNT = 2,
  /**
   * @deprecated
   *
   * Number of work-items in a wavefront for a given call convention. Must be a
   * power of 2 in the range [1,256]. The type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE = 3,
  /**
   * @deprecated
   *
   * Number of wavefronts per compute unit for a given call convention. In
   * practice, other factors (for example, the amount of group memory used by a
   * work-group) may further limit the number of wavefronts per compute
   * unit. The type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT = 4,
  /**
   * Machine models supported by the instruction set architecture. The type of
   * this attribute is a bool[2]. If the ISA supports the small machine model,
   * the element at index ::HSA_MACHINE_MODEL_SMALL is true. If the ISA supports
   * the large model, the element at index ::HSA_MACHINE_MODEL_LARGE is true.
   */
  HSA_ISA_INFO_MACHINE_MODELS = 5,
  /**
   * Profiles supported by the instruction set architecture. The type of this
   * attribute is a bool[2]. If the ISA supports the base profile, the element
   * at index ::HSA_PROFILE_BASE is true. If the ISA supports the full profile,
   * the element at index ::HSA_PROFILE_FULL is true.
   */
  HSA_ISA_INFO_PROFILES = 6,
  /**
   * Default floating-point rounding modes supported by the instruction set
   * architecture. The type of this attribute is a bool[3]. The value at a given
   * index is true if the corresponding rounding mode in
   * ::hsa_default_float_rounding_mode_t is supported. At least one default mode
   * has to be supported.
   *
   * If the default mode is supported, then
   * ::HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES must report that
   * both the zero and the near roundings modes are supported.
   */
  HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES = 7,
  /**
   * Default floating-point rounding modes supported by the instruction set
   * architecture in the Base profile. The type of this attribute is a
   * bool[3]. The value at a given index is true if the corresponding rounding
   * mode in ::hsa_default_float_rounding_mode_t is supported. The value at
   * index HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT must be false.  At least one
   * of the values at indexes ::HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO or
   * HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR must be true.
   */
  HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 8,
  /**
   * Flag indicating that the f16 HSAIL operation is at least as fast as the
   * f32 operation in the instruction set architecture. The type of this
   * attribute is bool.
   */
  HSA_ISA_INFO_FAST_F16_OPERATION = 9,
  /**
   * Maximum number of work-items of each dimension of a work-group.  Each
   * maximum must be greater than 0. No maximum can exceed the value of
   * ::HSA_ISA_INFO_WORKGROUP_MAX_SIZE. The type of this attribute is
   * uint16_t[3].
   */
  HSA_ISA_INFO_WORKGROUP_MAX_DIM = 12,
  /**
   * Maximum total number of work-items in a work-group. The type
   * of this attribute is uint32_t.
   */
  HSA_ISA_INFO_WORKGROUP_MAX_SIZE = 13,
  /**
   * Maximum number of work-items of each dimension of a grid. Each maximum must
   * be greater than 0, and must not be smaller than the corresponding value in
   * ::HSA_ISA_INFO_WORKGROUP_MAX_DIM. No maximum can exceed the value of
   * ::HSA_ISA_INFO_GRID_MAX_SIZE. The type of this attribute is
   * ::hsa_dim3_t.
   */
  HSA_ISA_INFO_GRID_MAX_DIM = 14,
  /**
   * Maximum total number of work-items in a grid. The type of this
   * attribute is uint64_t.
   */
  HSA_ISA_INFO_GRID_MAX_SIZE = 16,
  /**
   * Maximum number of fbarriers per work-group. Must be at least 32. The
   * type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_FBARRIER_MAX_SIZE = 17
} hsa_isa_info_t;

/**
 * @deprecated The concept of call convention has been deprecated. If the
 * application wants to query the value of an attribute for a given instruction
 * set architecture, use ::hsa_isa_get_info_alt instead. If the application
 * wants to query an attribute that is specific to a given combination of ISA
 * and wavefront, use ::hsa_wavefront_get_info.
 *
 * @brief Get the current value of an attribute for a given instruction set
 * architecture (ISA).
 *
 * @param[in] isa A valid instruction set architecture.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[in] index Call convention index. Used only for call convention
 * attributes, otherwise ignored. Must have a value between 0 (inclusive) and
 * the value of the attribute ::HSA_ISA_INFO_CALL_CONVENTION_COUNT (not
 * inclusive) in @p isa.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ISA The instruction set architecture is
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_INDEX The index is out of range.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * instruction set architecture attribute, or @p value is
 * NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_isa_get_info(
    hsa_isa_t isa,
    hsa_isa_info_t attribute,
    uint32_t index,
    void *value);

/**
 * @brief Get the current value of an attribute for a given instruction set
 * architecture (ISA).
 *
 * @param[in] isa A valid instruction set architecture.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ISA The instruction set architecture is
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * instruction set architecture attribute, or @p value is
 * NULL.
 */
hsa_status_t HSA_API hsa_isa_get_info_alt(
    hsa_isa_t isa,
    hsa_isa_info_t attribute,
    void *value);

/**
 * @brief Retrieve the exception policy support for a given combination of
 * instruction set architecture and profile.
 *
 * @param[in] isa A valid instruction set architecture.
 *
 * @param[in] profile Profile.
 *
 * @param[out] mask Pointer to a memory location where the HSA runtime stores a
 * mask of ::hsa_exception_policy_t values. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ISA The instruction set architecture is
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p profile is not a valid
 * profile, or @p mask is NULL.
 */
hsa_status_t HSA_API hsa_isa_get_exception_policies(
    hsa_isa_t isa,
    hsa_profile_t profile,
    uint16_t *mask);

/**
 * @brief Floating-point types.
 */
typedef enum {
  /**
   * 16-bit floating-point type.
   */
  HSA_FP_TYPE_16 = 1,
  /**
   * 32-bit floating-point type.
   */
  HSA_FP_TYPE_32 = 2,
  /**
   * 64-bit floating-point type.
   */
  HSA_FP_TYPE_64 = 4
} hsa_fp_type_t;

/**
 * @brief Flush to zero modes.
 */
typedef enum {
  /**
   * Flush to zero.
   */
  HSA_FLUSH_MODE_FTZ = 1,
  /**
   * Do not flush to zero.
   */
  HSA_FLUSH_MODE_NON_FTZ = 2
} hsa_flush_mode_t;

/**
 * @brief Round methods.
 */
typedef enum {
  /**
   * Single round method.
   */
  HSA_ROUND_METHOD_SINGLE = 1,
  /**
   * Double round method.
   */
  HSA_ROUND_METHOD_DOUBLE = 2
} hsa_round_method_t;

/**
 * @brief Retrieve the round method (single or double) used to implement the
 * floating-point multiply add instruction (mad) for a given combination of
 * instruction set architecture, floating-point type, and flush to zero
 * modifier.
 *
 * @param[in] isa Instruction set architecture.
 *
 * @param[in] fp_type Floating-point type.
 *
 * @param[in] flush_mode Flush to zero modifier.
 *
 * @param[out] round_method Pointer to a memory location where the HSA
 * runtime stores the round method used by the implementation. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ISA The instruction set architecture is
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p fp_type is not a valid
 * floating-point type, or @p flush_mode is not a valid flush to zero modifier,
 * or @p round_method is NULL.
 */
hsa_status_t HSA_API hsa_isa_get_round_method(
    hsa_isa_t isa,
    hsa_fp_type_t fp_type,
    hsa_flush_mode_t flush_mode,
    hsa_round_method_t *round_method);

/**
 * @brief Wavefront handle
 */
typedef struct hsa_wavefront_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_wavefront_t;

/**
 * @brief Wavefront attributes.
 */
typedef enum {
  /**
   * Number of work-items in the wavefront. Must be a power of 2 in the range
   * [1,256]. The type of this attribute is uint32_t.
   */
  HSA_WAVEFRONT_INFO_SIZE = 0
} hsa_wavefront_info_t;

/**
 * @brief Get the current value of a wavefront attribute.
 *
 * @param[in] wavefront A wavefront.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_WAVEFRONT The wavefront is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * wavefront attribute, or @p value is NULL.
 */
hsa_status_t HSA_API hsa_wavefront_get_info(
    hsa_wavefront_t wavefront,
    hsa_wavefront_info_t attribute,
    void *value);

/**
 * @brief Iterate over the different wavefronts supported by an instruction set
 * architecture, and invoke an application-defined callback on every iteration.
 *
 * @param[in] isa Instruction set architecture.
 *
 * @param[in] callback Callback to be invoked once per wavefront that is
 * supported by the agent. The HSA runtime passes two arguments to the callback:
 * the wavefront handle and the application data.  If @p callback returns a
 * status other than ::HSA_STATUS_SUCCESS for a particular iteration, the
 * traversal stops and that value is returned.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ISA The instruction set architecture is
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL.
 */
hsa_status_t HSA_API hsa_isa_iterate_wavefronts(
    hsa_isa_t isa,
    hsa_status_t (*callback)(hsa_wavefront_t wavefront, void *data),
    void *data);

/**
 * @deprecated Use ::hsa_agent_iterate_isas to query which instructions set
 * architectures are supported by a given agent.
 *
 * @brief Check if the instruction set architecture of a code object can be
 * executed on an agent associated with another architecture.
 *
 * @param[in] code_object_isa Instruction set architecture associated with a
 * code object.
 *
 * @param[in] agent_isa Instruction set architecture associated with an agent.
 *
 * @param[out] result Pointer to a memory location where the HSA runtime stores
 * the result of the check. If the two architectures are compatible, the result
 * is true; if they are incompatible, the result is false.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ISA @p code_object_isa or @p agent_isa are
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p result is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_isa_compatible(
    hsa_isa_t code_object_isa,
    hsa_isa_t agent_isa,
    bool *result);

/** @} */


/** \defgroup executable Executable
 *  @{
 */

/**
 * @brief Code object reader handle. A code object reader is used to
 * load a code object from file (when created using
 * ::hsa_code_object_reader_create_from_file), or from memory (if created using
 * ::hsa_code_object_reader_create_from_memory).
 */
typedef struct hsa_code_object_reader_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_code_object_reader_t;

/**
 * @brief Create a code object reader to operate on a file.
 *
 * @param[in] file File descriptor. The file must have been opened by
 * application with at least read permissions prior calling this function. The
 * file must contain a vendor-specific code object.
 *
 * The file is owned and managed by the application; the lifetime of the file
 * descriptor must exceed that of any associated code object reader.
 *
 * @param[out] code_object_reader Memory location to store the newly created
 * code object reader handle. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_FILE @p file is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p code_object_reader is NULL.
 */
hsa_status_t HSA_API hsa_code_object_reader_create_from_file(
    hsa_file_t file,
    hsa_code_object_reader_t *code_object_reader);

/**
 * @brief Create a code object reader to operate on memory.
 *
 * @param[in] code_object Memory buffer that contains a vendor-specific code
 * object. The buffer is owned and managed by the application; the lifetime of
 * the buffer must exceed that of any associated code object reader.
 *
 * @param[in] size Size of the buffer pointed to by @p code_object. Must not be
 * 0.
 *
 * @param[out] code_object_reader Memory location to store newly created code
 * object reader handle. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p code_object is NULL, @p size
 * is zero, or @p code_object_reader is NULL.
 */
hsa_status_t HSA_API hsa_code_object_reader_create_from_memory(
    const void *code_object,
    size_t size,
    hsa_code_object_reader_t *code_object_reader);

/**
 * @brief Destroy a code object reader.
 *
 * @details The code object reader handle becomes invalid after completion of
 * this function. Any file or memory used to create the code object read is not
 * closed, removed, or deallocated by this function.
 *
 * @param[in] code_object_reader Code object reader to destroy.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER @p code_object_reader
 * is invalid.
 */
hsa_status_t HSA_API hsa_code_object_reader_destroy(
    hsa_code_object_reader_t code_object_reader);

/**
 * @brief Struct containing an opaque handle to an executable, which contains
 * ISA for finalized kernels and indirect functions together with the allocated
 * global or readonly segment variables they reference.
 */
typedef struct hsa_executable_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_executable_t;

/**
 * @brief Executable state.
 */
typedef enum {
  /**
   * Executable state, which allows the user to load code objects and define
   * external variables. Variable addresses, kernel code handles, and
   * indirect function code handles are not available in query operations until
   * the executable is frozen (zero always returned).
   */
  HSA_EXECUTABLE_STATE_UNFROZEN = 0,
  /**
   * Executable state, which allows the user to query variable addresses,
   * kernel code handles, and indirect function code handles using query
   * operations. Loading new code objects, as well as defining external
   * variables, is not allowed in this state.
   */
  HSA_EXECUTABLE_STATE_FROZEN = 1
} hsa_executable_state_t;

/**
 * @deprecated Use ::hsa_executable_create_alt instead, which allows the
 * application to specify the default floating-point rounding mode of the
 * executable and assumes an unfrozen initial state.
 *
 * @brief Create an empty executable.
 *
 * @param[in] profile Profile used in the executable.
 *
 * @param[in] executable_state Executable state. If the state is
 * ::HSA_EXECUTABLE_STATE_FROZEN, the resulting executable is useless because no
 * code objects can be loaded, and no variables can be defined.
 *
 * @param[in] options Standard and vendor-specific options. Unknown options are
 * ignored. A standard option begins with the "-hsa_" prefix. Options beginning
 * with the "-hsa_ext_<extension_name>_" prefix are reserved for extensions. A
 * vendor-specific option begins with the "-<vendor_name>_" prefix. Must be a
 * NUL-terminated string. May be NULL.
 *
 * @param[out] executable Memory location where the HSA runtime stores the newly
 * created executable handle.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p profile is invalid, or
 * @p executable is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_executable_create(
    hsa_profile_t profile,
    hsa_executable_state_t executable_state,
    const char *options,
    hsa_executable_t *executable);

/**
 * @brief Create an empty executable.
 *
 * @param[in] profile Profile used in the executable.
 *
 * @param[in] default_float_rounding_mode Default floating-point rounding mode
 * used in the executable. Allowed rounding modes are near and zero (default is
 * not allowed).
 *
 * @param[in] options Standard and vendor-specific options. Unknown options are
 * ignored. A standard option begins with the "-hsa_" prefix. Options beginning
 * with the "-hsa_ext_<extension_name>_" prefix are reserved for extensions. A
 * vendor-specific option begins with the "-<vendor_name>_" prefix. Must be a
 * NUL-terminated string. May be NULL.
 *
 * @param[out] executable Memory location where the HSA runtime stores newly
 * created executable handle. The initial state of the executable is
 * ::HSA_EXECUTABLE_STATE_UNFROZEN.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p profile is invalid, or
 * @p executable is NULL.
 */
hsa_status_t HSA_API hsa_executable_create_alt(
    hsa_profile_t profile,
    hsa_default_float_rounding_mode_t default_float_rounding_mode,
    const char *options,
    hsa_executable_t *executable);

/**
 * @brief Destroy an executable.
 *
 * @details An executable handle becomes invalid after the executable has been
 * destroyed. Code object handles that were loaded into this executable are
 * still valid after the executable has been destroyed, and can be used as
 * intended. Resources allocated outside and associated with this executable
 * (such as external global or readonly variables) can be released after the
 * executable has been destroyed.
 *
 * Executable should not be destroyed while kernels are in flight.
 *
 * @param[in] executable Executable.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 */
hsa_status_t HSA_API hsa_executable_destroy(
    hsa_executable_t executable);

/**
 * @brief Loaded code object handle.
 */
typedef struct hsa_loaded_code_object_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_loaded_code_object_t;

/**
 * @brief Load a program code object into an executable.
 *
 * @details A program code object contains information about resources that are
 * accessible by all kernel agents that run the executable, and can be loaded
 * at most once into an executable.
 *
 * If the program code object uses extensions, the implementation must support
 * them for this operation to return successfully.
 *
 * @param[in] executable Executable.
 *
 * @param[in] code_object_reader A code object reader that holds the program
 * code object to load. If a code object reader is destroyed before all the
 * associated executables are destroyed, the behavior is undefined.
 *
 * @param[in] options Standard and vendor-specific options. Unknown options are
 * ignored. A standard option begins with the "-hsa_" prefix. Options beginning
 * with the "-hsa_ext_<extension_name>_" prefix are reserved for extensions. A
 * vendor-specific option begins with the "-<vendor_name>_" prefix. Must be a
 * NUL-terminated string. May be NULL.
 *
 * @param[out] loaded_code_object Pointer to a memory location where the HSA
 * runtime stores the loaded code object handle. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_FROZEN_EXECUTABLE The executable is frozen.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER @p code_object_reader
 * is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS The program code object is
 * not compatible with the executable or the implementation (for example, the
 * code object uses an extension that is not supported by the implementation).
 */
hsa_status_t HSA_API hsa_executable_load_program_code_object(
    hsa_executable_t executable,
    hsa_code_object_reader_t code_object_reader,
    const char *options,
    hsa_loaded_code_object_t *loaded_code_object);

/**
 * @brief Load an agent code object into an executable.
 *
 * @details The agent code object contains all defined agent
 * allocation variables, functions, indirect functions, and kernels in a given
 * program for a given instruction set architecture.
 *
 * Any module linkage declaration must have been defined either by a define
 * variable or by loading a code object that has a symbol with module linkage
 * definition.
 *
 * The default floating-point rounding mode of the code object associated with
 * @p code_object_reader must match that of the executable
 * (::HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE), or be default (in which
 * case the value of ::HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE is used).
 * If the agent code object uses extensions, the implementation and the agent
 * must support them for this operation to return successfully.
 *
 * @param[in] executable Executable.
 *
 * @param[in] agent Agent to load code object for. A code object can be loaded
 * into an executable at most once for a given agent. The instruction set
 * architecture of the code object must be supported by the agent.
 *
 * @param[in] code_object_reader A code object reader that holds the code object
 * to load. If a code object reader is destroyed before all the associated
 * executables are destroyed, the behavior is undefined.
 *
 * @param[in] options Standard and vendor-specific options. Unknown options are
 * ignored. A standard option begins with the "-hsa_" prefix. Options beginning
 * with the "-hsa_ext_<extension_name>_" prefix are reserved for extensions. A
 * vendor-specific option begins with the "-<vendor_name>_" prefix. Must be a
 * NUL-terminated string. May be NULL.
 *
 * @param[out] loaded_code_object Pointer to a memory location where the HSA
 * runtime stores the loaded code object handle. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_FROZEN_EXECUTABLE The executable is frozen.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER @p code_object_reader
 * is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS The code object read by @p
 * code_object_reader is not compatible with the agent (for example, the agent
 * does not support the instruction set architecture of the code object), the
 * executable (for example, there is a default floating-point mode mismatch
 * between the two), or the implementation.
 */
hsa_status_t HSA_API hsa_executable_load_agent_code_object(
    hsa_executable_t executable,
    hsa_agent_t agent,
    hsa_code_object_reader_t code_object_reader,
    const char *options,
    hsa_loaded_code_object_t *loaded_code_object);

/**
 * @brief Freeze the executable.
 *
 * @details No modifications to executable can be made after freezing: no code
 * objects can be loaded to the executable, and no external variables can be
 * defined. Freezing the executable does not prevent querying the executable's
 * attributes. The application must define all the external variables in an
 * executable before freezing it.
 *
 * @param[in] executable Executable.
 *
 * @param[in] options Standard and vendor-specific options. Unknown options are
 * ignored. A standard option begins with the "-hsa_" prefix. Options beginning
 * with the "-hsa_ext_<extension_name>_" prefix are reserved for extensions. A
 * vendor-specific option begins with the "-<vendor_name>_" prefix. Must be a
 * NUL-terminated string. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_VARIABLE_UNDEFINED One or more variables are
 * undefined in the executable.
 *
 * @retval ::HSA_STATUS_ERROR_FROZEN_EXECUTABLE @p executable is already frozen.
 */
hsa_status_t HSA_API hsa_executable_freeze(
    hsa_executable_t executable,
    const char *options);

/**
 * @brief Executable attributes.
 */
typedef enum {
  /**
   * Profile this executable is created for. The type of this attribute is
   * ::hsa_profile_t.
   */
  HSA_EXECUTABLE_INFO_PROFILE = 1,
  /**
   * Executable state. The type of this attribute is ::hsa_executable_state_t.
   */
  HSA_EXECUTABLE_INFO_STATE = 2,
  /**
   * Default floating-point rounding mode specified when executable was created.
   * The type of this attribute is ::hsa_default_float_rounding_mode_t.
   */
  HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 3
} hsa_executable_info_t;

/**
 * @brief Get the current value of an attribute for a given executable.
 *
 * @param[in] executable Executable.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * executable attribute, or @p value is NULL.
 */
hsa_status_t HSA_API hsa_executable_get_info(
    hsa_executable_t executable,
    hsa_executable_info_t attribute,
    void *value);

/**
 * @brief Define an external global variable with program allocation.
 *
 * @details This function allows the application to provide the definition
 * of a variable in the global segment memory with program allocation. The
 * variable must be defined before loading a code object into an executable.
 * In addition, code objects loaded must not define the variable.
 *
 * @param[in] executable Executable. Must not be in frozen state.
 *
 * @param[in] variable_name Name of the variable. The Programmer's Reference
 * Manual describes the standard name mangling scheme.
 *
 * @param[in] address Address where the variable is defined. This address must
 * be in global memory and can be read and written by any agent in the
 * system. The application cannot deallocate the buffer pointed by @p address
 * before @p executable is destroyed.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED The variable is
 * already defined.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SYMBOL_NAME There is no variable with the
 * @p variable_name.
 *
 * @retval ::HSA_STATUS_ERROR_FROZEN_EXECUTABLE @p executable is frozen.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p variable_name is NULL.
 */
hsa_status_t HSA_API hsa_executable_global_variable_define(
    hsa_executable_t executable,
    const char *variable_name,
    void *address);

/**
 * @brief Define an external global variable with agent allocation.
 *
 * @details This function allows the application to provide the definition
 * of a variable in the global segment memory with agent allocation. The
 * variable must be defined before loading a code object into an executable.
 * In addition, code objects loaded must not define the variable.
 *
 * @param[in] executable Executable. Must not be in frozen state.
 *
 * @param[in] agent Agent for which the variable is being defined.
 *
 * @param[in] variable_name Name of the variable. The Programmer's Reference
 * Manual describes the standard name mangling scheme.
 *
 * @param[in] address Address where the variable is defined. This address must
 * have been previously allocated using ::hsa_memory_allocate in a global region
 * that is only visible to @p agent. The application cannot deallocate the
 * buffer pointed by @p address before @p executable is destroyed.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT @p agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED The variable is
 * already defined.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SYMBOL_NAME There is no variable with the
 * @p variable_name.
 *
 * @retval ::HSA_STATUS_ERROR_FROZEN_EXECUTABLE @p executable is frozen.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p variable_name is NULL.
 */
hsa_status_t HSA_API hsa_executable_agent_global_variable_define(
    hsa_executable_t executable,
    hsa_agent_t agent,
    const char *variable_name,
    void *address);

/**
 * @brief Define an external readonly variable.
 *
 * @details This function allows the application to provide the definition
 * of a variable in the readonly segment memory. The variable must be defined
 * before loading a code object into an executable. In addition, code objects
 * loaded must not define the variable.
 *
 * @param[in] executable Executable. Must not be in frozen state.
 *
 * @param[in] agent Agent for which the variable is being defined.
 *
 * @param[in] variable_name Name of the variable. The Programmer's Reference
 * Manual describes the standard name mangling scheme.
 *
 * @param[in] address Address where the variable is defined. This address must
 * have been previously allocated using ::hsa_memory_allocate in a readonly
 * region associated with @p agent. The application cannot deallocate the buffer
 * pointed by @p address before @p executable is destroyed.
 *
 * @param[in] address Address where the variable is defined. The buffer pointed
 * by @p address is owned by the application, and cannot be deallocated before
 * @p executable is destroyed.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE Executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT @p agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED The variable is
 * already defined.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SYMBOL_NAME There is no variable with the
 * @p variable_name.
 *
 * @retval ::HSA_STATUS_ERROR_FROZEN_EXECUTABLE @p executable is frozen.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p variable_name is NULL.
 */
hsa_status_t HSA_API hsa_executable_readonly_variable_define(
    hsa_executable_t executable,
    hsa_agent_t agent,
    const char *variable_name,
    void *address);

/**
 * @brief Validate an executable. Checks that all code objects have matching
 * machine model, profile, and default floating-point rounding mode. Checks that
 * all declarations have definitions. Checks declaration-definition
 * compatibility (see the HSA Programming Reference Manual for compatibility
 * rules). Invoking this function is equivalent to invoking
 * ::hsa_executable_validate_alt with no options.
 *
 * @param[in] executable Executable. Must be in frozen state.
 *
 * @param[out] result Memory location where the HSA runtime stores the
 * validation result. If the executable passes validation, the result is 0.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE @p executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p result is NULL.
 */
hsa_status_t HSA_API hsa_executable_validate(
    hsa_executable_t executable,
    uint32_t *result);

/**
 * @brief Validate an executable. Checks that all code objects have matching
 * machine model, profile, and default floating-point rounding mode. Checks that
 * all declarations have definitions. Checks declaration-definition
 * compatibility (see the HSA Programming Reference Manual for compatibility
 * rules).
 *
 * @param[in] executable Executable. Must be in frozen state.
 *
 * @param[in] options Standard and vendor-specific options. Unknown options are
 * ignored. A standard option begins with the "-hsa_" prefix. Options beginning
 * with the "-hsa_ext_<extension_name>_" prefix are reserved for extensions. A
 * vendor-specific option begins with the "-<vendor_name>_" prefix. Must be a
 * NUL-terminated string. May be NULL.
 *
 * @param[out] result Memory location where the HSA runtime stores the
 * validation result. If the executable passes validation, the result is 0.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE @p executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p result is NULL.
 */
hsa_status_t HSA_API hsa_executable_validate_alt(
    hsa_executable_t executable,
    const char *options,
    uint32_t *result);

/**
 * @brief Executable symbol handle.
 *
 * The lifetime of an executable object symbol matches that of the executable
 * associated with it. An operation on a symbol whose associated executable has
 * been destroyed results in undefined behavior.
 */
typedef struct hsa_executable_symbol_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_executable_symbol_t;

/**
 * @deprecated Use ::hsa_executable_get_symbol_by_name instead.
 *
 * @brief Get the symbol handle for a given a symbol name.
 *
 * @param[in] executable Executable.
 *
 * @param[in] module_name Module name. Must be NULL if the symbol has
 * program linkage.
 *
 * @param[in] symbol_name Symbol name.
 *
 * @param[in] agent Agent associated with the symbol. If the symbol is
 * independent of any agent (for example, a variable with program
 * allocation), this argument is ignored.
 *
 * @param[in] call_convention Call convention associated with the symbol. If the
 * symbol does not correspond to an indirect function, this argument is ignored.
 *
 * @param[out] symbol Memory location where the HSA runtime stores the symbol
 * handle.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SYMBOL_NAME There is no symbol with a name
 * that matches @p symbol_name.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p symbol_name is NULL, or
 * @p symbol is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_executable_get_symbol(
    hsa_executable_t executable,
    const char *module_name,
    const char *symbol_name,
    hsa_agent_t agent,
    int32_t call_convention,
    hsa_executable_symbol_t *symbol);

/**
 * @brief Retrieve the symbol handle corresponding to a given a symbol name.
 *
 * @param[in] executable Executable.
 *
 * @param[in] symbol_name Symbol name. Must be a NUL-terminated character
 * array. The Programmer's Reference Manual describes the standard name mangling
 * scheme.
 *
 * @param[in] agent Pointer to the agent for which the symbol with the given
 * name is defined. If the symbol corresponding to the given name has program
 * allocation, @p agent must be NULL.
 *
 * @param[out] symbol Memory location where the HSA runtime stores the symbol
 * handle. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SYMBOL_NAME There is no symbol with a name
 * that matches @p symbol_name.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p symbol_name is NULL, or @p
 * symbol is NULL.
 */
hsa_status_t HSA_API hsa_executable_get_symbol_by_name(
    hsa_executable_t executable,
    const char *symbol_name,
    const hsa_agent_t *agent,
    hsa_executable_symbol_t *symbol);

/**
 * @brief Symbol type.
 */
typedef enum {
  /**
   * Variable.
   */
  HSA_SYMBOL_KIND_VARIABLE = 0,
  /**
   * Kernel.
   */
  HSA_SYMBOL_KIND_KERNEL = 1,
  /**
   * Indirect function.
   */
  HSA_SYMBOL_KIND_INDIRECT_FUNCTION = 2
} hsa_symbol_kind_t;

/**
 * @brief Linkage type of a symbol.
 */
typedef enum {
  /**
   * Module linkage.
   */
  HSA_SYMBOL_LINKAGE_MODULE = 0,
  /**
   * Program linkage.
   */
  HSA_SYMBOL_LINKAGE_PROGRAM = 1
} hsa_symbol_linkage_t;

/**
 * @brief Allocation type of a variable.
 */
typedef enum {
  /**
   * Agent allocation.
   */
  HSA_VARIABLE_ALLOCATION_AGENT = 0,
  /**
   * Program allocation.
   */
  HSA_VARIABLE_ALLOCATION_PROGRAM = 1
} hsa_variable_allocation_t;

/**
 * @brief Memory segment associated with a variable.
 */
typedef enum {
  /**
   * Global memory segment.
   */
  HSA_VARIABLE_SEGMENT_GLOBAL = 0,
  /**
   * Readonly memory segment.
   */
  HSA_VARIABLE_SEGMENT_READONLY = 1
} hsa_variable_segment_t;

/**
 * @brief Executable symbol attributes.
 */
typedef enum {
  /**
   * The kind of the symbol. The type of this attribute is ::hsa_symbol_kind_t.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_TYPE = 0,
  /**
   * The length of the symbol name in bytes, not including the NUL terminator.
   * The type of this attribute is uint32_t.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH = 1,
  /**
   * The name of the symbol. The type of this attribute is character array with
   * the length equal to the value of ::HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH
   * attribute.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_NAME = 2,
  /**
   * @deprecated
   *
   * The length of the module name in bytes (not including the NUL terminator)
   * to which this symbol belongs if this symbol has module linkage, otherwise 0
   * is returned. The type of this attribute is uint32_t.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH = 3,
  /**
   * @deprecated
   *
   * The module name to which this symbol belongs if this symbol has module
   * linkage, otherwise an empty string is returned. The type of this attribute
   * is character array with the length equal to the value of
   * ::HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH attribute.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME = 4,
  /**
   * @deprecated
   *
   * Agent associated with this symbol. If the symbol is a variable, the
   * value of this attribute is only defined if
   * ::HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION is
   * ::HSA_VARIABLE_ALLOCATION_AGENT. The type of this attribute is hsa_agent_t.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_AGENT = 20,
  /**
   * The address of the variable. The value of this attribute is undefined if
   * the symbol is not a variable. The type of this attribute is uint64_t.
   *
   * If executable's state is ::HSA_EXECUTABLE_STATE_UNFROZEN, then 0 is
   * returned.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = 21,
  /**
   * The linkage kind of the symbol. The type of this attribute is
   * ::hsa_symbol_linkage_t.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE = 5,
  /**
   * Indicates whether the symbol corresponds to a definition. The type of this
   * attribute is bool.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION = 17,
  /**
   * @deprecated
   *
   * The allocation kind of the variable. The value of this attribute is
   * undefined if the symbol is not a variable.  The type of this attribute is
   * ::hsa_variable_allocation_t.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION = 6,
  /**
   * @deprecated
   *
   * The segment kind of the variable. The value of this attribute is undefined
   * if the symbol is not a variable. The type of this attribute is
   * ::hsa_variable_segment_t.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT = 7,
  /**
   * @deprecated
   *
   * Alignment of the symbol in memory. The value of this attribute is undefined
   * if the symbol is not a variable. The type of this attribute is uint32_t.
   *
   * The current alignment of the variable in memory may be greater than the
   * value specified in the source program variable declaration.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT = 8,
  /**
   * @deprecated
   *
   * Size of the variable. The value of this attribute is undefined if
   * the symbol is not a variable. The type of this attribute is uint32_t.
   *
   * A value of 0 is returned if the variable is an external variable and has an
   * unknown dimension.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE = 9,
  /**
   * @deprecated
   *
   * Indicates whether the variable is constant. The value of this attribute is
   * undefined if the symbol is not a variable. The type of this attribute is
   * bool.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST = 10,
  /**
   * Kernel object handle, used in the kernel dispatch packet. The value of this
   * attribute is undefined if the symbol is not a kernel. The type of this
   * attribute is uint64_t.
   *
   * If the state of the executable is ::HSA_EXECUTABLE_STATE_UNFROZEN, then 0
   * is returned.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT = 22,
  /**
   * Size of kernarg segment memory that is required to hold the values of the
   * kernel arguments, in bytes. Must be a multiple of 16. The value of this
   * attribute is undefined if the symbol is not a kernel. The type of this
   * attribute is uint32_t.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = 11,
  /**
   * Alignment (in bytes) of the buffer used to pass arguments to the kernel,
   * which is the maximum of 16 and the maximum alignment of any of the kernel
   * arguments. The value of this attribute is undefined if the symbol is not a
   * kernel. The type of this attribute is uint32_t.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = 12,
  /**
   * Size of static group segment memory required by the kernel (per
   * work-group), in bytes. The value of this attribute is undefined
   * if the symbol is not a kernel. The type of this attribute is uint32_t.
   *
   * The reported amount does not include any dynamically allocated group
   * segment memory that may be requested by the application when a kernel is
   * dispatched.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = 13,
  /**
   * Size of static private, spill, and arg segment memory required by
   * this kernel (per work-item), in bytes. The value of this attribute is
   * undefined if the symbol is not a kernel. The type of this attribute is
   * uint32_t.
   *
   * If the value of ::HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK is
   * true, the kernel may use more private memory than the reported value, and
   * the application must add the dynamic call stack usage to @a
   * private_segment_size when populating a kernel dispatch packet.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = 14,
  /**
   * Dynamic callstack flag. The value of this attribute is undefined if the
   * symbol is not a kernel. The type of this attribute is bool.
   *
   * If this flag is set (the value is true), the kernel uses a dynamically
   * sized call stack. This can happen if recursive calls, calls to indirect
   * functions, or the HSAIL alloca instruction are present in the kernel.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = 15,
  /**
   * @deprecated
   *
   * Call convention of the kernel. The value of this attribute is undefined if
   * the symbol is not a kernel. The type of this attribute is uint32_t.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION = 18,
  /**
   * Indirect function object handle. The value of this attribute is undefined
   * if the symbol is not an indirect function, or the associated agent does
   * not support the Full Profile. The type of this attribute depends on the
   * machine model: the type is uint32_t for small machine model, and uint64_t
   * for large model.
   *
   * If the state of the executable is ::HSA_EXECUTABLE_STATE_UNFROZEN, then 0
   * is returned.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT = 23,
  /**
   * @deprecated
   *
   * Call convention of the indirect function. The value of this attribute is
   * undefined if the symbol is not an indirect function, or the associated
   * agent does not support the Full Profile. The type of this attribute is
   * uint32_t.
   */
  HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = 16
} hsa_executable_symbol_info_t;

/**
 * @brief Get the current value of an attribute for a given executable symbol.
 *
 * @param[in] executable_symbol Executable symbol.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL The executable symbol is
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * executable symbol attribute, or @p value is NULL.
 */
hsa_status_t HSA_API hsa_executable_symbol_get_info(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute,
    void *value);

/**
 * @deprecated
 *
 * @brief Iterate over the symbols in a executable, and invoke an
 * application-defined callback on every iteration.
 *
 * @param[in] executable Executable.
 *
 * @param[in] callback Callback to be invoked once per executable symbol. The
 * HSA runtime passes three arguments to the callback: the executable, a symbol,
 * and the application data.  If @p callback returns a status other than
 * ::HSA_STATUS_SUCCESS for a particular iteration, the traversal stops and
 * ::hsa_executable_iterate_symbols returns that status value.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_executable_iterate_symbols(
    hsa_executable_t executable,
    hsa_status_t (*callback)(hsa_executable_t exec,
                             hsa_executable_symbol_t symbol,
                             void *data),
    void *data);

/**
 * @brief Iterate over the kernels, indirect functions, and agent allocation
 * variables in an executable for a given agent, and invoke an application-
 * defined callback on every iteration.
 *
 * @param[in] executable Executable.
 *
 * @param[in] agent Agent.
 *
 * @param[in] callback Callback to be invoked once per executable symbol. The
 * HSA runtime passes three arguments to the callback: the executable, a symbol,
 * and the application data.  If @p callback returns a status other than
 * ::HSA_STATUS_SUCCESS for a particular iteration, the traversal stops and
 * ::hsa_executable_iterate_symbols returns that status value.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL.
 */
hsa_status_t HSA_API hsa_executable_iterate_agent_symbols(
    hsa_executable_t executable,
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t exec,
                             hsa_agent_t agent,
                             hsa_executable_symbol_t symbol,
                             void *data),
    void *data);

/**
 * @brief Iterate over the program allocation variables in an executable, and
 * invoke an application-defined callback on every iteration.
 *
 * @param[in] executable Executable.
 *
 * @param[in] callback Callback to be invoked once per executable symbol. The
 * HSA runtime passes three arguments to the callback: the executable, a symbol,
 * and the application data.  If @p callback returns a status other than
 * ::HSA_STATUS_SUCCESS for a particular iteration, the traversal stops and
 * ::hsa_executable_iterate_symbols returns that status value.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL.
 */
hsa_status_t HSA_API hsa_executable_iterate_program_symbols(
    hsa_executable_t executable,
    hsa_status_t (*callback)(hsa_executable_t exec,
                             hsa_executable_symbol_t symbol,
                             void *data),
    void *data);

/** @} */


/** \defgroup code-object Code Objects (deprecated).
 *  @{
 */

/**
 * @deprecated
 *
 * @brief Struct containing an opaque handle to a code object, which contains
 * ISA for finalized kernels and indirect functions together with information
 * about the global or readonly segment variables they reference.
 */
typedef struct hsa_code_object_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_code_object_t;

/**
 * @deprecated
 *
 * @brief Application data handle that is passed to the serialization
 * and deserialization functions.
 */
typedef struct hsa_callback_data_s {
  /**
   * Opaque handle.
   */
  uint64_t handle;
} hsa_callback_data_t;

/**
 * @deprecated
 *
 * @brief Serialize a code object. Can be used for offline finalization,
 * install-time finalization, disk code caching, etc.
 *
 * @param[in] code_object Code object.
 *
 * @param[in] alloc_callback Callback function for memory allocation. Must not
 * be NULL. The HSA runtime passes three arguments to the callback: the
 * allocation size, the application data, and a pointer to a memory location
 * where the application stores the allocation result. The HSA runtime invokes
 * @p alloc_callback once to allocate a buffer that contains the serialized
 * version of @p code_object.  If the callback returns a status code other than
 * ::HSA_STATUS_SUCCESS, this function returns the same code.
 *
 * @param[in] callback_data Application data that is passed to @p
 * alloc_callback. May be NULL.
 *
 * @param[in] options Standard and vendor-specific options. Unknown options are
 * ignored. A standard option begins with the "-hsa_" prefix. Options beginning
 * with the "-hsa_ext_<extension_name>_" prefix are reserved for extensions. A
 * vendor-specific option begins with the "-<vendor_name>_" prefix. Must be a
 * NUL-terminated string. May be NULL.
 *
 * @param[out] serialized_code_object Memory location where the HSA runtime
 * stores a pointer to the serialized code object. Must not be NULL.
 *
 * @param[out] serialized_code_object_size Memory location where the HSA runtime
 * stores the size (in bytes) of @p serialized_code_object. The returned value
 * matches the allocation size passed by the HSA runtime to @p
 * alloc_callback. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT @p code_object is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p alloc_callback, @p
 * serialized_code_object, or @p serialized_code_object_size are NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_code_object_serialize(
    hsa_code_object_t code_object,
    hsa_status_t (*alloc_callback)(size_t size,
                                   hsa_callback_data_t data,
                                   void **address),
    hsa_callback_data_t callback_data,
    const char *options,
    void **serialized_code_object,
    size_t *serialized_code_object_size);

/**
 * @deprecated
 *
 * @brief Deserialize a code object.
 *
 * @param[in] serialized_code_object A serialized code object. Must not be NULL.
 *
 * @param[in] serialized_code_object_size The size (in bytes) of @p
 * serialized_code_object. Must not be 0.
 *
 * @param[in] options Standard and vendor-specific options. Unknown options are
 * ignored. A standard option begins with the "-hsa_" prefix. Options beginning
 * with the "-hsa_ext_<extension_name>_" prefix are reserved for extensions. A
 * vendor-specific option begins with the "-<vendor_name>_" prefix. Must be a
 * NUL-terminated string. May be NULL.
 *
 * @param[out] code_object Memory location where the HSA runtime stores the
 * deserialized code object.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p serialized_code_object, or @p
 * code_object are NULL, or @p serialized_code_object_size is 0.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_code_object_deserialize(
    void *serialized_code_object,
    size_t serialized_code_object_size,
    const char *options,
    hsa_code_object_t *code_object);

/**
 * @deprecated
 *
 * @brief Destroy a code object.
 *
 * @details The lifetime of a code object must exceed that of any executable
 * where it has been loaded. If an executable that loaded @p code_object has not
 * been destroyed, the behavior is undefined.
 *
 * @param[in] code_object Code object. The handle becomes invalid after it has
 * been destroyed.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT @p code_object is invalid.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_code_object_destroy(
    hsa_code_object_t code_object);

/**
 * @deprecated
 *
 * @brief Code object type.
 */
typedef enum {
  /**
   * Produces code object that contains ISA for all kernels and indirect
   * functions in HSA source.
   */
  HSA_CODE_OBJECT_TYPE_PROGRAM = 0
} hsa_code_object_type_t;

/**
 * @deprecated
 *
 * @brief Code object attributes.
 */
typedef enum {
  /**
   * The version of the code object. The type of this attribute is a
   * NUL-terminated char[64]. The name must be at most 63 characters long (not
   * including the NUL terminator) and all array elements not used for the name
   * must be NUL.
   */
  HSA_CODE_OBJECT_INFO_VERSION = 0,
  /**
   * Type of code object. The type of this attribute is
   * ::hsa_code_object_type_t.
   */
  HSA_CODE_OBJECT_INFO_TYPE = 1,
  /**
   * Instruction set architecture this code object is produced for. The type of
   * this attribute is ::hsa_isa_t.
   */
  HSA_CODE_OBJECT_INFO_ISA = 2,
  /**
   * Machine model this code object is produced for. The type of this attribute
   * is ::hsa_machine_model_t.
   */
  HSA_CODE_OBJECT_INFO_MACHINE_MODEL = 3,
  /**
   * Profile this code object is produced for. The type of this attribute is
   * ::hsa_profile_t.
   */
  HSA_CODE_OBJECT_INFO_PROFILE = 4,
  /**
   * Default floating-point rounding mode used when the code object is
   * produced. The type of this attribute is
   * ::hsa_default_float_rounding_mode_t.
   */
  HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 5
} hsa_code_object_info_t;

/**
 * @deprecated
 *
 * @brief Get the current value of an attribute for a given code object.
 *
 * @param[in] code_object Code object.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT @p code_object is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * code object attribute, or @p value is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_code_object_get_info(
    hsa_code_object_t code_object,
    hsa_code_object_info_t attribute,
    void *value);

/**
 * @deprecated
 *
 * @brief Load code object into the executable.
 *
 * @details Every global or readonly variable that is external must be defined
 * before loading the code object. An internal global or readonly variable is
 * allocated once the code object, that is being loaded, references this
 * variable and this variable is not allocated.
 *
 * Any module linkage declaration must have been defined either by a define
 * variable or by loading a code object that has a symbol with module linkage
 * definition.
 *
 * @param[in] executable Executable.
 *
 * @param[in] agent Agent to load code object for. The agent must support the
 * default floating-point rounding mode used by @p code_object.
 *
 * @param[in] code_object Code object to load.  The lifetime of the code object
 * must exceed that of the executable: if @p code_object is destroyed before @p
 * executable, the behavior is undefined.
 *
 * @param[in] options Standard and vendor-specific options. Unknown options are
 * ignored. A standard option begins with the "-hsa_" prefix. Options beginning
 * with the "-hsa_ext_<extension_name>_" prefix are reserved for extensions. A
 * vendor-specific option begins with the "-<vendor_name>_" prefix. Must be a
 * NUL-terminated string. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_EXECUTABLE The executable is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT @p code_object is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS @p agent is not compatible
 * with @p code_object (for example, @p agent does not support the default
 * floating-point rounding mode specified by @p code_object), or @p code_object
 * is not compatible with @p executable (for example, @p code_object and @p
 * executable have different machine models or profiles).
 *
 * @retval ::HSA_STATUS_ERROR_FROZEN_EXECUTABLE @p executable is frozen.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_executable_load_code_object(
    hsa_executable_t executable,
    hsa_agent_t agent,
    hsa_code_object_t code_object,
    const char *options);

/**
 * @deprecated
 *
 * @brief Code object symbol handle.
 *
 * The lifetime of a code object symbol matches that of the code object
 * associated with it. An operation on a symbol whose associated code object has
 * been destroyed results in undefined behavior.
 */
typedef struct hsa_code_symbol_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_code_symbol_t;

/**
 * @deprecated
 *
 * @brief Get the symbol handle within a code object for a given a symbol name.
 *
 * @param[in] code_object Code object.
 *
 * @param[in] symbol_name Symbol name.
 *
 * @param[out] symbol Memory location where the HSA runtime stores the symbol
 * handle.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT @p code_object is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SYMBOL_NAME There is no symbol with a name
 * that matches @p symbol_name.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p symbol_name is NULL, or
 * @p symbol is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_code_object_get_symbol(
    hsa_code_object_t code_object,
    const char *symbol_name,
    hsa_code_symbol_t *symbol);

/**
 * @deprecated
 *
 * @brief Get the symbol handle within a code object for a given a symbol name.
 *
 * @param[in] code_object Code object.
 *
 * @param[in] module_name Module name. Must be NULL if the symbol has
 * program linkage.
 *
 * @param[in] symbol_name Symbol name.
 *
 * @param[out] symbol Memory location where the HSA runtime stores the symbol
 * handle.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT @p code_object is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SYMBOL_NAME There is no symbol with a name
 * that matches @p symbol_name.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p symbol_name is NULL, or
 * @p symbol is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_code_object_get_symbol_from_name(
    hsa_code_object_t code_object,
    const char *module_name,
    const char *symbol_name,
    hsa_code_symbol_t *symbol);

/**
 * @deprecated
 *
 * @brief Code object symbol attributes.
 */
typedef enum {
  /**
   * The type of the symbol. The type of this attribute is ::hsa_symbol_kind_t.
   */
  HSA_CODE_SYMBOL_INFO_TYPE = 0,
  /**
   * The length of the symbol name in bytes, not including the NUL terminator.
   * The type of this attribute is uint32_t.
   */
  HSA_CODE_SYMBOL_INFO_NAME_LENGTH = 1,
  /**
   * The name of the symbol. The type of this attribute is character array with
   * the length equal to the value of ::HSA_CODE_SYMBOL_INFO_NAME_LENGTH
   * attribute.
   */
  HSA_CODE_SYMBOL_INFO_NAME = 2,
  /**
   * The length of the module name in bytes (not including the NUL terminator)
   * to which this symbol belongs if this symbol has module linkage, otherwise 0
   * is returned. The type of this attribute is uint32_t.
   */
  HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH = 3,
  /**
   * The module name to which this symbol belongs if this symbol has module
   * linkage, otherwise an empty string is returned. The type of this attribute
   * is character array with the length equal to the value of
   * ::HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH attribute.
   */
  HSA_CODE_SYMBOL_INFO_MODULE_NAME = 4,
  /**
   * The linkage kind of the symbol. The type of this attribute is
   * ::hsa_symbol_linkage_t.
   */
  HSA_CODE_SYMBOL_INFO_LINKAGE = 5,
  /**
   * Indicates whether the symbol corresponds to a definition. The type of this
   * attribute is bool.
   */
  HSA_CODE_SYMBOL_INFO_IS_DEFINITION = 17,
  /**
   * The allocation kind of the variable. The value of this attribute is
   * undefined if the symbol is not a variable. The type of this attribute is
   * ::hsa_variable_allocation_t.
   */
  HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION = 6,
  /**
   * The segment kind of the variable. The value of this attribute is
   * undefined if the symbol is not a variable. The type of this attribute is
   * ::hsa_variable_segment_t.
   */
  HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT = 7,
  /**
   * Alignment of the symbol in memory. The value of this attribute is undefined
   * if the symbol is not a variable. The type of this attribute is uint32_t.
   *
   * The current alignment of the variable in memory may be greater than the
   * value specified in the source program variable declaration.
   */
  HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT = 8,
  /**
   * Size of the variable. The value of this attribute is undefined if the
   * symbol is not a variable. The type of this attribute is uint32_t.
   *
   * A size of 0 is returned if the variable is an external variable and has an
   * unknown dimension.
   */
  HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE = 9,
  /**
   * Indicates whether the variable is constant. The value of this attribute is
   * undefined if the symbol is not a variable. The type of this attribute is
   * bool.
   */
  HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST = 10,
  /**
   * Size of kernarg segment memory that is required to hold the values of the
   * kernel arguments, in bytes. Must be a multiple of 16. The value of this
   * attribute is undefined if the symbol is not a kernel. The type of this
   * attribute is uint32_t.
   */
  HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = 11,
  /**
   * Alignment (in bytes) of the buffer used to pass arguments to the kernel,
   * which is the maximum of 16 and the maximum alignment of any of the kernel
   * arguments. The value of this attribute is undefined if the symbol is not a
   * kernel. The type of this attribute is uint32_t.
   */
  HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = 12,
  /**
   * Size of static group segment memory required by the kernel (per
   * work-group), in bytes. The value of this attribute is undefined
   * if the symbol is not a kernel. The type of this attribute is uint32_t.
   *
   * The reported amount does not include any dynamically allocated group
   * segment memory that may be requested by the application when a kernel is
   * dispatched.
   */
  HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = 13,
  /**
   * Size of static private, spill, and arg segment memory required by
   * this kernel (per work-item), in bytes. The value of this attribute is
   * undefined if the symbol is not a kernel. The type of this attribute is
   * uint32_t.
   *
   * If the value of ::HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK is true,
   * the kernel may use more private memory than the reported value, and the
   * application must add the dynamic call stack usage to @a
   * private_segment_size when populating a kernel dispatch packet.
   */
  HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = 14,
  /**
   * Dynamic callstack flag. The value of this attribute is undefined if the
   * symbol is not a kernel. The type of this attribute is bool.
   *
   * If this flag is set (the value is true), the kernel uses a dynamically
   * sized call stack. This can happen if recursive calls, calls to indirect
   * functions, or the HSAIL alloca instruction are present in the kernel.
   */
  HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = 15,
  /**
   * Call convention of the kernel. The value of this attribute is undefined if
   * the symbol is not a kernel. The type of this attribute is uint32_t.
   */
  HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION = 18,
  /**
   * Call convention of the indirect function. The value of this attribute is
   * undefined if the symbol is not an indirect function. The type of this
   * attribute is uint32_t.
   */
  HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = 16
} hsa_code_symbol_info_t;

/**
 * @deprecated
 *
 * @brief Get the current value of an attribute for a given code symbol.
 *
 * @param[in] code_symbol Code symbol.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_SYMBOL The code symbol is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * code symbol attribute, or @p value is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_code_symbol_get_info(
    hsa_code_symbol_t code_symbol,
    hsa_code_symbol_info_t attribute,
    void *value);

/**
 * @deprecated
 *
 * @brief Iterate over the symbols in a code object, and invoke an
 * application-defined callback on every iteration.
 *
 * @param[in] code_object Code object.
 *
 * @param[in] callback Callback to be invoked once per code object symbol. The
 * HSA runtime passes three arguments to the callback: the code object, a
 * symbol, and the application data.  If @p callback returns a status other than
 * ::HSA_STATUS_SUCCESS for a particular iteration, the traversal stops and
 * ::hsa_code_object_iterate_symbols returns that status value.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT @p code_object is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL.
 */
hsa_status_t HSA_API HSA_DEPRECATED hsa_code_object_iterate_symbols(
    hsa_code_object_t code_object,
    hsa_status_t (*callback)(hsa_code_object_t code_object,
                             hsa_code_symbol_t symbol,
                             void *data),
    void *data);

/** @} */

#ifdef __cplusplus
}  // end extern "C" block
#endif

#endif  // header guard
