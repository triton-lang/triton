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

// HSA AMD extension for additional loader functionality.

#ifndef HSA_VEN_AMD_LOADER_H
#define HSA_VEN_AMD_LOADER_H

#include "hsa.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Queries equivalent host address for given @p device_address, and
 * records it in @p host_address.
 *
 *
 * @details Contents of memory pointed to by @p host_address would be identical
 * to contents of memory pointed to by @p device_address. Only difference
 * between the two is host accessibility: @p host_address is always accessible
 * from host, @p device_address might not be accessible from host.
 *
 * If @p device_address already points to host accessible memory, then the value
 * of @p device_address is simply copied into @p host_address.
 *
 * The lifetime of @p host_address is the same as the lifetime of @p
 * device_address, and both lifetimes are limited by the lifetime of the
 * executable that is managing these addresses.
 *
 *
 * @param[in] device_address Device address to query equivalent host address
 * for.
 *
 * @param[out] host_address Pointer to application-allocated buffer to record
 * queried equivalent host address in.
 *
 *
 * @retval HSA_STATUS_SUCCESS Function is executed successfully.
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED Runtime is not initialized.
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT @p device_address is invalid or
 * null, or @p host_address is null.
 */
hsa_status_t hsa_ven_amd_loader_query_host_address(
  const void *device_address,
  const void **host_address);

/**
 * @brief The storage type of the code object that is backing loaded memory
 * segment.
 */
typedef enum {
  /**
   * Loaded memory segment is not backed by any code object (anonymous), as the
   * case would be with BSS (uninitialized data).
   */
  HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE = 0,
  /**
   * Loaded memory segment is backed by the code object that is stored in the
   * file.
   */
  HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_FILE = 1,
  /**
   * Loaded memory segment is backed by the code object that is stored in the
   * memory.
   */
  HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY = 2
} hsa_ven_amd_loader_code_object_storage_type_t;

/**
 * @brief Loaded memory segment descriptor.
 *
 *
 * @details Loaded memory segment descriptor describes underlying loaded memory
 * segment. Loaded memory segment is created/allocated by the executable during
 * the loading of the code object that is backing underlying memory segment.
 *
 * The lifetime of underlying memory segment is limited by the lifetime of the
 * executable that is managing underlying memory segment.
 */
typedef struct hsa_ven_amd_loader_segment_descriptor_s {
  /**
   * Agent underlying memory segment is allocated on. If the code object that is
   * backing underlying memory segment is program code object, then 0.
   */
  hsa_agent_t agent;
  /**
   * Executable that is managing this underlying memory segment.
   */
  hsa_executable_t executable;
  /**
   * Storage type of the code object that is backing underlying memory segment.
   */
  hsa_ven_amd_loader_code_object_storage_type_t code_object_storage_type;
  /**
   * If the storage type of the code object that is backing underlying memory
   * segment is:
   *   - HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE, then null;
   *   - HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_FILE, then null-terminated
   *     filepath to the code object;
   *   - HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY, then host
   *     accessible pointer to the first byte of the code object.
   */
  const void *code_object_storage_base;
  /**
   * If the storage type of the code object that is backing underlying memory
   * segment is:
   *   - HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE, then 0;
   *   - HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_FILE, then the length of
   *     the filepath to the code object (including null-terminating character);
   *   - HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY, then the size, in
   *     bytes, of the memory occupied by the code object.
   */
  size_t code_object_storage_size;
  /**
   * If the storage type of the code object that is backing underlying memory
   * segment is:
   *   - HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE, then 0;
   *   - other, then offset, in bytes, from the beginning of the code object to
   *     the first byte in the code object data is copied from.
   */
  size_t code_object_storage_offset;
  /**
   * Starting address of the underlying memory segment.
   */
  const void *segment_base;
  /**
   * Size, in bytes, of the underlying memory segment.
   */
  size_t segment_size;
} hsa_ven_amd_loader_segment_descriptor_t;

/**
 * @brief Either queries loaded memory segment descriptors, or total number of
 * loaded memory segment descriptors.
 *
 *
 * @details If @p segment_descriptors is not null and @p num_segment_descriptors
 * points to number that exactly matches total number of loaded memory segment
 * descriptors, then queries loaded memory segment descriptors, and records them
 * in @p segment_descriptors. If @p segment_descriptors is null and @p
 * num_segment_descriptors points to zero, then queries total number of loaded
 * memory segment descriptors, and records it in @p num_segment_descriptors. In
 * all other cases returns appropriate error code (see below).
 *
 * The caller of this function is responsible for the allocation/deallocation
 * and the lifetime of @p segment_descriptors and @p num_segment_descriptors.
 *
 * The lifetime of loaded memory segments that are described by queried loaded
 * memory segment descriptors is limited by the lifetime of the executable that
 * is managing loaded memory segments.
 *
 * Queried loaded memory segment descriptors are always self-consistent: they
 * describe a complete set of loaded memory segments that are being backed by
 * fully loaded code objects that are present at the time (i.e. this function
 * is blocked until all executable manipulations are fully complete).
 *
 *
 * @param[out] segment_descriptors Pointer to application-allocated buffer to
 * record queried loaded memory segment descriptors in. Can be null if @p
 * num_segment_descriptors points to zero.
 *
 * @param[in,out] num_segment_descriptors Pointer to application-allocated
 * buffer that contains either total number of loaded memory segment descriptors
 * or zero.
 *
 *
 * @retval HSA_STATUS_SUCCESS Function is executed successfully.
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED Runtime is not initialized.
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT @p segment_descriptors is null
 * while @p num_segment_descriptors points to non-zero number, @p
 * segment_descriptors is not null while @p num_segment_descriptors points to
 * zero, or @p num_segment_descriptors is null.
 *
 * @retval HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS @p num_segment_descriptors
 * does not point to number that exactly matches total number of loaded memory
 * segment descriptors.
 */
hsa_status_t hsa_ven_amd_loader_query_segment_descriptors(
  hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
  size_t *num_segment_descriptors);

/**
 * @brief Obtains the handle of executable to which the device address belongs.
 *
 * @details This method should not be used to obtain executable handle by using
 * a host address. The executable returned is expected to be alive until its
 * destroyed by the user.
 *
 * @retval HSA_STATUS_SUCCESS Function is executed successfully.
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED Runtime is not initialized.
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT The input is invalid or there
 * is no exectuable found for this kernel code object.
 */
hsa_status_t hsa_ven_amd_loader_query_executable(
  const void *device_address,
  hsa_executable_t *executable);

//===----------------------------------------------------------------------===//

/**
 * @brief Iterate over the loaded code objects in an executable, and invoke
 * an application-defined callback on every iteration.
 *
 * @param[in] executable Executable.
 *
 * @param[in] callback Callback to be invoked once per loaded code object. The
 * HSA runtime passes three arguments to the callback: the executable, a
 * loaded code object, and the application data. If @p callback returns a
 * status other than ::HSA_STATUS_SUCCESS for a particular iteration, the
 * traversal stops and
 * ::hsa_ven_amd_loader_executable_iterate_loaded_code_objects returns that
 * status value.
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
hsa_status_t hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
  hsa_executable_t executable,
  hsa_status_t (*callback)(
    hsa_executable_t executable,
    hsa_loaded_code_object_t loaded_code_object,
    void *data),
  void *data);

/**
 * @brief Loaded code object kind.
 */
typedef enum {
  /**
   * Program code object.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_PROGRAM = 1,
  /**
   * Agent code object.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_AGENT = 2
} hsa_ven_amd_loader_loaded_code_object_kind_t;

/**
 * @brief Loaded code object attributes.
 */
typedef enum hsa_ven_amd_loader_loaded_code_object_info_e {
  /**
   * The executable in which this loaded code object is loaded. The
   * type of this attribute is ::hsa_executable_t.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE = 1,
  /**
   * The kind of this loaded code object. The type of this attribute is
   * ::uint32_t interpreted as ::hsa_ven_amd_loader_loaded_code_object_kind_t.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_KIND = 2,
  /**
   * The agent on which this loaded code object is loaded. The
   * value of this attribute is only defined if
   * ::HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_KIND is
   * ::HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_AGENT. The type of this
   * attribute is ::hsa_agent_t.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT = 3,
  /**
   * The storage type of the code object reader used to load the loaded code object.
   * The type of this attribute is ::uint32_t interpreted as a
   * ::hsa_ven_amd_loader_code_object_storage_type_t.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE = 4,
  /**
   * The memory address of the first byte of the code object that was loaaded.
   * The value of this attribute is only defined if
   * ::HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE is
   * ::HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY. The type of this
   * attribute is ::uint64_t.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE = 5,
  /**
   * The memory size in bytes of the code object that was loaaded.
   * The value of this attribute is only defined if
   * ::HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE is
   * ::HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY. The type of this
   * attribute is ::uint64_t.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE = 6,
  /**
   * The file descriptor of the code object that was loaaded.
   * The value of this attribute is only defined if
   * ::HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE is
   * ::HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_FILE. The type of this
   * attribute is ::int.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE = 7,
  /**
   * The signed byte address difference of the memory address at which the code
   * object is loaded minus the virtual address specified in the code object
   * that is loaded. The value of this attribute is only defined if the
   * executable in which the code object is loaded is froozen. The type of this
   * attribute is ::int64_t.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA = 8,
  /**
   * The base memory address at which the code object is loaded. This is the
   * base address of the allocation for the lowest addressed segment of the code
   * object that is loaded. Note that any non-loaded segments before the first
   * loaded segment are ignored. The value of this attribute is only defined if
   * the executable in which the code object is loaded is froozen. The type of
   * this attribute is ::uint64_t.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE = 9,
  /**
   * The byte size of the loaded code objects contiguous memory allocation. The
   * value of this attribute is only defined if the executable in which the code
   * object is loaded is froozen. The type of this attribute is ::uint64_t.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE = 10,
  /**
   * The length of the URI in bytes, not including the NUL terminator. The type
   * of this attribute is uint32_t.
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH = 11,
  /**
   * The URI name from which the code object was loaded. The type of this
   * attribute is a NUL terminated \p char* with the length equal to the value
   * of ::HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH attribute.
   * The URI name syntax is defined by the following BNF syntax:
   *
   *     code_object_uri ::== file_uri | memory_uri
   *     file_uri        ::== "file://" file_path [ range_specifier ]
   *     memory_uri      ::== "memory://" process_id range_specifier
   *     range_specifier ::== [ "#" | "?" ] "offset=" number "&" "size=" number
   *     file_path       ::== URI_ENCODED_OS_FILE_PATH
   *     process_id      ::== DECIMAL_NUMBER
   *     number          ::== HEX_NUMBER | DECIMAL_NUMBER | OCTAL_NUMBER
   *
   * ``number`` is a C integral literal where hexadecimal values are prefixed by
   * "0x" or "0X", and octal values by "0".
   *
   * ``file_path`` is the file's path specified as a URI encoded UTF-8 string.
   * In URI encoding, every character that is not in the regular expression
   * ``[a-zA-Z0-9/_.~-]`` is encoded as two uppercase hexidecimal digits
   * proceeded by "%".  Directories in the path are separated by "/".
   *
   * ``offset`` is a 0-based byte offset to the start of the code object.  For a
   * file URI, it is from the start of the file specified by the ``file_path``,
   * and if omitted defaults to 0. For a memory URI, it is the memory address
   * and is required.
   *
   * ``size`` is the number of bytes in the code object.  For a file URI, if
   * omitted it defaults to the size of the file.  It is required for a memory
   * URI.
   *
   * ``process_id`` is the identity of the process owning the memory.  For Linux
   * it is the C unsigned integral decimal literal for the process ID (PID).
   *
   * For example:
   *
   *     file:///dir1/dir2/file1
   *     file:///dir3/dir4/file2#offset=0x2000&size=3000
   *     memory://1234#offset=0x20000&size=3000
   */
  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI = 12,
} hsa_ven_amd_loader_loaded_code_object_info_t;

/**
 * @brief Get the current value of an attribute for a given loaded code
 * object.
 *
 * @param[in] loaded_code_object Loaded code object.
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
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT The loaded code object is
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * loaded code object attribute, or @p value is NULL.
 */
hsa_status_t hsa_ven_amd_loader_loaded_code_object_get_info(
  hsa_loaded_code_object_t loaded_code_object,
  hsa_ven_amd_loader_loaded_code_object_info_t attribute,
  void *value);

//===----------------------------------------------------------------------===//

/**
 * @brief Create a code object reader to operate on a file with size and offset.
 *
 * @param[in] file File descriptor. The file must have been opened by
 * application with at least read permissions prior calling this function. The
 * file must contain a vendor-specific code object.
 *
 * The file is owned and managed by the application; the lifetime of the file
 * descriptor must exceed that of any associated code object reader.
 *
 * @param[in] size Size of the code object embedded in @p file.
 *
 * @param[in] offset 0-based offset relative to the beginning of the @p file
 * that denotes the beginning of the code object embedded within the @p file.
 *
 * @param[out] code_object_reader Memory location to store the newly created
 * code object reader handle. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_FILE @p file is not opened with at least
 * read permissions. This condition may also be reported as
 * ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER by the
 * ::hsa_executable_load_agent_code_object function.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT The bytes starting at offset
 * do not form a valid code object. If file size is 0. Or offset > file size.
 * This condition may also be reported as
 * ::HSA_STATUS_ERROR_INVALID_CODE_OBJECT by the
 * ::hsa_executable_load_agent_code_object function.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to
 * allocate the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p code_object_reader is NULL.
 */
hsa_status_t
hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t file,
    size_t offset,
    size_t size,
    hsa_code_object_reader_t *code_object_reader);

//===----------------------------------------------------------------------===//

/**
 * @brief Iterate over the available executables, and invoke an
 * application-defined callback on every iteration. While
 * ::hsa_ven_amd_loader_iterate_executables is executing any calls to
 * ::hsa_executable_create, ::hsa_executable_create_alt, or
 * ::hsa_executable_destroy will be blocked.
 *
 * @param[in] callback Callback to be invoked once per executable. The HSA
 * runtime passes two arguments to the callback: the executable and the
 * application data. If @p callback returns a status other than
 * ::HSA_STATUS_SUCCESS for a particular iteration, the traversal stops and
 * ::hsa_ven_amd_loader_iterate_executables returns that status value. If
 * @p callback invokes ::hsa_executable_create, ::hsa_executable_create_alt, or
 * ::hsa_executable_destroy then the behavior is undefined.
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
hsa_status_t
hsa_ven_amd_loader_iterate_executables(
    hsa_status_t (*callback)(
      hsa_executable_t executable,
      void *data),
    void *data);

//===----------------------------------------------------------------------===//

/**
 * @brief Extension version.
 */
#define hsa_ven_amd_loader 001003

/**
 * @brief Extension function table version 1.00.
 */
typedef struct hsa_ven_amd_loader_1_00_pfn_s {
  hsa_status_t (*hsa_ven_amd_loader_query_host_address)(
    const void *device_address,
    const void **host_address);

  hsa_status_t (*hsa_ven_amd_loader_query_segment_descriptors)(
    hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
    size_t *num_segment_descriptors);

  hsa_status_t (*hsa_ven_amd_loader_query_executable)(
    const void *device_address,
    hsa_executable_t *executable);
} hsa_ven_amd_loader_1_00_pfn_t;

/**
 * @brief Extension function table version 1.01.
 */
typedef struct hsa_ven_amd_loader_1_01_pfn_s {
  hsa_status_t (*hsa_ven_amd_loader_query_host_address)(
    const void *device_address,
    const void **host_address);

  hsa_status_t (*hsa_ven_amd_loader_query_segment_descriptors)(
    hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
    size_t *num_segment_descriptors);

  hsa_status_t (*hsa_ven_amd_loader_query_executable)(
    const void *device_address,
    hsa_executable_t *executable);

  hsa_status_t (*hsa_ven_amd_loader_executable_iterate_loaded_code_objects)(
    hsa_executable_t executable,
    hsa_status_t (*callback)(
      hsa_executable_t executable,
      hsa_loaded_code_object_t loaded_code_object,
      void *data),
    void *data);

  hsa_status_t (*hsa_ven_amd_loader_loaded_code_object_get_info)(
    hsa_loaded_code_object_t loaded_code_object,
    hsa_ven_amd_loader_loaded_code_object_info_t attribute,
    void *value);
} hsa_ven_amd_loader_1_01_pfn_t;

/**
 * @brief Extension function table version 1.02.
 */
typedef struct hsa_ven_amd_loader_1_02_pfn_s {
  hsa_status_t (*hsa_ven_amd_loader_query_host_address)(
    const void *device_address,
    const void **host_address);

  hsa_status_t (*hsa_ven_amd_loader_query_segment_descriptors)(
    hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
    size_t *num_segment_descriptors);

  hsa_status_t (*hsa_ven_amd_loader_query_executable)(
    const void *device_address,
    hsa_executable_t *executable);

  hsa_status_t (*hsa_ven_amd_loader_executable_iterate_loaded_code_objects)(
    hsa_executable_t executable,
    hsa_status_t (*callback)(
      hsa_executable_t executable,
      hsa_loaded_code_object_t loaded_code_object,
      void *data),
    void *data);

  hsa_status_t (*hsa_ven_amd_loader_loaded_code_object_get_info)(
    hsa_loaded_code_object_t loaded_code_object,
    hsa_ven_amd_loader_loaded_code_object_info_t attribute,
    void *value);

  hsa_status_t
    (*hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size)(
      hsa_file_t file,
      size_t offset,
      size_t size,
      hsa_code_object_reader_t *code_object_reader);
} hsa_ven_amd_loader_1_02_pfn_t;

/**
 * @brief Extension function table version 1.03.
 */
typedef struct hsa_ven_amd_loader_1_03_pfn_s {
  hsa_status_t (*hsa_ven_amd_loader_query_host_address)(
    const void *device_address,
    const void **host_address);

  hsa_status_t (*hsa_ven_amd_loader_query_segment_descriptors)(
    hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
    size_t *num_segment_descriptors);

  hsa_status_t (*hsa_ven_amd_loader_query_executable)(
    const void *device_address,
    hsa_executable_t *executable);

  hsa_status_t (*hsa_ven_amd_loader_executable_iterate_loaded_code_objects)(
    hsa_executable_t executable,
    hsa_status_t (*callback)(
      hsa_executable_t executable,
      hsa_loaded_code_object_t loaded_code_object,
      void *data),
    void *data);

  hsa_status_t (*hsa_ven_amd_loader_loaded_code_object_get_info)(
    hsa_loaded_code_object_t loaded_code_object,
    hsa_ven_amd_loader_loaded_code_object_info_t attribute,
    void *value);

  hsa_status_t
    (*hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size)(
      hsa_file_t file,
      size_t offset,
      size_t size,
      hsa_code_object_reader_t *code_object_reader);

  hsa_status_t
    (*hsa_ven_amd_loader_iterate_executables)(
      hsa_status_t (*callback)(
        hsa_executable_t executable,
        void *data),
      void *data);
} hsa_ven_amd_loader_1_03_pfn_t;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* HSA_VEN_AMD_LOADER_H */
