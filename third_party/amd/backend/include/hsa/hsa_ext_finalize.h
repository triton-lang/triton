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

#ifndef HSA_RUNTIME_INC_HSA_EXT_FINALIZE_H_
#define HSA_RUNTIME_INC_HSA_EXT_FINALIZE_H_

#include "hsa.h"

#undef HSA_API
#ifdef HSA_EXPORT_FINALIZER
#define HSA_API HSA_API_EXPORT
#else
#define HSA_API HSA_API_IMPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct BrigModuleHeader;
typedef struct BrigModuleHeader* BrigModule_t;

/** \defgroup ext-alt-finalizer-extensions Finalization Extensions
 *  @{
 */

/**
 * @brief Enumeration constants added to ::hsa_status_t by this extension.
 */
enum {
  /**
   * The HSAIL program is invalid.
   */
  HSA_EXT_STATUS_ERROR_INVALID_PROGRAM = 0x2000,
  /**
   * The HSAIL module is invalid.
   */
  HSA_EXT_STATUS_ERROR_INVALID_MODULE = 0x2001,
  /**
   * Machine model or profile of the HSAIL module do not match the machine model
   * or profile of the HSAIL program.
   */
  HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE = 0x2002,
  /**
   * The HSAIL module is already a part of the HSAIL program.
   */
  HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED = 0x2003,
  /**
   * Compatibility mismatch between symbol declaration and symbol definition.
   */
  HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH = 0x2004,
  /**
   * The finalization encountered an error while finalizing a kernel or
   * indirect function.
   */
  HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED = 0x2005,
  /**
   * Mismatch between a directive in the control directive structure and in
   * the HSAIL kernel.
   */
  HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH = 0x2006
};

/** @} */

/** \defgroup ext-alt-finalizer-program Finalization Program
 *  @{
 */

/**
 * @brief HSAIL (BRIG) module. The HSA Programmer's Reference Manual contains
 * the definition of the BrigModule_t type.
 */
typedef BrigModule_t hsa_ext_module_t;

/**
 * @brief An opaque handle to a HSAIL program, which groups a set of HSAIL
 * modules that collectively define functions and variables used by kernels and
 * indirect functions.
 */
typedef struct hsa_ext_program_s {
  /**
   * Opaque handle.
   */
  uint64_t handle;
} hsa_ext_program_t;

/**
 * @brief Create an empty HSAIL program.
 *
 * @param[in] machine_model Machine model used in the HSAIL program.
 *
 * @param[in] profile Profile used in the HSAIL program.
 *
 * @param[in] default_float_rounding_mode Default float rounding mode used in
 * the HSAIL program.
 *
 * @param[in] options Vendor-specific options. May be NULL.
 *
 * @param[out] program Memory location where the HSA runtime stores the newly
 * created HSAIL program handle.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES There is a failure to allocate
 * resources required for the operation.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p machine_model is invalid,
 * @p profile is invalid, @p default_float_rounding_mode is invalid, or
 * @p program is NULL.
 */
hsa_status_t HSA_API hsa_ext_program_create(
    hsa_machine_model_t machine_model,
    hsa_profile_t profile,
    hsa_default_float_rounding_mode_t default_float_rounding_mode,
    const char *options,
    hsa_ext_program_t *program);

/**
 * @brief Destroy a HSAIL program.
 *
 * @details The HSAIL program handle becomes invalid after it has been
 * destroyed. Code object handles produced by ::hsa_ext_program_finalize are
 * still valid after the HSAIL program has been destroyed, and can be used as
 * intended. Resources allocated outside and associated with the HSAIL program
 * (such as HSAIL modules that are added to the HSAIL program) can be released
 * after the finalization program has been destroyed.
 *
 * @param[in] program HSAIL program.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_INVALID_PROGRAM The HSAIL program is
 * invalid.
 */
hsa_status_t HSA_API hsa_ext_program_destroy(
    hsa_ext_program_t program);

/**
 * @brief Add a HSAIL module to an existing HSAIL program.
 *
 * @details The HSA runtime does not perform a deep copy of the HSAIL module
 * upon addition. Instead, it stores a pointer to the HSAIL module. The
 * ownership of the HSAIL module belongs to the application, which must ensure
 * that @p module is not released before destroying the HSAIL program.
 *
 * The HSAIL module is successfully added to the HSAIL program if @p module is
 * valid, if all the declarations and definitions for the same symbol are
 * compatible, and if @p module specify machine model and profile that matches
 * the HSAIL program.
 *
 * @param[in] program HSAIL program.
 *
 * @param[in] module HSAIL module. The application can add the same HSAIL module
 * to @p program at most once. The HSAIL module must specify the same machine
 * model and profile as @p program. If the floating-mode rounding mode of @p
 * module is not default, then it should match that of @p program.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES There is a failure to allocate
 * resources required for the operation.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_INVALID_PROGRAM The HSAIL program is invalid.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_INVALID_MODULE The HSAIL module is invalid.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE The machine model of @p
 * module does not match machine model of @p program, or the profile of @p
 * module does not match profile of @p program.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED The HSAIL module is
 * already a part of the HSAIL program.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH Symbol declaration and symbol
 * definition compatibility mismatch. See the symbol compatibility rules in the
 * HSA Programming Reference Manual.
 */
hsa_status_t HSA_API hsa_ext_program_add_module(
    hsa_ext_program_t program,
    hsa_ext_module_t module);

/**
 * @brief Iterate over the HSAIL modules in a program, and invoke an
 * application-defined callback on every iteration.
 *
 * @param[in] program HSAIL program.
 *
 * @param[in] callback Callback to be invoked once per HSAIL module in the
 * program. The HSA runtime passes three arguments to the callback: the program,
 * a HSAIL module, and the application data.  If @p callback returns a status
 * other than ::HSA_STATUS_SUCCESS for a particular iteration, the traversal
 * stops and ::hsa_ext_program_iterate_modules returns that status value.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_INVALID_PROGRAM The program is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL.
 */
hsa_status_t HSA_API hsa_ext_program_iterate_modules(
    hsa_ext_program_t program,
    hsa_status_t (*callback)(hsa_ext_program_t program, hsa_ext_module_t module,
                             void* data),
    void* data);

/**
 * @brief HSAIL program attributes.
 */
typedef enum {
  /**
   * Machine model specified when the HSAIL program was created. The type
   * of this attribute is ::hsa_machine_model_t.
   */
  HSA_EXT_PROGRAM_INFO_MACHINE_MODEL = 0,
  /**
   * Profile specified when the HSAIL program was created. The type of
   * this attribute is ::hsa_profile_t.
   */
  HSA_EXT_PROGRAM_INFO_PROFILE = 1,
  /**
   * Default float rounding mode specified when the HSAIL program was
   * created. The type of this attribute is ::hsa_default_float_rounding_mode_t.
   */
  HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 2
} hsa_ext_program_info_t;

/**
 * @brief Get the current value of an attribute for a given HSAIL program.
 *
 * @param[in] program HSAIL program.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behaviour is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_INVALID_PROGRAM The HSAIL program is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p attribute is an invalid
 * HSAIL program attribute, or @p value is NULL.
 */
hsa_status_t HSA_API hsa_ext_program_get_info(
    hsa_ext_program_t program,
    hsa_ext_program_info_t attribute,
    void *value);

/**
 * @brief Finalizer-determined call convention.
 */
typedef enum {
 /**
  * Finalizer-determined call convention.
  */
  HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO = -1
} hsa_ext_finalizer_call_convention_t;

/**
 * @brief Control directives specify low-level information about the
 * finalization process.
 */
typedef struct hsa_ext_control_directives_s {
  /**
   * Bitset indicating which control directives are enabled. The bit assigned to
   * a control directive is determined by the corresponding value in
   * BrigControlDirective.
   *
   * If a control directive is disabled, its corresponding field value (if any)
   * must be 0. Control directives that are only present or absent (such as
   * partial workgroups) have no corresponding field as the presence of the bit
   * in this mask is sufficient.
   */
  uint64_t control_directives_mask;
  /**
   * Bitset of HSAIL exceptions that must have the BREAK policy enabled. The bit
   * assigned to an HSAIL exception is determined by the corresponding value
   * in BrigExceptionsMask. If the kernel contains a enablebreakexceptions
   * control directive, the finalizer uses the union of the two masks.
   */
  uint16_t break_exceptions_mask;
  /**
   * Bitset of HSAIL exceptions that must have the DETECT policy enabled. The
   * bit assigned to an HSAIL exception is determined by the corresponding value
   * in BrigExceptionsMask. If the kernel contains a enabledetectexceptions
   * control directive, the finalizer uses the union of the two masks.
   */
  uint16_t detect_exceptions_mask;
  /**
   * Maximum size (in bytes) of dynamic group memory that will be allocated by
   * the application for any dispatch of the kernel.  If the kernel contains a
   * maxdynamicsize control directive, the two values should match.
   */
  uint32_t max_dynamic_group_size;
  /**
   * Maximum number of grid work-items that will be used by the application to
   * launch the kernel. If the kernel contains a maxflatgridsize control
   * directive, the value of @a max_flat_grid_size must not be greater than the
   * value of the directive, and takes precedence.
   *
   * The value specified for maximum absolute grid size must be greater than or
   * equal to the product of the values specified by @a required_grid_size.
   *
   * If the bit at position BRIG_CONTROL_MAXFLATGRIDSIZE is set in @a
   * control_directives_mask, this field must be greater than 0.
   */
  uint64_t max_flat_grid_size;
  /**
   * Maximum number of work-group work-items that will be used by the
   * application to launch the kernel. If the kernel contains a
   * maxflatworkgroupsize control directive, the value of @a
   * max_flat_workgroup_size must not be greater than the value of the
   * directive, and takes precedence.
   *
   * The value specified for maximum absolute grid size must be greater than or
   * equal to the product of the values specified by @a required_workgroup_size.
   *
   * If the bit at position BRIG_CONTROL_MAXFLATWORKGROUPSIZE is set in @a
   * control_directives_mask, this field must be greater than 0.
   */
  uint32_t max_flat_workgroup_size;
  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;
  /**
   * Grid size that will be used by the application in any dispatch of the
   * kernel. If the kernel contains a requiredgridsize control directive, the
   * dimensions should match.
   *
   * The specified grid size must be consistent with @a required_workgroup_size
   * and @a required_dim. Also, the product of the three dimensions must not
   * exceed @a max_flat_grid_size. Note that the listed invariants must hold
   * only if all the corresponding control directives are enabled.
   *
   * If the bit at position BRIG_CONTROL_REQUIREDGRIDSIZE is set in @a
   * control_directives_mask, the three dimension values must be greater than 0.
   */
  uint64_t required_grid_size[3];
  /**
   * Work-group size that will be used by the application in any dispatch of the
   * kernel. If the kernel contains a requiredworkgroupsize control directive,
   * the dimensions should match.
   *
   * The specified work-group size must be consistent with @a required_grid_size
   * and @a required_dim. Also, the product of the three dimensions must not
   * exceed @a max_flat_workgroup_size. Note that the listed invariants must
   * hold only if all the corresponding control directives are enabled.
   *
   * If the bit at position BRIG_CONTROL_REQUIREDWORKGROUPSIZE is set in @a
   * control_directives_mask, the three dimension values must be greater than 0.
   */
  hsa_dim3_t required_workgroup_size;
  /**
   * Number of dimensions that will be used by the application to launch the
   * kernel. If the kernel contains a requireddim control directive, the two
   * values should match.
   *
   * The specified dimensions must be consistent with @a required_grid_size and
   * @a required_workgroup_size. This invariant must hold only if all the
   * corresponding control directives are enabled.
   *
   * If the bit at position BRIG_CONTROL_REQUIREDDIM is set in @a
   * control_directives_mask, this field must be 1, 2, or 3.
   */
  uint8_t required_dim;
  /**
   * Reserved. Must be 0.
   */
  uint8_t reserved2[75];
} hsa_ext_control_directives_t;

/**
 * @brief Finalize an HSAIL program for a given instruction set architecture.
 *
 * @details Finalize all of the kernels and indirect functions that belong to
 * the same HSAIL program for a specific instruction set architecture (ISA). The
 * transitive closure of all functions specified by call or scall must be
 * defined. Kernels and indirect functions that are being finalized must be
 * defined. Kernels and indirect functions that are referenced in kernels and
 * indirect functions being finalized may or may not be defined, but must be
 * declared. All the global/readonly segment variables that are referenced in
 * kernels and indirect functions being finalized may or may not be defined, but
 * must be declared.
 *
 * @param[in] program HSAIL program.
 *
 * @param[in] isa Instruction set architecture to finalize for.
 *
 * @param[in] call_convention A call convention used in a finalization. Must
 * have a value between ::HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO (inclusive)
 * and the value of the attribute ::HSA_ISA_INFO_CALL_CONVENTION_COUNT in @p
 * isa (not inclusive).
 *
 * @param[in] control_directives Low-level control directives that influence
 * the finalization process.
 *
 * @param[in] options Vendor-specific options. May be NULL.
 *
 * @param[in] code_object_type Type of code object to produce.
 *
 * @param[out] code_object Code object generated by the Finalizer, which
 * contains the machine code for the kernels and indirect functions in the HSAIL
 * program. The code object is independent of the HSAIL module that was used to
 * generate it.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES There is a failure to allocate
 * resources required for the operation.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_INVALID_PROGRAM The HSAIL program is
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ISA @p isa is invalid.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH The directive in
 * the control directive structure and in the HSAIL kernel mismatch, or if the
 * same directive is used with a different value in one of the functions used by
 * this kernel.
 *
 * @retval ::HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED The Finalizer
 * encountered an error while compiling a kernel or an indirect function.
 */
hsa_status_t HSA_API hsa_ext_program_finalize(
    hsa_ext_program_t program,
    hsa_isa_t isa,
    int32_t call_convention,
    hsa_ext_control_directives_t control_directives,
    const char *options,
    hsa_code_object_type_t code_object_type,
    hsa_code_object_t *code_object);

/** @} */

#define hsa_ext_finalizer_1_00

typedef struct hsa_ext_finalizer_1_00_pfn_s {
  hsa_status_t (*hsa_ext_program_create)(
      hsa_machine_model_t machine_model, hsa_profile_t profile,
      hsa_default_float_rounding_mode_t default_float_rounding_mode,
      const char *options, hsa_ext_program_t *program);

  hsa_status_t (*hsa_ext_program_destroy)(hsa_ext_program_t program);

  hsa_status_t (*hsa_ext_program_add_module)(hsa_ext_program_t program,
                                                 hsa_ext_module_t module);

  hsa_status_t (*hsa_ext_program_iterate_modules)(
      hsa_ext_program_t program,
      hsa_status_t (*callback)(hsa_ext_program_t program,
                               hsa_ext_module_t module, void *data),
      void *data);

  hsa_status_t (*hsa_ext_program_get_info)(
      hsa_ext_program_t program, hsa_ext_program_info_t attribute,
      void *value);

  hsa_status_t (*hsa_ext_program_finalize)(
      hsa_ext_program_t program, hsa_isa_t isa, int32_t call_convention,
      hsa_ext_control_directives_t control_directives, const char *options,
      hsa_code_object_type_t code_object_type, hsa_code_object_t *code_object);
} hsa_ext_finalizer_1_00_pfn_t;

#ifdef __cplusplus
} // extern "C" block
#endif // __cplusplus

#endif // HSA_RUNTIME_INC_HSA_EXT_FINALIZE_H_
