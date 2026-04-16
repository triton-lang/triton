// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <stdint.h>

/**
 * @defgroup SYMBOL_VERSIONING_GROUP Symbol Versions
 *
 * @brief The names used for the shared library versioned symbols.
 *
 * Every function is annotated with one of the version macros defined in this
 * section.  Each macro specifies a corresponding symbol version string.  After
 * dynamically loading the shared library with @p dlopen, the address of each
 * function can be obtained using @p dlsym with the name of the function and
 * its corresponding symbol version string.  An error will be reported by @p
 * dlvsym if the installed library does not support the version for the
 * function specified in this version of the interface.
 *
 * @{
 */

/**
 * @brief The function was introduced in version 0.0 of the interface and has
 * the symbol version string of ``"ROCPROFILER_SDK_0.0"``.
 */
#define ROCPROFILER_SDK_VERSION_0_0

/** @} */

/**
 * @def ROCPROFILER_SDK_BETA_COMPAT
 * @brief rocprofiler-sdk clients (i.e. tool using rocprofiler-sdk) should set
 * this definition to 1 before including any rocprofiler-sdk header if it wants
 * rocprofiler-sdk to provide preprocessor definitions to help with compilation
 * support for tools prior to v1.0.0 release. Note: for v1.0.0 release,
 * rocprofiler-sdk sets the ppdef to 1 by default. Eventually, rocprofiler-sdk
 * will remove defining this value and it will be up to the tools to define this
 * value.
 *
 * For example in version 0.6.0, there was a function
 * `rocprofiler_create_profile_config` and, prior to the 1.0.0 release, this
 * function was renamed to `rocprofiler_create_counter_config`.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_SDK_BETA_COMPAT_SUPPORTED
 * @brief rocprofiler-sdk will set this preprocessor definition to 1 if it can
 * honor
 * ::ROCPROFILER_SDK_BETA_COMPAT set to 1. Once backwards compatibility with the
 * beta rocprofiler-sdk can no longer be supported, this will always be set to
 * 0.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_SDK_DEPRECATED_WARNINGS
 * @brief Set this preprocessor definition to 0 to silent compiler warnings when
 * using features that are marked as deprecated. By default, rocprofiler-sdk
 * defines this to equal to 1.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_SDK_EXPERIMENTAL_WARNINGS
 * @brief Set this preprocessor definition to 1 to enable compiler warnings when
 * using experimental features. @see ::ROCPROFILER_SDK_EXPERIMENTAL
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_SDK_EXPERIMENTAL
 * @brief When this attribute is added to a type, object, expression, etc., the
 * developer should be aware that the API and/or ABI is subject to change in
 * subsequent releases.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_SDK_COMPUTE_VERSION_VALUE
 * @param[in] MAX_VERSION_VALUE set to +1 of the maximum value for a given
 * version, e.g. a value of 100 means that the max value of the
 * major/minor/patch version is 99
 * @param[in] MAJOR_VERSION major version of library (integral)
 * @param[in] MINOR_VERSION minor version of library (integral)
 * @param[in] PATCH_VERSION patch version of library (integral)
 * @brief Helper macro calculating a generate versioning integer for a library
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_SDK_COMPUTE_VERSION
 * @param[in] MAJOR_VERSION major version of library (integral)
 * @param[in] MINOR_VERSION minor version of library (integral)
 * @param[in] PATCH_VERSION patch version of library (integral)
 * @brief Helper macro for users to generate versioning int expected by
 * rocprofiler-sdk when the library maintains a major, minor, and patch version
 * numbers
 * @addtogroup VERSIONING_GROUP
 */

#if !defined(ROCPROFILER_ATTRIBUTE)
#if defined(_MSC_VER)
#define ROCPROFILER_ATTRIBUTE(...) __declspec(__VA_ARGS__)
#else
#define ROCPROFILER_ATTRIBUTE(...) __attribute__((__VA_ARGS__))
#endif
#endif

#if !defined(ROCPROFILER_PUBLIC_API)
#if defined(_MSC_VER)
#define ROCPROFILER_PUBLIC_API ROCPROFILER_ATTRIBUTE(dllexport)
#else
#define ROCPROFILER_PUBLIC_API ROCPROFILER_ATTRIBUTE(visibility("default"))
#endif
#endif

#if !defined(ROCPROFILER_HIDDEN_API)
#if defined(_MSC_VER)
#define ROCPROFILER_HIDDEN_API
#else
#define ROCPROFILER_HIDDEN_API ROCPROFILER_ATTRIBUTE(visibility("hidden"))
#endif
#endif

#if !defined(ROCPROFILER_EXPORT_DECORATOR)
#define ROCPROFILER_EXPORT_DECORATOR ROCPROFILER_PUBLIC_API
#endif

#if !defined(ROCPROFILER_IMPORT_DECORATOR)
#if defined(_MSC_VER)
#define ROCPROFILER_IMPORT_DECORATOR ROCPROFILER_ATTRIBUTE(dllimport)
#else
#define ROCPROFILER_IMPORT_DECORATOR
#endif
#endif

#define ROCPROFILER_EXPORT ROCPROFILER_EXPORT_DECORATOR
#define ROCPROFILER_IMPORT ROCPROFILER_IMPORT_DECORATOR

#if !defined(ROCPROFILER_API)
#if defined(rocprofiler_EXPORTS)
#define ROCPROFILER_API ROCPROFILER_EXPORT
#else
#define ROCPROFILER_API ROCPROFILER_IMPORT
#endif
#endif

#if defined(__has_attribute)
#if __has_attribute(nonnull)
#define ROCPROFILER_NONNULL(...) __attribute__((nonnull(__VA_ARGS__)))
#else
#define ROCPROFILER_NONNULL(...)
#endif
#else
#if defined(__GNUC__)
#define ROCPROFILER_NONNULL(...) __attribute__((nonnull(__VA_ARGS__)))
#else
#define ROCPROFILER_NONNULL(...)
#endif
#endif

#if __cplusplus >= 201103L // C++11
/* c++11 allows extended initializer lists.  */
#define ROCPROFILER_HANDLE_LITERAL(type, value) (type{value})
#elif __STDC_VERSION__ >= 199901L
/* c99 allows compound literals.  */
#define ROCPROFILER_HANDLE_LITERAL(type, value) ((type){value})
#else
#define ROCPROFILER_HANDLE_LITERAL(type, value) {value}
#endif

#ifdef __cplusplus
#define ROCPROFILER_EXTERN_C_INIT extern "C" {
#define ROCPROFILER_EXTERN_C_FINI }
#define ROCPROFILER_CXX_CODE(...) __VA_ARGS__
#else
#define ROCPROFILER_EXTERN_C_INIT
#define ROCPROFILER_EXTERN_C_FINI
#define ROCPROFILER_CXX_CODE(...)
#endif

#if __cplusplus
#define ROCPROFILER_UINT64_C(value) uint64_t(value)
#else
#define ROCPROFILER_UINT64_C(value) UINT64_C(value)
#endif

#if defined(__cplusplus) && __cplusplus >= 201402L
#define ROCPROFILER_SDK_DEPRECATED_MESSAGE(...) [[deprecated(__VA_ARGS__)]]
#elif !defined(__cplusplus) && defined(__STDC_VERSION__) &&                    \
    __STDC_VERSION__ >= 202311L
#define ROCPROFILER_SDK_DEPRECATED_MESSAGE(...) [[deprecated(__VA_ARGS__)]]
#else
#define ROCPROFILER_SDK_DEPRECATED_MESSAGE(...)                                \
  ROCPROFILER_ATTRIBUTE(deprecated)
#endif

// TODO(jomadsen): uncomment below code before v1.0.0
#if !defined(ROCPROFILER_SDK_DEPRECATED_WARNINGS)
#define ROCPROFILER_SDK_DEPRECATED_WARNINGS 0
#endif

#if !defined(ROCPROFILER_SDK_EXPERIMENTAL_WARNINGS)
#define ROCPROFILER_SDK_EXPERIMENTAL_WARNINGS 0
#endif

#if defined(ROCPROFILER_SDK_DEPRECATED_WARNINGS) &&                            \
    ROCPROFILER_SDK_DEPRECATED_WARNINGS > 0
#define ROCPROFILER_SDK_DEPRECATED(...)                                        \
  ROCPROFILER_SDK_DEPRECATED_MESSAGE(__VA_ARGS__)
#else
#define ROCPROFILER_SDK_DEPRECATED(...)
#endif

#define ROCPROFILER_SDK_EXPERIMENTAL_MESSAGE                                   \
  ROCPROFILER_SDK_DEPRECATED_MESSAGE(                                          \
      "Note: this feature has been marked as experimental. Define "            \
      "ROCPROFILER_SDK_EXPERIMENTAL_WARNINGS=0 to silence this message.")

#if defined(ROCPROFILER_SDK_EXPERIMENTAL_WARNINGS) &&                          \
    ROCPROFILER_SDK_EXPERIMENTAL_WARNINGS > 0
#define ROCPROFILER_SDK_EXPERIMENTAL ROCPROFILER_SDK_EXPERIMENTAL_MESSAGE
#else
#define ROCPROFILER_SDK_EXPERIMENTAL
#endif

//
// if ROCPROFILER_SDK_BETA_COMPAT is > 0, provide some macros to help with
// compatibility. For 1.0.0 release, we define this by default
//
#if !defined(ROCPROFILER_SDK_BETA_COMPAT)
#define ROCPROFILER_SDK_BETA_COMPAT 1
#endif

// rocprofiler-sdk will set ROCPROFILER_SDK_BETA_COMPAT_SUPPORTED to 1 if it can
// support compatibility with rocprofiler-sdk < v1.0.0
#if defined(ROCPROFILER_SDK_BETA_COMPAT) && ROCPROFILER_SDK_BETA_COMPAT > 0
#define ROCPROFILER_SDK_BETA_COMPAT_SUPPORTED 1
#endif

#define ROCPROFILER_SDK_COMPUTE_VERSION_VALUE(MAX_VERSION_VALUE, MAJOR, MINOR, \
                                              PATCH)                           \
  (((MAX_VERSION_VALUE * MAX_VERSION_VALUE) * MAJOR) +                         \
   (MAX_VERSION_VALUE * MINOR) + (PATCH))

#define ROCPROFILER_SDK_COMPUTE_VERSION(MAJOR, MINOR, PATCH)                   \
  ROCPROFILER_SDK_COMPUTE_VERSION_VALUE(100, MAJOR, MINOR, PATCH)
