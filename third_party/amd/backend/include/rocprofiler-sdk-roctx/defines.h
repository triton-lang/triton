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
 * dlsym if the installed library does not support the version for the
 * function specified in this version of the interface.
 *
 * @{
 */

/**
 * @brief The function was introduced in version 0.0 of the interface and has
 * the symbol version string of ``"ROCPROFILER_SDK_ROCTX_0.0"``.
 */
#define ROCPROFILER_SDK_ROCTX_VERSION_0_0

/** @} */

#if !defined(ROCTX_ATTRIBUTE)
#if defined(_MSC_VER)
#define ROCTX_ATTRIBUTE(...) __declspec(__VA_ARGS__)
#else
#define ROCTX_ATTRIBUTE(...) __attribute__((__VA_ARGS__))
#endif
#endif

#if !defined(ROCTX_PUBLIC_API)
#if defined(_MSC_VER)
#define ROCTX_PUBLIC_API ROCTX_ATTRIBUTE(dllexport)
#else
#define ROCTX_PUBLIC_API ROCTX_ATTRIBUTE(visibility("default"))
#endif
#endif

#if !defined(ROCTX_HIDDEN_API)
#if defined(_MSC_VER)
#define ROCTX_HIDDEN_API
#else
#define ROCTX_HIDDEN_API ROCTX_ATTRIBUTE(visibility("hidden"))
#endif
#endif

#if !defined(ROCTX_EXPORT_DECORATOR)
#define ROCTX_EXPORT_DECORATOR ROCTX_PUBLIC_API
#endif

#if !defined(ROCTX_IMPORT_DECORATOR)
#if defined(_MSC_VER)
#define ROCTX_IMPORT_DECORATOR ROCTX_ATTRIBUTE(dllimport)
#else
#define ROCTX_IMPORT_DECORATOR
#endif
#endif

#define ROCTX_EXPORT ROCTX_EXPORT_DECORATOR
#define ROCTX_IMPORT ROCTX_IMPORT_DECORATOR

#if !defined(ROCTX_API)
#if defined(roctx_EXPORTS)
#define ROCTX_API ROCTX_EXPORT
#else
#define ROCTX_API ROCTX_IMPORT
#endif
#endif

#if defined(__has_attribute)
#if __has_attribute(nonnull)
#define ROCTX_NONNULL(...) __attribute__((nonnull(__VA_ARGS__)))
#else
#define ROCTX_NONNULL(...)
#endif
#else
#if defined(__GNUC__)
#define ROCTX_NONNULL(...) __attribute__((nonnull(__VA_ARGS__)))
#else
#define ROCTX_NONNULL(...)
#endif
#endif

#ifdef __cplusplus
#define ROCTX_EXTERN_C_INIT extern "C" {
#define ROCTX_EXTERN_C_FINI }
#else
#define ROCTX_EXTERN_C_INIT
#define ROCTX_EXTERN_C_FINI
#endif
