/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

/** \mainpage ROCTX API Specification
 *
 * \section introduction Introduction
 * ROCTX is a library that implements the AMD code annotation API.  It provides
 * the support necessary to annotate events and code ranges in applications.
 */

/**
 * \file
 * ROCTX API interface.
 */

#ifndef ROCTX_H_
#define ROCTX_H_ 1

/* Placeholder for calling convention and import/export macros */
#if !defined(ROCTX_CALL)
#define ROCTX_CALL
#endif /* !defined (ROCTX_CALL) */

#if !defined(ROCTX_EXPORT_DECORATOR)
#if defined(__GNUC__)
#define ROCTX_EXPORT_DECORATOR __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define ROCTX_EXPORT_DECORATOR __declspec(dllexport)
#endif /* defined (_MSC_VER) */
#endif /* !defined (ROCTX_EXPORT_DECORATOR) */

#if !defined(ROCTX_IMPORT_DECORATOR)
#if defined(__GNUC__)
#define ROCTX_IMPORT_DECORATOR
#elif defined(_MSC_VER)
#define ROCTX_IMPORT_DECORATOR __declspec(dllimport)
#endif /* defined (_MSC_VER) */
#endif /* !defined (ROCTX_IMPORT_DECORATOR) */

#define ROCTX_EXPORT ROCTX_EXPORT_DECORATOR ROCTX_CALL
#define ROCTX_IMPORT ROCTX_IMPORT_DECORATOR ROCTX_CALL

#if !defined(ROCTX)
#if defined(ROCTX_EXPORTS)
#define ROCTX_API ROCTX_EXPORT
#else /* !defined (ROCTX_EXPORTS) */
#define ROCTX_API ROCTX_IMPORT
#endif /* !defined (ROCTX_EXPORTS) */
#endif /* !defined (ROCTX) */

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

/** \defgroup symbol_versions_group Symbol Versions
 *
 * The names used for the shared library versioned symbols.
 *
 * Every function is annotated with one of the version macros defined in this
 * section.  Each macro specifies a corresponding symbol version string.  After
 * dynamically loading the shared library with \p dlopen, the address of each
 * function can be obtained using \p dlvsym with the name of the function and
 * its corresponding symbol version string.  An error will be reported by \p
 * dlvsym if the installed library does not support the version for the
 * function specified in this version of the interface.
 *
 * @{
 */

/**
 * The function was introduced in version 4.1 of the interface and has the
 * symbol version string of ``"ROCTX_4.1"``.
 */
#define ROCTX_VERSION_4_1

/** @} */

/** \defgroup versioning_group Versioning
 *
 * Version information about the interface and the associated installed
 * library.
 *
 * @{
 */

/**
 * The semantic version of the interface following
 * [semver.org][semver] rules.
 *
 * A client that uses this interface is only compatible with the installed
 * library if the major version numbers match and the interface minor version
 * number is less than or equal to the installed library minor version number.
 */

/**
 * The major version of the interface as a macro so it can be used by the
 * preprocessor.
 */
#define ROCTX_VERSION_MAJOR 4

/**
 * The minor version of the interface as a macro so it can be used by the
 * preprocessor.
 */
#define ROCTX_VERSION_MINOR 1

/**
 * Query the major version of the installed library.
 *
 * Return the major version of the installed library. This can be used to check
 * if it is compatible with this interface version.
 *
 * \return Returns the major version number.
 */
ROCTX_API uint32_t roctx_version_major() ROCTX_VERSION_4_1;

/**
 * Query the minor version of the installed library.
 *
 * Return the minor version of the installed library. This can be used to check
 * if it is compatible with this interface version.
 *
 * \return Returns the minor version number.
 */
ROCTX_API uint32_t roctx_version_minor() ROCTX_VERSION_4_1;

/** @} */

/** \defgroup marker_group ROCTX Markers
 *
 * Marker annotations are used to describe events in a ROCm application.
 *
 * @{
 */

/**
 * Mark an event.
 *
 * \param[in] message The message associated with the event.
 */
ROCTX_API void roctxMarkA(const char* message) ROCTX_VERSION_4_1;
#define roctxMark(message) roctxMarkA(message)

/** @} */

/** \defgroup range_group ROCTX Ranges
 *
 * Range annotations are used to describe events in a ROCm application.
 *
 * @{
 */

/**
 * Start a new nested range.
 *
 * Nested ranges are stacked and local to the current CPU thread.
 *
 * \param[in] message The message associated with this range.
 *
 * \return Returns the level this nested range is started at. Nested range
 * levels are 0 based.
 */
ROCTX_API int roctxRangePushA(const char* message) ROCTX_VERSION_4_1;
#define roctxRangePush(message) roctxRangePushA(message)

/**
 * Stop the current nested range.
 *
 * Stop the current nested range, and pop it from the stack. If a nested range
 * was active before the last one was started, it becomes again the current
 * nested range.
 *
 * \return Returns the level the stopped nested range was started at, or a
 * negative value if there was no nested range active.
 */
ROCTX_API int roctxRangePop() ROCTX_VERSION_4_1;

/**
 * ROCTX range ID.
 *
 * This is the range ID used to identify start/end ranges.
 */
typedef uint64_t roctx_range_id_t;

/**
 * Starts a process range.
 *
 * Start/stop ranges can be started and stopped in different threads. Each
 * timespan is assigned a unique range ID.
 *
 * \param[in] message The message associated with this range.
 *
 * \return Returns the ID of the new range.
 */
ROCTX_API roctx_range_id_t roctxRangeStartA(const char* message)
    ROCTX_VERSION_4_1;
#define roctxRangeStart(message) roctxRangeStartA(message)

/**
 * Stop a process range.
 */
ROCTX_API void roctxRangeStop(roctx_range_id_t id) ROCTX_VERSION_4_1;

/** @} */

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined (__cplusplus) */

#endif /* ROCTX_H_ */
