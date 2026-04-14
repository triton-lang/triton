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
 * @def ROCPROFILER_IS_ROCPROFILER_SDK
 * @brief Preprocessor define indicating the rocprofiler header is a
 rocprofiler-sdk project
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_VERSION_MAJOR
 * @brief The major version of the interface as a macro so it can be used
 * by the preprocessor.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_VERSION_MINOR
 * @brief The minor version of the interface as a macro so it can be used
 * by the preprocessor.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_VERSION_PATCH
 * @brief The patch version of the interface as a macro so it can be used
 * by the preprocessor.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_VERSION
 * @brief Numerically increasing version number encoding major, minor, and patch
 via computing `((10000 * <MAJOR>) + (100 * <MINOR>) + <PATCH>)`.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_SOVERSION
 * @brief Shared object versioning value whose value is at least `(10000 *
 <MAJOR>)`.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_VERSION_STRING
 * @brief Version string in form: `<MAJOR>.<MINOR>.<PATCH>`.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_GIT_DESCRIBE
 * @brief String encoding of `git describe --tags` when rocprofiler was built.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_GIT_REVISION
 * @brief String encoding of `git rev-parse HEAD` when rocprofiler was built.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_LIBRARY_ARCH
 * @brief Architecture triplet of rocprofiler build.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_SYSTEM_NAME
 * @brief Target operating system for rocprofiler build, e.g. Linux.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_SYSTEM_PROCESSOR
 * @brief Target architecture for rocprofiler build.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_SYSTEM_VERSION
 * @brief Version of the operating system which built rocprofiler
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_COMPILER_ID
 * @brief C++ compiler identifier which built rocprofiler, e.g., GNU
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCPROFILER_COMPILER_VERSION
 * @brief C++ compiler version which built rocprofiler
 * @addtogroup VERSIONING_GROUP
 */

#define ROCPROFILER_IS_ROCPROFILER_SDK 1

// clang-format off
#define ROCPROFILER_VERSION_MAJOR   1
#define ROCPROFILER_VERSION_MINOR   0
#define ROCPROFILER_VERSION_PATCH   0
#define ROCPROFILER_SOVERSION       1
#define ROCPROFILER_VERSION_STRING "1.0.0"
#define ROCPROFILER_GIT_DESCRIBE   ""
#define ROCPROFILER_GIT_REVISION   "b590612966dbf678993fa357f21402858e86a88e"

// system info during compilation
#define ROCPROFILER_LIBRARY_ARCH     "x86_64-linux-gnu"
#define ROCPROFILER_SYSTEM_NAME      "Linux"
#define ROCPROFILER_SYSTEM_PROCESSOR "x86_64"
#define ROCPROFILER_SYSTEM_VERSION   "6.8.0-57-generic"

// compiler information
#define ROCPROFILER_COMPILER_ID      "GNU"
#define ROCPROFILER_COMPILER_VERSION "11.4.0"
// clang-format on

#define ROCPROFILER_VERSION                                                    \
  ((10000 * ROCPROFILER_VERSION_MAJOR) + (100 * ROCPROFILER_VERSION_MINOR) +   \
   ROCPROFILER_VERSION_PATCH)
