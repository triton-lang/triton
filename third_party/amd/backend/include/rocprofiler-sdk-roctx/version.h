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
 * @def ROCTX_IS_ROCPROFILER_SDK
 * @brief Preprocessor define indicating the roctx header is a rocprofiler-sdk
 project
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_VERSION_MAJOR
 * @brief The major version of the interface as a macro so it can be used
 * by the preprocessor.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_VERSION_MINOR
 * @brief The minor version of the interface as a macro so it can be used
 * by the preprocessor.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_VERSION_PATCH
 * @brief The patch version of the interface as a macro so it can be used
 * by the preprocessor.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_VERSION
 * @brief Numerically increasing version number encoding major, minor, and patch
 via computing `((10000 * <MAJOR>) + (100 * <MINOR>) + <PATCH>)`.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_SOVERSION
 * @brief Shared object versioning value whose value is at least `(10000 *
 <MAJOR>)`.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_VERSION_STRING
 * @brief Version string in form: `<MAJOR>.<MINOR>.<PATCH>`.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_GIT_DESCRIBE
 * @brief String encoding of `git describe --tags` when rocprofiler was built.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_GIT_REVISION
 * @brief String encoding of `git rev-parse HEAD` when rocprofiler was built.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_LIBRARY_ARCH
 * @brief Architecture triplet of rocprofiler build.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_SYSTEM_NAME
 * @brief Target operating system for rocprofiler build, e.g. Linux.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_SYSTEM_PROCESSOR
 * @brief Target architecture for rocprofiler build.
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_SYSTEM_VERSION
 * @brief Version of the operating system which built rocprofiler
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_COMPILER_ID
 * @brief C++ compiler identifier which built rocprofiler, e.g., GNU
 * @addtogroup VERSIONING_GROUP
 *
 * @def ROCTX_COMPILER_VERSION
 * @brief C++ compiler version which built rocprofiler
 * @addtogroup VERSIONING_GROUP
 */

#define ROCTX_IS_ROCPROFILER_SDK 1

// clang-format off
#define ROCTX_VERSION_MAJOR   1
#define ROCTX_VERSION_MINOR   2
#define ROCTX_VERSION_PATCH   1
#define ROCTX_SOVERSION       (10000 * 1)
#define ROCTX_VERSION_STRING "1.2.1"
#define ROCTX_GIT_DESCRIBE   ""
#define ROCTX_GIT_REVISION   "96e30b429c074f8a3fcf66be9c014df4d7cbc681"

// system info during compilation
#define ROCTX_LIBRARY_ARCH     "x86_64-unknown-linux-gnu"
#define ROCTX_SYSTEM_NAME      "Linux"
#define ROCTX_SYSTEM_PROCESSOR "x86_64"
#define ROCTX_SYSTEM_VERSION   "5.15.0-1102-azure"

// compiler information
#define ROCTX_COMPILER_ID      "Clang"
#define ROCTX_COMPILER_VERSION "23.0.0"

// clang-format on

#define ROCTX_VERSION                                                          \
  ((10000 * ROCTX_VERSION_MAJOR) + (100 * ROCTX_VERSION_MINOR) +               \
   ROCTX_VERSION_PATCH)
