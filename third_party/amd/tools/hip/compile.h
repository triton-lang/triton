// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

void unload_{kernel_name}(void);
void load_{kernel_name}(void);
hipError_t{_placeholder} {kernel_name}(hipStream_t stream, {signature});
