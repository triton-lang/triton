/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#if !defined(__HIPCC_RTC__)
#include <hip/hip_vector_types.h>
#endif

extern "C" {

#define ADDRESS_SPACE_CONSTANT __attribute__((address_space(4)))

__device__ float4::Native_vec_ __ockl_image_load_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, int c);

__device__ float4::Native_vec_ __ockl_image_load_1Db(unsigned int ADDRESS_SPACE_CONSTANT*i, int c);

__device__ float4::Native_vec_ __ockl_image_load_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int f);

__device__ float4::Native_vec_ __ockl_image_load_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int f);

__device__ float4::Native_vec_ __ockl_image_load_lod_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, int c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int f, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int f, int l);

__device__ void __ockl_image_store_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, int c, float4::Native_vec_ p);

__device__ void __ockl_image_store_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, float4::Native_vec_ p);

__device__ void __ockl_image_store_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, float4::Native_vec_ p);

__device__ void __ockl_image_store_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, float4::Native_vec_ p);

__device__ void __ockl_image_store_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, float4::Native_vec_ p);

__device__ void __ockl_image_store_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int f, float4::Native_vec_ p);

__device__ void __ockl_image_store_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int f, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, int c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, int2::Native_vec_ c, int f, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, int4::Native_vec_ c, int f, int l, float4::Native_vec_ p);

__device__ float4::Native_vec_ __ockl_image_sample_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float c);

__device__ float4::Native_vec_ __ockl_image_sample_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_grad_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float c, float dx, float dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c, float dx, float dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c, float2::Native_vec_ dx, float2::Native_vec_ dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float2::Native_vec_ dx, float2::Native_vec_ dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float4::Native_vec_ dx, float4::Native_vec_ dy);

__device__ float4::Native_vec_ __ockl_image_sample_lod_1D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_1Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_2Da(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_3D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_CM(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_CMa(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_gather4r_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_gather4g_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_gather4b_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_gather4a_2D(unsigned int ADDRESS_SPACE_CONSTANT*i, unsigned int ADDRESS_SPACE_CONSTANT*s, float2::Native_vec_ c);

__device__ int __ockl_image_channel_data_type_1D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_1Db(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2Dad(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2Dd(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_3D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_CM(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_1D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_1Db(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2Dad(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2Dd(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_3D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_CM(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i);

}
