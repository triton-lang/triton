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
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP16_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP16_H

#if defined(__HIPCC_RTC__)
  #define __HOST_DEVICE__ __device__
#else
  #define __HOST_DEVICE__ __host__ __device__
  #include <hip/amd_detail/amd_hip_common.h>
  #include "hip/amd_detail/host_defines.h"
  #include <assert.h>
  #if defined(__cplusplus)
    #include <algorithm>
    #include <type_traits>
    #include <utility>
#endif
#endif // !defined(__HIPCC_RTC__)

#if defined(__clang__) && defined(__HIP__)
    typedef _Float16 _Float16_2 __attribute__((ext_vector_type(2)));

    struct __half_raw {
        union {
            static_assert(sizeof(_Float16) == sizeof(unsigned short), "");

            _Float16 data;
            unsigned short x;
        };
    };

    struct __half2_raw {
        union {
            static_assert(sizeof(_Float16_2) == sizeof(unsigned short[2]), "");

            struct {
                __half_raw x;
                __half_raw y;
            };
            _Float16_2 data;
        };
    };

    #if defined(__cplusplus)
      #if !defined(__HIPCC_RTC__)
        #include "hip_fp16_math_fwd.h"
        #include "amd_hip_vector_types.h"
        #include "host_defines.h"
        #include "amd_device_functions.h"
        #include "amd_warp_functions.h"
      #endif
        namespace std
        {
            template<> struct is_floating_point<_Float16> : std::true_type {};
        }

        template<bool cond, typename T = void>
        using Enable_if_t = typename std::enable_if<cond, T>::type;

        // BEGIN STRUCT __HALF
        struct __half {
        protected:
            union {
                static_assert(sizeof(_Float16) == sizeof(unsigned short), "");

                _Float16 data;
                unsigned short __x;
            };
        public:
            // CREATORS
            __HOST_DEVICE__
            __half() = default;
            __HOST_DEVICE__
            __half(const __half_raw& x) : data{x.data} {}
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                __HOST_DEVICE__
                __half(decltype(data) x) : data{x} {}
                template<
                    typename T,
                    Enable_if_t<std::is_floating_point<T>{}>* = nullptr>
                __HOST_DEVICE__
                __half(T x) : data{static_cast<_Float16>(x)} {}
            #endif
            __HOST_DEVICE__
            __half(const __half&) = default;
            __HOST_DEVICE__
            __half(__half&&) = default;
            __HOST_DEVICE__
            ~__half() = default;

            // CREATORS - DEVICE ONLY
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                template<
                    typename T, Enable_if_t<std::is_integral<T>{}>* = nullptr>
                __HOST_DEVICE__
                __half(T x) : data{static_cast<_Float16>(x)} {}
            #endif

            // MANIPULATORS
            __HOST_DEVICE__
            __half& operator=(const __half&) = default;
            __HOST_DEVICE__
            __half& operator=(__half&&) = default;
            __HOST_DEVICE__
            __half& operator=(const __half_raw& x)
            {
                data = x.data;
                return *this;
            }
            __HOST_DEVICE__
            volatile __half& operator=(const __half_raw& x) volatile
            {
                data = x.data;
                return *this;
            }
            volatile __half& operator=(const volatile __half_raw& x) volatile
            {
                data = x.data;
                return *this;
            }
            __half& operator=(__half_raw&& x)
            {
                data = x.data;
                return *this;
            }
            volatile __half& operator=(__half_raw&& x) volatile
            {
                data = x.data;
                return *this;
            }
            volatile __half& operator=(volatile __half_raw&& x) volatile
            {
                data = x.data;
                return *this;
            }
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                template<
                    typename T,
                    Enable_if_t<std::is_floating_point<T>{}>* = nullptr>
                __HOST_DEVICE__
                __half& operator=(T x)
                {
                    data = static_cast<_Float16>(x);
                    return *this;
                }
            #endif

            // MANIPULATORS - DEVICE ONLY
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                template<
                    typename T, Enable_if_t<std::is_integral<T>{}>* = nullptr>
                __device__
                __half& operator=(T x)
                {
                    data = static_cast<_Float16>(x);
                    return *this;
                }
            #endif

            #if !defined(__HIP_NO_HALF_OPERATORS__)
                __device__
                __half& operator+=(const __half& x)
                {
                    data += x.data;
                    return *this;
                }
                __device__
                __half& operator-=(const __half& x)
                {
                    data -= x.data;
                    return *this;
                }
                __device__
                __half& operator*=(const __half& x)
                {
                    data *= x.data;
                    return *this;
                }
                __device__
                __half& operator/=(const __half& x)
                {
                    data /= x.data;
                    return *this;
                }
                __device__
                __half& operator++() { ++data; return *this; }
                __device__
                __half operator++(int)
                {
                    __half tmp{*this};
                    ++*this;
                    return tmp;
                }
                __device__
                __half& operator--() { --data; return *this; }
                __device__
                __half operator--(int)
                {
                    __half tmp{*this};
                    --*this;
                    return tmp;
                }
            #endif

            // ACCESSORS
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                template<
                    typename T,
                    Enable_if_t<std::is_floating_point<T>{}>* = nullptr>
                __HOST_DEVICE__
                operator T() const { return data; }
            #endif
            __HOST_DEVICE__
            operator __half_raw() const { return __half_raw{data}; }
            __HOST_DEVICE__
            operator __half_raw() const volatile
            {
                return __half_raw{data};
            }

            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                template<
                    typename T, Enable_if_t<std::is_integral<T>{}>* = nullptr>
                __HOST_DEVICE__
                operator T() const { return data; }
            #endif

            #if !defined(__HIP_NO_HALF_OPERATORS__)
                __device__
                __half operator+() const { return *this; }
                __device__
                __half operator-() const
                {
                    __half tmp{*this};
                    tmp.data = -tmp.data;
                    return tmp;
                }
            #endif

            // FRIENDS
            #if !defined(__HIP_NO_HALF_OPERATORS__)
                friend
                inline
                __device__
                __half operator+(const __half& x, const __half& y)
                {
                    return __half{x} += y;
                }
                friend
                inline
                __device__
                __half operator-(const __half& x, const __half& y)
                {
                    return __half{x} -= y;
                }
                friend
                inline
                __device__
                __half operator*(const __half& x, const __half& y)
                {
                    return __half{x} *= y;
                }
                friend
                inline
                __device__
                __half operator/(const __half& x, const __half& y)
                {
                    return __half{x} /= y;
                }
                friend
                inline
                __device__
                bool operator==(const __half& x, const __half& y)
                {
                    return x.data == y.data;
                }
                friend
                inline
                __device__
                bool operator!=(const __half& x, const __half& y)
                {
                    return !(x == y);
                }
                friend
                inline
                __device__
                bool operator<(const __half& x, const __half& y)
                {
                    return x.data < y.data;
                }
                friend
                inline
                __device__
                bool operator>(const __half& x, const __half& y)
                {
                    return y.data < x.data;
                }
                friend
                inline
                __device__
                bool operator<=(const __half& x, const __half& y)
                {
                    return !(y < x);
                }
                friend
                inline
                __device__
                bool operator>=(const __half& x, const __half& y)
                {
                    return !(x < y);
                }
            #endif // !defined(__HIP_NO_HALF_OPERATORS__)
        };
        // END STRUCT __HALF

        // BEGIN STRUCT __HALF2
        struct __half2 {
        public:
            union {
                static_assert(
                    sizeof(_Float16_2) == sizeof(unsigned short[2]), "");

                struct {
                    __half x;
                    __half y;
                };
                _Float16_2 data;
            };

            // CREATORS
            __HOST_DEVICE__
            __half2() = default;
            __HOST_DEVICE__
            __half2(const __half2_raw& xx) : data{xx.data} {}
            __HOST_DEVICE__
            __half2(decltype(data) xx) : data{xx} {}
            __HOST_DEVICE__
            __half2(const __half& xx, const __half& yy)
                :
                data{static_cast<__half_raw>(xx).data,
                     static_cast<__half_raw>(yy).data}
            {}
            __HOST_DEVICE__
            __half2(const __half2&) = default;
            __HOST_DEVICE__
            __half2(__half2&&) = default;
            __HOST_DEVICE__
            ~__half2() = default;

            // MANIPULATORS
            __HOST_DEVICE__
            __half2& operator=(const __half2&) = default;
            __HOST_DEVICE__
            __half2& operator=(__half2&&) = default;
            __HOST_DEVICE__
            __half2& operator=(const __half2_raw& xx)
            {
                data = xx.data;
                return *this;
            }

            // MANIPULATORS - DEVICE ONLY
            #if !defined(__HIP_NO_HALF_OPERATORS__)
                __device__
                __half2& operator+=(const __half2& xx)
                {
                    data += xx.data;
                    return *this;
                }
                __device__
                __half2& operator-=(const __half2& xx)
                {
                    data -= xx.data;
                    return *this;
                }
                __device__
                __half2& operator*=(const __half2& xx)
                {
                    data *= xx.data;
                    return *this;
                }
                __device__
                __half2& operator/=(const __half2& xx)
                {
                    data /= xx.data;
                    return *this;
                }
                __device__
                __half2& operator++() { return *this += _Float16_2{1, 1}; }
                __device__
                __half2 operator++(int)
                {
                    __half2 tmp{*this};
                    ++*this;
                    return tmp;
                }
                __device__
                __half2& operator--() { return *this -= _Float16_2{1, 1}; }
                __device__
                __half2 operator--(int)
                {
                    __half2 tmp{*this};
                    --*this;
                    return tmp;
                }
            #endif

            // ACCESSORS
            __HOST_DEVICE__
            operator decltype(data)() const { return data; }
            __HOST_DEVICE__
            operator __half2_raw() const {
              __half2_raw r;
              r.data = data;
              return r;
            }

            // ACCESSORS - DEVICE ONLY
            #if !defined(__HIP_NO_HALF_OPERATORS__)
                __device__
                __half2 operator+() const { return *this; }
                __device__
                __half2 operator-() const
                {
                    __half2 tmp{*this};
                    tmp.data = -tmp.data;
                    return tmp;
                }
            #endif

            // FRIENDS
            #if !defined(__HIP_NO_HALF_OPERATORS__)
                friend
                inline
                __device__
                __half2 operator+(const __half2& xx, const __half2& yy)
                {
                    return __half2{xx} += yy;
                }
                friend
                inline
                __device__
                __half2 operator-(const __half2& xx, const __half2& yy)
                {
                    return __half2{xx} -= yy;
                }
                friend
                inline
                __device__
                __half2 operator*(const __half2& xx, const __half2& yy)
                {
                    return __half2{xx} *= yy;
                }
                friend
                inline
                __device__
                __half2 operator/(const __half2& xx, const __half2& yy)
                {
                    return __half2{xx} /= yy;
                }
                friend
                inline
                __device__
                bool operator==(const __half2& xx, const __half2& yy)
                {
                    auto r = xx.data == yy.data;
                    return r.x != 0 && r.y != 0;
                }
                friend
                inline
                __device__
                bool operator!=(const __half2& xx, const __half2& yy)
                {
                    return !(xx == yy);
                }
                friend
                inline
                __device__
                bool operator<(const __half2& xx, const __half2& yy)
                {
                    auto r = xx.data < yy.data;
                    return r.x != 0 && r.y != 0;
                }
                friend
                inline
                __device__
                bool operator>(const __half2& xx, const __half2& yy)
                {
                    return yy < xx;
                }
                friend
                inline
                __device__
                bool operator<=(const __half2& xx, const __half2& yy)
                {
                    return !(yy < xx);
                }
                friend
                inline
                __device__
                bool operator>=(const __half2& xx, const __half2& yy)
                {
                    return !(xx < yy);
                }
            #endif // !defined(__HIP_NO_HALF_OPERATORS__)
        };
        // END STRUCT __HALF2

        namespace
        {
            inline
            __HOST_DEVICE__
            __half2 make_half2(__half x, __half y)
            {
                return __half2{x, y};
            }

            inline
            __HOST_DEVICE__
            __half __low2half(__half2 x)
            {
                return __half{__half_raw{static_cast<__half2_raw>(x).data.x}};
            }

            inline
            __HOST_DEVICE__
            __half __high2half(__half2 x)
            {
                return __half{__half_raw{static_cast<__half2_raw>(x).data.y}};
            }

            inline
            __HOST_DEVICE__
            __half2 __half2half2(__half x)
            {
                return __half2{x, x};
            }

            inline
            __HOST_DEVICE__
            __half2 __halves2half2(__half x, __half y)
            {
                return __half2{x, y};
            }

            inline
            __HOST_DEVICE__
            __half2 __low2half2(__half2 x)
            {
                return __half2{
                    _Float16_2{
                        static_cast<__half2_raw>(x).data.x,
                        static_cast<__half2_raw>(x).data.x}};
            }

            inline
            __HOST_DEVICE__
            __half2 __high2half2(__half2 x)
            {
                return __half2{
                    _Float16_2{
                        static_cast<__half2_raw>(x).data.y,
                        static_cast<__half2_raw>(x).data.y}};
            }

            inline
            __HOST_DEVICE__
            __half2 __lows2half2(__half2 x, __half2 y)
            {
                return __half2{
                    _Float16_2{
                        static_cast<__half2_raw>(x).data.x,
                        static_cast<__half2_raw>(y).data.x}};
            }

            inline
            __HOST_DEVICE__
            __half2 __highs2half2(__half2 x, __half2 y)
            {
                return __half2{
                    _Float16_2{
                        static_cast<__half2_raw>(x).data.y,
                        static_cast<__half2_raw>(y).data.y}};
            }

            inline
            __HOST_DEVICE__
            __half2 __lowhigh2highlow(__half2 x)
            {
                return __half2{
                    _Float16_2{
                        static_cast<__half2_raw>(x).data.y,
                        static_cast<__half2_raw>(x).data.x}};
            }

            // Bitcasts
            inline
            __device__
            short __half_as_short(__half x)
            {
                return static_cast<__half_raw>(x).x;
            }

            inline
            __device__
            unsigned short __half_as_ushort(__half x)
            {
                return static_cast<__half_raw>(x).x;
            }

            inline
            __device__
            __half __short_as_half(short x)
            {
                __half_raw r; r.x = x;
                return r;
            }

            inline
            __device__
            __half __ushort_as_half(unsigned short x)
            {
                __half_raw r; r.x = x;
                return r;
            }

            // float -> half | half2
            inline
            __HOST_DEVICE__
            __half __float2half(float x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __HOST_DEVICE__
            __half __float2half_rn(float x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            #if !defined(__HIPCC_RTC__)
            // TODO: rounding behaviour is not correct for host functions.
            inline
            __host__
            __half __float2half_rz(float x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __host__
            __half __float2half_rd(float x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __host__
            __half __float2half_ru(float x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            #endif
            inline
            __device__
            __half __float2half_rz(float x)
            {
                return __half_raw{__ocml_cvtrtz_f16_f32(x)};
            }
            inline
            __device__
            __half __float2half_rd(float x)
            {
                return __half_raw{__ocml_cvtrtn_f16_f32(x)};
            }
            inline
            __device__
            __half __float2half_ru(float x)
            {
                return __half_raw{__ocml_cvtrtp_f16_f32(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 __float2half2_rn(float x)
            {
                return __half2{
                    _Float16_2{
                        static_cast<_Float16>(x), static_cast<_Float16>(x)}};
            }
            inline
            __HOST_DEVICE__
            __half2 __floats2half2_rn(float x, float y)
            {
                return __half2{_Float16_2{
                    static_cast<_Float16>(x), static_cast<_Float16>(y)}};
            }
            inline
            __HOST_DEVICE__
            __half2 __float22half2_rn(float2 x)
            {
                return __floats2half2_rn(x.x, x.y);
            }

            // half | half2 -> float
            inline
            __HOST_DEVICE__
            float __half2float(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __HOST_DEVICE__
            float __low2float(__half2 x)
            {
                return static_cast<__half2_raw>(x).data.x;
            }
            inline
            __HOST_DEVICE__
            float __high2float(__half2 x)
            {
                return static_cast<__half2_raw>(x).data.y;
            }
            inline
            __HOST_DEVICE__
            float2 __half22float2(__half2 x)
            {
                return make_float2(
                    static_cast<__half2_raw>(x).data.x,
                    static_cast<__half2_raw>(x).data.y);
            }

            // half -> int
            inline
            __device__
            int __half2int_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            int __half2int_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            int __half2int_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            int __half2int_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // int -> half
            inline
            __device__
            __half __int2half_rn(int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __int2half_rz(int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __int2half_rd(int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __int2half_ru(int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // half -> short
            inline
            __device__
            short __half2short_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            short __half2short_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            short __half2short_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            short __half2short_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // short -> half
            inline
            __device__
            __half __short2half_rn(short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __short2half_rz(short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __short2half_rd(short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __short2half_ru(short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // half -> long long
            inline
            __device__
            long long __half2ll_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            long long __half2ll_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            long long __half2ll_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            long long __half2ll_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // long long -> half
            inline
            __device__
            __half __ll2half_rn(long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ll2half_rz(long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ll2half_rd(long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ll2half_ru(long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // half -> unsigned int
            inline
            __device__
            unsigned int __half2uint_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned int __half2uint_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned int __half2uint_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned int __half2uint_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // unsigned int -> half
            inline
            __device__
            __half __uint2half_rn(unsigned int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __uint2half_rz(unsigned int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __uint2half_rd(unsigned int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __uint2half_ru(unsigned int x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // half -> unsigned short
            inline
            __device__
            unsigned short __half2ushort_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned short __half2ushort_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned short __half2ushort_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned short __half2ushort_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // unsigned short -> half
            inline
            __device__
            __half __ushort2half_rn(unsigned short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ushort2half_rz(unsigned short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ushort2half_rd(unsigned short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ushort2half_ru(unsigned short x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // half -> unsigned long long
            inline
            __device__
            unsigned long long __half2ull_rn(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned long long __half2ull_rz(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned long long __half2ull_rd(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
            inline
            __device__
            unsigned long long __half2ull_ru(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }

            // unsigned long long -> half
            inline
            __device__
            __half __ull2half_rn(unsigned long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ull2half_rz(unsigned long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ull2half_rd(unsigned long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            inline
            __device__
            __half __ull2half_ru(unsigned long long x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }

            // Load primitives
            inline
            __device__
            __half __ldg(const __half* ptr) { return *ptr; }
            inline
            __device__
            __half __ldcg(const __half* ptr) { return *ptr; }
            inline
            __device__
            __half __ldca(const __half* ptr) { return *ptr; }
            inline
            __device__
            __half __ldcs(const __half* ptr) { return *ptr; }

            inline
            __HOST_DEVICE__
            __half2 __ldg(const __half2* ptr) { return *ptr; }
            inline
            __HOST_DEVICE__
            __half2 __ldcg(const __half2* ptr) { return *ptr; }
            inline
            __HOST_DEVICE__
            __half2 __ldca(const __half2* ptr) { return *ptr; }
            inline
            __HOST_DEVICE__
            __half2 __ldcs(const __half2* ptr) { return *ptr; }

            // Relations
            inline
            __device__
            bool __heq(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data ==
                    static_cast<__half_raw>(y).data;
            }
            inline
            __device__
            bool __hne(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data !=
                    static_cast<__half_raw>(y).data;
            }
            inline
            __device__
            bool __hle(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data <=
                    static_cast<__half_raw>(y).data;
            }
            inline
            __device__
            bool __hge(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data >=
                    static_cast<__half_raw>(y).data;
            }
            inline
            __device__
            bool __hlt(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data <
                    static_cast<__half_raw>(y).data;
            }
            inline
            __device__
            bool __hgt(__half x, __half y)
            {
                return static_cast<__half_raw>(x).data >
                    static_cast<__half_raw>(y).data;
            }
            inline __device__
            bool __hequ(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data < static_cast<__half_raw>(y).data) &&
                    !(static_cast<__half_raw>(x).data > static_cast<__half_raw>(y).data);
            }
            inline __device__
            bool __hneu(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data == static_cast<__half_raw>(y).data);
            }
            inline __device__
            bool __hleu(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data > static_cast<__half_raw>(y).data);
            }
            inline
            __device__
            bool __hgeu(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data < static_cast<__half_raw>(y).data);
            }
            inline
            __device__
            bool __hltu(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data >= static_cast<__half_raw>(y).data);
            }
            inline
            __device__
            bool __hgtu(__half x, __half y) {
                return !(static_cast<__half_raw>(x).data <= static_cast<__half_raw>(y).data);
            }

            inline
            __HOST_DEVICE__
            __half2 __heq2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data ==
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hne2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data !=
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hle2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data <=
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hge2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data >=
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hlt2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data <
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hgt2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(x).data >
                    static_cast<__half2_raw>(y).data;
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline __HOST_DEVICE__
            __half2 __hequ2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data < static_cast<__half2_raw>(y).data) &&
                    !(static_cast<__half2_raw>(x).data > static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hneu2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data == static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hleu2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data > static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hgeu2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data < static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hltu2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data >= static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }
            inline
            __HOST_DEVICE__
            __half2 __hgtu2(__half2 x, __half2 y) {
                auto r = !(static_cast<__half2_raw>(x).data <= static_cast<__half2_raw>(y).data);
                return __builtin_convertvector(-r, _Float16_2);
            }

            inline
            __HOST_DEVICE__
            bool __hbeq2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__heq2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hbne2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hne2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hble2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hle2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hbge2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hge2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hblt2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hlt2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hbgt2(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hgt2(x, y));
                return r.data.x != 0 && r.data.y != 0;
            }
            inline
            __HOST_DEVICE__
            bool __hbequ2(__half2 x, __half2 y) { return __hbeq2(x, y); }
            inline
            __HOST_DEVICE__
            bool __hbneu2(__half2 x, __half2 y) { return __hbne2(x, y); }
            inline
            __HOST_DEVICE__
            bool __hbleu2(__half2 x, __half2 y) { return __hble2(x, y); }
            inline
            __HOST_DEVICE__
            bool __hbgeu2(__half2 x, __half2 y) { return __hbge2(x, y); }
            inline
            __HOST_DEVICE__
            bool __hbltu2(__half2 x, __half2 y) { return __hblt2(x, y); }
            inline
            __HOST_DEVICE__
            bool __hbgtu2(__half2 x, __half2 y) { return __hbgt2(x, y); }
            inline
            __device__
            __half __hmax(const __half x, const __half y) {
              return __half_raw{__ocml_fmax_f16(static_cast<__half_raw>(x).data,
                                   static_cast<__half_raw>(y).data)};
            }
            inline
            __device__
            __half __hmax_nan(const __half x, const __half y) {
                if(__ocml_isnan_f16(static_cast<__half_raw>(x).data)) {
                  return x;
                } else if (__ocml_isnan_f16(static_cast<__half_raw>(y).data)) {
                  return y;
                }
                return __hmax(x, y);
            }
            inline
            __device__
            __half __hmin(const __half x, const __half y) {
              return __half_raw{__ocml_fmin_f16(static_cast<__half_raw>(x).data,
                                   static_cast<__half_raw>(y).data)};
            }
            inline
            __device__
            __half __hmin_nan(const __half x, const __half y) {
                if(__ocml_isnan_f16(static_cast<__half_raw>(x).data)) {
                  return x;
                } else if (__ocml_isnan_f16(static_cast<__half_raw>(y).data)) {
                  return y;
                }
                return __hmin(x, y);
            }

            // Arithmetic
            inline
            __device__
            __half __clamp_01(__half x)
            {
                auto r = static_cast<__half_raw>(x);

                if (__hlt(x, __half_raw{0})) return __half_raw{0};
                if (__hlt(__half_raw{1}, x)) return __half_raw{1};
                return r;
            }

            inline
            __device__
            __half __hadd(__half x, __half y)
            {
                return __half_raw{
                    static_cast<__half_raw>(x).data +
                    static_cast<__half_raw>(y).data};
            }
	    inline
	    __device__
	    __half __habs(__half x)
	    {
	        return __half_raw{
		    __ocml_fabs_f16(static_cast<__half_raw>(x).data)};
	    }
            inline
            __device__
            __half __hsub(__half x, __half y)
            {
                return __half_raw{
                    static_cast<__half_raw>(x).data -
                    static_cast<__half_raw>(y).data};
            }
            inline
            __device__
            __half __hmul(__half x, __half y)
            {
                return __half_raw{
                    static_cast<__half_raw>(x).data *
                    static_cast<__half_raw>(y).data};
            }
            inline
            __device__
            __half __hadd_sat(__half x, __half y)
            {
                return __clamp_01(__hadd(x, y));
            }
            inline
            __device__
            __half __hsub_sat(__half x, __half y)
            {
                return __clamp_01(__hsub(x, y));
            }
            inline
            __device__
            __half __hmul_sat(__half x, __half y)
            {
                return __clamp_01(__hmul(x, y));
            }
            inline
            __device__
            __half __hfma(__half x, __half y, __half z)
            {
                return __half_raw{__ocml_fma_f16(
                    static_cast<__half_raw>(x).data,
                    static_cast<__half_raw>(y).data,
                    static_cast<__half_raw>(z).data)};
            }
            inline
            __device__
            __half __hfma_sat(__half x, __half y, __half z)
            {
                return __clamp_01(__hfma(x, y, z));
            }
            inline
            __device__
            __half __hdiv(__half x, __half y)
            {
                return __half_raw{
                    static_cast<__half_raw>(x).data /
                    static_cast<__half_raw>(y).data};
            }

            inline
            __HOST_DEVICE__
            __half2 __hadd2(__half2 x, __half2 y)
            {
                return __half2{
                    static_cast<__half2_raw>(x).data +
                    static_cast<__half2_raw>(y).data};
            }
	    inline
	    __HOST_DEVICE__
	    __half2 __habs2(__half2 x)
	    {
	        return __half2{
		    __ocml_fabs_2f16(static_cast<__half2_raw>(x).data)};
	    }
            inline
            __HOST_DEVICE__
            __half2 __hsub2(__half2 x, __half2 y)
            {
                return __half2{
                    static_cast<__half2_raw>(x).data -
                    static_cast<__half2_raw>(y).data};
            }
            inline
            __HOST_DEVICE__
            __half2 __hmul2(__half2 x, __half2 y)
            {
                return __half2{
                    static_cast<__half2_raw>(x).data *
                    static_cast<__half2_raw>(y).data};
            }
            inline
            __HOST_DEVICE__
            __half2 __hadd2_sat(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hadd2(x, y));
                return __half2{
                    __clamp_01(__half_raw{r.data.x}),
                    __clamp_01(__half_raw{r.data.y})};
            }
            inline
            __HOST_DEVICE__
            __half2 __hsub2_sat(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hsub2(x, y));
                return __half2{
                    __clamp_01(__half_raw{r.data.x}),
                    __clamp_01(__half_raw{r.data.y})};
            }
            inline
            __HOST_DEVICE__
            __half2 __hmul2_sat(__half2 x, __half2 y)
            {
                auto r = static_cast<__half2_raw>(__hmul2(x, y));
                return __half2{
                    __clamp_01(__half_raw{r.data.x}),
                    __clamp_01(__half_raw{r.data.y})};
            }
            inline
            __HOST_DEVICE__
            __half2 __hfma2(__half2 x, __half2 y, __half2 z)
            {
                return __half2{__ocml_fma_2f16(x, y, z)};
            }
            inline
            __HOST_DEVICE__
            __half2 __hfma2_sat(__half2 x, __half2 y, __half2 z)
            {
                auto r = static_cast<__half2_raw>(__hfma2(x, y, z));
                return __half2{
                    __clamp_01(__half_raw{r.data.x}),
                    __clamp_01(__half_raw{r.data.y})};
            }
            inline
            __HOST_DEVICE__
            __half2 __h2div(__half2 x, __half2 y)
            {
                return __half2{
                    static_cast<__half2_raw>(x).data /
                    static_cast<__half2_raw>(y).data};
            }

            // Math functions
            #if defined(__clang__) && defined(__HIP__)
            inline
            __device__
            float amd_mixed_dot(__half2 a, __half2 b, float c, bool saturate) {
                return __ockl_fdot2(static_cast<__half2_raw>(a).data,
                                    static_cast<__half2_raw>(b).data,
                                    c, saturate);
            }
            #endif
            inline
            __device__
            __half htrunc(__half x)
            {
                return __half_raw{
                    __ocml_trunc_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hceil(__half x)
            {
                return __half_raw{
                    __ocml_ceil_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hfloor(__half x)
            {
                return __half_raw{
                   __ocml_floor_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hrint(__half x)
            {
                return __half_raw{
                    __ocml_rint_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hsin(__half x)
            {
                return __half_raw{
                    __ocml_sin_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hcos(__half x)
            {
                return __half_raw{
                    __ocml_cos_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hexp(__half x)
            {
                return __half_raw{
                    __ocml_exp_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hexp2(__half x)
            {
                return __half_raw{
                    __ocml_exp2_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hexp10(__half x)
            {
                return __half_raw{
                    __ocml_exp10_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hlog2(__half x)
            {
                return __half_raw{
                    __ocml_log2_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hlog(__half x)
            {
                return __half_raw{
                    __ocml_log_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hlog10(__half x)
            {
                return __half_raw{
                    __ocml_log10_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hrcp(__half x)
            {
                return __half_raw{
                    static_cast<_Float16>(1.0f) /static_cast<__half_raw>(x).data};
            }
            inline
            __device__
            __half hrsqrt(__half x)
            {
                return __half_raw{
                    __ocml_rsqrt_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            __half hsqrt(__half x)
            {
                return __half_raw{
                    __ocml_sqrt_f16(static_cast<__half_raw>(x).data)};
            }
            inline
            __device__
            bool __hisinf(__half x)
            {
                return __ocml_isinf_f16(static_cast<__half_raw>(x).data);
            }
            inline
            __device__
            bool __hisnan(__half x)
            {
                return __ocml_isnan_f16(static_cast<__half_raw>(x).data);
            }
            inline
            __device__
            __half __hneg(__half x)
            {
                return __half_raw{-static_cast<__half_raw>(x).data};
            }

            inline
            __HOST_DEVICE__
            __half2 h2trunc(__half2 x)
            {
                return __half2{__ocml_trunc_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2ceil(__half2 x)
            {
                return __half2{__ocml_ceil_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2floor(__half2 x)
            {
                return __half2{__ocml_floor_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2rint(__half2 x)
            {
                return __half2{__ocml_rint_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2sin(__half2 x)
            {
                return __half2{__ocml_sin_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2cos(__half2 x)
            {
                return __half2{__ocml_cos_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2exp(__half2 x)
            {
                return __half2{__ocml_exp_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2exp2(__half2 x)
            {
                return __half2{__ocml_exp2_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2exp10(__half2 x)
            {
                return __half2{__ocml_exp10_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2log2(__half2 x)
            {
                return __half2{__ocml_log2_2f16(x)};
            }
            inline
            __HOST_DEVICE__
            __half2 h2log(__half2 x) { return __ocml_log_2f16(x); }
            inline
            __HOST_DEVICE__
            __half2 h2log10(__half2 x) { return __ocml_log10_2f16(x); }
            inline
            __HOST_DEVICE__
            __half2 h2rcp(__half2 x) {
                return _Float16_2{
                    _Float16_2{static_cast<_Float16>(1.0f), static_cast<_Float16>(1.0f)} / x.data};
            }
            inline
            __HOST_DEVICE__
            __half2 h2rsqrt(__half2 x) { return __ocml_rsqrt_2f16(x); }
            inline
            __HOST_DEVICE__
            __half2 h2sqrt(__half2 x) { return __ocml_sqrt_2f16(x); }
            inline
            __HOST_DEVICE__
            __half2 __hisinf2(__half2 x)
            {
                auto r = __ocml_isinf_2f16(x);
                return __half2{_Float16_2{
                    static_cast<_Float16>(r.x), static_cast<_Float16>(r.y)}};
            }
            inline
            __HOST_DEVICE__
            __half2 __hisnan2(__half2 x)
            {
                auto r = __ocml_isnan_2f16(x);
                return __half2{_Float16_2{
                    static_cast<_Float16>(r.x), static_cast<_Float16>(r.y)}};
            }
            inline
            __HOST_DEVICE__
            __half2 __hneg2(__half2 x)
            {
                return __half2{-static_cast<__half2_raw>(x).data};
            }
        } // Anonymous namespace.

        #if !defined(HIP_NO_HALF)
            using half = __half;
            using half2 = __half2;
        #endif
        __device__
        inline
        __half __shfl(__half var, int src_lane, int width = warpSize) {
           union { int i; __half h; } tmp; tmp.h = var;
           tmp.i = __shfl(tmp.i, src_lane, width);
           return tmp.h;
        }
        __device__
        inline
        __half2 __shfl(__half2 var, int src_lane, int width = warpSize) {
           union { int i; __half2 h; } tmp; tmp.h = var;
           tmp.i = __shfl(tmp.i, src_lane, width);
           return tmp.h;
        }
        __device__
        inline
        __half __shfl_up(__half var, unsigned int lane_delta, int width = warpSize) {
           union { int i; __half h; } tmp; tmp.h = var;
           tmp.i = __shfl_up(tmp.i, lane_delta, width);
           return tmp.h;
        }
        __device__
        inline
         __half2 __shfl_up(__half2 var, unsigned int lane_delta, int width = warpSize) {
            union { int i; __half2 h; } tmp; tmp.h = var;
            tmp.i = __shfl_up(tmp.i, lane_delta, width);
            return tmp.h;
         }
         __device__
         inline
         __half __shfl_down(__half var, unsigned int lane_delta, int width = warpSize) {
            union { int i; __half h; } tmp; tmp.h = var;
            tmp.i = __shfl_down(tmp.i, lane_delta, width);
            return tmp.h;
         }
         __device__
         inline
         __half2 __shfl_down(__half2 var, unsigned int lane_delta, int width = warpSize) {
            union { int i; __half2 h; } tmp; tmp.h = var;
            tmp.i = __shfl_down(tmp.i, lane_delta, width);
            return tmp.h;
         }
         __device__
         inline
         __half __shfl_xor(__half var,  int lane_mask, int width = warpSize) {
            union { int i; __half h; } tmp; tmp.h = var;
            tmp.i = __shfl_xor(tmp.i, lane_mask, width);
            return tmp.h;
         }
         __device__
         inline
          __half2 __shfl_xor(__half2 var,  int lane_mask, int width = warpSize) {
             union { int i; __half2 h; } tmp; tmp.h = var;
             tmp.i = __shfl_xor(tmp.i, lane_mask, width);
             return tmp.h;
         }
    #endif // defined(__cplusplus)
#elif defined(__GNUC__)
    #if !defined(__HIPCC_RTC__)
      #include "hip_fp16_gcc.h"
    #endif
#endif // !defined(__clang__) && defined(__GNUC__)

#endif // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP16_H
