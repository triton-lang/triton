/*
Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

/**
 *  @file  amd_detail/hip_vector_types.h
 *  @brief Defines the different newt vector types for HIP runtime.
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_VECTOR_TYPES_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_VECTOR_TYPES_H

#include "hip/amd_detail/host_defines.h"

#if defined(__HIPCC_RTC__)
    #define __HOST_DEVICE__ __device__
#else
    #define __HOST_DEVICE__ __host__ __device__
#endif

#if defined(__has_attribute)
    #if __has_attribute(ext_vector_type)
        #define __HIP_USE_NATIVE_VECTOR__ 1
        #define __NATIVE_VECTOR__(n, T) T __attribute__((ext_vector_type(n)))
    #else
        #define __NATIVE_VECTOR__(n, T) T[n]
    #endif

#if defined(__cplusplus)
#if !defined(__HIPCC_RTC__)
    #include <array>
    #include <iosfwd>
    #include <type_traits>
#else
namespace std {
using ::size_t;

template <class _Tp, _Tp __v> struct integral_constant {
  static constexpr const _Tp value = __v;
  typedef _Tp value_type;
  typedef integral_constant type;
  constexpr operator value_type() const { return value; }
  constexpr value_type operator()() const { return value; }
};
template <class _Tp, _Tp __v> constexpr const _Tp integral_constant<_Tp, __v>::value;

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

template <bool B> using bool_constant = integral_constant<bool, B>;
typedef bool_constant<true> true_type;
typedef bool_constant<false> false_type;

template <bool __B, class __T = void> struct enable_if {};
template <class __T> struct enable_if<true, __T> { typedef __T type; };

template<bool _B> struct true_or_false_type : public false_type {};
template<> struct true_or_false_type<true> : public true_type {};

template <class _Tp> struct is_integral : public false_type {};
template <> struct is_integral<bool> : public true_type {};
template <> struct is_integral<char> : public true_type {};
template <> struct is_integral<signed char> : public true_type {};
template <> struct is_integral<unsigned char> : public true_type {};
template <> struct is_integral<wchar_t> : public true_type {};
template <> struct is_integral<short> : public true_type {};
template <> struct is_integral<unsigned short> : public true_type {};
template <> struct is_integral<int> : public true_type {};
template <> struct is_integral<unsigned int> : public true_type {};
template <> struct is_integral<long> : public true_type {};
template <> struct is_integral<unsigned long> : public true_type {};
template <> struct is_integral<long long> : public true_type {};
template <> struct is_integral<unsigned long long> : public true_type {};

template <class _Tp> struct is_arithmetic : public false_type {};
template <> struct is_arithmetic<bool> : public true_type {};
template <> struct is_arithmetic<char> : public true_type {};
template <> struct is_arithmetic<signed char> : public true_type {};
template <> struct is_arithmetic<unsigned char> : public true_type {};
template <> struct is_arithmetic<wchar_t> : public true_type {};
template <> struct is_arithmetic<short> : public true_type {};
template <> struct is_arithmetic<unsigned short> : public true_type {};
template <> struct is_arithmetic<int> : public true_type {};
template <> struct is_arithmetic<unsigned int> : public true_type {};
template <> struct is_arithmetic<long> : public true_type {};
template <> struct is_arithmetic<unsigned long> : public true_type {};
template <> struct is_arithmetic<long long> : public true_type {};
template <> struct is_arithmetic<unsigned long long> : public true_type {};
template <> struct is_arithmetic<float> : public true_type {};
template <> struct is_arithmetic<double> : public true_type {};

template<typename _Tp> struct is_floating_point : public false_type {};
template<> struct is_floating_point<float> : public true_type {};
template<> struct is_floating_point<double> : public true_type {};
template<> struct is_floating_point<long double> : public true_type {};

template <typename __T, typename __U> struct is_same : public false_type {};
template <typename __T> struct is_same<__T, __T> : public true_type {};

template<typename _Tp, bool = is_arithmetic<_Tp>::value>
  struct is_signed : public false_type {};
template<typename _Tp>
  struct is_signed<_Tp, true> : public true_or_false_type<_Tp(-1) < _Tp(0)> {};

template <class _T1, class _T2> struct is_convertible
  : public true_or_false_type<__is_convertible_to(_T1, _T2)> {};

template<typename _CharT> struct char_traits;
template<typename _CharT, typename _Traits = char_traits<_CharT>> class basic_istream;
template<typename _CharT, typename _Traits = char_traits<_CharT>> class basic_ostream;
typedef basic_istream<char> istream;
typedef basic_ostream<char> ostream;

template <typename __T> struct is_scalar : public integral_constant<bool, __is_scalar(__T)> {};
} // Namespace std.
#endif // defined(__HIPCC_RTC__)

    namespace hip_impl {
        inline
        constexpr
        unsigned int next_pot(unsigned int x) {
            // Precondition: x > 1.
	        return 1u << (32u - __builtin_clz(x - 1u));
        }
    } // Namespace hip_impl.

    template<typename T, unsigned int n> struct HIP_vector_base;

    template<typename T>
    struct HIP_vector_base<T, 1> {
        using Native_vec_ = __NATIVE_VECTOR__(1, T);

        union {
            Native_vec_ data;
            struct {
                T x;
            };
        };

        using value_type = T;

        __HOST_DEVICE__
        HIP_vector_base() = default;
        __HOST_DEVICE__
        explicit
        constexpr
        HIP_vector_base(T x_) noexcept : data{x_} {}
        __HOST_DEVICE__
        constexpr
        HIP_vector_base(const HIP_vector_base&) = default;
        __HOST_DEVICE__
        constexpr
        HIP_vector_base(HIP_vector_base&&) = default;
        __HOST_DEVICE__
        ~HIP_vector_base() = default;
        __HOST_DEVICE__
        HIP_vector_base& operator=(const HIP_vector_base&) = default;
    };

    template<typename T>
    struct HIP_vector_base<T, 2> {
        using Native_vec_ = __NATIVE_VECTOR__(2, T);

        union
        #if !__has_attribute(ext_vector_type)
            alignas(hip_impl::next_pot(2 * sizeof(T)))
        #endif
        {
            Native_vec_ data;
            struct {
                T x;
                T y;
            };
        };

        using value_type = T;

        __HOST_DEVICE__
        HIP_vector_base() = default;
        __HOST_DEVICE__
        explicit
        constexpr
        HIP_vector_base(T x_) noexcept : data{x_, x_} {}
        __HOST_DEVICE__
        constexpr
        HIP_vector_base(T x_, T y_) noexcept : data{x_, y_} {}
        __HOST_DEVICE__
        constexpr
        HIP_vector_base(const HIP_vector_base&) = default;
        __HOST_DEVICE__
        constexpr
        HIP_vector_base(HIP_vector_base&&) = default;
        __HOST_DEVICE__
        ~HIP_vector_base() = default;
        __HOST_DEVICE__
        HIP_vector_base& operator=(const HIP_vector_base&) = default;
    };

    template<typename T>
    struct HIP_vector_base<T, 3> {
        struct Native_vec_ {
            T d[3];

            __HOST_DEVICE__
            Native_vec_() = default;

            __HOST_DEVICE__
            explicit
            constexpr
            Native_vec_(T x_) noexcept : d{x_, x_, x_} {}
            __HOST_DEVICE__
            constexpr
            Native_vec_(T x_, T y_, T z_) noexcept : d{x_, y_, z_} {}
            __HOST_DEVICE__
            constexpr
            Native_vec_(const Native_vec_&) = default;
            __HOST_DEVICE__
            constexpr
            Native_vec_(Native_vec_&&) = default;
            __HOST_DEVICE__
            ~Native_vec_() = default;

            __HOST_DEVICE__
            Native_vec_& operator=(const Native_vec_&) = default;
            __HOST_DEVICE__
            Native_vec_& operator=(Native_vec_&&) = default;

            __HOST_DEVICE__
            T& operator[](unsigned int idx) noexcept { return d[idx]; }
            __HOST_DEVICE__
            T operator[](unsigned int idx) const noexcept { return d[idx]; }

            __HOST_DEVICE__
            Native_vec_& operator+=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] += x_.d[i];
                return *this;
            }
            __HOST_DEVICE__
            Native_vec_& operator-=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] -= x_.d[i];
                return *this;
            }

            __HOST_DEVICE__
            Native_vec_& operator*=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] *= x_.d[i];
                return *this;
            }
            __HOST_DEVICE__
            Native_vec_& operator/=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] /= x_.d[i];
                return *this;
            }

            template<
                typename U = T,
                typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
            __HOST_DEVICE__
            Native_vec_ operator-() const noexcept
            {
                auto r{*this};
                for (auto&& x : r.d) x = -x;
                return r;
            }

            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __HOST_DEVICE__
            Native_vec_ operator~() const noexcept
            {
                auto r{*this};
                for (auto&& x : r.d) x = ~x;
                return r;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __HOST_DEVICE__
            Native_vec_& operator%=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] %= x_.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __HOST_DEVICE__
            Native_vec_& operator^=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] ^= x_.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __HOST_DEVICE__
            Native_vec_& operator|=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] |= x_.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __HOST_DEVICE__
            Native_vec_& operator&=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] &= x_.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __HOST_DEVICE__
            Native_vec_& operator>>=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] >>= x_.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __HOST_DEVICE__
            Native_vec_& operator<<=(const Native_vec_& x_) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] <<= x_.d[i];
                return *this;
            }
#if defined (__INTEL_COMPILER)
            typedef struct {
              int values[4];
            } _Vec3_cmp;
            using Vec3_cmp = _Vec3_cmp;
#else
            using Vec3_cmp = int __attribute__((vector_size(4 * sizeof(int))));
#endif //INTEL
            __HOST_DEVICE__
            Vec3_cmp operator==(const Native_vec_& x_) const noexcept
            {
                return Vec3_cmp{d[0] == x_.d[0], d[1] == x_.d[1], d[2] == x_.d[2]};
            }
        };

        union {
            Native_vec_ data;
            struct {
                T x;
                T y;
                T z;
            };
        };

        using value_type = T;

        __HOST_DEVICE__
        HIP_vector_base() = default;
        __HOST_DEVICE__
        explicit
        constexpr
        HIP_vector_base(T x_) noexcept : data{x_, x_, x_} {}
        __HOST_DEVICE__
        constexpr
        HIP_vector_base(T x_, T y_, T z_) noexcept : data{x_, y_, z_} {}
        __HOST_DEVICE__
        constexpr
        HIP_vector_base(const HIP_vector_base&) = default;
        __HOST_DEVICE__
        constexpr
        HIP_vector_base(HIP_vector_base&&) = default;
        __HOST_DEVICE__
        ~HIP_vector_base() = default;

        __HOST_DEVICE__
        HIP_vector_base& operator=(const HIP_vector_base&) = default;
        __HOST_DEVICE__
        HIP_vector_base& operator=(HIP_vector_base&&) = default;
    };

    template<typename T>
    struct HIP_vector_base<T, 4> {
        using Native_vec_ = __NATIVE_VECTOR__(4, T);

        union
        #if !__has_attribute(ext_vector_type)
            alignas(hip_impl::next_pot(4 * sizeof(T)))
        #endif
        {
            Native_vec_ data;
            struct {
                T x;
                T y;
                T z;
                T w;
            };
        };

        using value_type = T;

        __HOST_DEVICE__
        HIP_vector_base() = default;
        __HOST_DEVICE__
        explicit
        constexpr
        HIP_vector_base(T x_) noexcept : data{x_, x_, x_, x_} {}
        __HOST_DEVICE__
        constexpr
        HIP_vector_base(T x_, T y_, T z_, T w_) noexcept : data{x_, y_, z_, w_} {}
        __HOST_DEVICE__
        constexpr
        HIP_vector_base(const HIP_vector_base&) = default;
        __HOST_DEVICE__
        constexpr
        HIP_vector_base(HIP_vector_base&&) = default;
        __HOST_DEVICE__
        ~HIP_vector_base() = default;
        __HOST_DEVICE__
        HIP_vector_base& operator=(const HIP_vector_base&) = default;
    };

    template<typename T, unsigned int rank>
    struct HIP_vector_type : public HIP_vector_base<T, rank> {
        using HIP_vector_base<T, rank>::data;
        using typename HIP_vector_base<T, rank>::Native_vec_;

        __HOST_DEVICE__
        HIP_vector_type() = default;
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>::value>::type* = nullptr>
        __HOST_DEVICE__
        explicit
        constexpr
        HIP_vector_type(U x_) noexcept
            : HIP_vector_base<T, rank>{static_cast<T>(x_)}
        {}
        template< // TODO: constrain based on type as well.
            typename... Us,
            typename std::enable_if<
                (rank > 1) && sizeof...(Us) == rank>::type* = nullptr>
        __HOST_DEVICE__
        constexpr
        HIP_vector_type(Us... xs) noexcept
            : HIP_vector_base<T, rank>{static_cast<T>(xs)...}
        {}
        __HOST_DEVICE__
        constexpr
        HIP_vector_type(const HIP_vector_type&) = default;
        __HOST_DEVICE__
        constexpr
        HIP_vector_type(HIP_vector_type&&) = default;
        __HOST_DEVICE__
        ~HIP_vector_type() = default;

        __HOST_DEVICE__
        HIP_vector_type& operator=(const HIP_vector_type&) = default;
        __HOST_DEVICE__
        HIP_vector_type& operator=(HIP_vector_type&&) = default;

        // Operators
        __HOST_DEVICE__
        HIP_vector_type& operator++() noexcept
        {
            return *this += HIP_vector_type{1};
        }
        __HOST_DEVICE__
        HIP_vector_type operator++(int) noexcept
        {
            auto tmp(*this);
            ++*this;
            return tmp;
        }

        __HOST_DEVICE__
        HIP_vector_type& operator--() noexcept
        {
            return *this -= HIP_vector_type{1};
        }
        __HOST_DEVICE__
        HIP_vector_type operator--(int) noexcept
        {
            auto tmp(*this);
            --*this;
            return tmp;
        }

        __HOST_DEVICE__
        HIP_vector_type& operator+=(const HIP_vector_type& x) noexcept
        {
#if __HIP_USE_NATIVE_VECTOR__
            data += x.data;
#else
            for (auto i = 0u; i != rank; ++i) data[i] += x.data[i];
#endif
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type& operator+=(U x) noexcept
        {
            return *this += HIP_vector_type{x};
        }

        __HOST_DEVICE__
        HIP_vector_type& operator-=(const HIP_vector_type& x) noexcept
        {
#if __HIP_USE_NATIVE_VECTOR__
            data -= x.data;
#else
            for (auto i = 0u; i != rank; ++i) data[i] -= x.data[i];
#endif
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type& operator-=(U x) noexcept
        {
            return *this -= HIP_vector_type{x};
        }

        __HOST_DEVICE__
        HIP_vector_type& operator*=(const HIP_vector_type& x) noexcept
        {
#if __HIP_USE_NATIVE_VECTOR__
            data *= x.data;
#else
            for (auto i = 0u; i != rank; ++i) data[i] *= x.data[i];
#endif
            return *this;
        }

        friend __HOST_DEVICE__ inline constexpr HIP_vector_type operator*(
        HIP_vector_type x, const HIP_vector_type& y) noexcept
        {
          return HIP_vector_type{ x } *= y;
        }

        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type& operator*=(U x) noexcept
        {
            return *this *= HIP_vector_type{x};
        }

        friend __HOST_DEVICE__ inline constexpr HIP_vector_type operator/(
        HIP_vector_type x, const HIP_vector_type& y) noexcept
        {
          return HIP_vector_type{ x } /= y;
        }

        __HOST_DEVICE__
        HIP_vector_type& operator/=(const HIP_vector_type& x) noexcept
        {
#if __HIP_USE_NATIVE_VECTOR__
            data /= x.data;
#else
            for (auto i = 0u; i != rank; ++i) data[i] /= x.data[i];
#endif
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type& operator/=(U x) noexcept
        {
            return *this /= HIP_vector_type{x};
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type operator-() const noexcept
        {
            auto tmp(*this);
#if __HIP_USE_NATIVE_VECTOR__
            tmp.data = -tmp.data;
#else
            for (auto i = 0u; i != rank; ++i) tmp.data[i] = -tmp.data[i];
#endif
            return tmp;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type operator~() const noexcept
        {
            HIP_vector_type r{*this};
#if __HIP_USE_NATIVE_VECTOR__
            r.data = ~r.data;
#else
            for (auto i = 0u; i != rank; ++i) r.data[i] = ~r.data[i];
#endif
            return r;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type& operator%=(const HIP_vector_type& x) noexcept
        {
#if __HIP_USE_NATIVE_VECTOR__
            data %= x.data;
#else
            for (auto i = 0u; i != rank; ++i) data[i] %= x.data[i];
#endif
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type& operator^=(const HIP_vector_type& x) noexcept
        {
#if __HIP_USE_NATIVE_VECTOR__
            data ^= x.data;
#else
            for (auto i = 0u; i != rank; ++i) data[i] ^= x.data[i];
#endif
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type& operator|=(const HIP_vector_type& x) noexcept
        {
#if __HIP_USE_NATIVE_VECTOR__
            data |= x.data;
#else
            for (auto i = 0u; i != rank; ++i) data[i] |= x.data[i];
#endif
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type& operator&=(const HIP_vector_type& x) noexcept
        {
#if __HIP_USE_NATIVE_VECTOR__
            data &= x.data;
#else
            for (auto i = 0u; i != rank; ++i) data[i] &= x.data[i];
#endif
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type& operator>>=(const HIP_vector_type& x) noexcept
        {
#if __HIP_USE_NATIVE_VECTOR__
            data >>= x.data;
#else
            for (auto i = 0u; i != rank; ++i) data[i] >>= x.data[i];
#endif
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __HOST_DEVICE__
        HIP_vector_type& operator<<=(const HIP_vector_type& x) noexcept
        {
#if __HIP_USE_NATIVE_VECTOR__
            data <<= x.data;
#else
            for (auto i = 0u; i != rank; ++i) data[i] <<= x.data[i];
#endif
            return *this;
        }
    };

    template<typename T, unsigned int n>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator+(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} += y;
    }
    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator+(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} += HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator+(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} += y;
    }

    template<typename T, unsigned int n>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator-(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= y;
    }
    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator-(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator-(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= y;
    }

    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator*(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} *= HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator*(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} *= y;
    }

    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator/(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} /= HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator/(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} /= y;
    }

    template<typename V>
    __HOST_DEVICE__
    inline
    constexpr
    bool _hip_compare(const V& x, const V& y, int n) noexcept
    {
        return
            (n == -1) ? true : ((x[n] != y[n]) ? false : _hip_compare(x, y, n - 1));
    }

    template<typename T, unsigned int n>
    __HOST_DEVICE__
    inline
    constexpr
    bool operator==(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return _hip_compare(x.data, y.data, n - 1);
    }
    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    bool operator==(const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return x == HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    bool operator==(U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} == y;
    }

    template<typename T, unsigned int n>
    __HOST_DEVICE__
    inline
    constexpr
    bool operator!=(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return !(x == y);
    }
    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    bool operator!=(const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return !(x == y);
    }
    template<typename T, unsigned int n, typename U>
    __HOST_DEVICE__
    inline
    constexpr
    bool operator!=(U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return !(x == y);
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator%(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator%(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator%(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator^(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator^(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator^(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator|(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator|(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator|(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator&(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator&(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator&(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator>>(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator>>(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator>>(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator<<(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator<<(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_arithmetic<U>::value>::type,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    __HOST_DEVICE__
    inline
    constexpr
    HIP_vector_type<T, n> operator<<(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= y;
    }

    /*
     * Map HIP_vector_type<U, rankU> to HIP_vector_type<T, rankT>
     */
    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    __forceinline__ __HOST_DEVICE__ typename std::enable_if<(rankT == 1 && rankU >= 1),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT>(static_cast<T>(u.x));
    };

    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    __forceinline__ __HOST_DEVICE__ typename std::enable_if<(rankT == 2 && rankU == 1),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT> (static_cast<T>(u.x), static_cast<T>(0));
    };

    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    __forceinline__ __HOST_DEVICE__ typename std::enable_if<(rankT == 2 && rankU >= 2),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT> (static_cast<T>(u.x), static_cast<T>(u.y));
    };

    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    __forceinline__ __HOST_DEVICE__ typename std::enable_if<(rankT == 4 && rankU == 1),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT> (static_cast<T>(u.x), static_cast<T>(0),
                                       static_cast<T>(0), static_cast<T>(0));
    };

    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    __forceinline__ __HOST_DEVICE__ typename std::enable_if<(rankT == 4 && rankU == 2),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(u.y),
                                       static_cast<T>(0), static_cast<T>(0));
    };

    template <typename T, unsigned int rankT, typename U, unsigned int rankU>
    __forceinline__ __HOST_DEVICE__ typename std::enable_if<(rankT == 4 && rankU == 4),
                                                            const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
      return HIP_vector_type<T, rankT> (static_cast<T>(u.x), static_cast<T>(u.y),
                                       static_cast<T>(u.z), static_cast<T>(u.w));
    };

    #define __MAKE_VECTOR_TYPE__(CUDA_name, T) \
        using CUDA_name##1 = HIP_vector_type<T, 1>;\
        using CUDA_name##2 = HIP_vector_type<T, 2>;\
        using CUDA_name##3 = HIP_vector_type<T, 3>;\
        using CUDA_name##4 = HIP_vector_type<T, 4>;
#else
    #define __MAKE_VECTOR_TYPE__(CUDA_name, T) \
        typedef struct {\
            T x;\
        } CUDA_name##1;\
        typedef struct {\
            T x;\
            T y;\
        } CUDA_name##2;\
        typedef struct {\
            T x;\
            T y;\
            T z;\
        } CUDA_name##3;\
        typedef struct {\
            T x;\
            T y;\
            T z;\
            T w;\
        } CUDA_name##4;
#endif

__MAKE_VECTOR_TYPE__(uchar, unsigned char);
__MAKE_VECTOR_TYPE__(char, char);
__MAKE_VECTOR_TYPE__(ushort, unsigned short);
__MAKE_VECTOR_TYPE__(short, short);
__MAKE_VECTOR_TYPE__(uint, unsigned int);
__MAKE_VECTOR_TYPE__(int, int);
__MAKE_VECTOR_TYPE__(ulong, unsigned long);
__MAKE_VECTOR_TYPE__(long, long);
__MAKE_VECTOR_TYPE__(ulonglong, unsigned long long);
__MAKE_VECTOR_TYPE__(longlong, long long);
__MAKE_VECTOR_TYPE__(float, float);
__MAKE_VECTOR_TYPE__(double, double);

#else // !defined(__has_attribute)

#if defined(_MSC_VER)
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

/*
this is for compatibility with CUDA as CUDA allows accessing vector components
in C++ program with MSVC
*/
typedef union {
  struct {
    char x;
  };
  char data;
} char1;
typedef union {
  struct {
    char x;
    char y;
  };
  char data[2];
} char2;
typedef union {
  struct {
    char x;
    char y;
    char z;
    char w;
  };
  char data[4];
} char4;
typedef union {
  struct {
    char x;
    char y;
    char z;
  };
  char data[3];
} char3;
typedef union {
  __m64 data;
} char8;
typedef union {
  __m128i data;
} char16;

typedef union {
  struct {
    unsigned char x;
  };
  unsigned char data;
} uchar1;
typedef union {
  struct {
    unsigned char x;
    unsigned char y;
  };
  unsigned char data[2];
} uchar2;
typedef union {
  struct {
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
  };
  unsigned char data[4];
} uchar4;
typedef union {
  struct {
    unsigned char x;
    unsigned char y;
    unsigned char z;
  };
  unsigned char data[3];
} uchar3;
typedef union {
  __m64 data;
} uchar8;
typedef union {
  __m128i data;
} uchar16;

typedef union {
  struct {
    short x;
  };
  short data;
} short1;
typedef union {
  struct {
    short x;
    short y;
  };
  short data[2];
} short2;
typedef union {
  struct {
    short x;
    short y;
    short z;
    short w;
  };
  __m64 data;
} short4;
typedef union {
  struct {
    short x;
    short y;
    short z;
  };
  short data[3];
} short3;
typedef union {
  __m128i data;
} short8;
typedef union {
  __m128i data[2];
} short16;

typedef union {
  struct {
    unsigned short x;
  };
  unsigned short data;
} ushort1;
typedef union {
  struct {
    unsigned short x;
    unsigned short y;
  };
  unsigned short data[2];
} ushort2;
typedef union {
  struct {
    unsigned short x;
    unsigned short y;
    unsigned short z;
    unsigned short w;
  };
  __m64 data;
} ushort4;
typedef union {
  struct {
    unsigned short x;
    unsigned short y;
    unsigned short z;
  };
  unsigned short data[3];
} ushort3;
typedef union {
  __m128i data;
} ushort8;
typedef union {
  __m128i data[2];
} ushort16;

typedef union {
  struct {
    int x;
  };
  int data;
} int1;
typedef union {
  struct {
    int x;
    int y;
  };
  __m64 data;
} int2;
typedef union {
  struct {
    int x;
    int y;
    int z;
    int w;
  };
  __m128i data;
} int4;
typedef union {
  struct {
    int x;
    int y;
    int z;
  };
  int data[3];
} int3;
typedef union {
  __m128i data[2];
} int8;
typedef union {
  __m128i data[4];
} int16;

typedef union {
  struct {
    unsigned int x;
  };
  unsigned int data;
} uint1;
typedef union {
  struct {
    unsigned int x;
    unsigned int y;
  };
  __m64 data;
} uint2;
typedef union {
  struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
  };
  __m128i data;
} uint4;
typedef union {
  struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
  };
  unsigned int data[3];
} uint3;
typedef union {
  __m128i data[2];
} uint8;
typedef union {
  __m128i data[4];
} uint16;

typedef union {
  struct {
    int x;
  };
  int data;
} long1;
typedef union {
  struct {
    int x;
    int y;
  };
  __m64 data;
} long2;
typedef union {
  struct {
    int x;
    int y;
    int z;
    int w;
  };
  __m128i data;
} long4;
typedef union {
  struct {
    int x;
    int y;
    int z;
  };
  int data[3];
} long3;
typedef union {
  __m128i data[2];
} long8;
typedef union {
  __m128i data[4];
} long16;

typedef union {
  struct {
    unsigned int x;
  };
  unsigned int data;
} ulong1;
typedef union {
  struct {
    unsigned int x;
    unsigned int y;
  };
  __m64 data;
} ulong2;
typedef union {
  struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
  };
  __m128i data;
} ulong4;
typedef union {
  struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
  };
  unsigned int data[3];
} ulong3;
typedef union {
  __m128i data[2];
} ulong8;
typedef union {
  __m128i data[4];
} ulong16;

typedef union {
  struct {
    long long x;
  };
  __m64 data;
} longlong1;
typedef union {
  struct {
    long long x;
    long long y;
  };
  __m128i data;
} longlong2;
typedef union {
  struct {
    long long x;
    long long y;
    long long z;
    long long w;
  };
  __m128i data[2];
} longlong4;
typedef union {
  struct {
    long long x;
    long long y;
    long long z;
  };
  __m64 data[3];
} longlong3;
typedef union {
  __m128i data[4];
} longlong8;
typedef union {
  __m128i data[8];
} longlong16;

typedef union {
  struct {
    __m64 x;
  };
  __m64 data;
} ulonglong1;
typedef union {
  struct {
    __m64 x;
    __m64 y;
  };
  __m128i data;
} ulonglong2;
typedef union {
  struct {
    __m64 x;
    __m64 y;
    __m64 z;
    __m64 w;
  };
  __m128i data[2];
} ulonglong4;
typedef union {
  struct {
    __m64 x;
    __m64 y;
    __m64 z;
  };
  __m64 data[3];
} ulonglong3;
typedef union {
  __m128i data[4];
} ulonglong8;
typedef union {
  __m128i data[8];
} ulonglong16;

typedef union {
  struct {
    float x;
  };
  float data;
} float1;
typedef union {
  struct {
    float x;
    float y;
  };
  __m64 data;
} float2;
typedef union {
  struct {
    float x;
    float y;
    float z;
    float w;
  };
  __m128 data;
} float4;
typedef union {
  struct {
    float x;
    float y;
    float z;
  };
  float data[3];
} float3;
typedef union {
  __m256 data;
} float8;
typedef union {
  __m256 data[2];
} float16;

typedef union {
  struct {
    double x;
  };
  double data;
} double1;
typedef union {
  struct {
    double x;
    double y;
  };
  __m128d data;
} double2;
typedef union {
  struct {
    double x;
    double y;
    double z;
    double w;
  };
  __m256d data;
} double4;
typedef union {
  struct {
    double x;
    double y;
    double z;
  };
  double data[3];
} double3;
typedef union {
  __m256d data[2];
} double8;
typedef union {
  __m256d data[4];
} double16;

#else  // !defined(_MSC_VER)

/*
this is for compatibility with CUDA as CUDA allows accessing vector components
in C++ program with MSVC
*/
typedef union {
  struct {
    char x;
  };
  char data;
} char1;
typedef union {
  struct {
    char x;
    char y;
  };
  char data[2];
} char2;
typedef union {
  struct {
    char x;
    char y;
    char z;
    char w;
  };
  char data[4];
} char4;
typedef union {
  char data[8];
} char8;
typedef union {
  char data[16];
} char16;
typedef union {
  struct {
    char x;
    char y;
    char z;
  };
  char data[3];
} char3;

typedef union {
  struct {
    unsigned char x;
  };
  unsigned char data;
} uchar1;
typedef union {
  struct {
    unsigned char x;
    unsigned char y;
  };
  unsigned char data[2];
} uchar2;
typedef union {
  struct {
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
  };
  unsigned char data[4];
} uchar4;
typedef union {
  unsigned char data[8];
} uchar8;
typedef union {
  unsigned char data[16];
} uchar16;
typedef union {
  struct {
    unsigned char x;
    unsigned char y;
    unsigned char z;
  };
  unsigned char data[3];
} uchar3;

typedef union {
  struct {
    short x;
  };
  short data;
} short1;
typedef union {
  struct {
    short x;
    short y;
  };
  short data[2];
} short2;
typedef union {
  struct {
    short x;
    short y;
    short z;
    short w;
  };
  short data[4];
} short4;
typedef union {
  short data[8];
} short8;
typedef union {
  short data[16];
} short16;
typedef union {
  struct {
    short x;
    short y;
    short z;
  };
  short data[3];
} short3;

typedef union {
  struct {
    unsigned short x;
  };
  unsigned short data;
} ushort1;
typedef union {
  struct {
    unsigned short x;
    unsigned short y;
  };
  unsigned short data[2];
} ushort2;
typedef union {
  struct {
    unsigned short x;
    unsigned short y;
    unsigned short z;
    unsigned short w;
  };
  unsigned short data[4];
} ushort4;
typedef union {
  unsigned short data[8];
} ushort8;
typedef union {
  unsigned short data[16];
} ushort16;
typedef union {
  struct {
    unsigned short x;
    unsigned short y;
    unsigned short z;
  };
  unsigned short data[3];
} ushort3;

typedef union {
  struct {
    int x;
  };
  int data;
} int1;
typedef union {
  struct {
    int x;
    int y;
  };
  int data[2];
} int2;
typedef union {
  struct {
    int x;
    int y;
    int z;
    int w;
  };
  int data[4];
} int4;
typedef union {
  int data[8];
} int8;
typedef union {
  int data[16];
} int16;
typedef union {
  struct {
    int x;
    int y;
    int z;
  };
  int data[3];
} int3;

typedef union {
  struct {
    unsigned int x;
  };
  unsigned int data;
} uint1;
typedef union {
  struct {
    unsigned int x;
    unsigned int y;
  };
  unsigned int data[2];
} uint2;
typedef union {
  struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
  };
  unsigned int data[4];
} uint4;
typedef union {
  unsigned int data[8];
} uint8;
typedef union {
  unsigned int data[16];
} uint16;
typedef union {
  struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
  };
  unsigned int data[3];
} uint3;

typedef union {
  struct {
    long x;
  };
  long data;
} long1;
typedef union {
  struct {
    long x;
    long y;
  };
  long data[2];
} long2;
typedef union {
  struct {
    long x;
    long y;
    long z;
    long w;
  };
  long data[4];
} long4;
typedef union {
  long data[8];
} long8;
typedef union {
  long data[16];
} long16;
typedef union {
  struct {
    long x;
    long y;
    long z;
  };
  long data[3];
} long3;

typedef union {
  struct {
    unsigned long x;
  };
  unsigned long data;
} ulong1;
typedef union {
  struct {
    unsigned long x;
    unsigned long y;
  };
  unsigned long data[2];
} ulong2;
typedef union {
  struct {
    unsigned long x;
    unsigned long y;
    unsigned long z;
    unsigned long w;
  };
  unsigned long data[4];
} ulong4;
typedef union {
  unsigned long data[8];
} ulong8;
typedef union {
  unsigned long data[16];
} ulong16;
typedef union {
  struct {
    unsigned long x;
    unsigned long y;
    unsigned long z;
  };
  unsigned long data[3];
} ulong3;

typedef union {
  struct {
    long long x;
  };
  long long data;
} longlong1;
typedef union {
  struct {
    long long x;
    long long y;
  };
  long long data[2];
} longlong2;
typedef union {
  struct {
    long long x;
    long long y;
    long long z;
    long long w;
  };
  long long data[4];
} longlong4;
typedef union {
  long long data[8];
} longlong8;
typedef union {
  long long data[16];
} longlong16;
typedef union {
  struct {
    long long x;
    long long y;
    long long z;
  };
  long long data[3];
} longlong3;

typedef union {
  struct {
    unsigned long long x;
  };
  unsigned long long data;
} ulonglong1;
typedef union {
  struct {
    unsigned long long x;
    unsigned long long y;
  };
  unsigned long long data[2];
} ulonglong2;
typedef union {
  struct {
    unsigned long long x;
    unsigned long long y;
    unsigned long long z;
    unsigned long long w;
  };
  unsigned long long data[4];
} ulonglong4;
typedef union {
  unsigned long long data[8];
} ulonglong8;
typedef union {
  unsigned long long data[16];
} ulonglong16;
typedef union {
  struct {
    unsigned long long x;
    unsigned long long y;
    unsigned long long z;
  };
  unsigned long long data[3];
} ulonglong3;

typedef union {
  struct {
    float x;
  };
  float data;
} float1;
typedef union {
  struct {
    float x;
    float y;
  };
  float data[2];
} float2;
typedef union {
  struct {
    float x;
    float y;
    float z;
    float w;
  };
  float data[4];
} float4;
typedef union {
  float data[8];
} float8;
typedef union {
  float data[16];
} float16;
typedef union {
  struct {
    float x;
    float y;
    float z;
  };
  float data[3];
} float3;

typedef union {
  struct {
    double x;
  };
  double data;
} double1;
typedef union {
  struct {
    double x;
    double y;
  };
  double data[2];
} double2;
typedef union {
  struct {
    double x;
    double y;
    double z;
    double w;
  };
  double data[4];
} double4;
typedef union {
  double data[8];
} double8;
typedef union {
  double data[16];
} double16;
typedef union {
  struct {
    double x;
    double y;
    double z;
  };
  double data[3];
} double3;

#endif // defined(_MSC_VER)
#endif // defined(__has_attribute)

#ifdef __cplusplus
#define DECLOP_MAKE_ONE_COMPONENT(comp, type)                                                      \
  static inline __HOST_DEVICE__ type make_##type(comp x) {                                         \
    type r{x};                                                                                     \
    return r;                                                                                      \
  }

#define DECLOP_MAKE_TWO_COMPONENT(comp, type)                                                      \
  static inline __HOST_DEVICE__ type make_##type(comp x, comp y) {                                 \
    type r{x, y};                                                                                  \
    return r;                                                                                      \
  }

#define DECLOP_MAKE_THREE_COMPONENT(comp, type)                                                    \
  static inline __HOST_DEVICE__ type make_##type(comp x, comp y, comp z) {                         \
    type r{x, y, z};                                                                               \
    return r;                                                                                      \
  }

#define DECLOP_MAKE_FOUR_COMPONENT(comp, type)                                                     \
  static inline __HOST_DEVICE__ type make_##type(comp x, comp y, comp z, comp w) {                 \
    type r{x, y, z, w};                                                                            \
    return r;                                                                                      \
  }
#else
#define DECLOP_MAKE_ONE_COMPONENT(comp, type)                                                      \
  static inline __HOST_DEVICE__ type make_##type(comp x) {                                         \
    type r;                                                                                        \
    r.x = x;                                                                                       \
    return r;                                                                                      \
  }

#define DECLOP_MAKE_TWO_COMPONENT(comp, type)                                                      \
  static inline __HOST_DEVICE__ type make_##type(comp x, comp y) {                                 \
    type r;                                                                                        \
    r.x = x;                                                                                       \
    r.y = y;                                                                                       \
    return r;                                                                                      \
  }

#define DECLOP_MAKE_THREE_COMPONENT(comp, type)                                                    \
  static inline __HOST_DEVICE__ type make_##type(comp x, comp y, comp z) {                         \
    type r;                                                                                        \
    r.x = x;                                                                                       \
    r.y = y;                                                                                       \
    r.z = z;                                                                                       \
    return r;                                                                                      \
  }

#define DECLOP_MAKE_FOUR_COMPONENT(comp, type)                                                     \
  static inline __HOST_DEVICE__ type make_##type(comp x, comp y, comp z, comp w) {                 \
    type r;                                                                                        \
    r.x = x;                                                                                       \
    r.y = y;                                                                                       \
    r.z = z;                                                                                       \
    r.w = w;                                                                                       \
    return r;                                                                                      \
  }
#endif

DECLOP_MAKE_ONE_COMPONENT(unsigned char, uchar1);
DECLOP_MAKE_TWO_COMPONENT(unsigned char, uchar2);
DECLOP_MAKE_THREE_COMPONENT(unsigned char, uchar3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned char, uchar4);

DECLOP_MAKE_ONE_COMPONENT(signed char, char1);
DECLOP_MAKE_TWO_COMPONENT(signed char, char2);
DECLOP_MAKE_THREE_COMPONENT(signed char, char3);
DECLOP_MAKE_FOUR_COMPONENT(signed char, char4);

DECLOP_MAKE_ONE_COMPONENT(unsigned short, ushort1);
DECLOP_MAKE_TWO_COMPONENT(unsigned short, ushort2);
DECLOP_MAKE_THREE_COMPONENT(unsigned short, ushort3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned short, ushort4);

DECLOP_MAKE_ONE_COMPONENT(signed short, short1);
DECLOP_MAKE_TWO_COMPONENT(signed short, short2);
DECLOP_MAKE_THREE_COMPONENT(signed short, short3);
DECLOP_MAKE_FOUR_COMPONENT(signed short, short4);

DECLOP_MAKE_ONE_COMPONENT(unsigned int, uint1);
DECLOP_MAKE_TWO_COMPONENT(unsigned int, uint2);
DECLOP_MAKE_THREE_COMPONENT(unsigned int, uint3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned int, uint4);

DECLOP_MAKE_ONE_COMPONENT(signed int, int1);
DECLOP_MAKE_TWO_COMPONENT(signed int, int2);
DECLOP_MAKE_THREE_COMPONENT(signed int, int3);
DECLOP_MAKE_FOUR_COMPONENT(signed int, int4);

DECLOP_MAKE_ONE_COMPONENT(float, float1);
DECLOP_MAKE_TWO_COMPONENT(float, float2);
DECLOP_MAKE_THREE_COMPONENT(float, float3);
DECLOP_MAKE_FOUR_COMPONENT(float, float4);

DECLOP_MAKE_ONE_COMPONENT(double, double1);
DECLOP_MAKE_TWO_COMPONENT(double, double2);
DECLOP_MAKE_THREE_COMPONENT(double, double3);
DECLOP_MAKE_FOUR_COMPONENT(double, double4);

DECLOP_MAKE_ONE_COMPONENT(unsigned long, ulong1);
DECLOP_MAKE_TWO_COMPONENT(unsigned long, ulong2);
DECLOP_MAKE_THREE_COMPONENT(unsigned long, ulong3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned long, ulong4);

DECLOP_MAKE_ONE_COMPONENT(signed long, long1);
DECLOP_MAKE_TWO_COMPONENT(signed long, long2);
DECLOP_MAKE_THREE_COMPONENT(signed long, long3);
DECLOP_MAKE_FOUR_COMPONENT(signed long, long4);

DECLOP_MAKE_ONE_COMPONENT(unsigned long long, ulonglong1);
DECLOP_MAKE_TWO_COMPONENT(unsigned long long, ulonglong2);
DECLOP_MAKE_THREE_COMPONENT(unsigned long long, ulonglong3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned long long, ulonglong4);

DECLOP_MAKE_ONE_COMPONENT(signed long long, longlong1);
DECLOP_MAKE_TWO_COMPONENT(signed long long, longlong2);
DECLOP_MAKE_THREE_COMPONENT(signed long long, longlong3);
DECLOP_MAKE_FOUR_COMPONENT(signed long long, longlong4);

#endif
