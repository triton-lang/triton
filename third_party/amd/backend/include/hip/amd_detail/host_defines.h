/*
Copyright (c) 2015 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
 *  @file  amd_detail/host_defines.h
 *  @brief TODO-doc
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HOST_DEFINES_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HOST_DEFINES_H

// Add guard to Generic Grid Launch method
#ifndef GENERIC_GRID_LAUNCH
#define GENERIC_GRID_LAUNCH 1
#endif

#if defined(__cplusplus)
namespace __hip_internal {
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int64_t;
#if defined(_MSC_VER)
typedef unsigned long long size_t;
#else
typedef unsigned long size_t;
#endif

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
template <class __T> struct enable_if<true, __T> {
  typedef __T type;
};

template <bool _B> struct true_or_false_type : public false_type {};
template <> struct true_or_false_type<true> : public true_type {};

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

template <typename _Tp> struct is_floating_point : public false_type {};
template <> struct is_floating_point<float> : public true_type {};
template <> struct is_floating_point<double> : public true_type {};
template <> struct is_floating_point<long double> : public true_type {};

template <typename __T, typename __U> struct is_same : public false_type {};
template <typename __T> struct is_same<__T, __T> : public true_type {};

template <typename _Tp, bool = is_arithmetic<_Tp>::value> struct is_signed : public false_type {};
template <typename _Tp> struct is_signed<_Tp, true> : public true_or_false_type<_Tp(-1) < _Tp(0)> {
};

template <class T> auto test_returnable(int)
    -> decltype(void(static_cast<T (*)()>(nullptr)), true_type{});
template <class> auto test_returnable(...) -> false_type;

template <class T> struct type_identity {
  using type = T;
};

template <class T>  // Note that `cv void&` is a substitution failure
auto try_add_lvalue_reference(int) -> type_identity<T&>;
template <class T>  // Handle T = cv void case
auto try_add_lvalue_reference(...) -> type_identity<T>;

template <class T> auto try_add_rvalue_reference(int) -> type_identity<T&&>;
template <class T> auto try_add_rvalue_reference(...) -> type_identity<T>;

template <class T> struct add_lvalue_reference : decltype(try_add_lvalue_reference<T>(0)) {};

template <class T> struct add_rvalue_reference : decltype(try_add_rvalue_reference<T>(0)) {};

template <typename T> typename add_rvalue_reference<T>::type declval() noexcept;

template <class From, class To> auto test_implicitly_convertible(int)
    -> decltype(void(declval<void (&)(To)>()(declval<From>())), true_type{});

template <class, class> auto test_implicitly_convertible(...) -> false_type;

template <class T> struct remove_cv {
  typedef T type;
};
template <class T> struct remove_cv<const T> {
  typedef T type;
};
template <class T> struct remove_cv<volatile T> {
  typedef T type;
};
template <class T> struct remove_cv<const volatile T> {
  typedef T type;
};

template <class T> struct is_void : public is_same<void, typename remove_cv<T>::type> {};

template <class From, class To> struct is_convertible
    : public integral_constant<bool, (decltype(test_returnable<To>(0))::value &&
                                      decltype(test_implicitly_convertible<From, To>(0))::value) ||
                                         (is_void<From>::value && is_void<To>::value)> {};

template <typename _CharT> struct char_traits;
template <typename _CharT, typename _Traits = char_traits<_CharT>> class basic_istream;
template <typename _CharT, typename _Traits = char_traits<_CharT>> class basic_ostream;
typedef basic_istream<char> istream;
typedef basic_ostream<char> ostream;

template <typename _Tp> struct is_standard_layout
    : public integral_constant<bool, __is_standard_layout(_Tp)> {};

template <typename _Tp> struct is_trivial : public integral_constant<bool, __is_trivial(_Tp)> {};


template <bool B, class T, class F> struct conditional {
  using type = T;
};
template <class T, class F> struct conditional<false, T, F> {
  using type = F;
};

template <class T> struct alignment_of : integral_constant<size_t, alignof(T)> {};

template <typename T, T... Ints> struct integer_sequence {
  using value_type = T;
  static constexpr size_t size() noexcept { return sizeof...(Ints); }
};

template <size_t... Ints> using index_sequence = integer_sequence<size_t, Ints...>;

template <size_t _hip_N, size_t... Ints> struct make_index_sequence_impl
    : make_index_sequence_impl<_hip_N - 1, _hip_N - 1, Ints...> {};

template <size_t... Ints> struct make_index_sequence_impl<0, Ints...> {
  using type = index_sequence<Ints...>;
};

template <size_t _hip_N> using make_index_sequence =
    typename make_index_sequence_impl<_hip_N>::type;

template <size_t... Ints>
constexpr index_sequence<Ints...> make_index_sequence_value(index_sequence<Ints...>) {
  return {};
}
}  // namespace __hip_internal
typedef __hip_internal::uint8_t __hip_uint8_t;
typedef __hip_internal::uint16_t __hip_uint16_t;
typedef __hip_internal::uint32_t __hip_uint32_t;
typedef __hip_internal::uint64_t __hip_uint64_t;
typedef __hip_internal::int8_t __hip_int8_t;
typedef __hip_internal::int16_t __hip_int16_t;
typedef __hip_internal::int32_t __hip_int32_t;
typedef __hip_internal::int64_t __hip_int64_t;
#endif  // defined(__cplusplus)

#if defined(__clang__) && defined(__HIP__)
#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#endif  // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__

#if !defined(__has_feature) || !__has_feature(cuda_noinline_keyword)
#define __noinline__ __attribute__((noinline))
#endif

#define __forceinline__ inline __attribute__((always_inline))

#if __HIP_NO_IMAGE_SUPPORT
#define __hip_img_chk__                                                                            \
  __attribute__((unavailable("The image/texture API not supported on the device")))
#else
#define __hip_img_chk__
#endif

#else

// Non-HCC compiler
/**
 * Function and kernel markers
 */
#define __host__
#define __device__

#define __global__

#define __noinline__
#define __forceinline__ inline

#define __shared__
#define __constant__

#define __hip_img_chk__
#endif  // defined(__clang__) && defined(__HIP__)

#endif
