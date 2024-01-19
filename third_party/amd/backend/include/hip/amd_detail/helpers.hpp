/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#include "concepts.hpp"

#include <type_traits>  // For std::conditional, std::decay, std::enable_if,
                        // std::false_type, std result_of and std::true_type.
#include <utility>      // For std::declval.

#ifdef __has_include                      // Check if __has_include is present
#  if __has_include(<version>)            // Check for version header
#    include <version>
#    if defined(__cpp_lib_is_invocable) && !defined(HIP_HAS_INVOCABLE)
#       define HIP_HAS_INVOCABLE __cpp_lib_is_invocable
#    endif
#    if defined(__cpp_lib_result_of_sfinae) && !defined(HIP_HAS_RESULT_OF_SFINAE)
#       define HIP_HAS_RESULT_OF_SFINAE __cpp_lib_result_of_sfinae
#    endif
#  endif
#endif

#ifndef HIP_HAS_INVOCABLE
#define HIP_HAS_INVOCABLE 0
#endif

#ifndef HIP_HAS_RESULT_OF_SFINAE
#define HIP_HAS_RESULT_OF_SFINAE 0
#endif

namespace std {  // TODO: these should be removed as soon as possible.
#if (__cplusplus < 201406L)
#if (__cplusplus < 201402L)
template <bool cond, typename T = void>
using enable_if_t = typename enable_if<cond, T>::type;
template <bool cond, typename T, typename U>
using conditional_t = typename conditional<cond, T, U>::type;
template <typename T>
using decay_t = typename decay<T>::type;
template <FunctionalProcedure F, typename... Ts>
using result_of_t = typename result_of<F(Ts...)>::type;
template <typename T>
using remove_reference_t = typename remove_reference<T>::type;
#endif
#endif
}  // namespace std

namespace hip_impl {
template <typename...>
using void_t_ = void;

#if HIP_HAS_INVOCABLE
template <typename, typename = void>
struct is_callable_impl;

template <FunctionalProcedure F, typename... Ts>
struct is_callable_impl<F(Ts...)> : std::is_invocable<F, Ts...> {};
#elif HIP_HAS_RESULT_OF_SFINAE
template <typename, typename = void>
struct is_callable_impl : std::false_type {};

template <FunctionalProcedure F, typename... Ts>
struct is_callable_impl<F(Ts...), void_t_<typename std::result_of<F(Ts...)>::type > > : std::true_type {};
#else
template <class Base, class T, class Derived>
auto simple_invoke(T Base::*pmd, Derived&& ref)
-> decltype(static_cast<Derived&&>(ref).*pmd);
 
template <class PMD, class Pointer>
auto simple_invoke(PMD&& pmd, Pointer&& ptr)
-> decltype((*static_cast<Pointer&&>(ptr)).*static_cast<PMD&&>(pmd));

template <class Base, class T, class Derived>
auto simple_invoke(T Base::*pmd, const std::reference_wrapper<Derived>& ref)
-> decltype(ref.get().*pmd);
 
template <class Base, class T, class Derived, class... Args>
auto simple_invoke(T Base::*pmf, Derived&& ref, Args&&... args)
-> decltype((static_cast<Derived&&>(ref).*pmf)(static_cast<Args&&>(args)...));
 
template <class PMF, class Pointer, class... Args>
auto simple_invoke(PMF&& pmf, Pointer&& ptr, Args&&... args)
-> decltype(((*static_cast<Pointer&&>(ptr)).*static_cast<PMF&&>(pmf))(static_cast<Args&&>(args)...));

template <class Base, class T, class Derived, class... Args>
auto simple_invoke(T Base::*pmf, const std::reference_wrapper<Derived>& ref, Args&&... args)
-> decltype((ref.get().*pmf)(static_cast<Args&&>(args)...));

template<class F, class... Ts>
auto simple_invoke(F&& f, Ts&&... xs) 
-> decltype(f(static_cast<Ts&&>(xs)...));

template <typename, typename = void>
struct is_callable_impl : std::false_type {};

template <FunctionalProcedure F, typename... Ts>
struct is_callable_impl<F(Ts...), void_t_<decltype(simple_invoke(std::declval<F>(), std::declval<Ts>()...))> >
    : std::true_type {};

#endif

template <typename Call>
struct is_callable : is_callable_impl<Call> {};

#define count_macro_args_impl_hip_(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13,     \
                                   _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25,     \
                                   _26, _27, _28, _29, _30, _31, _n, ...)                          \
    _n
#define count_macro_args_hip_(...)                                                                 \
    count_macro_args_impl_hip_(, ##__VA_ARGS__, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,    \
                               19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,  \
                               0)

#define overloaded_macro_expand_hip_(macro, arg_cnt) macro##arg_cnt
#define overload_macro_impl_hip_(macro, arg_cnt) overloaded_macro_expand_hip_(macro, arg_cnt)
#define overload_macro_hip_(macro, ...)                                                            \
    overload_macro_impl_hip_(macro, count_macro_args_hip_(__VA_ARGS__))(__VA_ARGS__)
}  // namespace hip_impl
