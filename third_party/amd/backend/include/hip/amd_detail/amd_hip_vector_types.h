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
#define __NATIVE_VECTOR__(n, T) alignas(n * sizeof(T)) T[n]
#endif

#if defined(__cplusplus)
#if !defined(__HIPCC_RTC__)
#include <array>
#include <iosfwd>
#include <type_traits>
#endif  // defined(__HIPCC_RTC__)

template <typename T, unsigned int n> struct HIP_vector_base;
template <typename T, unsigned int rank> struct HIP_vector_type;

namespace hip_impl {
template <typename T, unsigned int n> __attribute__((always_inline)) __HOST_DEVICE__
    typename HIP_vector_base<T, n>::Native_vec_*
    get_native_pointer(HIP_vector_base<T, n>& base_vec) {
  static_assert(sizeof(base_vec) == sizeof(typename HIP_vector_base<T, n>::Native_vec_));
  static_assert(
      (__hip_internal::alignment_of<HIP_vector_base<T, n>>::value %
       __hip_internal::alignment_of<typename HIP_vector_base<T, n>::Native_vec_>::value) == 0);
  return reinterpret_cast<typename HIP_vector_base<T, n>::Native_vec_*>(&base_vec);
};

template <typename T, unsigned int n>
__attribute__((always_inline)) __HOST_DEVICE__ const typename HIP_vector_base<T, n>::Native_vec_*
get_native_pointer(const HIP_vector_base<T, n>& base_vec) {
  static_assert(sizeof(base_vec) == sizeof(typename HIP_vector_base<T, n>::Native_vec_));
  static_assert(
      (__hip_internal::alignment_of<HIP_vector_base<T, n>>::value %
       __hip_internal::alignment_of<typename HIP_vector_base<T, n>::Native_vec_>::value) == 0);
  return reinterpret_cast<const typename HIP_vector_base<T, n>::Native_vec_*>(&base_vec);
};
}  // Namespace hip_impl.

template <typename T, unsigned int n> __attribute__((always_inline)) __HOST_DEVICE__
    typename HIP_vector_base<T, n>::Native_vec_&
    get_native_vector(HIP_vector_base<T, n>& base_vec) {
  return *hip_impl::get_native_pointer(base_vec);
};

template <typename T, unsigned int n>
__attribute__((always_inline)) __HOST_DEVICE__ const typename HIP_vector_base<T, n>::Native_vec_&
get_native_vector(const HIP_vector_base<T, n>& base_vec) {
  return *hip_impl::get_native_pointer(base_vec);
};

template <typename T> struct HIP_vector_base<T, 1> {
  using Native_vec_ = __NATIVE_VECTOR__(1, T);

  T x;

  using value_type = T;

  __HOST_DEVICE__
  HIP_vector_base() = default;
  __HOST_DEVICE__
  constexpr HIP_vector_base(const HIP_vector_base&) = default;
  __HOST_DEVICE__
  explicit constexpr HIP_vector_base(T x_) : x(x_) {}
  __HOST_DEVICE__
  constexpr HIP_vector_base(HIP_vector_base&&) = default;
  __HOST_DEVICE__
  ~HIP_vector_base() = default;
  __HOST_DEVICE__
  HIP_vector_base& operator=(const HIP_vector_base&) = default;
};

template <typename T> struct alignas(2 * sizeof(T)) HIP_vector_base<T, 2> {
  using Native_vec_ = __NATIVE_VECTOR__(2, T);

  T x, y;

  using value_type = T;

  __HOST_DEVICE__
  HIP_vector_base() = default;
  __HOST_DEVICE__
  constexpr HIP_vector_base(const HIP_vector_base&) = default;
  __HOST_DEVICE__
  constexpr HIP_vector_base(T x_, T y_ = T()) : x(x_), y(y_) {}
  __HOST_DEVICE__
  constexpr HIP_vector_base(HIP_vector_base&&) = default;
  __HOST_DEVICE__
  ~HIP_vector_base() = default;
  __HOST_DEVICE__
  HIP_vector_base& operator=(const HIP_vector_base&) = default;
};

template <typename T> struct HIP_vector_base<T, 3> {
  struct Native_vec_ {
    T d[3];

    __HOST_DEVICE__
    Native_vec_() = default;

    __HOST_DEVICE__
    explicit constexpr Native_vec_(T x_) noexcept : d{x_, x_, x_} {}
    __HOST_DEVICE__
    constexpr Native_vec_(T x_, T y_, T z_) noexcept : d{x_, y_, z_} {}
    __HOST_DEVICE__
    constexpr Native_vec_(const Native_vec_&) = default;
    __HOST_DEVICE__
    constexpr Native_vec_(Native_vec_&&) = default;
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
    Native_vec_& operator+=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] += x_.d[i];
      return *this;
    }
    __HOST_DEVICE__
    Native_vec_& operator-=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] -= x_.d[i];
      return *this;
    }

    __HOST_DEVICE__
    Native_vec_& operator*=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] *= x_.d[i];
      return *this;
    }
    __HOST_DEVICE__
    Native_vec_& operator/=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] /= x_.d[i];
      return *this;
    }

    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_signed<U>{}>::type* = nullptr>
    __HOST_DEVICE__ Native_vec_ operator-() const noexcept {
      auto r{*this};
      for (auto&& x : r.d) x = -x;
      return r;
    }

    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __HOST_DEVICE__ Native_vec_ operator~() const noexcept {
      auto r{*this};
      for (auto&& x : r.d) x = ~x;
      return r;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __HOST_DEVICE__ Native_vec_& operator%=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] %= x_.d[i];
      return *this;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __HOST_DEVICE__ Native_vec_& operator^=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] ^= x_.d[i];
      return *this;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __HOST_DEVICE__ Native_vec_& operator|=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] |= x_.d[i];
      return *this;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __HOST_DEVICE__ Native_vec_& operator&=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] &= x_.d[i];
      return *this;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __HOST_DEVICE__ Native_vec_& operator>>=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] >>= x_.d[i];
      return *this;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __HOST_DEVICE__ Native_vec_& operator<<=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] <<= x_.d[i];
      return *this;
    }
#if defined(__INTEL_COMPILER)
    typedef struct {
      int values[4];
    } _Vec3_cmp;
    using Vec3_cmp = _Vec3_cmp;
#else
    using Vec3_cmp = int __attribute__((vector_size(4 * sizeof(int))));
#endif  // INTEL
    __HOST_DEVICE__
    Vec3_cmp operator==(const Native_vec_& x_) const noexcept {
      return Vec3_cmp{d[0] == x_.d[0], d[1] == x_.d[1], d[2] == x_.d[2]};
    }
  };

  T x, y, z;

  using value_type = T;

  __HOST_DEVICE__
  HIP_vector_base() = default;
  __HOST_DEVICE__
  constexpr HIP_vector_base(const HIP_vector_base&) = default;
  __HOST_DEVICE__
  constexpr HIP_vector_base(T x_, T y_ = T(), T z_ = T()) : x(x_), y(y_), z(z_) {};
  __HOST_DEVICE__
  constexpr HIP_vector_base(HIP_vector_base&&) = default;
  __HOST_DEVICE__
  ~HIP_vector_base() = default;

  __HOST_DEVICE__
  HIP_vector_base& operator=(const HIP_vector_base&) = default;
  __HOST_DEVICE__
  HIP_vector_base& operator=(HIP_vector_base&&) = default;
};

template <typename T> struct alignas(4 * sizeof(T)) HIP_vector_base<T, 4> {
  using Native_vec_ = __NATIVE_VECTOR__(4, T);

  T x, y, z, w;

  using value_type = T;

  __HOST_DEVICE__
  HIP_vector_base() = default;
  __HOST_DEVICE__
  constexpr HIP_vector_base(const HIP_vector_base&) = default;
  __HOST_DEVICE__
  constexpr HIP_vector_base(T x_, T y_ = T(), T z_ = T(), T w_ = T())
      : x(x_), y(y_), z(z_), w(w_) {};
  __HOST_DEVICE__
  constexpr HIP_vector_base(HIP_vector_base&&) = default;
  __HOST_DEVICE__
  ~HIP_vector_base() = default;
  __HOST_DEVICE__
  HIP_vector_base& operator=(const HIP_vector_base&) = default;
};

template <typename T, size_t rank, size_t... indices>
constexpr inline __HOST_DEVICE__ HIP_vector_type<T, rank> make_vector_type_impl(
    T val, __hip_internal::index_sequence<indices...>) noexcept {
  // Fills vec with vals, and ignores the indices
  return HIP_vector_type<T, rank>{((void)indices, val)...};
}

template <typename T, unsigned int rank>
constexpr inline __HOST_DEVICE__ HIP_vector_type<T, rank> make_vector_type(T val) {
  return make_vector_type_impl<T, rank>(
      val, __hip_internal::make_index_sequence_value(__hip_internal::make_index_sequence<rank>{}));
}

template <typename T, unsigned int rank> struct HIP_vector_type : public HIP_vector_base<T, rank> {
  using typename HIP_vector_base<T, rank>::Native_vec_;

  __HOST_DEVICE__
  HIP_vector_type() = default;
  template <typename U, typename __hip_internal::enable_if<
                            __hip_internal::is_convertible<U, T>::value>::type* = nullptr>
  __HOST_DEVICE__ explicit constexpr HIP_vector_type(U x_) noexcept
      : HIP_vector_base<T, rank>{static_cast<T>(x_)} {}
  template <  // TODO: constrain based on type as well.
      typename... Us,
      typename __hip_internal::enable_if<(rank > 1) && sizeof...(Us) == rank>::type* = nullptr>
  __HOST_DEVICE__ constexpr HIP_vector_type(Us... xs) noexcept
      : HIP_vector_base<T, rank>{static_cast<T>(xs)...} {}
  __HOST_DEVICE__
  constexpr HIP_vector_type(const HIP_vector_type&) = default;
  __HOST_DEVICE__
  constexpr HIP_vector_type(HIP_vector_type&&) = default;
  __HOST_DEVICE__
  ~HIP_vector_type() = default;

  __HOST_DEVICE__
  HIP_vector_type& operator=(const HIP_vector_type&) = default;
  __HOST_DEVICE__
  HIP_vector_type& operator=(HIP_vector_type&&) = default;

  // Operators
  __HOST_DEVICE__
  T& operator[](size_t idx) noexcept { return reinterpret_cast<T*>(this)[idx]; }
  __HOST_DEVICE__
  const T& operator[](size_t idx) const noexcept { return reinterpret_cast<const T*>(this)[idx]; }

  __HOST_DEVICE__
  HIP_vector_type& operator++() noexcept {
    HIP_vector_type unity = make_vector_type<T, rank>(1);
    return *this += unity;
  }
  __HOST_DEVICE__
  HIP_vector_type operator++(int) noexcept {
    auto tmp(*this);
    ++*this;
    return tmp;
  }

  __HOST_DEVICE__
  HIP_vector_type& operator--() noexcept {
    HIP_vector_type unity = make_vector_type<T, rank>(1);
    return *this -= unity;
  }
  __HOST_DEVICE__
  HIP_vector_type operator--(int) noexcept {
    auto tmp(*this);
    --*this;
    return tmp;
  }

  __HOST_DEVICE__ HIP_vector_type& operator+=(const HIP_vector_type& x) noexcept {
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(*this) += get_native_vector(x);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(*this)[i] += get_native_vector(x)[i];
#endif
    return *this;
  }
  template <typename U, typename __hip_internal::enable_if<
                            __hip_internal::is_convertible<U, T>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type& operator+=(U x) noexcept {
    return *this += make_vector_type<T, rank>(x);
  }

  __HOST_DEVICE__ HIP_vector_type& operator-=(const HIP_vector_type& x) noexcept {
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(*this) -= get_native_vector(x);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(*this)[i] -= get_native_vector(x)[i];
#endif
    return *this;
  }
  template <typename U, typename __hip_internal::enable_if<
                            __hip_internal::is_convertible<U, T>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type& operator-=(U x) noexcept {
    return *this -= make_vector_type<T, rank>(x);
  }

  __HOST_DEVICE__ HIP_vector_type& operator*=(const HIP_vector_type& x) noexcept {
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(*this) *= get_native_vector(x);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(*this)[i] *= get_native_vector(x)[i];
#endif
    return *this;
  }

  friend __HOST_DEVICE__ inline constexpr HIP_vector_type operator*(
      HIP_vector_type x, const HIP_vector_type& y) noexcept {
    return HIP_vector_type{x} *= y;
  }

  template <typename U, typename __hip_internal::enable_if<
                            __hip_internal::is_convertible<U, T>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type& operator*=(U x) noexcept {
    return *this *= make_vector_type<T, rank>(x);
  }

  friend __HOST_DEVICE__ inline constexpr HIP_vector_type operator/(
      HIP_vector_type x, const HIP_vector_type& y) noexcept {
    return HIP_vector_type{x} /= y;
  }

  __HOST_DEVICE__ HIP_vector_type& operator/=(const HIP_vector_type& x) noexcept {
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(*this) /= get_native_vector(x);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(*this)[i] /= get_native_vector(x)[i];
#endif
    return *this;
  }
  template <typename U, typename __hip_internal::enable_if<
                            __hip_internal::is_convertible<U, T>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type& operator/=(U x) noexcept {
    return *this /= make_vector_type<T, rank>(x);
  }

  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_signed<U>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type operator-() const noexcept {
    auto tmp(*this);
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(tmp) = -get_native_vector(tmp);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(tmp)[i] = -get_native_vector(tmp)[i];
#endif
    return tmp;
  }

  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type operator~() const noexcept {
    HIP_vector_type r{*this};
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(r) = ~get_native_vector(r);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(r)[i] = ~get_native_vector(r)[i];
#endif
    return r;
  }

  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type& operator%=(const HIP_vector_type& x) noexcept {
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(*this) %= get_native_vector(x);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(*this)[i] %= get_native_vector(x)[i];
#endif
    return *this;
  }

  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type& operator^=(const HIP_vector_type& x) noexcept {
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(*this) ^= get_native_vector(x);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(*this)[i] ^= get_native_vector(x)[i];
#endif
    return *this;
  }

  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type& operator|=(const HIP_vector_type& x) noexcept {
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(*this) |= get_native_vector(x);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(*this)[i] |= get_native_vector(x)[i];
#endif
    return *this;
  }

  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type& operator&=(const HIP_vector_type& x) noexcept {
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(*this) &= get_native_vector(x);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(*this)[i] &= get_native_vector(x)[i];
#endif
    return *this;
  }

  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type& operator>>=(const HIP_vector_type& x) noexcept {
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(*this) >>= get_native_vector(x);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(*this)[i] >>= get_native_vector(x)[i];
#endif
    return *this;
  }

  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __HOST_DEVICE__ HIP_vector_type& operator<<=(const HIP_vector_type& x) noexcept {
#if __HIP_USE_NATIVE_VECTOR__
    get_native_vector(*this) <<= get_native_vector(x);
#else
    for (auto i = 0u; i != rank; ++i) get_native_vector(*this)[i] <<= get_native_vector(x)[i];
#endif
    return *this;
  }
};

template <typename T, unsigned int n>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator+(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} += y;
}
template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator+(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} += make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator+(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) += y;
}

template <typename T, unsigned int n>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator-(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} -= y;
}
template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator-(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} -= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator-(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) -= y;
}

template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator*(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} *= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator*(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) *= y;
}

template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator/(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} /= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator/(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) /= y;
}

template <typename T, unsigned int n> __HOST_DEVICE__ inline
#if __cplusplus >= 201402L && !defined(__HIPCC_RTC__)
    constexpr
#endif
    bool
    operator==(const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  bool isTrue = true;
  const auto& native_x = get_native_vector(x);
  const auto& native_y = get_native_vector(y);
  for (unsigned int i = 0; i < n; ++i) {
    isTrue = (isTrue && (native_x[i] == native_y[i]));
  }
  return isTrue;
}

template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr bool operator==(const HIP_vector_type<T, n>& x, U y) noexcept {
  return x == make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr bool operator==(U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) == y;
}

template <typename T, unsigned int n>
__HOST_DEVICE__ inline constexpr bool operator!=(const HIP_vector_type<T, n>& x,
                                                 const HIP_vector_type<T, n>& y) noexcept {
  return !(x == y);
}
template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr bool operator!=(const HIP_vector_type<T, n>& x, U y) noexcept {
  return !(x == y);
}
template <typename T, unsigned int n, typename U>
__HOST_DEVICE__ inline constexpr bool operator!=(U x, const HIP_vector_type<T, n>& y) noexcept {
  return !(x == y);
}

template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator%(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} %= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator%(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} %= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator%(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) %= y;
}

template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator^(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} ^= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator^(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} ^= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator^(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) ^= y;
}

template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator|(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} |= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator|(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} |= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator|(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) |= y;
}

template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator&(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} &= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator&(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} &= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator&(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) &= y;
}

template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator>>(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} >>= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator>>(const HIP_vector_type<T, n>& x,
                                                                  U y) noexcept {
  return HIP_vector_type<T, n>{x} >>= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator>>(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) >>= y;
}

template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator<<(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} <<= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator<<(const HIP_vector_type<T, n>& x,
                                                                  U y) noexcept {
  return HIP_vector_type<T, n>{x} <<= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_arithmetic<U>::value>::type,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__HOST_DEVICE__ inline constexpr HIP_vector_type<T, n> operator<<(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) <<= y;
}

/*
 * Map HIP_vector_type<U, rankU> to HIP_vector_type<T, rankT>
 */
template <typename T, unsigned int rankT, typename U, unsigned int rankU>
__forceinline__ __HOST_DEVICE__
    typename __hip_internal::enable_if<(rankT == 1 && rankU >= 1),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x));
};

template <typename T, unsigned int rankT, typename U, unsigned int rankU>
__forceinline__ __HOST_DEVICE__
    typename __hip_internal::enable_if<(rankT == 2 && rankU == 1),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(0));
};

template <typename T, unsigned int rankT, typename U, unsigned int rankU>
__forceinline__ __HOST_DEVICE__
    typename __hip_internal::enable_if<(rankT == 2 && rankU >= 2),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(u.y));
};

template <typename T, unsigned int rankT, typename U, unsigned int rankU>
__forceinline__ __HOST_DEVICE__
    typename __hip_internal::enable_if<(rankT == 4 && rankU == 1),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(0), static_cast<T>(0),
                                   static_cast<T>(0));
};

template <typename T, unsigned int rankT, typename U, unsigned int rankU>
__forceinline__ __HOST_DEVICE__
    typename __hip_internal::enable_if<(rankT == 4 && rankU == 2),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(u.y), static_cast<T>(0),
                                   static_cast<T>(0));
};

template <typename T, unsigned int rankT, typename U, unsigned int rankU>
__forceinline__ __HOST_DEVICE__
    typename __hip_internal::enable_if<(rankT == 4 && rankU == 4),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(u.y), static_cast<T>(u.z),
                                   static_cast<T>(u.w));
};

#define __MAKE_VECTOR_TYPE__(CUDA_name, T)                                                         \
  using CUDA_name##1 = HIP_vector_type<T, 1>;                                                      \
  using CUDA_name##2 = HIP_vector_type<T, 2>;                                                      \
  using CUDA_name##3 = HIP_vector_type<T, 3>;                                                      \
  using CUDA_name##4 = HIP_vector_type<T, 4>;
#else
#define __MAKE_VECTOR_TYPE__(CUDA_name, T)                                                         \
  typedef struct {                                                                                 \
    T x;                                                                                           \
  } CUDA_name##1;                                                                                  \
  typedef struct {                                                                                 \
    T x;                                                                                           \
    T y;                                                                                           \
  } CUDA_name##2;                                                                                  \
  typedef struct {                                                                                 \
    T x;                                                                                           \
    T y;                                                                                           \
    T z;                                                                                           \
  } CUDA_name##3;                                                                                  \
  typedef struct {                                                                                 \
    T x;                                                                                           \
    T y;                                                                                           \
    T z;                                                                                           \
    T w;                                                                                           \
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

#else  // !defined(__has_attribute)

#if defined(_MSC_VER)
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

/*
this is for compatibility with CUDA as CUDA allows accessing vector components
in C++ program with MSVC
structs are wrapped with templates so that mangled names match templated implementation
*/

template <typename T, unsigned int n> struct HIP_vector_type;

// One template per vector size
template <typename T> struct HIP_vector_type<T, 1> {
  union {
    struct {
      T x;
    };
    T data;
  };
};
template <typename T> struct HIP_vector_type<T, 2> {
  union {
    struct {
      T x;
      T y;
    };
    T data[2];
  };
};
template <typename T> struct HIP_vector_type<T, 3> {
  union {
    struct {
      T x;
      T y;
      T z;
    };
    T data[3];
  };
};
template <typename T> struct HIP_vector_type<T, 4> {
  union {
    struct {
      T x;
      T y;
      T z;
      T w;
    };
    T data[4];
  };
};
// 8- and 16-length vectors do not have CUDA-style accessible components
template <typename T> struct HIP_vector_type<T, 8> {
  union {
    T data[8];
  };
};
template <typename T> struct HIP_vector_type<T, 16> {
  union {
    T data[16];
  };
};

// Explicit specialization for vectors using MSVC-specific definitions
template <> struct HIP_vector_type<char, 8> {
  union {
    __m64 data;
  };
};
template <> struct HIP_vector_type<char, 16> {
  union {
    __m128i data;
  };
};

template <> struct HIP_vector_type<unsigned char, 8> {
  union {
    __m64 data;
  };
};
template <> struct HIP_vector_type<unsigned char, 16> {
  union {
    __m128i data;
  };
};

template <> struct HIP_vector_type<short, 4> {
  union {
    struct {
      short x;
      short y;
      short z;
      short w;
    };
    __m64 data;
  };
};
template <> struct HIP_vector_type<short, 8> {
  union {
    __m128i data;
  };
};
template <> struct HIP_vector_type<short, 16> {
  union {
    __m128i data[2];
  };
};

template <> struct HIP_vector_type<unsigned short, 4> {
  union {
    struct {
      unsigned short x;
      unsigned short y;
      unsigned short z;
      unsigned short w;
    };
    __m64 data;
  };
};
template <> struct HIP_vector_type<unsigned short, 8> {
  union {
    __m128i data;
  };
};
template <> struct HIP_vector_type<unsigned short, 16> {
  union {
    __m128i data[2];
  };
};

template <> struct HIP_vector_type<int, 2> {
  union {
    struct {
      int x;
      int y;
    };
    __m64 data;
  };
};
template <> struct HIP_vector_type<int, 4> {
  union {
    struct {
      int x;
      int y;
      int z;
      int w;
    };
    __m128i data;
  };
};
template <> struct HIP_vector_type<int, 8> {
  union {
    __m128i data[2];
  };
};
template <> struct HIP_vector_type<int, 16> {
  union {
    __m128i data[4];
  };
};

template <> struct HIP_vector_type<unsigned int, 2> {
  union {
    struct {
      unsigned int x;
      unsigned int y;
    };
    __m64 data;
  };
};
template <> struct HIP_vector_type<unsigned int, 4> {
  union {
    struct {
      unsigned int x;
      unsigned int y;
      unsigned int z;
      unsigned int w;
    };
    __m128i data;
  };
};
template <> struct HIP_vector_type<unsigned int, 8> {
  union {
    __m128i data[2];
  };
};
template <> struct HIP_vector_type<unsigned int, 16> {
  union {
    __m128i data[4];
  };
};

// MSVC uses 32-bit longs and 64-bit long longs, explicitly defining for clarity
template <> struct HIP_vector_type<long, 1> {
  union {
    struct {
      std::int32_t x;
    };
    std::int32_t data;
  };
};
template <> struct HIP_vector_type<long, 2> {
  union {
    struct {
      std::int32_t x;
      std::int32_t y;
    };
    __m64 data;
  };
};
template <> struct HIP_vector_type<long, 3> {
  union {
    struct {
      std::int32_t x;
      std::int32_t y;
      std::int32_t z;
    };
    std::int32_t data[3];
  };
};
template <> struct HIP_vector_type<long, 4> {
  union {
    struct {
      std::int32_t x;
      std::int32_t y;
      std::int32_t z;
      std::int32_t w;
    };
    __m128i data;
  };
};
template <> struct HIP_vector_type<long, 8> {
  union {
    __m128i data[2];
  };
};
template <> struct HIP_vector_type<long, 16> {
  union {
    __m128i data[4];
  };
};

template <> struct HIP_vector_type<unsigned long, 1> {
  union {
    struct {
      std::uint32_t x;
    };
    std::uint32_t data;
  };
};
template <> struct HIP_vector_type<unsigned long, 2> {
  union {
    struct {
      std::uint32_t x;
      std::uint32_t y;
    };
    __m64 data;
  };
};
template <> struct HIP_vector_type<unsigned long, 3> {
  union {
    struct {
      std::uint32_t x;
      std::uint32_t y;
      std::uint32_t z;
    };
    std::uint32_t data[3];
  };
};
template <> struct HIP_vector_type<unsigned long, 4> {
  union {
    struct {
      std::uint32_t x;
      std::uint32_t y;
      std::uint32_t z;
      std::uint32_t w;
    };
    __m128i data;
  };
};
template <> struct HIP_vector_type<unsigned long, 8> {
  union {
    __m128i data[2];
  };
};
template <> struct HIP_vector_type<unsigned long, 16> {
  union {
    __m128i data[4];
  };
};

template <> struct HIP_vector_type<long long, 1> {
  union {
    struct {
      std::int64_t x;
    };
    __m64 data;
  };
};
template <> struct HIP_vector_type<long long, 2> {
  union {
    struct {
      std::int64_t x;
      std::int64_t y;
    };
    __m128i data;
  };
};
template <> struct HIP_vector_type<long long, 3> {
  union {
    struct {
      std::int64_t x;
      std::int64_t y;
      std::int64_t z;
    };
    __m64 data[3];
  };
};
template <> struct HIP_vector_type<long long, 4> {
  union {
    struct {
      std::int64_t x;
      std::int64_t y;
      std::int64_t z;
      std::int64_t w;
    };
    __m128i data[2];
  };
};
template <> struct HIP_vector_type<long long, 8> {
  union {
    __m128i data[4];
  };
};
template <> struct HIP_vector_type<long long, 16> {
  union {
    __m128i data[8];
  };
};

template <> struct HIP_vector_type<unsigned long long, 1> {
  union {
    struct {
      std::uint64_t x;
    };
    __m64 data;
  };
};
template <> struct HIP_vector_type<unsigned long long, 2> {
  union {
    struct {
      std::uint64_t x;
      std::uint64_t y;
    };
    __m128i data;
  };
};
template <> struct HIP_vector_type<unsigned long long, 3> {
  union {
    struct {
      std::uint64_t x;
      std::uint64_t y;
      std::uint64_t z;
    };
    __m64 data[3];
  };
};
template <> struct HIP_vector_type<unsigned long long, 4> {
  union {
    struct {
      std::uint64_t x;
      std::uint64_t y;
      std::uint64_t z;
      std::uint64_t w;
    };
    __m128i data[2];
  };
};
template <> struct HIP_vector_type<unsigned long long, 8> {
  union {
    __m128i data[4];
  };
};
template <> struct HIP_vector_type<unsigned long long, 16> {
  union {
    __m128i data[8];
  };
};

template <> struct HIP_vector_type<float, 2> {
  union {
    struct {
      float x;
      float y;
    };
    __m64 data;
  };
};
template <> struct HIP_vector_type<float, 4> {
  union {
    struct {
      float x;
      float y;
      float z;
      float w;
    };
    __m128 data;
  };
};
template <> struct HIP_vector_type<float, 8> {
  union {
    __m256 data;
  };
};
template <> struct HIP_vector_type<float, 16> {
  union {
    __m256 data[2];
  };
};

template <> struct HIP_vector_type<double, 2> {
  union {
    struct {
      double x;
      double y;
    };
    __m128d data;
  };
};
template <> struct HIP_vector_type<double, 4> {
  union {
    struct {
      double x;
      double y;
      double z;
      double w;
    };
    __m256d data;
  };
};
template <> struct HIP_vector_type<double, 8> {
  union {
    __m256d data[2];
  };
};
template <> struct HIP_vector_type<double, 16> {
  union {
    __m256d data[4];
  };
};

// Type aliasing
using char1 = HIP_vector_type<char, 1>;
using char2 = HIP_vector_type<char, 2>;
using char3 = HIP_vector_type<char, 3>;
using char4 = HIP_vector_type<char, 4>;
using char8 = HIP_vector_type<char, 8>;
using char16 = HIP_vector_type<char, 16>;
using uchar1 = HIP_vector_type<unsigned char, 1>;
using uchar2 = HIP_vector_type<unsigned char, 2>;
using uchar3 = HIP_vector_type<unsigned char, 3>;
using uchar4 = HIP_vector_type<unsigned char, 4>;
using uchar8 = HIP_vector_type<unsigned char, 8>;
using uchar16 = HIP_vector_type<unsigned char, 16>;
using short1 = HIP_vector_type<short, 1>;
using short2 = HIP_vector_type<short, 2>;
using short3 = HIP_vector_type<short, 3>;
using short4 = HIP_vector_type<short, 4>;
using short8 = HIP_vector_type<short, 8>;
using short16 = HIP_vector_type<short, 16>;
using ushort1 = HIP_vector_type<unsigned short, 1>;
using ushort2 = HIP_vector_type<unsigned short, 2>;
using ushort3 = HIP_vector_type<unsigned short, 3>;
using ushort4 = HIP_vector_type<unsigned short, 4>;
using ushort8 = HIP_vector_type<unsigned short, 8>;
using ushort16 = HIP_vector_type<unsigned short, 16>;
using int1 = HIP_vector_type<int, 1>;
using int2 = HIP_vector_type<int, 2>;
using int3 = HIP_vector_type<int, 3>;
using int4 = HIP_vector_type<int, 4>;
using int8 = HIP_vector_type<int, 8>;
using int16 = HIP_vector_type<int, 16>;
using uint1 = HIP_vector_type<unsigned int, 1>;
using uint2 = HIP_vector_type<unsigned int, 2>;
using uint3 = HIP_vector_type<unsigned int, 3>;
using uint4 = HIP_vector_type<unsigned int, 4>;
using uint8 = HIP_vector_type<unsigned int, 8>;
using uint16 = HIP_vector_type<unsigned int, 16>;
using long1 = HIP_vector_type<long, 1>;
using long2 = HIP_vector_type<long, 2>;
using long3 = HIP_vector_type<long, 3>;
using long4 = HIP_vector_type<long, 4>;
using long8 = HIP_vector_type<long, 8>;
using long16 = HIP_vector_type<long, 16>;
using ulong1 = HIP_vector_type<unsigned long, 1>;
using ulong2 = HIP_vector_type<unsigned long, 2>;
using ulong3 = HIP_vector_type<unsigned long, 3>;
using ulong4 = HIP_vector_type<unsigned long, 4>;
using ulong8 = HIP_vector_type<unsigned long, 8>;
using ulong16 = HIP_vector_type<unsigned long, 16>;
using longlong1 = HIP_vector_type<long long, 1>;
using longlong2 = HIP_vector_type<long long, 2>;
using longlong3 = HIP_vector_type<long long, 3>;
using longlong4 = HIP_vector_type<long long, 4>;
using longlong8 = HIP_vector_type<long long, 8>;
using longlong16 = HIP_vector_type<long long, 16>;
using ulonglong1 = HIP_vector_type<unsigned long long, 1>;
using ulonglong2 = HIP_vector_type<unsigned long long, 2>;
using ulonglong3 = HIP_vector_type<unsigned long long, 3>;
using ulonglong4 = HIP_vector_type<unsigned long long, 4>;
using ulonglong8 = HIP_vector_type<unsigned long long, 8>;
using ulonglong16 = HIP_vector_type<unsigned long long, 16>;
using float1 = HIP_vector_type<float, 1>;
using float2 = HIP_vector_type<float, 2>;
using float3 = HIP_vector_type<float, 3>;
using float4 = HIP_vector_type<float, 4>;
using float8 = HIP_vector_type<float, 8>;
using float16 = HIP_vector_type<float, 16>;
using double1 = HIP_vector_type<double, 1>;
using double2 = HIP_vector_type<double, 2>;
using double3 = HIP_vector_type<double, 3>;
using double4 = HIP_vector_type<double, 4>;
using double8 = HIP_vector_type<double, 8>;
using double16 = HIP_vector_type<double, 16>;

#else  // !defined(_MSC_VER)

/*
this is for compatibility with CUDA as CUDA allows accessing vector components
in C++ program with MSVC
structs are wrapped with templates so that mangled names match templated implementation
*/

template <typename T, unsigned int n> struct HIP_vector_type;

// One template per vector size
template <typename T> struct HIP_vector_type<T, 1> {
  union {
    struct {
      T x;
    };
    T data;
  };
};
template <typename T> struct HIP_vector_type<T, 2> {
  union {
    struct {
      T x;
      T y;
    };
    T data[2];
  };
};
template <typename T> struct HIP_vector_type<T, 3> {
  union {
    struct {
      T x;
      T y;
      T z;
    };
    T data[3];
  };
};
template <typename T> struct HIP_vector_type<T, 4> {
  union {
    struct {
      T x;
      T y;
      T z;
      T w;
    };
    T data[4];
  };
};
// 8- and 16-length vectors do not have CUDA-style accessible components
template <typename T> struct HIP_vector_type<T, 8> {
  union {
    T data[8];
  };
};
template <typename T> struct HIP_vector_type<T, 16> {
  union {
    T data[16];
  };
};

// Type aliasing
using char1 = HIP_vector_type<char, 1>;
using char2 = HIP_vector_type<char, 2>;
using char3 = HIP_vector_type<char, 3>;
using char4 = HIP_vector_type<char, 4>;
using char8 = HIP_vector_type<char, 8>;
using char16 = HIP_vector_type<char, 16>;
using uchar1 = HIP_vector_type<unsigned char, 1>;
using uchar2 = HIP_vector_type<unsigned char, 2>;
using uchar3 = HIP_vector_type<unsigned char, 3>;
using uchar4 = HIP_vector_type<unsigned char, 4>;
using uchar8 = HIP_vector_type<unsigned char, 8>;
using uchar16 = HIP_vector_type<unsigned char, 16>;
using short1 = HIP_vector_type<short, 1>;
using short2 = HIP_vector_type<short, 2>;
using short3 = HIP_vector_type<short, 3>;
using short4 = HIP_vector_type<short, 4>;
using short8 = HIP_vector_type<short, 8>;
using short16 = HIP_vector_type<short, 16>;
using ushort1 = HIP_vector_type<unsigned short, 1>;
using ushort2 = HIP_vector_type<unsigned short, 2>;
using ushort3 = HIP_vector_type<unsigned short, 3>;
using ushort4 = HIP_vector_type<unsigned short, 4>;
using ushort8 = HIP_vector_type<unsigned short, 8>;
using ushort16 = HIP_vector_type<unsigned short, 16>;
using int1 = HIP_vector_type<int, 1>;
using int2 = HIP_vector_type<int, 2>;
using int3 = HIP_vector_type<int, 3>;
using int4 = HIP_vector_type<int, 4>;
using int8 = HIP_vector_type<int, 8>;
using int16 = HIP_vector_type<int, 16>;
using uint1 = HIP_vector_type<unsigned int, 1>;
using uint2 = HIP_vector_type<unsigned int, 2>;
using uint3 = HIP_vector_type<unsigned int, 3>;
using uint4 = HIP_vector_type<unsigned int, 4>;
using uint8 = HIP_vector_type<unsigned int, 8>;
using uint16 = HIP_vector_type<unsigned int, 16>;
using long1 = HIP_vector_type<long, 1>;
using long2 = HIP_vector_type<long, 2>;
using long3 = HIP_vector_type<long, 3>;
using long4 = HIP_vector_type<long, 4>;
using long8 = HIP_vector_type<long, 8>;
using long16 = HIP_vector_type<long, 16>;
using ulong1 = HIP_vector_type<unsigned long, 1>;
using ulong2 = HIP_vector_type<unsigned long, 2>;
using ulong3 = HIP_vector_type<unsigned long, 3>;
using ulong4 = HIP_vector_type<unsigned long, 4>;
using ulong8 = HIP_vector_type<unsigned long, 8>;
using ulong16 = HIP_vector_type<unsigned long, 16>;
using longlong1 = HIP_vector_type<long long, 1>;
using longlong2 = HIP_vector_type<long long, 2>;
using longlong3 = HIP_vector_type<long long, 3>;
using longlong4 = HIP_vector_type<long long, 4>;
using longlong8 = HIP_vector_type<long long, 8>;
using longlong16 = HIP_vector_type<long long, 16>;
using ulonglong1 = HIP_vector_type<unsigned long long, 1>;
using ulonglong2 = HIP_vector_type<unsigned long long, 2>;
using ulonglong3 = HIP_vector_type<unsigned long long, 3>;
using ulonglong4 = HIP_vector_type<unsigned long long, 4>;
using ulonglong8 = HIP_vector_type<unsigned long long, 8>;
using ulonglong16 = HIP_vector_type<unsigned long long, 16>;
using float1 = HIP_vector_type<float, 1>;
using float2 = HIP_vector_type<float, 2>;
using float3 = HIP_vector_type<float, 3>;
using float4 = HIP_vector_type<float, 4>;
using float8 = HIP_vector_type<float, 8>;
using float16 = HIP_vector_type<float, 16>;
using double1 = HIP_vector_type<double, 1>;
using double2 = HIP_vector_type<double, 2>;
using double3 = HIP_vector_type<double, 3>;
using double4 = HIP_vector_type<double, 4>;
using double8 = HIP_vector_type<double, 8>;
using double16 = HIP_vector_type<double, 16>;

#endif  // defined(_MSC_VER)
#endif  // defined(__has_attribute)

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
