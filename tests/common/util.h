#pragma once

#ifndef _TRITON_TESTS_UTIL_H
#define _TRITON_TESTS_UTIL_H

#include <iomanip>
#include "triton/runtime/function.h"

namespace drv = triton::driver;
namespace rt = triton::runtime;

inline size_t ceil(size_t x, size_t y) {
  return (x + y - 1) / y;
};

inline rt::function::grid_fn_ty grid1d(size_t N) {
  return [N](const rt::function::options_t& x) {
    return rt::grid_t{ceil(N, x.D<int>("TN"))};
  };
}

inline rt::function::grid_fn_ty grid2d(size_t M, size_t N) {
  return [M, N](const rt::function::options_t& x) {
    return rt::grid_t{ceil(M, x.D<int>("TM")),
                      ceil(N, x.D<int>("TN"))};
  };
}

enum order_t {
  ROWMAJOR,
  COLMAJOR
};

template<class T>
void init_rand(std::vector<T>& x) {
  for(size_t i = 0; i < x.size(); i++)
    x[i] = static_cast<T>((double)rand()/RAND_MAX);
}



namespace aux{
template<std::size_t...> struct seq{};

template<std::size_t N, std::size_t... Is>
struct gen_seq : gen_seq<N-1, N-1, Is...>{};

template<std::size_t... Is>
struct gen_seq<0, Is...> : seq<Is...>{};

template<class Ch, class Tr, class Tuple, std::size_t... Is>
void print_tuple(std::basic_ostream<Ch,Tr>& os, Tuple const& t, seq<Is...>){
  using swallow = int[];
  (void)swallow{0, (void(os << (Is == 0? "" : ", ") << std::setfill(' ') << std::setw(3) << std::get<Is>(t)), 0)...};
}
} // aux::

template<class Ch, class Tr, class... Args>
auto operator<<(std::basic_ostream<Ch, Tr>& os, std::tuple<Args...> const& t)
    -> std::basic_ostream<Ch, Tr>&
{
  os << "(";
  aux::print_tuple(os, t, aux::gen_seq<sizeof...(Args)>());
  return os << ")";
}


namespace testing {

    template<class T>
    bool diff(const std::vector<T>& hc, const std::vector<T>& rc) {
    if(hc.size() != rc.size())
        return false;
    for(size_t i = 0; i < hc.size(); i++)
      if(std::isinf(hc[i]) || std::isnan(hc[i]) || std::abs(hc[i] - rc[i])/std::max(hc[i], rc[i]) > 1e-2){
        std::cout << i << " " << hc[i] << " " << rc[i] << std::endl;

        return false;
      }
    return true;
    }

}

#endif
