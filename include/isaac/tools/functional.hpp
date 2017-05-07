/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#ifndef ISAAC_CPP_FUNCTIONAL_HPP
#define ISAAC_CPP_FUNCTIONAL_HPP

#include <type_traits>
#include <tuple>

namespace isaac
{
namespace cpp
{

template <typename T>
struct function_traits
    : public function_traits<decltype(&T::operator())>
{};
// For generic types, directly use the result of the signature of its 'operator()'

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const>
// we specialize for pointers to member function
{
  enum { arity = sizeof...(Args) };
  // arity is the number of arguments.

  typedef ReturnType result_type;

  template <size_t i>
  struct arg
  {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
    // the i-th argument is equivalent to the i-th tuple element of a tuple
    // composed of those arguments.
  };
};


template<class U, class FN, class V>
V forward_dyncast(U const & x, FN const & fn, V const &backup)
{
  typedef typename function_traits<FN>::template arg<0>::type RT;
  typedef typename std::remove_reference<RT>::type T;
  if(T const * p = dynamic_cast<T const *>(&x))
    return fn(*p);
  return backup;
}

template<class U, class FN>
void forward_dyncast(U const & x, FN const & fn)
{
  typedef typename function_traits<FN>::template arg<0>::type RT;
  typedef typename std::remove_reference<RT>::type T;
  if(T const * p = dynamic_cast<T const *>(&x))
    fn(*p);
}

template<class U, class FN>
bool compare_if_same(U const & base, FN const & f)
{ return cpp::forward_dyncast(base, f, false); }

}
}

#endif
