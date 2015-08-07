
//  Copyright (c) 2014 Agustin Berge
//
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).
//
//  See http://www.boost.org/libs/type_traits for most recent version including documentation.


#ifndef BOOST_TT_IS_FINAL_HPP_INCLUDED
#define BOOST_TT_IS_FINAL_HPP_INCLUDED

#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/config.hpp>
#include <boost/type_traits/intrinsics.hpp>

// should be the last #include
#include <boost/type_traits/detail/bool_trait_def.hpp>

namespace boost {

namespace detail {
template <typename T> struct is_final_impl
{
#ifdef BOOST_IS_FINAL
   typedef typename remove_cv<T>::type cvt;
   BOOST_STATIC_CONSTANT(bool, value = BOOST_IS_FINAL(cvt));
#else
   BOOST_STATIC_CONSTANT(bool, value = false);
#endif
};
} // namespace detail

BOOST_TT_AUX_BOOL_TRAIT_DEF1(is_final,T,::boost::detail::is_final_impl<T>::value)

} // namespace boost

#include <boost/type_traits/detail/bool_trait_undef.hpp>

#endif // BOOST_TT_IS_FINAL_HPP_INCLUDED
