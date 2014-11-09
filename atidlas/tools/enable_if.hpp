#ifndef VIENNACL_META_ENABLE_IF_HPP_
#define VIENNACL_META_ENABLE_IF_HPP_

/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/meta/enable_if.hpp
    @brief Simple enable-if variant that uses the SFINAE pattern
*/

namespace viennacl
{

/** @brief Simple enable-if variant that uses the SFINAE pattern */
template<bool b, class T = void>
struct enable_if
{
  typedef T   type;
};

/** \cond */
template<class T>
struct enable_if<false, T> {};
/** \endcond */

} //namespace viennacl


#endif
