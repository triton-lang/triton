// Copyright Jim Bosch 2010-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_NUMPY_NUMPY_OBJECT_MGR_TRAITS_HPP_INCLUDED
#define BOOST_NUMPY_NUMPY_OBJECT_MGR_TRAITS_HPP_INCLUDED

/**
 *  @file boost/numpy/numpy_object_mgr_traits.hpp
 *  @brief Macro that specializes object_manager_traits by requiring a 
 *         source-file implementation of get_pytype().
 */

#define NUMPY_OBJECT_MANAGER_TRAITS(manager)                            \
template <>								\
struct object_manager_traits<manager>					\
{									\
  BOOST_STATIC_CONSTANT(bool, is_specialized = true);			\
  static inline python::detail::new_reference adopt(PyObject* x)	\
  {									\
    return python::detail::new_reference(python::pytype_check((PyTypeObject*)get_pytype(), x)); \
  }									\
  static bool check(PyObject* x)					\
  {									\
    return ::PyObject_IsInstance(x, (PyObject*)get_pytype());		\
  }									\
  static manager* checked_downcast(PyObject* x)				\
  {									\
    return python::downcast<manager>((checked_downcast_impl)(x, (PyTypeObject*)get_pytype())); \
  }									\
  static PyTypeObject const * get_pytype();				\
}

#endif // !BOOST_NUMPY_NUMPY_OBJECT_MGR_TRAITS_HPP_INCLUDED

