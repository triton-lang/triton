// Copyright Jim Bosch 2010-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_NUMPY_INTERNAL
#include <boost/numpy/internal.hpp>
#include <boost/numpy/matrix.hpp>

namespace boost 
{
namespace numpy 
{ 
namespace detail 
{
inline python::object get_matrix_type() 
{
  python::object module = python::import("numpy");
  return module.attr("matrix");
}
} // namespace boost::numpy::detail
} // namespace boost::numpy

namespace python
{
namespace converter 
{

PyTypeObject const * object_manager_traits<numpy::matrix>::get_pytype()
{
  return reinterpret_cast<PyTypeObject*>(numpy::detail::get_matrix_type().ptr());
}

} // namespace boost::python::converter
} // namespace boost::python

namespace numpy 
{

python::object matrix::construct(python::object const & obj, dtype const & dt, bool copy) 
{
  return numpy::detail::get_matrix_type()(obj, dt, copy);
}

python::object matrix::construct(python::object const & obj, bool copy) 
{
  return numpy::detail::get_matrix_type()(obj, object(), copy);
}

matrix matrix::view(dtype const & dt) const 
{
  return matrix(python::detail::new_reference
    (PyObject_CallMethod(this->ptr(), const_cast<char*>("view"), const_cast<char*>("O"), dt.ptr())));
}

matrix matrix::copy() const 
{
  return matrix(python::detail::new_reference
    (PyObject_CallMethod(this->ptr(), const_cast<char*>("copy"), const_cast<char*>(""))));
}

matrix matrix::transpose() const 
{
  return matrix(python::extract<matrix>(ndarray::transpose()));
}

} // namespace boost::numpy
} // namespace boost
