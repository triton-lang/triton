// Copyright Jim Bosch 2010-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_NUMPY_INTERNAL
#include <boost/numpy/internal.hpp>
#include <boost/scoped_array.hpp>

namespace boost 
{ 
namespace python 
{
namespace converter 
{
NUMPY_OBJECT_MANAGER_TRAITS_IMPL(PyArray_Type, numpy::ndarray)
} // namespace boost::python::converter
} // namespace boost::python

namespace numpy 
{
namespace detail 
{

ndarray::bitflag numpy_to_bitflag(int const f) 
{
  ndarray::bitflag r = ndarray::NONE;
  if (f & NPY_C_CONTIGUOUS) r = (r | ndarray::C_CONTIGUOUS);
  if (f & NPY_F_CONTIGUOUS) r = (r | ndarray::F_CONTIGUOUS);
  if (f & NPY_ALIGNED) r = (r | ndarray::ALIGNED);
  if (f & NPY_WRITEABLE) r = (r | ndarray::WRITEABLE);
  return r;
}

int const bitflag_to_numpy(ndarray::bitflag f) 
{
  int r = 0;
  if (f & ndarray::C_CONTIGUOUS) r |= NPY_C_CONTIGUOUS;
  if (f & ndarray::F_CONTIGUOUS) r |= NPY_F_CONTIGUOUS;
  if (f & ndarray::ALIGNED) r |= NPY_ALIGNED;
  if (f & ndarray::WRITEABLE) r |= NPY_WRITEABLE;
  return r;
}

bool is_c_contiguous(std::vector<Py_intptr_t> const & shape,
		     std::vector<Py_intptr_t> const & strides,
		     int itemsize)
{
  std::vector<Py_intptr_t>::const_reverse_iterator j = strides.rbegin();
  int total = itemsize;
  for (std::vector<Py_intptr_t>::const_reverse_iterator i = shape.rbegin(); i != shape.rend(); ++i, ++j) 
  {
    if (total != *j) return false;
    total *= (*i);
  }
  return true;
}

bool is_f_contiguous(std::vector<Py_intptr_t> const & shape,
		     std::vector<Py_intptr_t> const & strides,
		     int itemsize)
{
  std::vector<Py_intptr_t>::const_iterator j = strides.begin();
  int total = itemsize;
  for (std::vector<Py_intptr_t>::const_iterator i = shape.begin(); i != shape.end(); ++i, ++j)
  {
    if (total != *j) return false;
    total *= (*i);
  }
  return true;
}

bool is_aligned(std::vector<Py_intptr_t> const & strides,
		int itemsize)
{
  for (std::vector<Py_intptr_t>::const_iterator i = strides.begin(); i != strides.end(); ++i) 
  {
    if (*i % itemsize) return false;
  }
  return true;
}

inline PyArray_Descr * incref_dtype(dtype const & dt) 
{
  Py_INCREF(dt.ptr());
  return reinterpret_cast<PyArray_Descr*>(dt.ptr());
}

ndarray from_data_impl(void * data, 
		       dtype const & dt, 
		       python::object const & shape,
		       python::object const & strides,
		       python::object const & owner,
		       bool writeable)
{
  std::vector<Py_intptr_t> shape_(len(shape));
  std::vector<Py_intptr_t> strides_(len(strides));
  if (shape_.size() != strides_.size()) 
  {
    PyErr_SetString(PyExc_ValueError, "Length of shape and strides arrays do not match.");
    python::throw_error_already_set();
  }
  for (std::size_t i = 0; i < shape_.size(); ++i) 
  {
    shape_[i] = python::extract<Py_intptr_t>(shape[i]);
    strides_[i] = python::extract<Py_intptr_t>(strides[i]);
  }
  return from_data_impl(data, dt, shape_, strides_, owner, writeable);
}

ndarray from_data_impl(void * data, 
		       dtype const & dt, 
		       std::vector<Py_intptr_t> const & shape,
		       std::vector<Py_intptr_t> const & strides,
		       python::object const & owner,
		       bool writeable)
{
  if (shape.size() != strides.size()) 
  {
    PyErr_SetString(PyExc_ValueError, "Length of shape and strides arrays do not match.");
    python::throw_error_already_set();
  }
  int itemsize = dt.get_itemsize();
  int flags = 0;
  if (writeable) flags |= NPY_WRITEABLE;
  if (is_c_contiguous(shape, strides, itemsize)) flags |= NPY_C_CONTIGUOUS;
  if (is_f_contiguous(shape, strides, itemsize)) flags |= NPY_F_CONTIGUOUS;
  if (is_aligned(strides, itemsize)) flags |= NPY_ALIGNED;
  ndarray r(python::detail::new_reference
    (PyArray_NewFromDescr(&PyArray_Type,
			  incref_dtype(dt),
			  shape.size(),
			  const_cast<Py_intptr_t*>(&shape.front()),
			  const_cast<Py_intptr_t*>(&strides.front()),
			  data,
			  flags,
			  NULL)));
    r.set_base(owner);
    return r;
}

} // namespace detail

ndarray ndarray::view(dtype const & dt) const
{
  return ndarray(python::detail::new_reference
    (PyObject_CallMethod(this->ptr(), const_cast<char*>("view"), const_cast<char*>("O"), dt.ptr())));
}
    
ndarray ndarray::astype(dtype const & dt) const
{
  return ndarray(python::detail::new_reference
    (PyObject_CallMethod(this->ptr(), const_cast<char*>("astype"), const_cast<char*>("O"), dt.ptr())));
}

ndarray ndarray::copy() const 
{
  return ndarray(python::detail::new_reference
    (PyObject_CallMethod(this->ptr(), const_cast<char*>("copy"), const_cast<char*>(""))));
}

dtype ndarray::get_dtype() const 
{
  return dtype(python::detail::borrowed_reference(get_struct()->descr));
}

python::object ndarray::get_base() const 
{
  if (get_struct()->base == NULL) return object();
  return python::object(python::detail::borrowed_reference(get_struct()->base));
}

void ndarray::set_base(object const & base) 
{
  Py_XDECREF(get_struct()->base);
  if (base != object()) 
  {
    Py_INCREF(base.ptr());
    get_struct()->base = base.ptr();
  }
  else
  {
    get_struct()->base = NULL;
  }
}

ndarray::bitflag const ndarray::get_flags() const 
{
  return numpy::detail::numpy_to_bitflag(get_struct()->flags);
}

ndarray ndarray::transpose() const 
{
  return ndarray(python::detail::new_reference
    (PyArray_Transpose(reinterpret_cast<PyArrayObject*>(this->ptr()), NULL)));
}

ndarray ndarray::squeeze() const 
{
  return ndarray(python::detail::new_reference
    (PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(this->ptr()))));
}

ndarray ndarray::reshape(python::tuple const & shape) const 
{
  return ndarray(python::detail::new_reference
    (PyArray_Reshape(reinterpret_cast<PyArrayObject*>(this->ptr()), shape.ptr())));
}

python::object ndarray::scalarize() const 
{
  Py_INCREF(ptr());
  return python::object(python::detail::new_reference(PyArray_Return(reinterpret_cast<PyArrayObject*>(ptr()))));
}

ndarray zeros(python::tuple const & shape, dtype const & dt) 
{
  int nd = len(shape);
  boost::scoped_array<Py_intptr_t> dims(new Py_intptr_t[nd]);
  for (int n=0; n<nd; ++n) dims[n] = python::extract<Py_intptr_t>(shape[n]);
  return ndarray(python::detail::new_reference
                 (PyArray_Zeros(nd, dims.get(), detail::incref_dtype(dt), 0)));
}

ndarray zeros(int nd, Py_intptr_t const * shape, dtype const & dt) 
{
  return ndarray(python::detail::new_reference
    (PyArray_Zeros(nd, const_cast<Py_intptr_t*>(shape), detail::incref_dtype(dt), 0)));
}

ndarray empty(python::tuple const & shape, dtype const & dt) 
{
  int nd = len(shape);
  boost::scoped_array<Py_intptr_t> dims(new Py_intptr_t[nd]);
  for (int n=0; n<nd; ++n) dims[n] = python::extract<Py_intptr_t>(shape[n]);
  return ndarray(python::detail::new_reference
                 (PyArray_Empty(nd, dims.get(), detail::incref_dtype(dt), 0)));    
}

ndarray empty(int nd, Py_intptr_t const * shape, dtype const & dt)
{
  return ndarray(python::detail::new_reference
    (PyArray_Empty(nd, const_cast<Py_intptr_t*>(shape), detail::incref_dtype(dt), 0)));
}

ndarray array(python::object const & obj) 
{
  return ndarray(python::detail::new_reference
    (PyArray_FromAny(obj.ptr(), NULL, 0, 0, NPY_ENSUREARRAY, NULL)));
}

ndarray array(python::object const & obj, dtype const & dt) 
{
  return ndarray(python::detail::new_reference
    (PyArray_FromAny(obj.ptr(), detail::incref_dtype(dt), 0, 0, NPY_ENSUREARRAY, NULL)));
}

ndarray from_object(python::object const & obj, dtype const & dt, int nd_min, int nd_max, ndarray::bitflag flags)
{
  int requirements = detail::bitflag_to_numpy(flags);
  return ndarray(python::detail::new_reference
    (PyArray_FromAny(obj.ptr(),
		     detail::incref_dtype(dt),
		     nd_min, nd_max,
		     requirements,
		     NULL)));
}

ndarray from_object(python::object const & obj, int nd_min, int nd_max, ndarray::bitflag flags) 
{
  int requirements = detail::bitflag_to_numpy(flags);
  return ndarray(python::detail::new_reference
    (PyArray_FromAny(obj.ptr(),
		     NULL,
		     nd_min, nd_max,
		     requirements,
		     NULL)));
}

}
}
