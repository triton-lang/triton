// Copyright Jim Bosch 2010-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_NUMPY_UFUNC_HPP_INCLUDED
#define BOOST_NUMPY_UFUNC_HPP_INCLUDED

/**
 *  @file boost/numpy/ufunc.hpp
 *  @brief Utilities to create ufunc-like broadcasting functions out of C++ functors.
 */

#include <boost/python.hpp>
#include <boost/numpy/numpy_object_mgr_traits.hpp>
#include <boost/numpy/dtype.hpp>
#include <boost/numpy/ndarray.hpp>

namespace boost 
{
namespace numpy 
{

/**
 *  @brief A boost.python "object manager" (subclass of object) for PyArray_MultiIter.
 *
 *  multi_iter is a Python object, but a very low-level one.  It should generally only be used
 *  in loops of the form:
 *  @code
 *  while (iter.not_done()) {
 *      ...
 *      iter.next();
 *  }
 *  @endcode
 *
 *  @todo I can't tell if this type is exposed in Python anywhere; if it is, we should use that name.
 *        It's more dangerous than most object managers, however - maybe it actually belongs in
 *        a detail namespace?
 */
class multi_iter : public python::object 
{
public:

  BOOST_PYTHON_FORWARD_OBJECT_CONSTRUCTORS(multi_iter, python::object);

  /// @brief Increment the iterator.
  void next();

  /// @brief Check if the iterator is at its end.
  bool not_done() const;

  /// @brief Return a pointer to the element of the nth broadcasted array.
  char * get_data(int n) const;

  /// @brief Return the number of dimensions of the broadcasted array expression.
  int const get_nd() const;
    
  /// @brief Return the shape of the broadcasted array expression as an array of integers.
  Py_intptr_t const * get_shape() const;

  /// @brief Return the shape of the broadcasted array expression in the nth dimension.
  Py_intptr_t const shape(int n) const;
    
};

/// @brief Construct a multi_iter over a single sequence or scalar object.
multi_iter make_multi_iter(python::object const & a1);

/// @brief Construct a multi_iter by broadcasting two objects.
multi_iter make_multi_iter(python::object const & a1, python::object const & a2);

/// @brief Construct a multi_iter by broadcasting three objects.
multi_iter make_multi_iter(python::object const & a1, python::object const & a2, python::object const & a3);

/**
 *  @brief Helps wrap a C++ functor taking a single scalar argument as a broadcasting ufunc-like
 *         Python object.
 *
 *  Typical usage looks like this:
 *  @code
 *  struct TimesPI 
 *  {
 *    typedef double argument_type;
 *    typedef double result_type;
 *    double operator()(double input) const { return input * M_PI; }
 *  };
 *  
 *  BOOST_PYTHON_MODULE(example)
 *  {
 *    class_< TimesPI >("TimesPI")
 *      .def("__call__", unary_ufunc<TimesPI>::make());
 *  }
 *  @endcode
 *  
 */
template <typename TUnaryFunctor, 
          typename TArgument=typename TUnaryFunctor::argument_type,
          typename TResult=typename TUnaryFunctor::result_type>
struct unary_ufunc 
{

  /**
   *  @brief A C++ function with object arguments that broadcasts its arguments before
   *         passing them to the underlying C++ functor.
   */
  static python::object call(TUnaryFunctor & self, python::object const & input, python::object const & output) 
  {
    dtype in_dtype = dtype::get_builtin<TArgument>();
    dtype out_dtype = dtype::get_builtin<TResult>();
    ndarray in_array = from_object(input, in_dtype, ndarray::ALIGNED);
    ndarray out_array = (output != python::object()) ? 
      from_object(output, out_dtype, ndarray::ALIGNED | ndarray::WRITEABLE)
      : zeros(in_array.get_nd(), in_array.get_shape(), out_dtype);
    multi_iter iter = make_multi_iter(in_array, out_array);
    while (iter.not_done()) 
    {
      TArgument * argument = reinterpret_cast<TArgument*>(iter.get_data(0));
      TResult * result = reinterpret_cast<TResult*>(iter.get_data(1));
      *result = self(*argument);
      iter.next();
    } 
    return out_array.scalarize();
  }

  /**
   *  @brief Construct a boost.python function object from call() with reasonable keyword names.
   *
   *  Users will often want to specify their own keyword names with the same signature, but this
   *  is a convenient shortcut.
   */
  static python::object make() 
  {
    namespace p = python;
    return p::make_function(call, p::default_call_policies(), (p::arg("input"), p::arg("output")=p::object())); 
  }
};

/**
 *  @brief Helps wrap a C++ functor taking a pair of scalar arguments as a broadcasting ufunc-like
 *         Python object.
 *
 *  Typical usage looks like this:
 *  @code
 *  struct CosSum 
 *  {
 *    typedef double first_argument_type;
 *    typedef double second_argument_type;
 *    typedef double result_type;
 *    double operator()(double input1, double input2) const { return std::cos(input1 + input2); }
 *  };
 *  
 *  BOOST_PYTHON_MODULE(example) 
 *  {
 *    class_< CosSum >("CosSum")
 *      .def("__call__", binary_ufunc<CosSum>::make());
 *  }
 *  @endcode
 *  
 */
template <typename TBinaryFunctor, 
          typename TArgument1=typename TBinaryFunctor::first_argument_type,
          typename TArgument2=typename TBinaryFunctor::second_argument_type,
          typename TResult=typename TBinaryFunctor::result_type>
struct binary_ufunc 
{

  static python::object 
  call(TBinaryFunctor & self, python::object const & input1, python::object const & input2, 
       python::object const & output) 
  {
    dtype in1_dtype = dtype::get_builtin<TArgument1>();
    dtype in2_dtype = dtype::get_builtin<TArgument2>();
    dtype out_dtype = dtype::get_builtin<TResult>();
    ndarray in1_array = from_object(input1, in1_dtype, ndarray::ALIGNED);
    ndarray in2_array = from_object(input2, in2_dtype, ndarray::ALIGNED);
    multi_iter iter = make_multi_iter(in1_array, in2_array);
    ndarray out_array = (output != python::object())
      ? from_object(output, out_dtype, ndarray::ALIGNED | ndarray::WRITEABLE)
      : zeros(iter.get_nd(), iter.get_shape(), out_dtype);
    iter = make_multi_iter(in1_array, in2_array, out_array);
    while (iter.not_done()) 
    {
      TArgument1 * argument1 = reinterpret_cast<TArgument1*>(iter.get_data(0));
      TArgument2 * argument2 = reinterpret_cast<TArgument2*>(iter.get_data(1));
      TResult * result = reinterpret_cast<TResult*>(iter.get_data(2));
      *result = self(*argument1, *argument2);
      iter.next();
    } 
    return out_array.scalarize();
  }

  static python::object make() 
  {
    namespace p = python;
    return p::make_function(call, p::default_call_policies(), 
			    (p::arg("input1"), p::arg("input2"), p::arg("output")=p::object())); 
  }

};

} // namespace boost::numpy

namespace python
{
namespace converter 
{

NUMPY_OBJECT_MANAGER_TRAITS(numpy::multi_iter);

} // namespace boost::python::converter
} // namespace boost::python
} // namespace boost

#endif // !BOOST_NUMPY_UFUNC_HPP_INCLUDED
