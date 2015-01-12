// Copyright Jim Bosch & Ankit Daftery 2010-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numpy.hpp>
#include <boost/cstdint.hpp>

namespace p = boost::python;
namespace np = boost::numpy;

template <typename T>
np::dtype accept(T) {
  return np::dtype::get_builtin<T>();
}

BOOST_PYTHON_MODULE(dtype_mod) 
{
  np::initialize();
  // wrap dtype equivalence test, since it isn't available in Python API.
  p::def("equivalent", np::equivalent);
  // integers, by number of bits
  p::def("accept_int8", accept<boost::int8_t>);
  p::def("accept_uint8", accept<boost::uint8_t>);
  p::def("accept_int16", accept<boost::int16_t>);
  p::def("accept_uint16", accept<boost::uint16_t>);
  p::def("accept_int32", accept<boost::int32_t>);
  p::def("accept_uint32", accept<boost::uint32_t>);
  p::def("accept_int64", accept<boost::int64_t>);
  p::def("accept_uint64", accept<boost::uint64_t>);
  // integers, by C name according to NumPy
  p::def("accept_bool_", accept<bool>);
  p::def("accept_byte", accept<signed char>);
  p::def("accept_ubyte", accept<unsigned char>);
  p::def("accept_short", accept<short>);
  p::def("accept_ushort", accept<unsigned short>);
  p::def("accept_intc", accept<int>);
  p::def("accept_uintc", accept<unsigned int>);
  // floats and complex
  p::def("accept_float32", accept<float>);
  p::def("accept_complex64", accept< std::complex<float> >);
  p::def("accept_float64", accept<double>);
  p::def("accept_complex128", accept< std::complex<double> >);
  if (sizeof(long double) > sizeof(double)) {
      p::def("accept_longdouble", accept<long double>);
      p::def("accept_clongdouble", accept< std::complex<long double> >);
  }
}
