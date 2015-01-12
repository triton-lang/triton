// Copyright Jim Bosch & Ankit Daftery 2010-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numpy.hpp>

namespace p = boost::python;
namespace np = boost::numpy;

struct UnaryCallable 
{
  typedef double argument_type;
  typedef double result_type;

  double operator()(double r) const { return r * 2;}
};

struct BinaryCallable 
{
  typedef double first_argument_type;
  typedef double second_argument_type;
  typedef double result_type;

  double operator()(double a, double b) const { return a * 2 + b * 3;}
};

BOOST_PYTHON_MODULE(ufunc_mod) 
{
  np::initialize();
  p::class_<UnaryCallable>("UnaryCallable")
    .def("__call__", np::unary_ufunc<UnaryCallable>::make());
  p::class_< BinaryCallable>("BinaryCallable")
    .def("__call__", np::binary_ufunc<BinaryCallable>::make());
}
