// Copyright Jim Bosch & Ankit Daftery 2010-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numpy.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/vector_c.hpp>

namespace p = boost::python;
namespace np = boost::numpy;

struct ArrayFiller 
{

  typedef boost::mpl::vector< short, int, float, std::complex<double> > TypeSequence;
  typedef boost::mpl::vector_c< int, 1, 2 > DimSequence;

  explicit ArrayFiller(np::ndarray const & arg) : argument(arg) {}

  template <typename T, int N>
  void apply() const 
  {
    if (N == 1) 
    {
      char * p = argument.get_data();
      int stride = argument.strides(0);
      int size = argument.shape(0);
      for (int n = 0; n != size; ++n, p += stride) 
	*reinterpret_cast<T*>(p) = static_cast<T>(n);
    }
    else
    {
      char * row_p = argument.get_data();
      int row_stride = argument.strides(0);
      int col_stride = argument.strides(1);
      int rows = argument.shape(0);
      int cols = argument.shape(1);
      int i = 0;
      for (int n = 0; n != rows; ++n, row_p += row_stride) 
      {
	char * col_p = row_p;
	for (int m = 0; m != cols; ++i, ++m, col_p += col_stride)
	  *reinterpret_cast<T*>(col_p) = static_cast<T>(i);
      }
    }
  }

  np::ndarray argument;
};

void fill(np::ndarray const & arg) 
{
  ArrayFiller filler(arg);
  np::invoke_matching_array<ArrayFiller::TypeSequence, ArrayFiller::DimSequence >(arg, filler);
}

BOOST_PYTHON_MODULE(templates_mod)
{
  np::initialize();
  p::def("fill", fill);
}
