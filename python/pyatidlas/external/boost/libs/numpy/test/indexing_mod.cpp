// Copyright Jim Bosch & Ankit Daftery 2010-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numpy.hpp>
#include <boost/python/slice.hpp>

namespace p = boost::python;
namespace np = boost::numpy;

p::object single(np::ndarray ndarr, int i) { return ndarr[i];}
p::object slice(np::ndarray ndarr, p::slice sl) { return ndarr[sl];}
p::object indexarray(np::ndarray ndarr, np::ndarray d1) { return ndarr[d1];}
p::object indexarray_2d(np::ndarray ndarr, np::ndarray d1,np::ndarray d2) { return ndarr[p::make_tuple(d1,d2)];}
p::object indexslice(np::ndarray ndarr, np::ndarray d1, p::slice sl) { return ndarr[p::make_tuple(d1, sl)];}

BOOST_PYTHON_MODULE(indexing_mod) 
{
  np::initialize();
  p::def("single", single);
  p::def("slice", slice);
  p::def("indexarray", indexarray);
  p::def("indexarray", indexarray_2d);
  p::def("indexslice", indexslice);

}
