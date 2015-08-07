// Copyright Jim Bosch 2010-2012.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_NUMPY_INTERNAL_MAIN
#include <boost/numpy/internal.hpp>
#include <boost/numpy/dtype.hpp>

namespace boost 
{
namespace numpy 
{

#if PY_MAJOR_VERSION == 2
static void wrap_import_array() {
    import_array();
}
#else
static void * wrap_import_array() {
    import_array();
}
#endif

void initialize(bool register_scalar_converters) 
{
  wrap_import_array();
  import_ufunc();
  if (register_scalar_converters)
	dtype::register_scalar_converters();
}

}
}
