#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include "driver.h"
#include "core.h"
#include "model.h"

namespace bp = boost::python;
namespace np = boost::numpy;

BOOST_PYTHON_MODULE(_isaac)
{
  Py_Initialize();
  boost::numpy::initialize();

  // specify that this module is actually a package
  bp::object package = bp::scope();
  package.attr("__path__") = "_isaac";

  export_core();
  export_driver();
  export_model();
}
