#include "atidlas/exception/unknown_datatype.h"
#include "atidlas/tools/to_string.hpp"

namespace atidlas
{

unknown_datatype::unknown_datatype(int v) :
  message_("ATIDLAS: The data-type provided was not recognized. The datatype code provided is " + tools::to_string(v)) {}

const char* unknown_datatype::what() const throw()
{ return message_.c_str(); }

unknown_datatype::~unknown_datatype() throw()
{}

}
