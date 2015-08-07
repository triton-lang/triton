#include "isaac/exception/unknown_datatype.h"

#include <string>

namespace isaac
{

unknown_datatype::unknown_datatype(int v) :
  message_("ISAAC: The data-type provided was not recognized. The datatype code provided is " + std::to_string(v)) {}

const char* unknown_datatype::what() const throw()
{ return message_.c_str(); }

unknown_datatype::~unknown_datatype() throw()
{}

}
