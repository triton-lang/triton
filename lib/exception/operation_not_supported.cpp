#include "atidlas/exception/operation_not_supported.h"

namespace atidlas
{

operation_not_supported_exception::operation_not_supported_exception() : message_()
{}

operation_not_supported_exception::operation_not_supported_exception(std::string message) :
  message_("ATIDLAS: Internal error: The internal generator cannot handle the operation provided: " + message) {}

const char* operation_not_supported_exception::what() const throw()
{ return message_.c_str(); }

operation_not_supported_exception::~operation_not_supported_exception() throw()
{}

}
