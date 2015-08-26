#ifndef ISAAC_EXCEPTION_OPERATION_NOT_SUPPORTED_H
#define ISAAC_EXCEPTION_OPERATION_NOT_SUPPORTED_H

#include <string>
#include <exception>

#include "isaac/defines.h"

namespace isaac
{

/** @brief Exception for the case the generator is unable to deal with the operation */
DISABLE_MSVC_WARNING_C4275
class operation_not_supported_exception : public std::exception
{
public:
  operation_not_supported_exception();
  operation_not_supported_exception(std::string message);
  virtual const char* what() const throw();
  virtual ~operation_not_supported_exception() throw();
private:
DISABLE_MSVC_WARNING_C4251
  std::string message_;
RESTORE_MSVC_WARNING_C4251
};
RESTORE_MSVC_WARNING_C4275

}

#endif
