#ifndef ISAAC_EXCEPTION_UNKNOWN_DATATYPE_H
#define ISAAC_EXCEPTION_UNKNOWN_DATATYPE_H

#include <string>
#include <exception>
#include "isaac/defines.h"

namespace isaac
{

/** @brief Exception for the case the generator is unable to deal with the operation */
DISABLE_MSVC_WARNING_C4275
class ISAACAPI unknown_datatype : public std::exception
{
public:
  unknown_datatype(int);
  virtual const char* what() const throw();
  virtual ~unknown_datatype() throw();
private:
DISABLE_MSVC_WARNING_C4251
  std::string message_;
RESTORE_MSVC_WARNING_C4251
};
RESTORE_MSVC_WARNING_C4275

}

#endif
