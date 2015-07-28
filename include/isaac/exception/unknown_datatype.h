#ifndef ISAAC_EXCEPTION_UNKNOWN_DATATYPE_H
#define ISAAC_EXCEPTION_UNKNOWN_DATATYPE_H

#include <string>
#include <exception>
#include "isaac/defines.h"

namespace isaac
{

/** @brief Exception for the case the generator is unable to deal with the operation */
class ISAACAPI unknown_datatype : public std::exception
{
public:
  unknown_datatype(int);
  virtual const char* what() const throw();
  virtual ~unknown_datatype() throw();
private:
  std::string message_;
};

}

#endif
