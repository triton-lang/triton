#ifndef ISAAC_EXCEPTION_OPERATION_NOT_SUPPORTED_H
#define ISAAC_EXCEPTION_OPERATION_NOT_SUPPORTED_H

#include <string>
#include <exception>

namespace isaac
{

/** @brief Exception for the case the generator is unable to deal with the operation */
class operation_not_supported_exception : public std::exception
{
public:
  operation_not_supported_exception();
  operation_not_supported_exception(std::string message);
  virtual const char* what() const throw();
  virtual ~operation_not_supported_exception() throw();
private:
  std::string message_;
};

}

#endif
