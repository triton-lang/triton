/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef ISAAC_EXCEPTION_API_H
#define ISAAC_EXCEPTION_API_H

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
private:
DISABLE_MSVC_WARNING_C4251
  std::string message_;
RESTORE_MSVC_WARNING_C4251
};
RESTORE_MSVC_WARNING_C4275


/** @brief Exception for the case the generator is unable to deal with the operation */
DISABLE_MSVC_WARNING_C4275
class ISAACAPI unknown_datatype : public std::exception
{
public:
  unknown_datatype(int);
  virtual const char* what() const throw();
private:
DISABLE_MSVC_WARNING_C4251
  std::string message_;
RESTORE_MSVC_WARNING_C4251
};
RESTORE_MSVC_WARNING_C4275


/** @brief Exception for the case the generator is unable to deal with the operation */
DISABLE_MSVC_WARNING_C4275
class ISAACAPI semantic_error : public std::exception
{
public:
  semantic_error(std::string const & message);
  virtual const char* what() const throw();
private:
DISABLE_MSVC_WARNING_C4251
  std::string message_;
RESTORE_MSVC_WARNING_C4251
};
RESTORE_MSVC_WARNING_C4275


}

#endif
