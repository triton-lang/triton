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

#include "isaac/exception/api.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{

//
operation_not_supported_exception::operation_not_supported_exception() : message_()
{}

operation_not_supported_exception::operation_not_supported_exception(std::string message) :
  message_("ISAAC: Internal error: The internal generator cannot handle the operation provided: " + message) {}

const char* operation_not_supported_exception::what() const throw()
{ return message_.c_str(); }

//
unknown_datatype::unknown_datatype(int v) :
  message_("ISAAC: The data-type provided was not recognized. The datatype code provided is " + tools::to_string(v)) {}

const char* unknown_datatype::what() const throw()
{ return message_.c_str(); }

//
semantic_error::semantic_error(std::string const & str) :
  message_("Semantic error: " + str) {}

const char* semantic_error::what() const throw()
{ return message_.c_str(); }



}
