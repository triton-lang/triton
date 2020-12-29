#pragma once

#ifndef _TRITON_RUNTIME_ERROR_H_
#define _TRITON_RUNTIME_ERROR_H_

#include <exception>
#include <string>

namespace triton {
namespace runtime{
namespace exception {

class base: public std::exception {};
#define TRITON_CREATE_RUNTIME_EXCEPTION(name, msg) class name: public base { public: const char * what() const throw(){ return "Triton: Error - Runtime: " msg; } };

TRITON_CREATE_RUNTIME_EXCEPTION(out_of_shared_memory, "out of shared memory")
TRITON_CREATE_RUNTIME_EXCEPTION(out_of_registers, "out of registers")

class no_valid_configuration: public exception::base {
public:
  no_valid_configuration(const std::string& err): err_(err) { }
  const char * what() const throw(){ return err_.c_str(); }
private:
  std::string err_;
};


#undef TRITON_CREATE_RUNTIME_EXCEPTION

}
}
}

#endif
