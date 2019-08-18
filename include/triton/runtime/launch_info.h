#ifndef TRITON_INCLUDE_RUNTIME_LAUNCH_INFO_H
#define TRITON_INCLUDE_RUNTIME_LAUNCH_INFO_H

#include <map>

namespace triton{
namespace runtime{

struct launch_information{
  unsigned num_threads;
  std::map<std::string, unsigned> globals;
};

}
}

#endif
