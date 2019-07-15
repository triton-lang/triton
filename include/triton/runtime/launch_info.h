#ifndef TRITON_INCLUDE_RUNTIME_LAUNCH_INFO_H
#define TRITON_INCLUDE_RUNTIME_LAUNCH_INFO_H

#include <vector>
#include <map>

namespace triton{
namespace runtime{

struct launch_information{
  std::vector<unsigned> global_range_size;
  unsigned num_threads;
  std::map<std::string, unsigned> globals;
};

}
}

#endif
