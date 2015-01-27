#include "atidlas/array.h"
#include "atidlas/tools/timer.hpp"

#include <vector>

namespace ad = atidlas;

int main()
{
  for(ad::cl_ext::queues_t::iterator it = ad::cl_ext::queues.begin() ; it != ad::cl_ext::queues.end() ; ++it)
  {
    ad::array x(10, ad::FLOAT_TYPE, it->first);
    cl::Device device = it->second[0].getInfo<CL_QUEUE_DEVICE>();
    ad::tools::timer t;
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "-------------------------" << std::endl;
    x = x + x;
    ad::cl_ext::synchronize(x.context());
    t.start();\
    for(unsigned int i = 0 ; i < 100 ; ++i){
      x = x + x;
      ad::cl_ext::synchronize(x.context());
    }
    std::cout << "Kernel launch overhead: " << t.get()/100 << std::endl;
    std::cout << "Expression tree creation:" << std::endl;
  #define BENCH(CREATE, STR) \
    {\
    ad::array_expression tmp1(CREATE);\
    t.start();\
    for(unsigned int i = 0 ; i < 1000 ; ++i)\
      ad::array_expression tmp2(CREATE);\
    std::cout << STR << ": " << t.get()/1000 << std::endl;\
    }

    BENCH(x + x, "2 terms");
    BENCH(x + x + x, "3 terms");
    BENCH(x + x + x + x, "4 terms");
    BENCH(x + x + x + x + x, "5 terms");
  #undef BENCH
    std::cout << "-------------------------" << std::endl;
  }

}
