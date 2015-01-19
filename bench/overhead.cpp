#include "atidlas/array.h"
#include "atidlas/tools/timer.hpp"

#include <vector>

namespace ad = atidlas;

int main()
{
  ad::array x(10, ad::FLOAT_TYPE), y(10, ad::FLOAT_TYPE), z(10, ad::FLOAT_TYPE);
  ad::tools::timer t;
//  std::cout << "-------------------------" << std::endl;
//  std::cout << "Expression tree creation:" << std::endl;
//#define BENCH(CREATE, STR) \
//  {\
//  ad::array_expression tmp1(CREATE);\
//  t.start();\
//  for(unsigned int i = 0 ; i < 1000 ; ++i)\
//    ad::array_expression tmp2(CREATE);\
//  std::cout << STR << ": " << t.get()/1000 << std::endl;\
//  }

//  BENCH(x + y, "2 terms");
//  BENCH(x + y + x, "3 terms");
//  BENCH(x + y + x + y, "4 terms");
//  BENCH(x + y + x + y + x, "5 terms");
//#undef BENCH
  std::cout << "-------------------------" << std::endl;
  x = y + z;
  ad::cl::synchronize(x.context());
  t.start();\
  for(unsigned int i = 0 ; i < 100 ; ++i){
    x = y + z;
    ad::cl::synchronize(x.context());
  }
  std::cout << "Kernel launch overhead: " << t.get()/100 << std::endl;

}
