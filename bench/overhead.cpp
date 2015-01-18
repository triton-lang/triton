#include "atidlas/array.h"
#include "atidlas/tools/timer.hpp"

#include <vector>

namespace ad = atidlas;

int main()
{
  ad::array x(10, ad::FLOAT_TYPE), y(10, ad::FLOAT_TYPE), z(10, ad::FLOAT_TYPE);
  ad::tools::timer t;
  std::cout << "-------------------------" << std::endl;
  std::cout << "Expression tree creation:" << std::endl;
#define BENCH(CREATE, STR) \
  {\
  std::vector<int> flusher(10000000, 1);\
  t.start();\
  ad::array_expression tmp(CREATE);\
  std::cout << STR << ": " << t.get() << std::endl;\
  }

  BENCH(x + y, "2 terms");
  BENCH(x + y + x, "3 terms");
  BENCH(x + y + x + y, "4 terms");
  BENCH(x + y + x + y + x, "5 terms");
  std::cout << "-------------------------" << std::endl;
}
