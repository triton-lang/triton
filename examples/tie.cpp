#include "isaac/array.h"

namespace sc = isaac;

int main()
{
    sc::array A = sc::array(sc::zeros(10, 10, sc::FLOAT_TYPE));
    std::cout << A({2,sc::end}, {3, sc::end}) << std::endl;
}
