#include "isaac/array.h"

namespace sc = isaac;

int main()
{

    std::vector<float> data(70);
    for(unsigned int i = 0 ; i < data.size(); ++i)
      data[i] = i;
    sc::array A = sc::array(10, 7, data);
    std::cout << A << std::endl;
    std::cout << sc::diag(A, 1) << std::endl;
    std::cout << sc::diag(A, -1) << std::endl;
    std::cout << sc::diag(A, -7) << std::endl;
    std::cout << A(3, {2,sc::end}) << std::endl;
    std::cout << A({2,sc::end}, 4) << std::endl;
    std::cout << sc::row(A, 3) << std::endl;
}
