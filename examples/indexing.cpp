#include "isaac/array.h"

namespace sc = isaac;

int main()
{
//    static const char * sline = "--------------------";
    static const char * dline = "====================";

    std::cout << dline << std::endl;
    std::cout << "Tutorial: Indexing " << std::endl;
    std::cout << dline << std::endl;

    sc::int_t M = 5, N = 12;

    std::vector<float> data(M*N);
    for(unsigned int i = 0 ; i < data.size(); ++i)
      data[i] = i;
    sc::array A = sc::array(M, N, data);

    sc::array s = sc::array({1,1}, std::vector<float>{5});
    sc::array x = sc::array({1,3},std::vector<float>{1,2,3});
    sc::array y = sc::array({3,3},std::vector<float>{1,2,3,4,5,6,7,8,9});

    sc::array B({4,3},std::vector<float>{0,1,2,3,4,5,6,7,8,9,10,11});

//    std::cout << sc::sum(y, 1)*sc::sum(x) << std::endl;
//    std::cout << sc::dot(B.T, B + B) << std::endl;
    std::cout << 1*s*x + x << std::endl;
//    std::cout << sc::sum(B) << std::endl;
//    std::cout << sc::reshape(x, {3,1}) + sc::sum(x)*sc::sum(sc::dot(B.T,B) + x + y, 1) + sc::sum(B)*sc::sum(B, 0)<< std::endl;
//    std::cout << sline << std::endl;
//    std::cout << "A[3, 2:end]:" << A(3, {2,sc::end}) << std::endl;

//    std::cout << sline << std::endl;
//    std::cout << "A[2:end, 4]:" << A({2,sc::end}, 4) << std::endl;

//    std::cout << sline << std::endl;
//    std::cout << "diag(A,  1): " << sc::diag(A, 1) << std::endl;

//    std::cout << sline << std::endl;
//    std::cout << "diag(A, -7): " << sc::diag(A, -7) << std::endl;
}
