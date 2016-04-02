#include <iostream>

#include "isaac/symbolic/scheduler/strategies/heft.h"

namespace sch = isaac::symbolic::scheduler;


int main()
{
    //Adjacency list
    sch::adjacency_t
          succ = { {1, {2,3,4,5,6}},
                   {2, {8,9}},
                   {3, {7}},
                   {4, {8,9}},
                   {5, {9}},
                   {6, {8}},
                   {7, {10}},
                   {8, {10}},
                   {9, {10}} };

    //Computation cost
    std::map<size_t, std::vector<size_t> >
         compcosts = {{1, {14,16,9}},
                      {2, {13, 19, 18}},
                      {3, {11, 13, 19}},
                      {4, {13, 8, 17}},
                      {5, {12, 13, 10}},
                      {6, {13, 16, 9}},
                      {7, {7, 15, 11}},
                      {8, {5, 11, 14}},
                      {9, {18, 12, 20}},
                      {10, {21, 7, 16}}};
    auto compcost = [&](size_t n, size_t p){ return compcosts[n][p]; };

    //Communication cost
    std::map<std::pair<size_t, size_t>, size_t>
        commcosts = { {{1,2},18},
                      {{1,3},12},
                      {{1,4},9},
                      {{1,5},11},
                      {{1,6},14},
                      {{2,8},19},
                      {{2,9},16},
                      {{3,7}, 23},
                      {{4,8}, 27},
                      {{4,9}, 23},
                      {{5,9}, 13},
                      {{6,8}, 15},
                      {{7,10}, 17},
                      {{8,10}, 11},
                      {{9,10}, 13} };
    auto commcost = [&](size_t n1, size_t n2, size_t p1, size_t p2){ return (p1==p2)?0:commcosts[{n1, n2}]; };

    //Schedule tasks
    sch::orders_t orders;
    sch::jobson_t jobson;
    sch::heft::schedule(succ, {0,1,2}, compcost, commcost, orders, jobson);

    //Verify paper results
    std::cout << "Makespan...";
    if(sch::heft::makespan(orders)!=80){
      std::cout << " [Failure!]" << std::endl;
      return EXIT_FAILURE;
    }
    else
      std::cout << std::endl;

    return EXIT_SUCCESS;
}
