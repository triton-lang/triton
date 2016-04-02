#include "isaac/array.h"
#include "isaac/symbolic/scheduler/dag.h"

namespace sc = isaac;

class carma_generator
{
  void apply_impl(sc::array_base const & A, sc::array_base const & B, sc::view C, size_t depth)
  {
    if(depth>=split_.size()){
      dag_.append(sc::assign(C, sc::dot(A, B)), "C = dot(A, B)");
    }
    else
    {
      sc::int_t M = C.shape()[0], N = C.shape()[1], K = A.shape()[1];
      size_t new_depth = depth + 1;
      //Split along M
      if(M >= N && M >= K){
        apply_impl(A({0, M/2}, {sc::all}), B, C({0, M/2}, sc::all), new_depth);
        apply_impl(A({M/2, sc::end}, {sc::all}), B, C({M/2, sc::end}, sc::all), new_depth);
      }
      //Split along N
      else if(N >= M && N >= K){
        apply_impl(A, B(sc::all, {0, N/2}), C(sc::all, {0, N/2}), new_depth);
        apply_impl(A, B(sc::all, {N/2, sc::end}), C(sc::all, {N/2, sc::end}), new_depth);
      }
      //Split along K
      else{
        sc::array_base & C1 = dag_.create_temporary(new sc::array(C.shape(), C.dtype(), C.context()));
        sc::array_base & C2 = dag_.create_temporary(new sc::array(C.shape(), C.dtype(), C.context()));
        apply_impl(A(sc::all, {0, K/2}), B({0, K/2}, sc::all), C1, new_depth);
        apply_impl(A(sc::all, {K/2, sc::end}), B({K/2, sc::end}, sc::all), C2, new_depth);
        dag_.append(sc::assign(C, C1 + C2), "C = C1 + C2");
      }
    }
  }

public:
  carma_generator(size_t depth): split_(depth)
  { }

  void apply(sc::array_base const & A, sc::array_base const & B, sc::array_base & C)
  {
    apply_impl(A, B, sc::view(C), 0);
    dag_.export_graphviz("test.dot");
  }

private:
  sc::symbolic::scheduler::dag dag_;
  std::vector<sc::int_t> split_;
};


int main()
{
  sc::int_t M = 131, N = 1402, K = 5023;
  sc::array C(M, N), A(M, K), B(K, N);
  carma_generator generator(3);
  generator.apply(A, B, C);
}
