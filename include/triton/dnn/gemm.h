#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include <string>

namespace triton{
namespace dnn{

class gemm {
public:
  static void init(driver::stream* stream, driver::buffer* locks);

  static void set_arg(driver::kernel *kernel,
                      driver::buffer *a, driver::buffer *b, driver::buffer *c,
                      int32_t M, int32_t N, int32_t K,
                      driver::buffer *locks, int32_t grid_0, int32_t grid_1);

  static std::vector<unsigned> default_params(bool AT, bool BT);

  static std::string src(bool AT, bool BT,
                         std::string a_ty, std::string b_ty,
                         unsigned alignment_lda, unsigned alignment_ldb);

  template<class T, bool AT, bool BT>
  static void cpu_ref(std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b, size_t M, size_t N, size_t K){
    for(size_t m = 0; m < M; m++)
    for(size_t n = 0; n < N; n++){
      T acc = 0;
      for(size_t k = 0; k < K; k++)
        acc += (AT?a[k + m*K]:a[m + k*M]) * (BT?b[n + k*N]:b[k + n*K]);
      c[m + n*M] = acc;
    }
  }

  template<class T>
  static void cpu_ref(bool AT, bool BT, std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b, size_t M, size_t N, size_t K) {
    if(AT && BT)
      gemm::cpu_ref<T, true, true>(c, a, b, M, N, K);
    else if(AT && !BT)
      gemm::cpu_ref<T, true, false>(c, a, b, M, N, K);
    else if(!AT && BT)
      gemm::cpu_ref<T, false, true>(c, a, b, M, N, K);
    else
      gemm::cpu_ref<T, false, false>(c, a, b, M, N, K);
  }
};

}
}
