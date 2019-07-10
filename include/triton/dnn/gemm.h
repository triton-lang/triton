#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/dnn/base.h"
#include <string>

namespace triton{
namespace dnn{

class gemm: public base {
private:
  // initialize
  void init_impl(driver::stream *, driver::cu_module *);
  // enqueue
  void enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                    std::vector<driver::buffer*> args,
                    const std::vector<unsigned>& ranges,
                    size_t nthreads);
  // number of flops
  size_t num_flops() const;
  // comparison for maps
  bool operator<(const base& other) const;


public:
  gemm(int M, int N, int K, bool AT, bool BT,
       std::string a_ty, std::string b_ty,
       unsigned alignment_lda, unsigned alignment_ldb);

  // triton-c source
  void triton_c_src(std::ostream &os) const;

  // clone
  base* clone() const;

  // default params
  std::vector<unsigned> default_params();

  // CPU reference implementation
  template<class T, bool AT, bool BT>
  static void cpu_ref(std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b,
                      size_t M, size_t N, size_t K){
    for(size_t m = 0; m < M; m++)
    for(size_t n = 0; n < N; n++){
      T acc = 0;
      for(size_t k = 0; k < K; k++)
        acc += (AT?a[k + m*K]:a[m + k*M]) * (BT?b[n + k*N]:b[k + n*K]);
      c[m + n*M] = acc;
    }
  }
  template<class T>
  void cpu_ref(std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b) {
    if(AT_ && BT_)
      gemm::cpu_ref<T, true, true>(c, a, b, M_, N_, K_);
    else if(AT_ && !BT_)
      gemm::cpu_ref<T, true, false>(c, a, b, M_, N_, K_);
    else if(!AT_ && BT_)
      gemm::cpu_ref<T, false, true>(c, a, b, M_, N_, K_);
    else
      gemm::cpu_ref<T, false, false>(c, a, b, M_, N_, K_);
  }

private:
  int32_t M_;
  int32_t N_;
  int32_t K_;
  bool AT_;
  bool BT_;
  std::string a_ty_;
  std::string b_ty_;
  unsigned align_lda_;
  unsigned align_ldb_;
  driver::buffer *locks_;
};

}
}
