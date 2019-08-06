#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/dnn/dot.h"
#include "triton/dnn/heuristics.h"
#include <string>

namespace triton{
namespace dnn{

dot::dot(int M, int N, int K,
           bool AT, bool BT,
           std::string a_ty, std::string b_ty,
           unsigned align_lda, unsigned align_ldb, unsigned align_ldc)
  : base("matmul"),
    M_(M), N_(N), K_(K), AT_(AT), BT_(BT),
    a_ty_(a_ty), b_ty_(b_ty),
    align_lda_(align_lda), align_ldb_(align_ldb), align_ldc_(align_ldc),
    locks_(nullptr) {

}

size_t dot::num_flops() const {
  return 2.*M_*N_*K_;
}

// retune parameters
std::vector<int64_t> dot::retune_params() const {
  return {M_, N_, K_, AT_, BT_,
          (int)align_lda_, (int)align_ldb_};
}

// clone
base* dot::clone() const {
  return new dot(*this);
}

void dot::init_impl(driver::stream* stream, driver::cu_module *, runtime::launch_information) {
  std::vector<int32_t> hlocks(2048, 0);
  if(locks_ == nullptr)
    locks_ = triton::driver::buffer::create(stream->context(), hlocks.size()*4);
  stream->write(locks_, false, 0, hlocks);
}

void dot::enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                        std::vector<driver::buffer*> args,
                        runtime::launch_information info) {
  driver::buffer *a = args[0], *b = args[1], *c = args[2];
  unsigned TM = info.globals.at("TM");
  unsigned TN = info.globals.at("TN");
  unsigned TK = info.globals.at("TK");
  unsigned grid_0 = (M_ + TM - 1)/TM;
  unsigned grid_1 = (N_ + TN - 1)/TN;
  unsigned grid_2 = 1;
  int32_t lda = AT_ ? K_ : M_;
  int32_t ldb = BT_ ? N_ : K_;
  int32_t ldc = M_;
  std::array<size_t, 3> grid = {grid_0, grid_1, grid_2};
  kernel->setArg(0, a);
  kernel->setArg(1, b);
  kernel->setArg(2, c);
  kernel->setArg(3, M_);
  kernel->setArg(4, N_);
  kernel->setArg(5, K_);
  kernel->setArg(6, lda);
  kernel->setArg(7, ldb);
  kernel->setArg(8, ldc);
  kernel->setArg(9, TK);
  kernel->setArg(10, locks_);
  kernel->setArg(11, grid_0);
  kernel->setArg(12, grid_1);
  stream->enqueue(kernel, grid, {info.num_threads, 1, 1});
}

void dot::triton_c_src(std::ostream &os) const {
  std::string AS0 = "TM", AS1 = "TK";
  std::string BS0 = "TK", BS1 = "TN";
  std::string XAS0 = "TM", XAS1 = "TK/1", XAS2 = "1";
  std::string XBS0 = "TK/1", XBS1 = "1", XBS2 = "TN";
  std::string bca0 = "[newaxis, :]", bca1 = "[:, newaxis]";
  std::string bcb0 = "[:, newaxis]", bcb1 = "[newaxis, :]";
  std::string lda0 = "*lda", lda1 = "";
  std::string ldb0 = "", ldb1 = "*ldb";
  std::string usea = AT_ ? "trans(xa, 0, 2, 1)" : "xa";
  std::string useb = BT_ ? "trans(xb, 1, 0, 2)" : "trans(xb, 0, 2, 1)";
  if(AT_){
    std::swap(AS0, AS1);
    std::swap(XAS0, XAS1);
    std::swap(XAS1, XAS2);
    std::swap(bca0, bca1);
    std::swap(lda0, lda1);
  }
  if(BT_){
    std::swap(BS0, BS1);
    std::swap(XBS1, XBS2);
    std::swap(XBS0, XBS1);
    std::swap(bcb0, bcb1);
    std::swap(ldb0, ldb1);
  }
  std::string AS = AS0 + ", " + AS1;
  std::string BS = BS0 + ", " + BS1;
  std::string XAS = XAS0 + ", " + XAS1 + ", " + XAS2;
  std::string XBS = XBS0 + ", " + XBS1 + ", " + XBS2;
  std::string XCS = "TM, TN, 1";
  std::string align_lda_str = "multiple_of(" + std::to_string(align_lda_) + ")";
  std::string align_ldb_str = "multiple_of(" + std::to_string(align_ldb_) + ")";
  std::string res =
R"(
const tunable int TM = {32};
const tunable int TN = {32};
const tunable int TK = {32};
const tunable int GZ = {1};

void matmul(restrict read_only align(16) )" + a_ty_ + R"( *A,
            restrict read_only align(16) )" + b_ty_ + R"( *B,
            restrict read_only align(16) float *C,
            int M, int N, int K,
            )" + align_lda_str + R"( int lda, )" + align_ldb_str + R"(" int ldb, int ldc,
            int bound, int *locks, int grid0, int grid1) {
  int ridx = get_range_id(0);
  int ridy = get_range_id(1);
  int rxa[TM] = ridx * TM + (0 ... TM);
  int ryb[TN] = ridy * TN + (0 ... TN);
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  float xc[)" + XCS + R"(] = 0;
  )" + a_ty_ + R"(* pa[)" + AS + "] = A + rka" + bca0 + lda0 + " + rxa" + bca1 + lda1 + R"(;
  )" + b_ty_ + R"(* pb[)" + BS + "] = B + rkb" + bcb0 + ldb0 + " + ryb" + bcb1 + ldb1 + R"(;
  bool checka[)" + AS + R"(] = (rka < K))" + bca0 + " && (rxa < M)" + bca1 + R"(;
  bool checkb[)" + BS + R"(] = (rkb < K))" + bcb0 + " && (ryb < N)" + bcb1 + R"(;
  )" + a_ty_ + R"( a[)" + AS + R"(] = checka ? *pa : 0;
  )" + b_ty_ + R"( b[)" + BS + R"(] = checkb ? *pb : 0;
  for(int k = K; k > 0; k = k - TK){
    )" + a_ty_ + R"( xa[)" + XAS + "] = __reshape(a, " + XAS + R"();
    )" + b_ty_ + R"( xb[)" + XBS + "] = __reshape(b, " + XBS + R"();
    xc = dot()" + usea + ", " + useb + R"(, xc);
    pa = pa + TK)" + lda0 + R"(;
    pb = pb + TK)" + ldb0 + R"(;
    bool checka[)" + AS + R"(] = k > TK;
    bool checkb[)" + BS + R"(] = k > TK;
    a = checka ? *pa : 0;
    b = checkb ? *pb : 0;
  }
  int rxc[TM] =  ridx * TM + (0 ... TM);
  int ryc[TN] =  ridy * TN + (0 ... TN);
  float* pc[TM, TN] = C + ryc[newaxis, :]*ldc + rxc[:, newaxis];
  float c[TM, TN] = __sum(xc, 2);
  *pc = c;
}
)";

//  std::cout << res << std::endl;
  os << res;
}

// small search space for partial auto-tuning
std::vector<params_t> dot::search_space() const {
  return dot_search_space(AT_, BT_);
}

// simple parameter heuristics
params_t dot::heuristics() const {
  return dot_heuristics(AT_, BT_, M_, N_, K_);
}

}
}
