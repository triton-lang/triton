#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/dnn/gemm.h"
#include <string>

namespace triton{
namespace dnn{

dot::dot(int M, int N, int K,
           bool AT, bool BT,
           std::string a_ty, std::string b_ty,
           unsigned alignment_lda, unsigned alignment_ldb)
  : base("matmul"),
    M_(M), N_(N), K_(K), AT_(AT), BT_(BT),
    a_ty_(a_ty), b_ty_(b_ty),
    align_lda_(alignment_lda), align_ldb_(alignment_ldb),
    locks_(nullptr) {

}

size_t dot::num_flops() const {
  return 2.*M_*N_*K_;
}

// comparison for maps
bool dot::operator<(const base& other) const {
  auto *y = dynamic_cast<const dot*>(&other);
  if(!y)
    return true;
  return  std::tie(M_, N_, K_, AT_, BT_,
                   a_ty_, b_ty_, align_lda_, align_ldb_)
        < std::tie(y->M_, y->N_, y->K_, y->AT_, y->BT_,
                   y->a_ty_, y->b_ty_, y->align_lda_, y->align_ldb_);
}

// clone
base* dot::clone() const {
  return new dot(*this);
}

void dot::init_impl(driver::stream* stream, driver::cu_module *) {
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
  std::string bca0 = "[newaxis, :]", bca1 = "[:, newaxis]";
  std::string bcb0 = "[:, newaxis]", bcb1 = "[newaxis, :]";
  std::string lda0 = "*lda", lda1 = "";
  std::string ldb0 = "", ldb1 = "*ldb";
  std::string usea = AT_ ? "trans(a)" : "a";
  std::string useb = BT_ ? "trans(b)" : "b";
  if(AT_){
    std::swap(AS0, AS1);
    std::swap(bca0, bca1);
    std::swap(lda0, lda1);
  }
  if(BT_){
    std::swap(BS0, BS1);
    std::swap(bcb0, bcb1);
    std::swap(ldb0, ldb1);
  }
  std::string AS = AS0 + ", " + AS1;
  std::string BS = BS0 + ", " + BS1;
  std::string align_lda_str = "multiple_of(" + std::to_string(align_lda_) + ")";
  std::string align_ldb_str = "multiple_of(" + std::to_string(align_ldb_) + ")";
  std::string res =
R"(
const tunable int32 TM = {16, 32, 64, 128};
const tunable int32 TN = {16, 32, 64, 128};
const tunable int32 TK = {32};
const tunable int32 GZ = {1};

void matmul(restrict read_only align(16) )" + a_ty_ + R"( *A,
            restrict read_only align(16) )" + b_ty_ + R"( *B,
            fp32 *C,
            int32 M, int32 N, int32 K,
            )" + align_lda_str + R"( int32 lda, )" + align_ldb_str + R"(" int32 ldb, int32 ldc,
            int32 bound, int32 *locks, int32 grid0, int32 grid1) {
  int32 ridx = get_range_id(0);
  int32 ridy = get_range_id(1);
  int32 rxa[TM] = ridx * TM + (0 ... TM);
  int32 ryb[TN] = ridy * TN + (0 ... TN);
  int32 rka[TK] = 0 ... TK;
  int32 rkb[TK] = 0 ... TK;
  fp32 c[TM, TN] = 0;
  )" + a_ty_ + R"(* pa[)" + AS + "] = A + rka" + bca0 + lda0 + " + rxa" + bca1 + lda1 + R"(;
  )" + b_ty_ + R"(* pb[)" + BS + "] = B + rkb" + bcb0 + ldb0 + " + ryb" + bcb1 + ldb1 + R"(;
  int1 checka[)" + AS + R"(] = (rka < K))" + bca0 + " && (rxa < M)" + bca1 + R"(;
  int1 checkb[)" + BS + R"(] = (rkb < K))" + bcb0 + " && (ryb < N)" + bcb1 + R"(;
  )" + a_ty_ + R"( a[)" + AS + R"(] = checka ? *pa : 0;
  )" + b_ty_ + R"( b[)" + BS + R"(] = checkb ? *pb : 0;
  for(int32 k = K; k > 0; k = k - TK){
    c = dot()" + usea + ", " + useb + R"(, c);
    pa = pa + TK)" + lda0 + R"(;
    pb = pb + TK)" + ldb0 + R"(;
    int1 checka[)" + AS + R"(] = k > bound;
    int1 checkb[)" + BS + R"(] = k > bound;
    @checka a = *pa;
    @checkb b = *pb;
  }
  int32 rxc[TM] = ridx*TM + (0 ... TM);
  int32 ryc[TN] = ridy*TN + (0 ... TN);
  int1 checkc0[TM] = rxc < M;
  int1 checkc1[TN] = ryc < N;
  int1 checkc[TM, TN] = checkc0[:, newaxis] && checkc1[newaxis, :];
  fp32* pc[TM, TN] = C + ryc[newaxis, :]*ldc + rxc[:, newaxis];
  @checkc *pc = c;
}
)";
  os << res;
}

// small search space for partial auto-tuning
std::vector<params_t> dot::search_space() const {
  typedef std::vector<unsigned> params_t;
  typedef std::tuple<size_t, size_t> key_t;
  static std::vector<key_t> keys  = {
    {16, 16}, {16, 32}, {16, 64}, {16, 128},
    {32, 16}, {32, 32}, {32, 64}, {32, 128},
    {64, 16}, {64, 32}, {64, 64}, {64, 128},
    {128, 16},{128, 32},{128, 64},{128, 128}
  };
  static std::vector<params_t> space_nn = {
    {4, 4, 16, 8, 16, 2, 2, 1, 1, 8, 32, 4, 8, 1},
    {2, 8, 16, 8, 32, 2, 2, 1, 1, 16, 32, 4, 8, 1},
    {4, 4, 16, 4, 64, 2, 2, 1, 1, 8, 32, 8, 4, 1},
    {4, 4, 16, 16, 128, 2, 2, 1, 2, 16, 32, 4, 8, 1},
    {4, 8, 32, 8, 16, 2, 2, 1, 1, 8, 32, 4, 8, 1},
    {4, 8, 32, 8, 32, 2, 2, 1, 1, 8, 32, 4, 8, 1},
    {8, 4, 32, 8, 64, 2, 2, 1, 1, 4, 32, 4, 8, 1},
    {8, 4, 32, 16, 128, 2, 2, 1, 4, 16, 32, 8, 4, 1},
    {8, 8, 64, 4, 16, 2, 2, 1, 1, 4, 32, 8, 4, 1},
    {8, 8, 64, 8, 32, 2, 2, 1, 1, 4, 32, 4, 8, 1},
    {8, 8, 64, 16, 64, 2, 2, 2, 1, 8, 32, 4, 8, 1},
    {16, 4, 64, 16, 128, 2, 2, 2, 2, 8, 32, 8, 4, 1},
    {8, 8, 128, 8, 16, 2, 2, 2, 1, 8, 32, 8, 4, 1},
    {8, 8, 128, 16, 32, 2, 2, 2, 1, 8, 32, 4, 8, 1},
    {8, 8, 128, 32, 64, 2, 2, 2, 2, 16, 32, 4, 8, 1},
    {8, 8, 128, 32, 128, 2, 2, 1, 4, 16, 32, 4, 8, 1},
  };
  static std::vector<params_t> space_nt = {
    {4, 4, 16, 2, 8, 16, 2, 2, 1, 1, 8, 32, 16, 1},
    {4, 4, 16, 4, 8, 32, 2, 2, 1, 1, 8, 32, 8, 1},
    {4, 4, 16, 8, 8, 64, 2, 2, 1, 4, 32, 32, 16, 1},
    {4, 4, 16, 32, 4, 128, 2, 2, 1, 2, 16, 32, 2, 1},
    {8, 4, 32, 2, 8, 16, 2, 2, 1, 1, 4, 32, 16, 1},
    {4, 8, 32, 4, 8, 32, 2, 2, 1, 1, 8, 32, 8, 1},
    {16, 8, 128, 4, 4, 64, 2, 2, 1, 4, 8, 32, 32, 1},
    {4, 8, 32, 8, 8, 128, 2, 2, 1, 2, 16, 32, 8, 1},
    {8, 8, 64, 2, 8, 16, 2, 2, 1, 1, 4, 32, 16, 1},
    {8, 8, 64, 4, 8, 32, 2, 2, 1, 1, 4, 32, 8, 1},
    {8, 8, 64, 8, 8, 64, 2, 2, 1, 2, 8, 32, 8, 1},
    {8, 8, 64, 16, 8, 128, 2, 2, 1, 4, 16, 32, 8, 1},
    {8, 8, 128, 2, 8, 16, 2, 2, 2, 1, 8, 32, 32, 1},
    {16, 8, 128, 4, 8, 32, 2, 2, 2, 1, 4, 32, 16, 1},
    {8, 8, 128, 8, 8, 64, 2, 2, 2, 2, 16, 32, 16, 1},
    {8, 8, 128, 8, 8, 128, 2, 2, 4, 1, 16, 32, 16, 1},
  };
  static std::vector<params_t> space_tn = {
    {8, 16, 16, 16, 2, 2, 1, 1, 4, 8, 32, 2, 8, 1},
    {4, 16, 8, 32, 2, 2, 1, 1, 8, 4, 32, 4, 8, 1},
    {4, 16, 4, 64, 2, 2, 1, 1, 8, 4, 32, 8, 4, 1},
    {16, 16, 16, 128, 2, 2, 1, 2, 4, 8, 32, 4, 8, 1},
    {4, 32, 8, 16, 2, 2, 1, 1, 8, 4, 32, 4, 8, 1},
    {8, 32, 8, 32, 2, 2, 1, 1, 4, 8, 32, 4, 8, 1},
    {8, 32, 8, 64, 2, 2, 1, 1, 4, 8, 32, 4, 8, 1},
    {32, 32, 64, 128, 2, 2, 2, 2, 4, 8, 32, 2, 8, 1},
    {8, 64, 8, 16, 2, 2, 1, 1, 4, 8, 32, 4, 8, 1},
    {8, 64, 8, 32, 2, 2, 1, 1, 4, 8, 32, 4, 8, 1},
    {16, 64, 16, 64, 2, 2, 2, 1, 4, 8, 32, 4, 8, 1},
    {32, 64, 16, 128, 2, 2, 2, 2, 4, 8, 32, 8, 4, 1},
    {16, 128, 16, 16, 2, 2, 2, 1, 4, 8, 32, 4, 8, 1},
    {32, 128, 32, 32, 2, 2, 4, 1, 4, 8, 32, 4, 8, 1},
    {32, 128, 32, 64, 2, 2, 4, 1, 4, 8, 32, 4, 8, 1},
    {32, 128, 32, 128, 2, 2, 4, 1, 4, 8, 32, 4, 8, 1},
  };
  static std::vector<params_t> space_tt = {
    {4, 16, 2, 8, 16, 2, 2, 1, 1, 8, 4, 32, 16, 1},
    {8, 16, 4, 8, 32, 2, 2, 1, 1, 4, 8, 32, 8, 1},
    {16, 16, 4, 8, 64, 2, 2, 1, 4, 8, 4, 32, 32, 1},
    {16, 16, 8, 4, 128, 2, 2, 1, 2, 4, 8, 32, 8, 1},
    {4, 32, 2, 8, 16, 2, 2, 1, 1, 8, 4, 32, 16, 1},
    {8, 32, 4, 8, 32, 2, 2, 1, 1, 4, 8, 32, 8, 1},
    {16, 64, 4, 8, 64, 2, 2, 2, 1, 4, 8, 32, 16, 1},
    {32, 32, 8, 8, 128, 2, 2, 1, 4, 4, 8, 32, 16, 1},
    {8, 64, 2, 8, 16, 2, 2, 1, 1, 4, 8, 32, 16, 1},
    {8, 64, 4, 8, 32, 2, 2, 1, 1, 4, 8, 32, 8, 1},
    {16, 64, 8, 8, 64, 2, 2, 2, 1, 4, 8, 32, 8, 1},
    {32, 64, 8, 8, 128, 2, 2, 1, 4, 4, 8, 32, 16, 1},
    {16, 128, 2, 8, 16, 2, 2, 2, 1, 4, 8, 32, 32, 1},
    {32, 128, 8, 4, 32, 2, 2, 4, 1, 4, 8, 32, 16, 1},
    {32, 128, 16, 4, 64, 2, 2, 4, 1, 4, 8, 32, 8, 1},
    {32, 128, 8, 8, 128, 2, 2, 4, 1, 4, 8, 32, 16, 1}
  };
  if(!AT_ && !BT_)
    return space_nn;
  else if(!AT_ && BT_)
    return space_nt;
  else if(AT_ && !BT_)
    return space_tn;
  else
    return space_tt;
}

// simple parameter heuristics
params_t dot::heuristics() const {
  return search_space().back();
}

}
}
