#include "opts.hpp"
#include "isaac/scalar.h"
#include "isaac/driver/backend.h"
#include "isaac/driver/context.h"
#include "isaac/driver/stream.h"
#include "isaac/runtime/predict.h"
#include "isaac/templates/gemm.h"

namespace sc = isaac;

int main(int argc, char* argv[]){
  opts::Application program("dump", "source code dumping suite for ISAAC");
  program.add<sc::DType>("dtype", "data-type", "float32", {{"float16", sc::HALF_TYPE}, {"float32", sc::FLOAT_TYPE}, {"float64", sc::DOUBLE_TYPE}});
  program.add<std::string>("layout", "Transposition layout for A and B", "NT", {{"NN", "NN"}, {"NT", "NT"}, {"TN", "TN"}, {"TT", "TT"}});
  program.add<std::tuple<size_t, size_t, size_t>>("shape", "tensor shapes to generate the kernel for", std::make_tuple(2048, 2048, 2048));
  program.parse(argc, argv);

  // Data-Type
  sc::DType dtype = program.get<sc::DType>("dtype");
  // Shapes
  size_t M, N, K;
  std::tie(M, N, K) = program.get<std::tuple<size_t, size_t, size_t>>("shape");

  //Get Source
  std::string layout = program.get<std::string>("layout");
  sc::IsaacOperation_t AT = layout[0]=='T'?sc::ISAAC_OP_T:sc::ISAAC_OP_N;
  sc::IsaacOperation_t BT = layout[1]=='T'?sc::ISAAC_OP_T:sc::ISAAC_OP_N;
  size_t offa = 0, offb = 0, offc = 0;
  size_t ldc = M;
  size_t lda = (AT==sc::ISAAC_OP_N)?M:K;
  size_t ldb = (BT==sc::ISAAC_OP_N)?K:N;

  sc::driver::Context context = sc::driver::backend::contexts::get_default();
  sc::driver::Stream stream(context);
  sc::driver::Device device = context.device();

  sc::runtime::GEMMProfile* profile = (sc::runtime::GEMMProfile*)sc::runtime::database.at({device.architecture(), sc::runtime::GEMM}).get();
  sc::templates::GEMM generator = profile->predict(stream, dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc);
  std::cout << generator.dump(device, "gemm") << std::endl;
}
