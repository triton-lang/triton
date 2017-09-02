#include "opts.hpp"
#include "isaac/scalar.h"
#include "isaac/api.h"
#include "isaac/driver/cublas.h"
#include "isaac/driver/backend.h"
#include "isaac/driver/context.h"
#include "isaac/driver/stream.h"
#include "isaac/runtime/predict.h"
#include "isaac/templates/gemm.h"
#include "isaac/templates/error.hpp"
#include "isaac/tools/bench.hpp"

namespace sc = isaac;
namespace drv = sc::driver;
using sc::param_t;

enum Code {
  RESET = 0,
  BOLD = 1,
  ITALIC = 3,
  FG_RED = 31,
  FG_GREEN = 32,
  FG_YELLOW = 33,
  FG_BLUE = 34,
  FG_MAGENTA = 35,
  FG_CYAN = 36,
  FG_LIGHT_GRAY = 37,
  FG_DARK_GRAY = 90,
  FG_LIGHT_RED = 91,
  FG_LIGHT_GREEN = 92,
  FG_LIGHT_YELLOW = 93,
  FG_LIGHT_BLUE = 94,
  FG_LIGHT_MAGENTA = 95,
  FG_LIGHT_CYAN = 96,
  FG_WHITE = 97
};

class color_stream {
    Code code;
public:
    color_stream(Code pCode) : code(pCode) {}
    friend std::ostream&
    operator<<(std::ostream& os, const color_stream& mod) {
        return os << "\033[" << mod.code << "m";
    }
};

/* Helpers for benchmarking */
typedef std::tuple<sc::DType, sc::IsaacOperation_t, sc::IsaacOperation_t, sc::param_t, sc::param_t, sc::param_t> gemm_params_t;
typedef std::tuple<sc::DType, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t, param_t> conv_params_t;

struct TestBench{
  // GEMM
  static std::vector<gemm_params_t> gemm(sc::DType dtype){
    std::vector<gemm_params_t> shapes;
    // LinPack
    for(param_t N: std::vector<param_t>{512, 1024, 2048})
      shapes.push_back(std::make_tuple(dtype, sc::ISAAC_OP_N, sc::ISAAC_OP_T, N, N, N));
    // DeepBench
    for(sc::IsaacOperation_t AT: std::vector<sc::IsaacOperation_t>{sc::ISAAC_OP_N, sc::ISAAC_OP_T})
      for(param_t M: std::vector<param_t>{1760})
        for(param_t N: std::vector<param_t>{16, 32, 64, 128})
          shapes.push_back(std::make_tuple(dtype, AT, sc::ISAAC_OP_N, M, N, M));
    // PCA/ICA
    for(param_t N: std::vector<param_t>{16, 64, 256})
      for(param_t K: std::vector<param_t>{64000})
        shapes.push_back(std::make_tuple(dtype, sc::ISAAC_OP_N, sc::ISAAC_OP_T, N, N, K));
    // LaPACK
    for(param_t N: std::vector<param_t>{1024, 2048, 4096})
      for(param_t K: std::vector<param_t>{32})
        shapes.push_back(std::make_tuple(dtype, sc::ISAAC_OP_N, sc::ISAAC_OP_T, N, N, K));
    return shapes;
  }

  // CONV
  static std::vector<conv_params_t> conv(sc::DType dtype){
    std::vector<conv_params_t> shapes;

    //DeepSpeech
    for(size_t N: std::vector<size_t>{16})
      shapes.push_back(std::make_tuple(dtype, 700, 161, 1, N, 32, 5, 20, 0, 0, 2, 2));
    for(size_t N: std::vector<size_t>{16})
      shapes.push_back(std::make_tuple(dtype, 341, 79, 32, N, 32, 5, 10, 0, 0, 2, 2));

    //OCR
//    shapes.push_back(std::make_tuple(dtype, 480, 48, 1, 16, 16, 3, 3, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 240, 24, 16, 16, 32, 3, 3, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 120, 12, 32, 16, 64, 3, 3, 1, 1, 1, 1));
//    shapes.push_back(std::make_tuple(dtype, 60, 6, 64, 16, 128, 3, 3, 1, 1, 1, 1));

    //Face Recognition [1]
//    shapes.push_back(std::make_tuple(dtype, 108, 108, 3, 8, 64, 3, 3, 1, 1, 2, 2));
    shapes.push_back(std::make_tuple(dtype, 54, 54, 64, 8, 64, 3, 3, 1, 1, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 27, 27, 128, 8, 128, 3, 3, 1, 1, 1, 1));
//    shapes.push_back(std::make_tuple(dtype, 14, 14, 128, 8, 256, 3, 3, 1, 1, 1, 1));
//    shapes.push_back(std::make_tuple(dtype, 7, 7, 256, 8, 512, 3, 3, 1, 1, 1, 1));

    //Face Recognition [2]
//    shapes.push_back(std::make_tuple(dtype, 224, 224, 3, 16, 64, 7, 7, 3, 3, 2, 2));
//    shapes.push_back(std::make_tuple(dtype, 28, 28, 192, 16, 32, 5, 5, 2, 2, 1, 1));
//    shapes.push_back(std::make_tuple(dtype, 28, 28, 192, 16, 64, 1, 1, 0, 0, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 14, 14, 512, 16, 48, 5, 5, 2, 2, 1, 1));
//    shapes.push_back(std::make_tuple(dtype, 14, 14, 512, 16, 192, 1, 1, 0, 0, 1, 1));
//    shapes.push_back(std::make_tuple(dtype, 7, 7, 832, 16, 256, 1, 1, 0, 0, 1, 1));
    shapes.push_back(std::make_tuple(dtype, 7, 7, 832, 16, 128, 5, 5, 2, 2, 1, 1));

    //Vision
    for(size_t N: std::vector<size_t>{8}){
//      shapes.push_back(std::make_tuple(dtype, 224, 224, 3, N, 64, 3, 3, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 112, 112, 64, N, 128, 3, 3, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 56, 56, 128, N, 256, 3, 3, 1, 1, 1, 1));
//      shapes.push_back(std::make_tuple(dtype, 28, 28, 256, N, 512, 3, 3, 1, 1, 1, 1));
//      shapes.push_back(std::make_tuple(dtype, 14, 14, 512, N, 512, 3, 3, 1, 1, 1, 1));
//      shapes.push_back(std::make_tuple(dtype, 7, 7, 512, N, 512, 3, 3, 1, 1, 1, 1));
    }

    //Speaker ID
    shapes.push_back(std::make_tuple(dtype, 350, 80, 64, 16, 128, 5, 5, 1, 1, 2, 2));
    shapes.push_back(std::make_tuple(dtype, 175, 40, 128, 16, 256, 5, 5, 1, 1, 2, 2));

    //ResNET
    for(size_t N: std::vector<size_t>{16}){
      shapes.push_back(std::make_tuple(dtype, 7, 7, 512, N, 512, 3, 3, 1, 1, 1, 1));
      shapes.push_back(std::make_tuple(dtype, 14, 14, 1024, N, 2048, 1, 1, 0, 0, 2, 2));
    }

    return shapes;
  }
};
/* Metrics for benchmarking */
struct Metric{
  virtual std::function<bool(double, double)> cmp() const = 0;
  virtual double conv(param_t P, param_t Q, param_t K, param_t N, param_t C, param_t R, param_t S, double tsec) const = 0;
  virtual double gemm(param_t M, param_t N, param_t K, double tsec) const = 0;
};

class FLOPS: public Metric{
public:
  FLOPS(double scale): scale_(scale){}
  std::function<bool(double, double)> cmp() const { return std::greater<double>(); }
  double conv(param_t P, param_t Q, param_t K, param_t N, param_t C, param_t R, param_t S, double tsec) const
  { return  sc::templates::Conv::tflops(P,Q,K,N,C,R,S,tsec) * 1e12 / scale_; }
  double gemm(param_t M, param_t N, param_t K, double tsec) const
  { return  sc::templates::GEMM::tflops(M, N, K, tsec) * 1e12 / scale_; }
private:
  double scale_;
};

class Time: public Metric{
public:
  Time(double scale): scale_(scale){}
  std::function<bool(double, double)> cmp() const { return std::less<double>(); }
  double conv(param_t, param_t, param_t, param_t, param_t, param_t, param_t, double tsec) const { return tsec*1e-9/scale_; }
  double gemm(param_t, param_t, param_t, double tsec) const { return tsec*1e-9/scale_; }
private:
  double scale_;
};

void print_results_header(std::vector<std::string> sections){
    std::cout << color_stream(ITALIC) << color_stream(BOLD) ;
    std::copy(sections.begin(), sections.end(), std::ostream_iterator<std::string>(std::cout, "\t"));
    std::cout << color_stream(RESET) << std::endl;
}

void print_results(std::vector<double> const & times, std::vector<std::string> const & prefix, std::function<bool(double, double)> cmp, std::function<double(double)> fn){
    std::copy(prefix.begin(), prefix.end(), std::ostream_iterator<std::string>(std::cout, "\t"));
    std::vector<double> perf;
    std::transform(times.begin(), times.end(), std::back_inserter(perf), fn);
    auto fastest = perf;
    std::sort(fastest.begin(), fastest.end(), cmp);

    for(auto x: perf){
      if(x == fastest[0] && x / fastest[1] > 1.05)
        std::cout << color_stream(FG_LIGHT_BLUE) << x << color_stream(RESET);
      else
        std::cout << x;
      std::cout << "\t";
    }
    std::cout << std::endl;
}

void benchmark_gemm(Metric const & metric, sc::driver::Context& ctx, sc::driver::Device& device, sc::driver::Stream& stream,
                    sc::DType dtype, sc::IsaacOperation_t AT, sc::IsaacOperation_t BT, size_t M, size_t N, size_t K,
                    sc::templates::Generator* generator){
  size_t ldc = M;
  size_t lda = (AT==sc::ISAAC_OP_N)?M:K;
  size_t ldb = (BT==sc::ISAAC_OP_N)?K:N;

  size_t dtsize = sc::size_of(dtype);
  sc::scalar alpha(1., dtype);
  sc::scalar beta(0., dtype);
  char cuAT = (AT==sc::ISAAC_OP_T)?'T':'N';
  char cuBT = (BT==sc::ISAAC_OP_T)?'T':'N';

  sc::driver::Buffer C(ctx, M*N*dtsize);
  sc::driver::Buffer A(ctx, M*K*dtsize);
  sc::driver::Buffer B(ctx, K*N*dtsize);

  std::vector<double> times;
  times.push_back(bench([&](){ sc::GEMM(device, stream, dtype, AT, BT, M, N, K, 0, lda, 0, ldb, 0, ldc, alpha, A, B, beta, C, (sc::templates::GEMM*)generator); }, [&](){ stream.synchronize(); }, device));
  if(sc::driver::dispatch::cublasinit()){
//    cublasGemmAlgo_t fastest;
//    sc::driver::cublasGemm(dtype, stream, cuAT, cuBT, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, &fastest);
//    times.push_back(bench([&](){ sc::driver::cublasGemm(dtype, stream, cuAT, cuBT, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, NULL, fastest); }, [&](){ stream.synchronize();  }, device));
    times.push_back(bench([&](){ sc::driver::cublasGemm(dtype, stream, cuAT, cuBT, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }, [&](){ stream.synchronize();  }, device));
  }
  print_results(times, {str(AT), str(BT), str(M), str(N), str(K)}, metric.cmp(), [&](double tsec){ return metric.gemm(M, N, K, tsec);});
}

void benchmark_conv(Metric const & metric, sc::driver::Context& ctx, sc::driver::Device& device, sc::driver::Stream& stream,
                    sc::DType dtype, size_t W, size_t H, size_t C, size_t N, size_t K, size_t R, size_t S, size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w,
                    sc::templates::Generator* generator){
  size_t P = (H - R + 1 + 2*pad_h + stride_h - 1)/stride_h;
  size_t Q = (W - S + 1 + 2*pad_w + stride_w - 1)/stride_w;

  size_t dtsize = sc::size_of(dtype);
  sc::scalar alpha(1., dtype);
  sc::scalar beta(0., dtype);

  sc::driver::Buffer O(ctx, N*K*P*Q*dtsize);
  sc::driver::Buffer I(ctx, C*H*W*N*dtsize);
  sc::driver::Buffer F(ctx, K*C*R*S*dtsize);

  std::vector<double> times;
  times.push_back(bench([&](){ sc::CONV(device, stream, dtype, N, K, P, Q, C, R, S, H, W, pad_h, pad_w, stride_h, stride_w, alpha, I, F, beta, O, (sc::templates::Conv*)generator); }, [&](){ stream.synchronize(); }, device));
  if(sc::driver::dispatch::cudnninit())
    times.push_back(bench([&](){ sc::driver::cudnnConv(dtype, stream, H, W, N, K, P, Q, C, R, S, pad_h, pad_w, stride_h, stride_w, alpha, I, F, beta, O); }, [&](){ stream.synchronize();  }, device));
  print_results(times, {str(N), str(K), str(P), str(Q), str(C), str(R), str(S)}, metric.cmp(), [&](double tsec){ return metric.conv(P, Q, K, N, C, R, S, tsec);});
}

/* ------------------------------- */

void loop_nest(std::vector<size_t> const & ranges, std::function<void(std::vector<size_t> const &)> const & f){
  size_t D = ranges.size();
  std::vector<size_t> values(D, 0);
  // Start with innermost loop
  size_t i = D - 1;
  while(true){
    //Execute function
    f(values);
    //Increment counters
    while(values[i]++ == ranges[i] - 1){
      if(i == 0)
        return;
      values[i--] = 0;
    }
    i = D - 1;
  }
}

template<class T>
void loop_nest(std::vector<std::vector<T>> const & iterates, std::function<void(std::vector<T>)> const & f){
  //Ranges to iterate over
  std::vector<size_t> ranges;
  for(auto const & x: iterates)
    ranges.push_back(x.size());
  //Proxy function
  auto proxy = [&](std::vector<size_t> const & idx){
    std::vector<T> x(iterates.size());
    for(size_t i = 0; i < x.size(); ++i)
    x[i] = iterates[i][idx[i]];
  f(x);
  };
  //Iterate
  loop_nest(ranges, proxy);
}


void search_conv(int32_t W, int32_t H, int32_t C, int32_t N, int32_t K, int32_t R, int32_t S, int32_t pad_h, int32_t pad_w, int32_t stride_h, int32_t stride_w, sc::DType dtype){
  auto ctx = drv::backend::contexts::get_default();
  size_t dtsize = sc::size_of(dtype);
  int32_t P = (H - R + 1 + 2*pad_h)/stride_h, Q = (W - S + 1 + 2*pad_w)/stride_w;

  //Setup
  drv::Buffer O(ctx, K*P*Q*N*dtsize);
  drv::Buffer I(ctx, C*H*W*N*dtsize);
  drv::Buffer F(ctx, C*R*S*K*dtsize);
  drv::Stream stream(ctx);
  sc::scalar alpha(1., dtype),  beta(1., dtype);

  //Exhaustive search
  std::vector<sc::param_t> rv = {4};
  std::vector<sc::param_t> rl = {1, 2, 4, 8};
  std::vector<sc::param_t> rs = {1, 2, 4, 8};
  std::vector<sc::param_t> r1 = {1};
  double best;
  loop_nest<sc::param_t>({rv,rl,rl,rl,rl,rl,rl,rl,rs,rs,rl,r1,r1,rl,rl}, [&](std::vector<sc::param_t> const & x){
    sc::templates::Conv generator(dtype, C, H, W, N, K, P, Q, R, S, pad_h, pad_w, stride_h, stride_w, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14]);
    //Compile
    try{
      std::string src = generator.dump(ctx.device(), "conv");
      drv::Module program(ctx, src);
      drv::Kernel kernel(program, "conv");
      double tsec = bench([&](){ generator.enqueue(kernel, stream, alpha, I, F, beta, O); }, [&](){ stream.synchronize(); }, ctx.device());
      double tflops = sc::templates::Conv::tflops(P,Q,K,N,C,R,S,tsec);
      best = std::max(tflops, best);
      std::cout << "//";
      std::copy(x.begin(), x.end(), std::ostream_iterator<int>(std::cout, " "));
      std::cout << ": " << tflops << " TFLOPS [BEST: " << best << "]" << std::endl;
    }catch(isaac::templates::invalid_parameters const &){
      return;
    }catch(drv::exception::cuda::launch_out_of_resources const &){
      return;
    }
  });
  std::cout << "ISAAC: " << best << std::endl;
}

void search_gemm(int32_t M, int32_t N, int32_t K, sc::IsaacOperation_t AT, sc::IsaacOperation_t BT, sc::DType dtype){
  auto ctx = drv::backend::contexts::get_default();
  size_t dtsize = sc::size_of(dtype);

  // Setup
  size_t ldc = M;
  size_t lda = (AT==sc::ISAAC_OP_N)?M:K;
  size_t ldb = (BT==sc::ISAAC_OP_N)?K:N;
  int32_t offc = 0, offa = 0, offb = 0;
  drv::Buffer C(ctx, M*N*dtsize);
  drv::Buffer A(ctx, M*K*dtsize);
  drv::Buffer B(ctx, K*N*dtsize);
  drv::Stream stream(ctx);
  sc::scalar alpha(1., dtype), beta(0., dtype);

  // Exhaustive search
  std::vector<sc::param_t> r1 = {1};
  std::vector<sc::param_t> rv = {4};
  std::vector<sc::param_t> rr = {1, 2, 4, 8};
  std::vector<sc::param_t> rl = {4, 8, 16, 32};
  std::vector<sc::param_t> rs = {4, 8, 16};
  double best = 0;

  loop_nest<sc::param_t>({rv, rl, rl, rl, rs, r1, rs, rl, rl, rl, rl, r1, rr, rr}, [&](std::vector<sc::param_t> const & x){
    isaac::templates::GEMM generator(dtype, AT, BT, M, N, K, offa, lda, offb, ldb, offc, ldc, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]);
    // Compile
    try{
      std::string src = generator.dump(ctx.device(), "gemm");
      drv::Module program(ctx, src);
      drv::Kernel kernel(program, "gemm");
      double time = bench([&](){ generator.enqueue(kernel, stream, alpha, A, B, beta, C); }, [&](){ stream.synchronize(); }, ctx.device());
      double tflops = 2*1e-3*M*N*K/time;
      best = std::max(tflops, best);
      std::cout << "//";
      std::copy(x.begin(), x.end(), std::ostream_iterator<int>(std::cout, " "));
      std::cout << ": " << tflops << " TFLOPS [BEST: " << best << "]" << std::endl;
    }catch(isaac::templates::invalid_parameters const &){
      return;
    }catch(drv::exception::cuda::launch_out_of_resources const &){
      return;
    }
  });
  std::cout << "ISAAC: " << best << std::endl;
}

/* Helpers for dumping source code */
void dump_source(sc::driver::Device const & device, sc::templates::Generator& generator, opts::Options* options, std::string const & name){
  if(options->get<std::string>("format") == "ptx")
    std::cout << generator.dump(device, name) << std::endl;
  else{
    auto x = generator.tuning_params();
    std::cout << "Tuning parameters: " << std::flush;
    for(size_t i = 0; i < x.size(); ++i)
      std::cout << ((i>0)?", ":"") << x[i] << std::flush;
    std::cout << std::endl;
  }
}

/* Application code */
int main(int argc, char* argv[]){
  opts::Application program("isaac-tools", "Command-line interface for ISAAC");
  // Options
  opts::Options* options = program.options();
  options->add<size_t>("device", "Device to run on", 0);
  options->add<sc::DType>("dtype", "Data-type to use for computations", "float32", {{"float16", sc::HALF_TYPE}, {"float32", sc::FLOAT_TYPE}, {"float64", sc::DOUBLE_TYPE}});
  options->add<std::string>("name", "Name to give to the generated kernel", "kernel");
  opts::Options* gemm = options->add_group("gemm", "Use matrix-multiplication");
  gemm->add("layout", "Transposition layout for A and B", "NT", {"NN", "NT", "TN", "TT"});
  gemm->add<std::vector<size_t>>("shape", "Matrix shapes (M,N,K)", {2048, 2048, 2048}, opts::SizeConstraint(3));
  gemm->add<std::vector<size_t>>("kernel", "Bypass predictive model to use given tuning parameters", opts::SizeConstraint(13));
  opts::Options* conv = options->add_group("conv", "Use convolutions");
  conv->add<std::vector<size_t>>("shape", "Tensor shapes (W, H, C, N, K, R, S, pad_h, pad_w, stride_h, stride_w)", {112, 112, 64, 8, 128, 3, 3, 1, 1, 1, 1}, opts::SizeConstraint(11));
  conv->add<std::vector<size_t>>("kernel", "Bypass predictive model to use given tuning parameters", opts::SizeConstraint(15));
  options->add_group("search", "Exhaustively search for best tuning parameters");
  opts::Options* dump = options->add_group("dump", "Dump source-code generated by ISAAC");
  dump->add("format", "Format to generate", "ptx", {"ptx", "params"});
  dump->add("target", "Target GPU (sm_xx)", {"sm_50", "sm_52", "sm_60", "sm_61", "sm_70"});
  opts::Options* bench = options->add_group("bench", "Benchmark source code generated by ISAAC");
  bench->add("suite", "Benchmarking suite to run", "custom", {"custom", "deepbench"});
  bench->add<std::shared_ptr<Metric>>("metric", "performance metric for the results", "tflops", {{"tflops", std::make_shared<FLOPS>(1e12)}, {"ms", std::make_shared<Time>(1e-3)},  {"us", std::make_shared<Time>(1e-6)}});
  // Constraints
  options->add_constraint(opts::OneOf({"bench", "dump", "search"}));
  options->add_constraint(opts::OneOf({"gemm", "conv"}));
  program.parse(argc, argv);

  if(options->has("bench"))
    std::cout << std::fixed << std::setprecision(2);
  //Device
  sc::driver::Device device = sc::driver::backend::devices()[options->get<size_t>("device")];
  if(options->has("dump") && dump->has("target")){
    std::string target = dump->get<std::string>("target");
    char major = target[3];
    char minor = target[4];
    device.interpret_as(std::make_pair((size_t)std::atoi(&major), (size_t)std::atoi(&minor)));
  }
  static sc::driver::Context context(device);
  sc::driver::Stream stream(context);
  // Data-Type
  sc::DType dtype = options->get<sc::DType>("dtype");
  // Kernel name
  std::string name = options->get<std::string>("name");

  /* Get optimized kernel generator */
  std::unique_ptr<sc::templates::Generator> generator;

  // GEMM
  if(options->has("gemm")){
    std::string layout = gemm->get<std::string>("layout");
    sc::IsaacOperation_t AT = layout[0]=='T'?sc::ISAAC_OP_T:sc::ISAAC_OP_N;
    sc::IsaacOperation_t BT = layout[1]=='T'?sc::ISAAC_OP_T:sc::ISAAC_OP_N;
    auto shape = gemm->get<std::vector<size_t>>("shape");
    size_t M = shape[0], N = shape[1], K = shape[2];
    //Get Source
    size_t ldc = M;
    size_t lda = (AT==sc::ISAAC_OP_N)?M:K;
    size_t ldb = (BT==sc::ISAAC_OP_N)?K:N;
    if(gemm->has("kernel")){
      auto x = gemm->get<std::vector<size_t>>("kernel");
      param_t VEC = x[0], BM = x[1], BN = x[2], MS = x[3], NS = x[4], U = x[5],  BA0 = x[6], BA1 = x[7], BB0 = x[8], BB1 = x[9], KS = x[10], BK = x[11], KG = x[12];
      generator.reset(new sc::templates::GEMM(dtype, AT, BT, M, N, K, 0, lda, 0, ldb, 0, ldc, VEC, BM, U, BN, MS, 1, NS, BA0, BA1, BB0, BB1, KS, BK, KG));
    }
    else{
      sc::runtime::GEMMProfile* profile = (sc::runtime::GEMMProfile*)sc::runtime::database.at({device.architecture(), sc::runtime::GEMM}).get();
      generator.reset(new sc::templates::GEMM(profile->predict(stream, dtype, AT, BT, M, N, K, 0, lda, 0, ldb, 0, ldc)));
    }
    if(options->has("search")){
      search_gemm(M, N, K, AT, BT, dtype);
    }
    if(options->has("dump"))
      dump_source(device, *generator, dump, name);
    if(options->has("bench")){
      auto metric = bench->get<std::shared_ptr<Metric>>("metric");
      print_results_header({"AT", "BT", "M", "N", "K", "ISAAC", "cuBLAS"});
      std::vector<gemm_params_t> shapes;
      //User provided shapes
      if(bench->get<std::string>("suite")=="custom")
        shapes = {std::make_tuple(dtype, AT, BT, M, N, K)};

      //SC17 paper shapes
      if(bench->get<std::string>("suite")=="deepbench")
        shapes = TestBench::gemm(dtype);

      //Print results
      for(auto x: shapes){
        std::tie(dtype, AT, BT, M, N, K) = x;
        benchmark_gemm(*metric, context, device, stream, dtype, AT, BT, M, N, K, gemm->has("kernel")?generator.get():NULL);
      }
    }
  }

  // CONV
  if(options->has("conv")){
    auto x = conv->get<std::vector<size_t>>("shape");
    size_t W = x[0], H = x[1], C = x[2], N = x[3], K = x[4], R = x[5], S = x[6], pad_h = x[7], pad_w = x[8], stride_h = x[9], stride_w = x[10];
    size_t P = (H - R + 1 + 2*pad_h + stride_h - 1)/stride_h;
    size_t Q = (W - S + 1 + 2*pad_w + stride_w - 1)/stride_w;
    if(conv->has("kernel")){
      auto x = conv->get<std::vector<size_t>>("kernel");
      generator.reset(new sc::templates::Conv(dtype, C, H, W, N, K, P, Q, R, S, pad_h, pad_w, stride_h, stride_w, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14]));
    }
    else{
      sc::runtime::ConvProfile* profile = (sc::runtime::ConvProfile*)sc::runtime::database.at({device.architecture(), sc::runtime::CONV}).get();
      generator.reset(new sc::templates::Conv(profile->predict(stream, dtype, C, H, W, N, K, P, Q, R, S, pad_h, pad_w, stride_h, stride_w)));
    }
    if(options->has("search"))
      search_conv(W, H, C, N, K, R, S, pad_h, pad_w, stride_h, stride_w, dtype);
    if(options->has("dump"))
      dump_source(device, *generator, dump, name);
    if(options->has("bench")){
      auto metric = bench->get<std::shared_ptr<Metric>>("metric");
      print_results_header({"N", "K", "P", "Q", "C", "R", "S", "ISAAC", "cuDNN"});
      std::vector<conv_params_t> shapes;
      //User provided shapes
      if(bench->get<std::string>("suite")=="custom")
        shapes = {std::make_tuple(dtype, W, H, C, N, K, R, S, pad_h, pad_w, stride_h, stride_w)};
      //SuperComputing17 shapes
      if(bench->get<std::string>("suite")=="deepbench")
        shapes = TestBench::conv(dtype);
      //Print results
      for(auto x: shapes){
        std::tie(dtype, W, H, C, N, K, R, S, pad_h, pad_w, stride_h, stride_w) = x;
        benchmark_conv(*metric, context, device, stream, dtype, W, H, C, N, K, R, S, pad_h, pad_w, stride_h, stride_w, conv->has("kernel")?generator.get():NULL);
      }
    }
  }
}
