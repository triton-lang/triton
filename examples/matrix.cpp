#include <cstring>
#include <cstdio>
#include "cuda.h"
#include "ast/ast.h"
#include "ir/context.h"
#include "ir/module.h"
#include "codegen/selection.h"
#include "codegen/tune.h"
#include "codegen/shared_copy.h"
#include "codegen/allocation.h"
#include "codegen/liveness.h"
#include "codegen/vectorize.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"

typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yyparse();
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
using tdl::ast::translation_unit;
extern translation_unit *ast_root;

const char src[] =
"\
void test(fp32 *a, fp32 *b, fp32 *c, int32 M, int32 N, int32 K){\
  int32 rxa[16] = get_global_range[16](0);\
  int32 ryb[16] = get_global_range[16](1);\
  int32 rka[8] = 0 ... 8;\
  int32 rkb[8] = 0 ... 8;\
  int32 rxc[16] = get_global_range[16](0);\
  int32 ryc[16] = get_global_range[16](1);\
  fp32 C[16, 16] = 0;\
  int32 k;\
  fp32* pa[16, 8] = a + rxa[:, newaxis] + rka[newaxis, :]*M;\
  fp32* pb[16, 8] = b + ryb[:, newaxis] + rkb[newaxis, :]*K;\
  fp32* pc[16, 16] = c + rxc[:, newaxis] + ryc[newaxis, :]*M;\
  for(k = K; k > 0; k = k - 8){\
    fp32 a[16, 8] = *pa;\
    fp32 b[16, 8] = *pb;\
    C = dot(a, b, C);\
    pa = pa + 8*M;\
    pb = pb + 8*K;\
  }\
  *pc = C;\
}\
";

static std::string compute_data_layout(bool is64Bit, bool UseShortPointers) {
  std::string Ret = "e";
  if (!is64Bit)
    Ret += "-p:32:32";
  else if (UseShortPointers)
    Ret += "-p3:32:32-p4:32:32-p5:32:32";
  Ret += "-i64:64-i128:128-v16:16-v32:32-n16:32:64";
  return Ret;
}

static std::string generate_machine_code(llvm::Module &module, const std::string &target_triple, const std::string &data_layout) {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  module.setTargetTriple(target_triple);
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  llvm::TargetMachine *machine = target->createTargetMachine(module.getTargetTriple(), "sm_52", "",
                                                             llvm::TargetOptions(), llvm::Reloc::Model(),
                                                             llvm::None, llvm::CodeGenOpt::Aggressive);
  module.setDataLayout(data_layout);

  // emit machine code
  llvm::legacy::PassManager pass;
  llvm::SmallVector<char, 0> buffer;
  llvm::raw_svector_ostream stream(buffer);
  machine->addPassesToEmitFile(pass, stream, nullptr, llvm::TargetMachine::CGFT_AssemblyFile);
  pass.run(module);
  std::string src(buffer.begin(), buffer.end());
  return src;
}

static void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

static void compile_machine_code(CUdevice &device, CUcontext &context, CUmodule &module,
                                 CUfunction &function, CUstream &stream, int &major, int &minor,
                                 const std::string &src, const std::string &name) {
  int numDevices;

  // Initialize
  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGetCount(&numDevices));
  checkCudaErrors(cuDeviceGet(&device, 0));
  checkCudaErrors(cuDeviceComputeCapability(&major, &minor, device));
  checkCudaErrors(cuCtxCreate(&context, 0, device));
  checkCudaErrors(cuStreamCreate(&stream, 0));

  // Compile program
  CUjit_option opt[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_ERROR_LOG_BUFFER};
  unsigned int errbufsize = 8096;
  std::string errbuf(errbufsize, 0);
  const void *cpterr = static_cast<const void*>(errbuf.data());
  void *pterr =  const_cast<void*>(cpterr);
  void* optval[] = {(void*)(uintptr_t)errbufsize, pterr};
  int err = cuModuleLoadDataEx(&module, src.data(), 2, opt, optval);
  if(err != CUDA_SUCCESS){
    std::cerr << "Compilation Failed! Log: " << std::endl;
    std::cerr << errbuf << std::endl;
  }

  // Get function
  checkCudaErrors(cuModuleGetFunction(&function, module, name.c_str()));
}

template<class T>
void simple_gemm(std::vector<T> &c, const std::vector<T> &a, const std::vector<T> &b, size_t M, size_t N, size_t K){
  for(size_t m = 0; m < M; m++)
  for(size_t n = 0; n < N; n++){
    T acc = 0;
    for(size_t k = 0; k < K; k++)
      acc += a[m + k*M] * b[n + k*N];
    c[m + n*M] = acc;
  }
}

int main() {
  // create AST from Triton-C source
  YY_BUFFER_STATE buffer = yy_scan_string(src);
  yyparse();
  yy_delete_buffer(buffer);
  translation_unit *program = ast_root;

  // create Triton-IR from AST
  tdl::ir::context context;
  tdl::ir::module module("matrix", context);
  program->codegen(&module);
  llvm::LLVMContext llvm_context;
  llvm::Module llvm_module("test", llvm_context);

  // create passes
  tdl::codegen::place_shared_copy shared;
  tdl::codegen::tune tune;
  tdl::codegen::liveness liveness;
  tdl::codegen::allocation allocation(&liveness);
  tdl::codegen::vectorize vectorize(&tune);
  tdl::codegen::selection selection(&allocation, &tune);

  // tuning parameters
  tune.run(module);
  std::vector<unsigned> params = {
    // a0
    2, 8, 1,
    // b0
    4, 4, 1,
    // c0
    2, 8, 1,
    // c1
    4, 4, 1,
    // a1
    2, 4, 1,
    // b1
    1, 8, 1
  };
  std::map<tdl::ir::value*, std::vector<std::string>> errors;
  unsigned i = 0;
  std::cout << tune.get_params(module).size() << std::endl;
  for(unsigned *x: tune.get_params(module))
    *x = params[i++];
  tune.check_constraints(module, errors);
  std::cout << "errors: " << errors.size() << std::endl;
  for(auto &x: errors){
  for(auto &e: x.second)
    std::cout << e << std::endl;
  }

  // run passes
  shared.run(module);
  liveness.run(module);
  allocation.run();
  vectorize.run(module);
  selection.run(module, llvm_module);

  // llvm source
  llvm::PrintModulePass print(llvm::outs());
  llvm::AnalysisManager<llvm::Module> analysis;
  print.run(llvm_module, analysis);

  // generate machine code
  std::string src = generate_machine_code(llvm_module, "nvptx64-nvidia-cuda", compute_data_layout(true, true));
//  std::cout << src << std::endl;

  // compile machine code
  CUdevice   cu_device;
  CUcontext  cu_context;
  CUmodule   cu_module;
  CUfunction cu_kernel;
  CUstream cu_stream;
  int major, minor;
  compile_machine_code(cu_device, cu_context, cu_module, cu_kernel, cu_stream, major, minor, src, "test");
  std::cout << src << std::endl;

  // execute machine code
  // Allocate buffers
  typedef float numeric_t;
  size_t M = 32, N = 32, K = 32;
  std::vector<numeric_t> c(M*N);
  std::vector<numeric_t> rc(M*N);
  std::vector<numeric_t> a(M*K);
  std::vector<numeric_t> b(K*N);
  for(size_t i = 0; i < a.size(); i++)
    a[i] = (float)rand() / RAND_MAX;
  for(size_t i = 0; i < b.size(); i++)
    b[i] = (float)rand() / RAND_MAX;
  for(size_t i = 0; i < c.size(); i++)
    c[i] = 0;
  CUdeviceptr d_a, d_b, d_c;
  checkCudaErrors(cuMemAlloc(&d_a, sizeof(numeric_t) * a.size()));
  checkCudaErrors(cuMemAlloc(&d_b, sizeof(numeric_t) * b.size()));
  checkCudaErrors(cuMemAlloc(&d_c, sizeof(numeric_t) * c.size()));
  // Copy buffers
  checkCudaErrors(cuMemcpyHtoD(d_a, a.data(), sizeof(numeric_t) * a.size()));
  checkCudaErrors(cuMemcpyHtoD(d_b, b.data(), sizeof(numeric_t) * b.size()));
  checkCudaErrors(cuMemcpyHtoD(d_c, c.data(), sizeof(numeric_t) * c.size()));
  // Launch kernel
  void *args[] = { &d_a, &d_b, &d_c, &M, &N, &K};
  int num_regs;
  cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, cu_kernel);
  unsigned TM = 16;
  unsigned TN = 16;
  unsigned nthreads = 32;
  checkCudaErrors(cuLaunchKernel(cu_kernel, M/TM, N/TN, 1, nthreads, 1, 1, 0, cu_stream, args, NULL));
  checkCudaErrors(cuStreamSynchronize(cu_stream));
  // Write back
  checkCudaErrors(cuMemcpyDtoH(c.data(), d_c, sizeof(numeric_t) * c.size()));
  simple_gemm(rc, a, b, M, N, K);
  for(size_t i = 0; i < M*N; i++)
    if(std::abs(c[i] - rc[i])/std::max(c[i], rc[i]) > 1e-4)
      std::cout << i << " " << c[i] << " " << rc[i] << std::endl;
}
