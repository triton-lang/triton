#include <cstring>
#include <cstdio>
#include "cuda.h"
#include "llvm/IR/Verifier.h"
#include "triton/ast/ast.h"
#include "triton/ir/context.h"
#include "triton/ir/module.h"
#include "triton/ir/print.h"
#include "triton/ir/context_impl.h"
#include "triton/codegen/selection.h"
#include "triton/codegen/tune.h"
#include "triton/codegen/shared_copy.h"
#include "triton/codegen/allocation.h"
#include "triton/codegen/liveness.h"
#include "triton/codegen/vectorize.h"
#include "triton/codegen/buffer_info.h"
#include "triton/codegen/barriers.h"
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
#include "llvm/Analysis/LoopPass.h"

typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yyparse();
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
using triton::ast::translation_unit;
extern translation_unit *ast_root;

const char src[] =
"\
const tunable int32 TM;\
const tunable int32 TN;\
const tunable int32 TK;\
\
void matmul(restrict readonly fp32 *a, restrict readonly fp32 *b, fp32 *c, int32 M, int32 N, int32 K, int32 bound){\
  int32 rxa[TM] = get_global_range[TM](0);\
  int32 ryb[TN] = get_global_range[TN](1);\
  int32 rka[TK] = 0 ... TK;\
  int32 rkb[TK] = 0 ... TK;\
  int32 rxc[TM] = get_global_range[TM](0);\
  int32 ryc[TN] = get_global_range[TN](1);\
  fp32 C[TM, TN] = 0;\
  int32 k;\
  fp32* pa[TM, TK] = a + rxa[:, newaxis] + rka[newaxis, :]*M;\
  fp32* pb[TN, TK] = b + ryb[:, newaxis] + rkb[newaxis, :]*K;\
  fp32* pc[TM, TN] = c + rxc[:, newaxis] + ryc[newaxis, :]*M;\
  fp32 a[TM, TK] = *pa;\
  fp32 b[TN, TK] = *pb;\
  int1 checkc0[TM];\
  int1 checkc1[TN];\
  int1 checkc[TM, TN];\
   for(k = K; k > 0; k = k - TK){\
     int1 checka[TM, TK] = (k > bound);\
     int1 checkb[TN, TK] = (k > bound);\
     int1 checka0[TM];\
     int1 checka1[TK];\
     int1 checkb0[TN];\
     int1 checkb1[TK];\
     C = dot(a, b, C);\
     pa = pa + TK*M;\
     pb = pb + TK*K;\
     @checka a = *pa;\
     @checkb b = *pb;\
     if(k > bound)\
       continue;\
     checka0 = rxa < M;\
     checka1 = rka < k;\
     checkb0 = ryb < N;\
     checkb1 = rkb < k;\
     checka = checka0[:, newaxis] && checka1[newaxis, :];\
     checkb = checkb0[:, newaxis] && checkb1[newaxis, :];\
     @checka a = *pa;\
     @checkb b = *pb;\
   }\
  checkc0 = rxc < M;\
  checkc1 = ryc < N;\
  checkc = checkc0[:, newaxis] && checkc1[newaxis, :];\
  @checkc *pc = C;\
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

int main() {
  // create AST from Triton-C source
  YY_BUFFER_STATE buffer = yy_scan_string(src);
  yyparse();
  yy_delete_buffer(buffer);
  translation_unit *program = ast_root;

  // create Triton-IR from AST
  triton::ir::context context;
  triton::ir::module module("matrix", context);
  program->codegen(&module);
  llvm::LLVMContext llvm_context;
  llvm::Module llvm_module("matmul", llvm_context);



  // create passes
  triton::codegen::buffer_info_pass buffer_info;
  triton::codegen::place_shared_copy shared(&buffer_info);
  triton::codegen::tune tune;
  triton::codegen::liveness liveness(&buffer_info);
  triton::codegen::allocation allocation(&liveness, &buffer_info);
  triton::codegen::barriers barriers(&allocation, &buffer_info);
  triton::codegen::vectorize vectorize(&tune);
  triton::codegen::selection selection(&allocation, &tune, &buffer_info);

  // tuning parameters
  tune.run(module);
  std::vector<unsigned> params = {
    // shapes
    16, 16, 8,
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

  // meta-parameters
  unsigned i = 0;
  context.p_impl->mp_constants_[0]->set_value(params[0]);
  context.p_impl->mp_constants_[1]->set_value(params[1]);
  context.p_impl->mp_constants_[2]->set_value(params[2]);
  for(unsigned *x: tune.get_params(module))
    *x = params[3 + i++];
  // constraints
  std::map<triton::ir::value*, std::vector<std::string>> errors;
  tune.check_constraints(module, errors);
  std::cout << "errors: " << errors.size() << std::endl;
  for(auto &x: errors){
  for(auto &e: x.second)
    std::cout << e << std::endl;
  }
  if(errors.size())
    exit(EXIT_FAILURE);


  // run passes
  buffer_info.run(module);
  shared.run(module);
  liveness.run(module);
  allocation.run();
  barriers.run(module);
  vectorize.run(module);
  triton::ir::print(module, std::cout);
  selection.run(module, llvm_module);

  // llvm source
  llvm::legacy::PassManager manager;
//  manager.add(llvm::createPrintModulePass(llvm::outs()));
  manager.add(llvm::createVerifierPass(true));
  manager.run(llvm_module);

  std::string src = generate_machine_code(llvm_module, "nvptx64-nvidia-cuda", compute_data_layout(true, true));
//  std::cout << src << std::endl;

  // compile machine code
  CUdevice   cu_device;
  CUcontext  cu_context;
  CUmodule   cu_module;
  CUfunction cu_kernel;
  CUstream cu_stream;
  int major, minor;
  compile_machine_code(cu_device, cu_context, cu_module, cu_kernel, cu_stream, major, minor, src, "matmul");

  // execute machine code
  // Allocate buffers
  typedef float numeric_t;
  size_t M = 128, N = 128, K = 128;
  size_t bound = 8;
  std::vector<numeric_t> c(M*N);
  std::vector<numeric_t> rc(M*N);
  std::vector<numeric_t> a(M*K);
  std::vector<numeric_t> b(K*N);
  srand(0);
  for(size_t i = 0; i < a.size(); i++)
    a[i] = 1;
  for(size_t i = 0; i < b.size(); i++)
    b[i] = 1;
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
  void *args[] = { &d_a, &d_b, &d_c, &M, &N, &K, &bound};
  int num_regs;
  cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, cu_kernel);
  unsigned TM = context.p_impl->mp_constants_[0]->get_value();
  unsigned TN = context.p_impl->mp_constants_[1]->get_value();
  unsigned nthreads = params[10]*params[13]*params[11]*params[14];
  checkCudaErrors(cuLaunchKernel(cu_kernel, (M + TM - 1)/TM, (N + TN - 1)/TN, 1, nthreads, 1, 1, 0, cu_stream, args, NULL));
  checkCudaErrors(cuStreamSynchronize(cu_stream));
  // Write back
  checkCudaErrors(cuMemcpyDtoH(c.data(), d_c, sizeof(numeric_t) * c.size()));
  simple_gemm(rc, a, b, M, N, K);
  for(size_t i = 0; i < M*N; i++)
    if(std::abs(c[i] - rc[i])/std::max(c[i], rc[i]) > 1e-4){
      std::cout << i << " " << c[i] << " " << rc[i] << std::endl;
      exit(EXIT_FAILURE);
    }
  std::cout << "Pass!" << std::endl;
}
