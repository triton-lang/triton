#include <cstring>
#include <cstdio>
#include "ast/ast.h"
#include "ir/context.h"
#include "ir/module.h"
#include "codegen/selection.h"
#include "codegen/tune.h"
#include "codegen/shared_copy.h"
#include "codegen/allocation.h"
#include "codegen/liveness.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"

typedef struct yy_buffer_state * YY_BUFFER_STATE;
extern int yyparse();
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
using tdl::ast::translation_unit;
extern translation_unit *ast_root;

const char src[] =
"\
void test(fp32 *a, fp32 *b, fp32 *c, int32 M, int32 N, int32 K){\
  int32 rx[16] = get_global_range[16](0);\
  int32 ry[16] = get_global_range[16](1);\
  int32 rka[8] = 0 ... 8;\
  int32 rkb[8] = 0 ... 8;\
  fp32 C[16, 16] = 0;\
  int32 k;\
  fp32* pa[16, 8] = a + rx[:, newaxis] + rka[newaxis, :]*M;\
  fp32* pb[16, 8] = b + ry[:, newaxis] + rkb[newaxis, :]*K;\
  fp32* pc[16, 16];\
  for(k = 0; k < K; k = k + 8){\
    fp32 A[16, 8] = *pa;\
    fp32 B[16, 8] = *pb;\
    C = dot(A, B, C);\
    pa = pa + 8*M;\
    pb = pb + 8*K;\
  }\
  pc = c + rx[:, newaxis] + ry[newaxis, :];\
  *pc = C;\
}\
";

int main() {
   YY_BUFFER_STATE buffer = yy_scan_string(src);
   yyparse();
   yy_delete_buffer(buffer);
   translation_unit *program = ast_root;
   tdl::ir::context context;
   tdl::ir::module module("matrix", context);
   program->codegen(&module);
   llvm::LLVMContext llvm_context;
   llvm::Module llvm_module("test", llvm_context);
   // lowering passes
   tdl::codegen::place_shared_copy shared;
   tdl::codegen::tune tune;
   tdl::codegen::liveness liveness;
   tdl::codegen::allocation allocation(&liveness);
   tune.run(module);
   shared.run(module);
   liveness.run(module);
   allocation.run();
   std::vector<unsigned*> params;
   tune.get_params(module, params);
   std::cout << params.size() << std::endl;
//   selection.run(module, llvm_module);
//   // print LLVM program
//   llvm::PrintModulePass print(llvm::outs());
//   llvm::AnalysisManager<llvm::Module> analysis;
//   print.run(llvm_module, analysis);
   return 0;
}
