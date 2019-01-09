#include <cstring>
#include <cstdio>
#include "ast/ast.h"
#include "ir/context.h"
#include "ir/module.h"
#include "codegen/selection.h"
#include "codegen/tune.h"
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
void test(fp32 *A, fp32 *B, fp32 *C, int32 M, int32 N, int32 K){\
  int32 rx[16] = get_global_range[16](0);\
  int32 ry[16] = get_global_range[16](1);\
  int32 rk[8]  = 0 ... 8;\
  fp32 acc[16, 16] = 0;\
  fp32 *pa[16, 8] = A + rx[:,newaxis] + rk[newaxis,:]*M;\
  fp32 *pb[16, 8] = B + ry[:,newaxis] + rk[newaxis,:]*K;\
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
   tdl::codegen::selection selection;
   tdl::codegen::tune tune;
   tune.run(module);
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
