#include <iostream>
#include <memory>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>


#include "triton/ast/ast.hpp"
#include "triton/ast/llvm/llvm.hpp"

int main() {
    
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::LLVMContext context;


    triton::ast::SharedAbstractNode node = triton::ast::bvadd(
        triton::ast::bv(10, 32),
        triton::ast::bv(20, 32)
    );

    
    triton::ast::TritonToLLVM lifter(context);
    std::shared_ptr<llvm::Module> module = lifter.convert(node);

    
    if (llvm::verifyModule(*module, &llvm::errs())) {
        std::cerr << "Error: Module verification failed." << std::endl;
        return 1;
    }

    
    std::string errStr;
    llvm::ExecutionEngine* engine = llvm::EngineBuilder(std::move(module))
        .setErrorStr(&errStr)
        .setEngineKind(llvm::EngineKind::JIT)
        .create();

    if (!engine) {
        std::cerr << "Failed to create ExecutionEngine: " << errStr << std::endl;
        return 1;
    }

    
    llvm::Function* func = engine->FindFunctionNamed("__triton");
    if (!func) {
        std::cerr << "Function '__triton' not found in module." << std::endl;
        return 1;
    }

    
    llvm::GenericValue result = engine->runFunction(func, {});
    std::cout << "Result: " << result.IntVal << std::endl;

    // Clean up
    delete engine;
    return 0;
}