#pragma once

#ifndef _PRINT_IR_H_
#define _PRINT_IR_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Module.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "triton/ir/module.h"
#include "triton/ir/print.h"
#include <iomanip>

#define PRINT_CURRENT_FUNCTION() std::cout << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << std::endl;

static int print_count = 0;

inline std::string return_current_time_and_date()
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d--%I-%M-%S");
    return ss.str();
}

template <typename T>
inline void print_vector(std::vector<T> &vec, std::string name = "")
{
    std::cout << name << ": ";
    for (auto v : vec)
    {
        std::cout << v << ", ";
    }

    std::cout << '\b';
    std::cout << std::endl;
}

// dump llvm ir to tmp file
inline void write_llvm_ir(llvm::Module &llvm_module, std::string filename = "", bool tracked = false)
{

    if (filename.empty())
        filename = llvm_module.getModuleIdentifier();

    std::string count_str = "";
    if (tracked)
    {
        count_str = "_" + std::to_string(print_count);
    }

    std::string ir_path = std::string("/tmp/") + filename + count_str + std::string(".ll");
    std::error_code ec;
    std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
        new llvm::raw_fd_ostream(ir_path, ec));
    llvm_module.print(*ir_fs, nullptr);
    ir_fs->flush();
    if (tracked)
    {
        print_count += 1;
    }
}

inline void print_triton_ir(triton::ir::module ir_ref, std::string name)
{
    std::ofstream ir_out(std::string("/tmp/") + name + std::string("_") + return_current_time_and_date() + std::string(".ttir"));
    ir_out.flush();
    triton::ir::print(ir_ref, ir_out);
    ir_out.close();
}

inline void print_triton_ir(std::string ir_ref, std::string name)
{
    std::ofstream ir_out(std::string("/tmp/") + name + std::string("_") + return_current_time_and_date() + std::string(".ttir"));
    ir_out.flush();
    ir_out << ir_ref << std::endl;
    ir_out.close();
}

inline std::string get_llvm_value_as_str(llvm::Value *llvm_value)
{
    std::string value_str;
    llvm::raw_string_ostream rso(value_str);
    llvm_value->print(rso);
    return rso.str();
}

inline void print_llvm_value(llvm::Value *llvm_value, std::string name = "")
{
    if (llvm_value)
        std::cout << "\t" << name << ": " << get_llvm_value_as_str(llvm_value) << std::endl;
    else
        std::cout << "\t" << name << ": " << "is nullptr" << std::endl;
}

inline void print_llvm_type(llvm::Type *llvm_type, std::string name = "")
{
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    llvm_type->print(rso);
    std::cout << name << " type: " << rso.str() << std::endl;
}

inline void print_llvm_value_type(llvm::Value *llvm_value, std::string name = "")
{
    print_llvm_type(llvm_value->getType(), name);
}

inline void write_ptx(std::string ptx_str)
{
    std::ofstream file("/tmp/kernel.ptx");
    file << ptx_str;
}
#endif