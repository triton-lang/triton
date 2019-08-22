#include "triton/lang/wgtcc/code_gen.h"
#include "triton/lang/wgtcc/cpp.h"
#include "triton/lang/wgtcc/error.h"
#include "triton/lang/wgtcc/parser.h"
#include "triton/lang/wgtcc/scanner.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>


std::string program;
std::string filename_in;
std::string filename_out;
bool debug = false;
static bool only_preprocess = false;
static bool only_compile = false;
static bool specified_out_name = false;
static std::list<std::string> filenames_in;
static std::list<std::string> gcc_filenames_in;
static std::list<std::string> gcc_args;
static std::list<std::string> defines;
static std::list<std::string> include_paths;
