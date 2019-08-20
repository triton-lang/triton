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


static void Usage() {
  printf("Usage: wgtcc [options] file...\n"
       "Options: \n"
       "  -h        Display this information\n"
       "  -D        Define object like macro\n"
       "  -I        Add search path\n"
       "  -E        Preprocess only; do not compile, assemble or link\n"
       "  -S        Compile only; do not assemble or link\n"
       "  -o        specify output file\n");

  exit(0);
}


static std::string GetExtension(const std::string& filename) {
  return filename.substr(filename.size() >= 2 ? filename.size() - 2 : 0);
}


static void ValidateFileName(const std::string& filename) {
  auto ext = GetExtension(filename);
  if (ext != ".c" && ext != ".s" && ext != ".o" && ext != ".a")
    Error("bad file name format:'%s'", filename.c_str());
}


static void DefineMacro(Preprocessor& cpp, const std::string& def) {
  auto pos = def.find('=');
  std::string macro;
  std::string* replace;
  if (pos == std::string::npos) {
    macro = def;
    replace = new std::string();
  } else {
    macro = def.substr(0, pos);
    replace = new std::string(def.substr(pos + 1));
  }
  cpp.AddMacro(macro, replace);
}


static std::string GetName(const std::string& path) {
  auto pos = path.rfind('/');
  if (pos == std::string::npos)
    return path;
  return path.substr(pos + 1);
}

static int RunWgtcc() {
  if (GetExtension(filename_in) != ".c")
    return -3;

  Preprocessor cpp(&filename_in);
  for (auto& def: defines)
    DefineMacro(cpp, def);
  for (auto& path: include_paths)
    cpp.AddSearchPath(path);

  FILE* fp = stdout;
  if (specified_out_name) {
    fp = fopen(filename_out.c_str(), "w");
  }
  TokenSequence ts;
  cpp.Process(ts);
  if (only_preprocess) {
    ts.Print(fp);
    return 0;
  }

  if (!only_compile || !specified_out_name) {
    filename_out = GetName(filename_in);
    filename_out.back() = 's';
  }
  fp = fopen(filename_out.c_str(), "w");

  Parser parser(ts);
  parser.Parse();
  Generator::SetInOut(&parser, fp);
  Generator().Gen();
  fclose(fp);
  return 0;
}


static int RunGcc() {
  // Froce C11
  bool spec_std = false;
  for (auto& arg: gcc_args) {
    if (arg.substr(0, 4) == "-std") {
      arg = "-std=c11";
      spec_std = true;
    }
  }
  if (!spec_std) {
    gcc_args.push_front("-std=c11");
  }

  std::string systemArg = "gcc";
  for (const auto& arg: gcc_args) {
    systemArg += " " + arg;
  }
  auto ret = system(systemArg.c_str());
  return ret;
}


static void ParseInclude(int argc, char* argv[], int& i) {
  if (argv[i][2]) {
    include_paths.push_front(&argv[i][2]);
    return;
  }

  if (i == argc - 1) {
    Error("missing argument to '%s'", argv[i]);
  }
  include_paths.push_front(argv[++i]);
  gcc_args.push_back(argv[i]);
}


static void ParseDefine(int argc, char* argv[], int& i) {
  if (argv[i][2]) {
    defines.push_back(&argv[i][2]);
    return;
  }

  if (i == argc - 1)
    Error("missing argument to '%s'", argv[i]);
  defines.push_back(argv[++i]);
  gcc_args.push_back(argv[i]);
}


static void ParseOut(int argc, char* argv[], int& i) {
  if (i == argc - 1)
    Error("missing argument to '%s'", argv[i]);
  filename_out = argv[++i];
  gcc_args.push_back(argv[i]);
}


/* Use:
 *   wgtcc: compile
 *   gcc: assemble and link
 * Allowing multi file may not be a good idea...
 */
int main(int argc, char* argv[]) {
  if (argc < 2)
    Usage();

  program = std::string(argv[0]);
  for (auto i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') {
      filename_in = std::string(argv[i]);
      ValidateFileName(filename_in);
      filenames_in.push_back(filename_in);
      continue;
    }

    gcc_args.push_back(argv[i]);
    switch (argv[i][1]) {
    case 'h': Usage(); break;
    case 'E': only_preprocess = true; break;
    case 'S': only_compile = true; break;
    case 'I': ParseInclude(argc, argv, i); break;
    case 'D': ParseDefine(argc, argv, i); break;
    case 'o':
      specified_out_name = true;
      ParseOut(argc, argv, i); break;
    case 'g': gcc_args.pop_back(); debug = true; break;
    default:;
    }
  }

#ifdef DEBUG
  RunWgtcc();
#else
  for (const auto& filename: filenames_in) {
    filename_in = filename;
    pid_t pid = fork();
    if (pid < 0) {
      Error("fork error");
    } else if (pid == 0) {
      // Do work in child process
      return RunWgtcc();
    }
  }

  for (size_t i = 0; i < filenames_in.size(); ++i) {
      int stat;
      wait(&stat);
      // Child process terminate normaly if :
      // 1. terminate with `exit()`, that is, WIFEXITED(stat) if true.
      // 2. the status code is 0, that is, WEXITSTATUS(stat) == 0
      if (!WIFEXITED(stat) || WEXITSTATUS(stat))
        return 0;
  }
#endif

  if (only_preprocess || only_compile) {
    if (specified_out_name && filenames_in.size() > 1)
      Error("cannot specifier output filename with multiple input file");
    return 0;
  }

  std::list<std::string> filenames_out;
  for (auto& filename: filenames_in) {
    if (GetExtension(filename) == ".c") {
      gcc_args.push_back(GetName(filename));
      gcc_args.back().back() = 's';
    } else {
      gcc_args.clear();
      for (int i = 1; i < argc; ++i)
        gcc_args.push_back(argv[i]);
      break;
    }
  }
  auto ret = RunGcc();
  remove(filename_out.c_str());
  return ret;
}
