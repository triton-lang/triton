// Umar Arshad
// Copyright 2014


#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>

using namespace std;
typedef map<string, string> opt_t;

static
void print_usage() {
    cout << R"delimiter(BIN2CPP
Converts files from a binary file to C++ headers. It is similar to bin2c and
xxd but adds support for namespaces.

| --name        | name of the variable (default: var)                               |
| --file        | input file                                                        |
| --output      | output file (If no output is specified then it prints to stdout   |
| --type        | Type of variable (default: char)                                  |
| --namespace   | A space seperated list of namespaces                              |
| --formatted   | Tabs for formatting                                               |
| --version     | Prints my name                                                    |
| --help        | Prints usage info                                                 |

Example
-------
Command:
./bin2cpp --file blah.txt --namespace blah detail --formatted --name blah_var

Will produce:
#pragma once
#include <cstddef>
namespace blah {
    namespace detail {
        static const char blah_var[] = {
            0x2f,    0x2f,    0x20,    0x62,    0x6c,    0x61,    0x68,    0x2e,    0x74,    0x78,
            0x74,    0xa,    0x62,    0x6c,    0x61,    0x68,    0x20,    0x62,    0x6c,    0x61,
            0x68,    0x20,    0x62,    0x6c,    0x61,    0x68,    0xa,    };
        static const size_t blah_var_len = 27;
    }
})delimiter";
        exit(0);
}

static bool formatted;

static
void add_tabs(const int level ){
    if(formatted) {
        for(int i =0; i < level; i++) {
            cout << "\t";
        }
    }
}

static
opt_t
parse_options(const vector<string>& args) {
    opt_t options;

    options["--name"]       = "";
    options["--type"]       = "";
    options["--file"]       = "";
    options["--output"]     = "";
    options["--namespace"]  = "";
    options["--eof"]        = "";

    //Parse Arguments
    string curr_opt;
    bool verbose = false;
    for(auto arg : args) {
        if(arg == "--verbose") {
            verbose = true;
        }
        else if(arg == "--formatted") {
            formatted = true;
        }
        else if(arg == "--version") {
            cout << args[0] << " By Umar Arshad" << endl;
        }
        else if(arg == "--help") {
            print_usage();
        }
        else if(options.find(arg) != options.end()) {
            curr_opt = arg;
        }
        else if(curr_opt.empty()) {
            //cerr << "Invalid Argument: " << arg << endl;
        }
        else {
            if(options[curr_opt] != "") {
                options[curr_opt] += " " + arg;
            }
            else {
                options[curr_opt] += arg;
            }
        }
    }

    if(verbose) {
        for(auto opts : options) {
            cout << get<0>(opts) << " " << get<1>(opts) << endl;
        }
    }
    return options;
}

int main(int argc, const char * const * const argv)
{

    vector<string> args(argv, argv+argc);

    opt_t&& options = parse_options(args);

    //Save default cout buffer. Need this to prevent crash.
    auto bak = cout.rdbuf();
    unique_ptr<ofstream> outfile;

    // Set defaults
    if(options["--name"] == "")     { options["--name"]     = "var"; }
    if(options["--output"] != "")   {
        //redirect stream if output file is specified
        outfile.reset(new ofstream(options["--output"]));
        cout.rdbuf(outfile->rdbuf());
    }

    cout << "#pragma once\n";
    cout << "#include <cstddef>\n"; // defines size_t

    int ns_cnt = 0;
    int level = 0;
    if(options["--namespace"] != "") {
        std::stringstream namespaces(options["--namespace"]);
        string name;
        namespaces >> name;
        do {
            add_tabs(level++);
            cout << "namespace " << name << " { \n";
            ns_cnt++;
            namespaces >> name;
        } while(!namespaces.fail());
    }

    if(options["--type"] == "") {
        options["--type"]     = "char";
    }
    add_tabs(level);
    cout << "static const " << options["--type"] << " " << options["--name"] << "[] = {\n";


    ifstream input(options["--file"]);
    size_t char_cnt = 0;
    add_tabs(++level);
    for(char i; input.get(i);) {
        cout << "0x" << std::hex << static_cast<int>(i) << ",\t";
        char_cnt++;
        if(!(char_cnt % 10)) {
            cout << endl;
            add_tabs(level);
        }
    }

    if (options["--eof"].c_str()[0] == '1') {
        // Add end of file character
        cout << "0x0";
        char_cnt++;
    }

    cout << "};\n";
    add_tabs(--level);
    cout << "static const size_t " << options["--name"] << "_len" << " = " << std::dec << char_cnt << ";\n";

    while(ns_cnt--) {
        add_tabs(--level);
        cout << "}\n";
    }
    cout.rdbuf(bak);
}
