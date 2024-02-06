import argparse
import re
import sys
from pathlib import Path


def parseIRDumps(dumpName):
    # mkdir for the parsed dumps at the dumpName level
    if not isinstance(dumpName, Path):
        dumpName = Path(dumpName)
    parent = dumpName.parent
    parseDir = parent / dumpName.stem
    if not parseDir.exists():
        parseDir.mkdir()

    with open(dumpName, 'r') as input_file:
        file_content = input_file.read()

        # Use regex to find all occurrences of "IR Dump" and the content between them
        filenames = re.findall(r'IR Dump(.*?)\(', file_content)
        ir_dump_sections = re.findall(r'(^\/\/ -----\/\/ IR Dump.*?)\n\n', file_content, re.DOTALL|re.MULTILINE)
        # Check if the regex finds match of title and content
        if len(filenames) != len(ir_dump_sections):
            print(f"{len(filenames)} lines of IR Dump header found but only {len(ir_dump_sections)} bodies parsed")
            actual_len = min(len(filenames), len(ir_dump_sections))
            filenames = filenames[:actual_len]
            ir_dump_sections = ir_dump_sections[:actual_len]

        # write them out to individual files
        for i, (fname, dump) in enumerate(zip(filenames, ir_dump_sections)):
            subFileName = f"{i:02d}_" + fname.strip().replace(" ", "_") + ".mlir"
            subFileName =  parseDir / subFileName
            print(subFileName)
            with open(subFileName, 'w') as f:
                f.write(dump)
        
        # dumb way to extract LLVM IR
        numFiles = len(filenames)
        amdgcnDump = re.findall(r'(\/\/ -----\/\/ LLVM IR Dump.*DILocation.*?\n\n)', file_content, re.DOTALL|re.MULTILINE)
        subFileName = f"{len(filenames):02d}_" + "LLVM_IR" + ".mlir"
        subFileName =  parseDir / subFileName
        if not len(amdgcnDump):
            print("No LLVM IR")
            return
        else:
            numFiles += 1
        print(subFileName)
        with open(subFileName, 'w') as f:
            f.write(amdgcnDump[0])

        # dumb way to extract AMDGCN
        amdgcnDump = re.findall(r'(\/\/ -----\/\/ AMDGCN.*\.Lline_table_start0:)', file_content, re.DOTALL|re.MULTILINE)
        subFileName = f"{numFiles:02d}_" + "AMDGCN" + ".mlir"
        subFileName =  parseDir / subFileName
        if not len(amdgcnDump):
            print("No AMDGCN")
            return
        print(subFileName)
        with open(subFileName, 'w') as f:
            f.write(amdgcnDump[0])


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Parse Triton IR dump; no assembly for now",
        # allow_abbrev=False,
    )

    parser.add_argument("name", default='tmp.txt', help="provide the dump text file name")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    dumpName = args.name
    parseIRDumps(dumpName)


if __name__ == "__main__":
    sys.exit(main())