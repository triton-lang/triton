import argparse
import subprocess


def generate_math_libdevice(output):
	pass


def extract_symbols(output):
	for line in output:
		print(line)


parser = argparse.ArgumentParser()
parser.add_argument('--llvm-dis-path', help='path to llvm-dis', default='llvm-dis')
parser.add_argument('--libdevice-path', help='path to libdevice.10.bc',
                    default="/usr/local/cuda/libdevice.10.bc")
args = parser.parse_args()
llvm_dis = args.llvm_dis_path
libdevice = args.libdevice_path
output = subprocess.check_output([llvm_dis, libdevice, '|', 'grep', 'define']).decode().splitlines()
