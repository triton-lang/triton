'''
Compare cached triton kernels in 2 directories.

example:
python compare_asm.py --dir0=triton-works/ --dir1=triton-fails/ --asm=ttir \
       --diff-out0=diff-works.ll --diff-out1=diff-fails.ll
'''
import argparse
import os
import pickle

parser = argparse.ArgumentParser(description="unpickle")
parser.add_argument('--dir0', dest='dir0', required=True,
                    help="Triton cache dir 0")
parser.add_argument('--dir1', dest='dir1', required=True,
                    help="Triton cache dir 1")
parser.add_argument('--asm', dest='asm',
                    choices=['ttir', 'llir', 'ptx', 'cubin'], required=True)
parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                    help="Stop after first diff")
parser.set_defaults(early_stop=True)
parser.add_argument('--diff-out0', dest='diff_out0', required=True,
                    help="output file path for kernels in dir0")
parser.add_argument('--diff-out1', dest='diff_out1', required=True,
                    help="output file path for kernels in dir1")
args = parser.parse_args()
dir0 = args.dir0
dir1 = args.dir1
asm = args.asm

dir0_files = {}
dir1_files = {}
for root, _, files in os.walk(dir0):
    for file in files:
        if not file.endswith('.lock'):
            path = os.path.join(root, file)
            with open(path, 'rb') as f:
                loaded_file = pickle.load(f)
                bin = loaded_file['binary']
                key = loaded_file['key']
                info = key.split('-')[-3:]  # num_warps, num_stages, signature
                dict_key = bin.name + '-'.join(info)
                dir0_files[dict_key] = bin.asm

for root, _, files in os.walk(dir1):
    for file in files:
        if not file.endswith('.lock'):
            path = os.path.join(root, file)
            with open(path, 'rb') as f:
                loaded_file = pickle.load(f)
                bin = loaded_file['binary']
                key = loaded_file['key']
                info = key.split('-')[-3:]  # num_warps, num_stages, signature
                dict_key = bin.name + '-'.join(info)
                dir1_files[dict_key] = bin.asm

diff_keys = []
for key in dir0_files:
    asm0 = dir0_files[key]
    if key not in dir1_files:
        continue
    asm1 = dir1_files[key]
    if asm0[asm] != asm1[asm]:
        diff_keys.append(key)

if args.early_stops:
    diff_keys = diff_keys[:1]
if diff_keys:
    with open(args.diff_out0, 'w') as f0, open(args.diff_out1, 'w') as f1:
        for key in diff_keys:
            f0.write(f'{asm} mismatch at {key}')
            f0.write(dir0_files[key][asm])
            f0.write('\n')
            f1.write(f'{asm} mismatch at {key}')
            f1.write(dir1_files[key][asm])
            f1.write('\n')
