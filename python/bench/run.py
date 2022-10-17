import argparse
import inspect
import os
import sys

import triton


def run_all(result_dir, names):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for mod in os.listdir(os.path.dirname(os.path.realpath(__file__))):
        # skip non python files
        if not mod.endswith('.py'):
            continue
        # skip file not in provided names
        if names and names not in mod:
            continue
        # skip files that don't start with 'bench_'
        if not mod.startswith('bench_'):
            continue
        print(f'running {mod}...')
        mod = __import__(os.path.splitext(mod)[0])
        benchmarks = inspect.getmembers(mod, lambda x: isinstance(x, triton.testing.Mark))
        for name, bench in benchmarks:
            curr_dir = os.path.join(result_dir, mod.__name__.replace('bench_', ''))
            if len(benchmarks) > 1:
                curr_dir = os.path.join(curr_dir, name.replace('bench_', ''))
            if not os.path.exists(curr_dir):
                os.makedirs(curr_dir)
            bench.run(save_path=curr_dir)


def main(args):
    parser = argparse.ArgumentParser(description="Run the benchmark suite.")
    parser.add_argument("-r", "--result-dir", type=str, default='results', required=False)
    parser.add_argument("-n", "--names", type=str, default='', required=False)
    parser.set_defaults(feature=False)
    args = parser.parse_args(args)
    run_all(args.result_dir, args.names)


if __name__ == '__main__':
    main(sys.argv[1:])
