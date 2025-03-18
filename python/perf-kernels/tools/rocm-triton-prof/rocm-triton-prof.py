#!/usr/bin/python3

import argparse
import os
import pandas as pd
import yaml
import subprocess
import shutil
import re
from collections import OrderedDict


def get_perf_metrics():
    pmc0 = OrderedDict()
    pmc0['SQ_INSTS_VALU_ADD_F16'] = {'value': 0, 'factor': 64, 'flop': 0}
    pmc0['SQ_INSTS_VALU_MUL_F16'] = {'value': 0, 'factor': 64, 'flop': 0}
    pmc0['SQ_INSTS_VALU_FMA_F16'] = {'value': 0, 'factor': 128, 'flop': 0}
    pmc0['SQ_INSTS_VALU_TRANS_F16'] = {'value': 0, 'factor': 64, 'flop': 0}

    pmc1 = OrderedDict()
    pmc1['SQ_INSTS_VALU_ADD_F32'] = {'value': 0, 'factor': 64, 'flop': 0}
    pmc1['SQ_INSTS_VALU_MUL_F32'] = {'value': 0, 'factor': 64, 'flop': 0}
    pmc1['SQ_INSTS_VALU_FMA_F32'] = {'value': 0, 'factor': 128, 'flop': 0}
    pmc1['SQ_INSTS_VALU_TRANS_F32'] = {'value': 0, 'factor': 64, 'flop': 0}

    pmc2 = OrderedDict()
    pmc2['SQ_INSTS_VALU_ADD_F64'] = {'value': 0, 'factor': 64, 'flop': 0}
    pmc2['SQ_INSTS_VALU_MUL_F64'] = {'value': 0, 'factor': 64, 'flop': 0}
    pmc2['SQ_INSTS_VALU_FMA_F64'] = {'value': 0, 'factor': 128, 'flop': 0}
    pmc2['SQ_INSTS_VALU_TRANS_F64'] = {'value': 0, 'factor': 64, 'flop': 0}

    pmc3 = OrderedDict()
    pmc3['SQ_INSTS_VALU_MFMA_MOPS_F16'] = {'value': 0, 'factor': 512, 'flop': 0}
    pmc3['SQ_INSTS_VALU_MFMA_MOPS_BF16'] = {'value': 0, 'factor': 512, 'flop': 0}
    pmc3['SQ_INSTS_VALU_MFMA_MOPS_F32'] = {'value': 0, 'factor': 512, 'flop': 0}
    pmc3['SQ_INSTS_VALU_MFMA_MOPS_F64'] = {'value': 0, 'factor': 512, 'flop': 0}

    pmc4 = OrderedDict()
    pmc4['GRBM_COUNT'] = {'value': 0}
    pmc4['TCC_HIT_sum'] = {'value': 0}
    pmc4['TCC_MISS_sum'] = {'value': 0}

    jobs = OrderedDict()
    jobs[0] = pmc0
    jobs[1] = pmc1
    jobs[2] = pmc2
    jobs[3] = pmc3
    jobs[4] = pmc4
    return jobs


def get_metrics_as_yaml():
    perf_metrics = get_perf_metrics()
    pmcs = [
        {'pmc': list(perf_metrics[0].keys())},
        {'pmc': list(perf_metrics[1].keys())},
        {'pmc': list(perf_metrics[2].keys())},
        {'pmc': list(perf_metrics[3].keys())},
        {'pmc': list(perf_metrics[4].keys())},
    ]

    spec = {}
    spec['jobs'] = pmcs

    spec_str = yaml.dump(spec)
    return spec_str


def run_external_binary(binary_path, arguments=[], verbose=False):
    try:
        # Run the external binary and capture its standard output
        cmd = [binary_path] + arguments if binary_path else arguments
        if verbose:
            print(f"CURR.CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check if the process was successful
        if result.returncode == 0:
            if result.stderr:
                print(result.stderr.strip())
            return result.stdout.strip()
        else:
            cmd = ' '.join(cmd)
            raise RuntimeError(f'Error: The external binary returned non-zero exit code {result.returncode}. '
                               f'Attempted command:\n{cmd}')
    except FileNotFoundError:
        raise RuntimeError(f'Error: The binary could not be found - i.e., {binary_path}')
    except Exception as err:
        raise RuntimeError(f'Error: {str(err)}')


def check_rocprofv3():
    run_external_binary('which', ['rocprofv3'])


def find_file(rootdir, regex):
    for root, _, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                return os.path.join(root, file)


def filter(df, name):
    return df[df['Kernel_Name'] == name]


def process_files(metrics_dir, timing_dir, kernel_name, verbose):
    timing_file = find_file(timing_dir, re.compile(r'.*kernel_trace.csv'))
    df = pd.read_csv(timing_file)
    df = filter(df, kernel_name)
    timing = df['End_Timestamp'] - df['Start_Timestamp']
    print('Timing info in `nsec`:')
    print(timing.describe())
    print()

    # post process all passes
    num_flop_sum = 0
    perf_metrics = get_perf_metrics()
    num_passes = 5
    metrics_file_regex = re.compile(r'.*counter_collection.csv')
    for pass_id in range(1, num_passes + 1):
        search_dir = os.path.join(metrics_dir, f'pass_{pass_id}')
        metrics_file = find_file(search_dir, metrics_file_regex)
        df = pd.read_csv(metrics_file)
        df = filter(df, kernel_name)

        curr_metrics = perf_metrics[pass_id - 1]
        curr_metrics_names = list(curr_metrics.keys())
        for name in curr_metrics_names:
            data = df[df['Counter_Name'] == name]
            value = data['Counter_Value'].mean()

            if 'flop' in curr_metrics[name].keys():
                curr_metrics[name]['value'] = value
                num_flops = value * curr_metrics[name]['factor']

                num_flop_sum += num_flops
                curr_metrics[name]['flop'] = num_flops
            else:
                curr_metrics[name]['value'] = value
    print()

    # Print data from non-flop-passes
    print('NON-FLOP related data:')
    table = {'Counter Name': [], 'Max': [], 'Min': [], 'Mean': [], 'Median': []}
    non_flop_passes = [5]
    for pass_id in non_flop_passes:
        search_dir = os.path.join(metrics_dir, f'pass_{pass_id}')
        metrics_file = find_file(search_dir, metrics_file_regex)
        df = pd.read_csv(metrics_file)
        df = filter(df, kernel_name)

        curr_metrics = perf_metrics[pass_id - 1]
        curr_metrics_names = list(curr_metrics.keys())
        for name in curr_metrics_names:
            data = df[df['Counter_Name'] == name]
            values = data['Counter_Value']
            table['Counter Name'].append(name)
            table['Max'].append(values.max())
            table['Min'].append(values.min())
            table['Mean'].append(values.mean())
            table['Median'].append(values.median())
    print(pd.DataFrame(table))
    print()

    # Print data from flop-passes
    print('FLOP related data:')
    table = {'Counter Name': [], 'Raw Data': [], 'FLOP': [], 'Relative FLOP, %': []}
    flop_passes = [1, 2, 3, 4]
    for pass_id in flop_passes:
        search_dir = os.path.join(metrics_dir, f'pass_{pass_id}')
        metrics_file = find_file(search_dir, metrics_file_regex)
        df = pd.read_csv(metrics_file)
        df = filter(df, kernel_name)

        curr_metrics = perf_metrics[pass_id - 1]
        curr_metrics_names = list(curr_metrics.keys())
        for name in curr_metrics_names:
            data = df[df['Counter_Name'] == name]
            value = data['Counter_Value'].mean()

            num_flops = curr_metrics[name]['flop']
            relative_value = 100 * num_flops / num_flop_sum
            table['Counter Name'].append(name)
            table['Raw Data'].append(value)
            table['FLOP'].append(num_flops)
            table['Relative FLOP, %'].append(relative_value)
    print(pd.DataFrame(table))
    print()

    print('Performance info in TFLOP/s:')
    performance = num_flop_sum / (timing * 1000)
    print(performance.describe())
    print()


def main(args):
    check_rocprofv3()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_spec = get_metrics_as_yaml()
    metrics_spec_path = os.path.join(curr_dir, "metrics_spec.yaml")
    with open(metrics_spec_path, 'w') as file:
        file.write(metrics_spec)

    metrics_dir = os.path.join(curr_dir, "metrics_dir")
    timing_dir = os.path.join(curr_dir, "timing_dir")

    if not args.display_only:
        # test original command
        if (args.verbose):
            print('running original program...')
        user_cmd = args.cmd
        output = run_external_binary([], user_cmd, args.verbose)

        if (args.verbose):
            print(output)

        # collect performance metrices
        if (args.verbose):
            print('running rocprofv3 passes...')

        if os.path.exists(metrics_dir):
            shutil.rmtree(metrics_dir)

        rocprof_cmd = ['rocprofv3', '-i', metrics_spec_path, '-d', metrics_dir, '--']
        output = run_external_binary([], rocprof_cmd + user_cmd, args.verbose)

        if (args.verbose):
            print(output)

        # collect timing
        if (args.verbose):
            print('running rocprofv3 for timing info...')

        if os.path.exists(timing_dir):
            shutil.rmtree(timing_dir)

        rocprof_cmd = ['rocprofv3', '--kernel-trace', '-d', timing_dir, '--']
        output = run_external_binary([], rocprof_cmd + user_cmd, args.verbose)

        if (args.verbose):
            print(output)

    process_files(metrics_dir, timing_dir, args.kernel, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--kernel", type=str, required=True, help="name of a kernel")
    parser.add_argument('-c', '--cmd', required=True, nargs=argparse.REMAINDER, help='user command')
    parser.add_argument("--display-only", action='store_true', help='display info without running')
    parser.add_argument("-v", "--verbose", action='store_true', help='verbose output')
    args = parser.parse_args()

    main(args)
