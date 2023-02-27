import argparse
import csv
from collections import namedtuple

# Create a named tuple for the output of the benchmark
BenchmarkOutput = namedtuple(
    'BenchmarkOutput', ['dev', 'name', 'batch_size', 'speedup', 'latency'])


def parse_output(file_path: str) -> dict:
    entries = {}
    with open(file_path) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0 or len(row) < 5:
                continue
            dev = row[0]
            name = row[1]
            batch_size = row[2]
            speedup = float(row[3])
            latency = float(row[4])
            entries[name] = BenchmarkOutput(dev, name, batch_size, speedup, latency)
    return entries


def compare(baseline: dict, new: dict, threshold: float) -> bool:
    for key in new:
        if key not in baseline:
            print(f"New benchmark {key} not found in baseline")
        baseline_latency = baseline[key].latency
        new_latency = new[key].latency
        if new_latency < baseline_latency * (1 - threshold):
            print(
                f"New benchmark {key} is slower than baseline: {new_latency} vs {baseline_latency}")
        elif new_latency > baseline_latency * (1 + threshold):
            print(
                f"New benchmark {key} is faster than baseline: {new_latency} vs {baseline_latency}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--new', required=True)
    parser.add_argument('--threshold', type=float, default=0.05)
    args = parser.parse_args()
    baseline = parse_output(args.baseline)
    new = parse_output(args.new)
    compare(baseline, new, args.threshold)


main()
