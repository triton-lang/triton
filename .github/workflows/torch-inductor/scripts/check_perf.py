import argparse
import csv
from collections import namedtuple

# Create a named tuple for the output of the benchmark
BenchmarkOutput = namedtuple('BenchmarkOutput', ['dev', 'name', 'batch_size', 'speedup', 'latency'])


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


def compare(baseline: dict, new: dict, threshold: float, geomean_threshold: float) -> bool:
    baseline_geomean = 1.0
    new_geomean = 1.0
    for key in new:
        if key not in baseline:
            print(f"New benchmark {key} not found in baseline")
        baseline_latency = baseline[key].latency
        new_latency = new[key].latency
        if baseline_latency == 0:
            print(f"Baseline latency for {key} is 0")
            continue
        elif new_latency == 0:
            print(f"New latency for {key} is 0")
            continue

        if new_latency < baseline_latency * (1 - threshold):
            print(f"New benchmark {key} is faster than baseline: {new_latency} vs {baseline_latency}")
        elif new_latency > baseline_latency * (1 + threshold):
            print(f"New benchmark {key} is slower than baseline: {new_latency} vs {baseline_latency}")
        else:
            print(f"New benchmark {key} is within threshold: {new_latency} vs {baseline_latency}")
        baseline_geomean *= baseline[key].speedup
        new_geomean *= new[key].speedup

    baseline_geomean = baseline_geomean**(1 / len(baseline))
    new_geomean = new_geomean**(1 / len(new))
    print(f"Baseline geomean: {baseline_geomean}")
    print(f"New geomean: {new_geomean}")
    assert new_geomean >= baseline_geomean * (1 - geomean_threshold), \
        f"New geomean is slower than baseline: {new_geomean} vs {baseline_geomean}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--new', required=True)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--geomean-threshold', type=float, default=0.02)
    args = parser.parse_args()
    baseline = parse_output(args.baseline)
    new = parse_output(args.new)
    compare(baseline, new, args.threshold, args.geomean_threshold)


if __name__ == "__main__":
    main()
