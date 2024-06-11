import argparse
from collections import namedtuple
import json
import pandas as pd

import hatchet as ht
from triton.profiler.hook import COMPUTE_METADATA_SCOPE_NAME, TritonHook


def match_available_metrics(metrics, raw_metrics):
    ret = []
    if metrics:
        for metric in metrics:
            metric = metric.lower()
            for raw_metric in raw_metrics:
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                if metric in (raw_metric, raw_metric_no_unit):
                    ret.append(raw_metric + " (inc)")
                    break
    else:
        ret = [raw_metrics[0]] + " (inc)"
    return ret


def get_raw_metrics(file):
    database = json.load(file)
    device_info = database.pop(1)
    gf = ht.GraphFrame.from_literal(database)
    return gf, gf.show_metric_columns(), device_info


def get_min_time_flops(df, device_info):
    min_time_flops = pd.DataFrame(0.0, index=df.index, columns=["min_time"])
    for device_type in device_info:
        for device_index in device_info[device_type]:
            arch = device_info[device_type][device_index]["arch"]
            num_sms = device_info[device_type][device_index]["num_sms"]
            clock_rate = device_info[device_type][device_index]["clock_rate"]
            for width in TritonHook.flops_width:
                idx = df["DeviceId"] == device_index
                device_frames = df[idx]
                if f"flops{width}" not in device_frames.columns:
                    continue
                max_flops = 0
                if device_type == "CUDA":
                    if arch == "80":
                        max_flops = 624e12 / (width / 8)
                    elif arch == "89":
                        # TODO(Keren): Implement fp16 acc-> 660.6 fp8
                        max_flops = (330.3 * 1e12) / (width / 8)
                    elif arch == "90":
                        # 114 sms and 1755mhz is the base number of sms and clock rate of H100 pcie
                        max_flops = ((num_sms / 114 * clock_rate / (1755 * 1e3) * 1513) * 1e12) / (width / 8)
                elif device_type == "HIP":
                    if arch == "gfx90a":
                        max_flops = 383e12 / (width / 8)
                    elif arch == "gfx941" or arch == "gfx942":
                        max_flops = 2614.9e12 / (width / 8)
                else:
                    raise ValueError(f"Unsupported device type: {device_type}")
                min_time_flops.loc[idx, "min_time"] += device_frames[f"flops{width}"].fillna(0) / max_flops
    return min_time_flops


def get_min_time_bytes(df, device_info):
    min_time_bytes = pd.DataFrame(0.0, index=df.index, columns=["min_time"])
    for device_type in device_info:
        for device_index in device_info[device_type]:
            idx = df["DeviceId"] == device_index
            device_frames = df[idx]
            memory_clock_rate = device_info[device_type][device_index]["memory_clock_rate"]  # in khz
            bus_width = device_info[device_type][device_index]["bus_width"]  # in bits
            peak_bandwidth = 2 * bus_width * memory_clock_rate * 1e3 / 8
            min_time_bytes.loc[idx, "min_time"] += device_frames["bytes"] / peak_bandwidth
    return min_time_bytes


FactorDict = namedtuple("FactorDict", ["name", "factor"])
time_factor_dict = FactorDict("time", {"time/s": 1, "time/ms": 1e-3, "time/us": 1e-6, "time/ns": 1e-9})
flops_factor_dict = FactorDict("flops", {"flop/s": 1, "gflop/s": 1e9, "tflop/s": 1e12})
bytes_factor_dict = FactorDict("bytes", {"byte/s": 1, "gbyte/s": 1e9, "tbyte/s": 1e12})

derivable_metrics = {
    **{key: flops_factor_dict
       for key in flops_factor_dict.factor.keys()},
    **{key: bytes_factor_dict
       for key in bytes_factor_dict.factor.keys()},
}


def derive_metrics(gf, metrics, raw_metrics, device_info):
    derived_metrics = []
    original_metrics = []
    time_metric_name = match_available_metrics([time_factor_dict.name], raw_metrics)[0]
    time_unit = (time_factor_dict.name + "/" + time_metric_name.split("(")[1].split(")")[0])
    for metric in metrics:
        if metric == "util":  # Tensor core only
            min_time_bytes = get_min_time_bytes(gf.dataframe, device_info)
            min_time_flops = get_min_time_flops(gf.dataframe, device_info)
            time_sec = gf.dataframe[time_metric_name] * (time_factor_dict.factor[time_unit] /
                                                         time_factor_dict.factor["time/s"])
            gf.dataframe["util (inc)"] = min_time_flops["min_time"].combine(min_time_bytes["min_time"], max) / time_sec
            derived_metrics.append("util (inc)")
        elif metric in derivable_metrics:
            deriveable_metric = derivable_metrics[metric]
            metric_name = deriveable_metric.name
            metric_factor_dict = deriveable_metric.factor
            matched_metric_name = match_available_metrics([metric_name], raw_metrics)[0]
            gf.dataframe[f"{metric} (inc)"] = (gf.dataframe[matched_metric_name] /
                                               (gf.dataframe[time_metric_name] * time_factor_dict.factor[time_unit]) /
                                               metric_factor_dict[metric])
            derived_metrics.append(f"{metric} (inc)")
        elif metric in time_factor_dict.factor:
            metric_time_unit = time_factor_dict.name + "/" + metric.split("/")[1]
            gf.dataframe[f"{metric} (inc)"] = gf.dataframe[time_metric_name] * (
                time_factor_dict.factor[time_unit] / time_factor_dict.factor[metric_time_unit])
            derived_metrics.append(f"{metric} (inc)")
        else:
            original_metrics.append(metric)

    if original_metrics:
        original_metrics = match_available_metrics(original_metrics, raw_metrics)
    return derived_metrics + original_metrics


def parse(metrics, filename, include, exclude, threshold, depth):
    with open(filename, "r") as f:
        gf, raw_metrics, device_info = get_raw_metrics(f)
        assert len(raw_metrics) > 0, "No metrics found in the input file"
        gf.update_inclusive_columns()
        metrics = derive_metrics(gf, metrics, raw_metrics, device_info)
        if include or exclude:
            # make regex do negative match
            name_filter = f"^(?!{exclude}).*" if exclude else include
            query = ["*", {"name": name_filter}]
            gf = gf.filter(query, squash=True)
        # filter out metadata computation
        query = [{"name": f"^(?!{COMPUTE_METADATA_SCOPE_NAME}).*"}]
        gf = gf.filter(query, squash=True)
        if threshold:
            # TODO: generalize to support multiple metrics
            query = ["*", {metrics[0]: f">= {threshold}"}]
            gf = gf.filter(query, squash=True)
        print(gf.tree(metric_column=metrics, expand_name=True, depth=depth, render_header=False))


def show_metrics(file_name):
    with open(file_name, "r") as f:
        _, raw_metrics, _ = get_raw_metrics(f)
        print("Available metrics:")
        if raw_metrics:
            for raw_metric in raw_metrics:
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                print(f"- {raw_metric_no_unit}")
        return


def main():
    argparser = argparse.ArgumentParser(
        description="Performance data viewer for proton profiles.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    argparser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="""List available metrics. Metric names are case insensitive and ignore units.
Derived metrics can be created when source metrics are available.
- time/s, time/ms, time/us, time/ns: time
- flop/s, gflop/s, tflop/s: flops / time
- byte/s, gbyte/s, tbyte/s: bytes / time
- util: max(sum(flops<width>) / peak_flops<width>_time, bytes / peak_bandwidth_time))
""",
    )
    argparser.add_argument(
        "-m",
        "--metrics",
        type=str,
        default=None,
        help="""At maximum two metrics can be specified, separated by comma.
There are two modes:
1) Choose the output metric to display. It's case insensitive and ignore units.
2) Derive a new metric from existing metrics.
""",
    )
    argparser.add_argument(
        "-i",
        "--include",
        type=str,
        default=None,
        help="Include frames(kernels) that match the given regular expression",
    )
    argparser.add_argument(
        "-e",
        "--exclude",
        type=str,
        default=None,
        help="Exclude frames(kernels) that match the given regular expression",
    )

    argparser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help=
        "Exclude frames(kernels) whose metrics are below the given threshold. This filter only applies on the first metric.",
    )

    argparser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=100,
        help="The depth of the tree to display",
    )

    args, target_args = argparser.parse_known_args()
    assert len(target_args) == 1, "Must specify a file to read"

    file_name = target_args[0]
    metrics = args.metrics.split(",") if args.metrics else None
    include = args.include
    exclude = args.exclude
    threshold = args.threshold
    depth = args.depth
    if include and exclude:
        raise ValueError("Cannot specify both include and exclude")
    if args.list:
        show_metrics(file_name)
    elif metrics:
        parse(metrics, file_name, include, exclude, threshold, depth)


if __name__ == "__main__":
    main()
