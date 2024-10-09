import argparse
from collections import namedtuple
import json
import pandas as pd
try:
    import hatchet as ht
    from hatchet.query import NegationQuery
except ImportError:
    raise ImportError("Failed to import hatchet. `pip install llnl-hatchet` to get the correct version.")
import numpy as np
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
        ret = [raw_metrics[0] + " (inc)"]
    if len(ret) == 0:
        raise RuntimeError(f"Metric {metric} is not found. Use the --list flag to list available metrics")
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
                idx = df["device_id"] == device_index
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
            idx = df["device_id"] == device_index
            device_frames = df[idx]
            memory_clock_rate = device_info[device_type][device_index]["memory_clock_rate"]  # in khz
            bus_width = device_info[device_type][device_index]["bus_width"]  # in bits
            peak_bandwidth = 2 * bus_width * memory_clock_rate * 1e3 / 8
            min_time_bytes.loc[idx, "min_time"] += device_frames["bytes"] / peak_bandwidth
    return min_time_bytes


FactorDict = namedtuple("FactorDict", ["name", "factor"])
time_factor_dict = FactorDict("time", {"time/s": 1, "time/ms": 1e-3, "time/us": 1e-6, "time/ns": 1e-9})
avg_time_factor_dict = FactorDict("avg_time", {f"avg_{key}": value for key, value in time_factor_dict.factor.items()})
bytes_factor_dict = FactorDict("bytes", {"byte/s": 1, "gbyte/s": 1e9, "tbyte/s": 1e12})

derivable_metrics = {
    **{key: bytes_factor_dict
       for key in bytes_factor_dict.factor.keys()},
}

# FLOPS have a specific width to their metric
default_flop_factor_dict = {f"flop/s": 1, f"gflop/s": 1e9, f"tflop/s": 1e12}
derivable_metrics.update(
    {key: FactorDict("flops", default_flop_factor_dict)
     for key in default_flop_factor_dict.keys()})
for width in TritonHook.flops_width:
    factor_name = f"flops{width}"
    factor_dict = {f"flop{width}/s": 1, f"gflop{width}/s": 1e9, f"tflop{width}/s": 1e12}
    derivable_metrics.update({key: FactorDict(factor_name, factor_dict) for key in factor_dict.keys()})


def derive_metrics(gf, metrics, raw_metrics, device_info):
    derived_metrics = []
    internal_frame_indices = gf.dataframe["device_id"].isna()

    def get_time_seconds(df):
        time_metric_name = match_available_metrics([time_factor_dict.name], raw_metrics)[0]
        time_unit = (time_factor_dict.name + "/" + time_metric_name.split("(")[1].split(")")[0])
        return df[time_metric_name] * time_factor_dict.factor[time_unit]

    for metric in metrics:
        if metric == "util":  # Tensor core only
            min_time_bytes = get_min_time_bytes(gf.dataframe, device_info)
            min_time_flops = get_min_time_flops(gf.dataframe, device_info)
            time_sec = get_time_seconds(gf.dataframe)
            gf.dataframe["util (inc)"] = min_time_flops["min_time"].combine(min_time_bytes["min_time"], max) / time_sec
            gf.dataframe.loc[internal_frame_indices, "util (inc)"] = np.nan
            derived_metrics.append("util (inc)")
        elif metric in derivable_metrics:  # flop<width>/s, <t/g>byte/s
            derivable_metric = derivable_metrics[metric]
            metric_name = derivable_metric.name
            metric_factor_dict = derivable_metric.factor
            matched_metric_name = match_available_metrics([metric_name], raw_metrics)[0]
            gf.dataframe[f"{metric} (inc)"] = (gf.dataframe[matched_metric_name] / (get_time_seconds(gf.dataframe)) /
                                               metric_factor_dict[metric])
            derived_metrics.append(f"{metric} (inc)")
        elif metric in time_factor_dict.factor:
            metric_time_unit = time_factor_dict.name + "/" + metric.split("/")[1]
            gf.dataframe[f"{metric} (inc)"] = (get_time_seconds(gf.dataframe) /
                                               time_factor_dict.factor[metric_time_unit])
            derived_metrics.append(f"{metric} (inc)")
        elif metric in avg_time_factor_dict.factor:
            metric_time_unit = avg_time_factor_dict.name + "/" + metric.split("/")[1]
            gf.dataframe[f"{metric} (inc)"] = (get_time_seconds(gf.dataframe) / gf.dataframe['count'] /
                                               avg_time_factor_dict.factor[metric_time_unit])
            gf.dataframe.loc[internal_frame_indices, f"{metric} (inc)"] = np.nan
            derived_metrics.append(f"{metric} (inc)")
        else:
            metric_name_and_unit = metric.split("/")
            metric_name = metric_name_and_unit[0]
            if len(metric_name_and_unit) > 1:
                metric_unit = metric_name_and_unit[1]
                if metric_unit != "%":
                    raise ValueError(f"Unsupported unit {metric_unit}")
                matched_metric_name = match_available_metrics([metric_name], raw_metrics)[0]
                single_frame = gf.dataframe[matched_metric_name]
                total = gf.dataframe[matched_metric_name].iloc[0]
                gf.dataframe[f"{metric_name}/% (inc)"] = (single_frame / total) * 100.0
                derived_metrics.append(f"{metric_name}/% (inc)")
            else:
                matched_metric_name = match_available_metrics([metric_name], raw_metrics)[0]
                derived_metrics.append(matched_metric_name)
    return derived_metrics


def format_frames(gf, format):
    if format == "file_function_line":
        gf.dataframe["name"] = gf.dataframe["name"].apply(lambda x: x.split("/")[-1])
    elif format == "function_line":
        gf.dataframe["name"] = gf.dataframe["name"].apply(lambda x: x.split(":")[-1])
    elif format == "file_function":
        gf.dataframe["name"] = gf.dataframe["name"].apply(lambda x: x.split("/")[-1].split("@")[0])
    return gf


def filter_frames(gf, include=None, exclude=None, threshold=None, metric=None):
    if include:
        query = f"""
MATCH ("*")->(".", p)->("*")
WHERE p."name" =~ "{include}"
"""
        gf = gf.filter(query, squash=True)
    if exclude:
        inclusion_query = f"""
MATCH (".", p)->("*")
WHERE p."name" =~ "{exclude}"
"""
        query = NegationQuery(inclusion_query)
        gf = gf.filter(query, squash=True)
    # filter out metadata computation
    query = [{"name": f"^(?!{COMPUTE_METADATA_SCOPE_NAME}).*"}]
    gf = gf.filter(query, squash=True)
    if threshold:
        query = ["*", {metric: f">= {threshold}"}]
        gf = gf.filter(query, squash=True)
    return gf


def parse(metrics, filename, include=None, exclude=None, threshold=None, depth=100, format=None):
    with open(filename, "r") as f:
        gf, raw_metrics, device_info = get_raw_metrics(f)
        gf = format_frames(gf, format)
        assert len(raw_metrics) > 0, "No metrics found in the input file"
        gf.update_inclusive_columns()
        metrics = derive_metrics(gf, metrics, raw_metrics, device_info)
        # TODO: generalize to support multiple metrics, not just the first one
        gf = filter_frames(gf, include, exclude, threshold, metrics[0])
        print(gf.tree(metric_column=metrics, expand_name=True, depth=depth, render_header=False))
        emit_warnings(gf, metrics)


def emit_warnings(gf, metrics):
    if "bytes (inc)" in metrics:
        byte_values = gf.dataframe["bytes (inc)"].values
        min_byte_value = np.nanmin(byte_values)
        if min_byte_value < 0:
            print("Warning: Negative byte values detected, this is usually the result of a datatype overflow\n")


def show_metrics(file_name):
    with open(file_name, "r") as f:
        _, raw_metrics, _ = get_raw_metrics(f)
        print("Available metrics:")
        if raw_metrics:
            for raw_metric in raw_metrics:
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                print(f"- {raw_metric_no_unit}")


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
- avg_time/s, avg_time/ms, avg_time/us, avg_time/ns: time / count
- flop[<8/16/32/64>]/s, gflop[<8/16/32/64>]/s, tflop[<8/16/32/64>]/s: flops / time
- byte/s, gbyte/s, tbyte/s: bytes / time
- util: max(sum(flops<width>) / peak_flops<width>_time, sum(bytes) / peak_bandwidth_time)

For inclusive metrics (e.g. time) an additional column (i.e. metric-name/%) can be printed to show the percentage
each frame is of the full model.

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
        help=
        """Find frames that match the given regular expression and return all nodes in the paths that pass through the matching frames.
For example, the following command will display all paths that contain frames that contains "test":
```
proton-viewer -i ".*test.*" path/to/file.json
```
""",
    )
    argparser.add_argument(
        "-e",
        "--exclude",
        type=str,
        default=None,
        help="""Exclude frames that match the given regular expression and their children.
For example, the following command will exclude all paths that contain frames that contains "test":
```
proton-viewer -e ".*test.*" path/to/file.json
```
""",
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
    argparser.add_argument(
        "-f", "--format", type=str, choices=["full", "file_function_line", "function_line", "file_function"],
        default="full", help="""Formatting the frame name.
- full: include the path, file name, function name and line number.
- file_function_line: include the file name, function name and line number.
- function_line: include the function name and line number.
- file_function: include the file name and function name.
""")

    args, target_args = argparser.parse_known_args()
    assert len(target_args) == 1, "Must specify a file to read"

    file_name = target_args[0]
    metrics = args.metrics.split(",") if args.metrics else None
    include = args.include
    exclude = args.exclude
    threshold = args.threshold
    depth = args.depth
    format = args.format
    if include and exclude:
        raise ValueError("Cannot specify both include and exclude")
    if args.list:
        show_metrics(file_name)
    elif metrics:
        parse(metrics, file_name, include, exclude, threshold, depth, format)


if __name__ == "__main__":
    main()
