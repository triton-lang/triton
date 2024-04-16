import argparse
from collections import namedtuple
import json

import hatchet as ht
import triton._C.libproton.proton as libproton
from triton.profiler.hook import COMPUTE_METADATA_SCOPE_NAME


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
    gf = ht.GraphFrame.from_literal(json.load(file))
    return gf, gf.show_metric_columns()


FactorDict = namedtuple("FactorDict", ["name", "factor"])
time_factor_dict = FactorDict("time", {"time/s": 1, "time/ms": 1e-3, "time/us": 1e-6, "time/ns": 1e-9})
flops_factor_dict = FactorDict("flops", {"flop/s": 1, "tflop/s": 1e12, "gflop/s": 1e9})
bytes_factor_dict = FactorDict("bytes", {"byte/s": 1, "tbyte/s": 1e12, "gbyte/s": 1e9})

derivable_metrics = {
    **{key: flops_factor_dict
       for key in flops_factor_dict.factor.keys()},
    **{key: bytes_factor_dict
       for key in bytes_factor_dict.factor.keys()},
}


def max_flops(width):
    device_info = libproton.device_info(0)
    capability_major = device_info["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR"]
    capability_minor = device_info["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR"]
    capability = (capability_major, capability_minor)
    # hardcoded, because we can't get max tensor core clock from cuDeviceGetAttribute
    flops_8bits = {(8, 0): 610e12, (9, 0): 1960e12}[capability]
    return flops_8bits / (width / 8)


def max_bandwidth():
    # XXX(Keren): It assumes that GPUs are availble, need to get device info from the input
    # profiled at runtime
    device_info = libproton.device_info(0)
    capability_major = device_info["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR"]
    capability_minor = device_info["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR"]
    memory_clock_rate = device_info["CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE"]
    capability = (capability_major, capability_minor)
    memory_num_buses = {(8, 0): 5, (9, 0): 5}[capability]
    memory_bus_width = {(8, 0): 128, (9, 0): 128}[capability]
    return 2. * memory_num_buses * memory_bus_width * memory_clock_rate * 1e3


def derive_metrics(gf, metrics, raw_metrics):
    derived_metrics = []
    original_metrics = []
    for metric in metrics:
        if metric == "util":
            time = gf.dataframe["Time (ns) (inc)"] * 1e-9
            min_time_bytes = gf.dataframe["bytes (inc)"] / max_bandwidth()
            min_time_flops = 0
            # XXX(Keren): Make it more general
            for width in [16]:
                min_time_flops += gf.dataframe[f"flops{width} (inc)"].fillna(0) / max_flops(width)
            gf.dataframe["util (inc)"] = min_time_flops.combine(min_time_bytes, max) / time
            derived_metrics.append("util (inc)")
        elif metric in derivable_metrics:
            deriveable_metric = derivable_metrics[metric]
            metric_name = deriveable_metric.name
            metric_factor_dict = deriveable_metric.factor
            matched_metric_name = match_available_metrics([metric_name], raw_metrics)[0]
            time_metric_name = match_available_metrics([time_factor_dict.name], raw_metrics)[0]
            time_unit = (time_factor_dict.name + "/" + time_metric_name.split("(")[1].split(")")[0])
            gf.dataframe[f"{metric} (inc)"] = (gf.dataframe[matched_metric_name] /
                                               (gf.dataframe[time_metric_name] * time_factor_dict.factor[time_unit]) /
                                               metric_factor_dict[metric])
            derived_metrics.append(f"{metric} (inc)")
        elif metric in time_factor_dict.factor:
            time_metric_name = match_available_metrics([time_factor_dict.name], raw_metrics)[0]
            original_time_unit = (time_factor_dict.name + "/" + time_metric_name.split("(")[1].split(")")[0])
            metric_time_unit = time_factor_dict.name + "/" + metric.split("/")[1]
            gf.dataframe[f"{metric} (inc)"] = gf.dataframe[time_metric_name] * (
                time_factor_dict.factor[original_time_unit] / time_factor_dict.factor[metric_time_unit])
            derived_metrics.append(f"{metric} (inc)")
        else:
            original_metrics.append(metric)

    if original_metrics:
        original_metrics = match_available_metrics(original_metrics, raw_metrics)
    return derived_metrics + original_metrics


def parse(metrics, filename, include, exclude, threshold, depth):
    with open(filename, "r") as f:
        gf, raw_metrics = get_raw_metrics(f)
        assert len(raw_metrics) > 0, "No metrics found in the input file"
        gf.update_inclusive_columns()
        metrics = derive_metrics(gf, metrics, raw_metrics)
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
        print(gf.tree(metric_column=metrics, expand_name=True, depth=depth))


def show_metrics(file_name):
    with open(file_name, "r") as f:
        _, raw_metrics = get_raw_metrics(f)
        print("Available metrics:")
        if raw_metrics:
            print("\n".join(raw_metrics))
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
- flop/s, tflop/s, gflop/s: flops / time
- byte/s, tbyte/s, gbyte/s: bytes / time
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
