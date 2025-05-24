flops_by_device = {
    "CUDA": {
        "80":
        lambda width, **kwargs: 624e12 / (width / 8),
        "89":
        lambda width, **kwargs: (330.3 * 1e12) / (width / 8),  # TODO(Keren): Implement fp16 acc-> 660.6 fp8
        "90":
        lambda width, num_sms, clock_rate, **kwargs: ((num_sms / 114 * clock_rate / (1755 * 1e3) * 1513) * 1e12) /
        (width / 8),
        "100":
        lambda width, num_sms, clock_rate, **kwargs: (num_sms * 16384 * (clock_rate / 1e3) * 1e6) / (width / 8),
    },
    "HIP": {
        "gfx90a": lambda width, **kwargs: 383e12 / (width / 8),
        "gfx942": lambda width, **kwargs: 2614.9e12 / (width / 8),
    },
}


def max_flops(device_type, arch, width, num_sms, clock_rate):
    """
    Calculate the maximum FLOPS for a given device type and width.

    Args:
        device_type (str): The type of device (e.g., "CUDA", "HIP").
        arch (str): The architecture of the device (e.g., "80", "90").
        width (int): The width in bits.
        num_sms (int): The number of streaming multiprocessors.
        clock_rate (float): The clock rate in GHz.

    Returns:
        float: The maximum FLOPS for the given device type and width.
    """
    if device_type not in flops_by_device:
        raise ValueError(f"Unsupported device type: {device_type}")

    if arch not in flops_by_device[device_type]:
        raise ValueError(f"Unsupported architecture: {arch}")

    flops_func = flops_by_device[device_type][arch]

    return flops_func(width, num_sms=num_sms, clock_rate=clock_rate)


def max_bps(bus_width, memory_clock_rate):
    """
    Calculate the maximum bytes per second for a given bus width and memory clock rate.

    Args:
        bus_width (int): The bus width in bits.
        memory_clock_rate (float): The memory clock rate in GHz.

    Returns:
        float: The maximum bytes per second.
    """
    return 2 * bus_width * memory_clock_rate * 1e3 / 8
