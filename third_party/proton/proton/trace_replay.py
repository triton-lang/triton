from dataclasses import dataclass
import copy
import torch
from typing import List


@dataclass
class IntraKernelConfig(object):
    slots: int
    header: int
    wg_num: int
    word_per_slot: int


@dataclass
class ProfileEvent(object):
    region_id: int
    start: int
    end: int
    wg: int


def _get_events(index: int, wg_id: int, data: List[int], word_per_slot: int, block_id: int):
    assert word_per_slot == 2
    size = index if len(data) > index else len(data)
    event_list = []
    active_event = {}
    # Each region has 3 parsing states: init, start, end
    parse_state = {}
    for i in range(0, size, word_per_slot):
        metadata = data[i]
        cycle = data[i + 1]
        is_start = False if metadata >> 31 == 1 else True
        region_id = metadata & 0x7FFFFFFF
        if region_id not in active_event:
            active_event[region_id] = ProfileEvent(region_id, 0, 0, wg_id)
        event = active_event[region_id]
        prev_state = parse_state.get(region_id, "init")
        suffix = f"(region={region_id}, wg={wg_id}, block={block_id})"
        # We check clock overflow (32-bit) and ignore the extra event.
        # For example, two consecutive start events of the same region in the event stream.
        if is_start:
            if prev_state in ["init", "end"]:
                event.start = cycle
                parse_state[region_id] = "start"
            else:
                print("Warning: ignore an extra start record", suffix)
                continue
        else:
            if prev_state == "start":
                event.end = cycle
                parse_state[region_id] = "end"
                if event.end < event.start:
                    print("Warning: ignore an event due to clock overflow", suffix)
                    continue
                event_list.append(copy.deepcopy(event))
            else:
                print("Warning: ignore an extra end record", suffix)
                continue
    return event_list


def _shift_start(event_list: List[ProfileEvent]):
    start_time = []
    for event in event_list:
        start_time.append(event.start)
    if len(start_time) == 0:
        return

    min_start = min(start_time)
    for event in event_list:
        event.start -= min_start
        event.end -= min_start


def intra_kernel_smem(config: IntraKernelConfig):
    return config.header + config.slots * config.word_per_slot


def _get_chrome_event_str(event: ProfileEvent, block_id: int, sm_id: int):
    return f'{{"name": "region_{event.region_id}", "cat": "triton", \
        "ph": "X", "ts": {event.start}, "dur": {event.end - event.start}, \
        "pid": "{block_id}", "tid": "{event.wg}", \
        "args":{{"sm_id": "{sm_id}", "frequency": "1MHz"}}}}'


def dump_chrome_trace(block_num: int, config: IntraKernelConfig, profile_mem: torch.Tensor, file_name: str):
    scratch = intra_kernel_smem(config)
    trace_str = "{\"traceEvents\": ["
    for i in range(block_num):
        workspace = profile_mem[i * scratch:(i + 1) * scratch]
        block_id = workspace[0].item()
        sm_id = workspace[1].item()
        index = workspace[2].item()
        data = workspace[3:].tolist()
        event_list = []
        wg_data_len = int(len(data) / config.wg_num)
        words = config.word_per_slot
        for j in range(config.wg_num):
            ws = j * wg_data_len
            wg_events = _get_events(index, j, data[ws:ws + wg_data_len], words, block_id)
            event_list += wg_events

        _shift_start(event_list)
        for event in event_list:
            chrome_event_str = _get_chrome_event_str(event, block_id, sm_id)
            trace_str += chrome_event_str + ",\n"

    trace_str = trace_str[:-2] + "]}"

    with open(file_name, "w") as f:
        f.write(trace_str)
