import json
import msgpack
import os
import select
import sys
import triton.profiler.viewer as viewer


def handle_document(database, output_format: str, messages: int, kernel_count: int):
    messages += 1
    if output_format == "chrome_trace":
        kernel_count += sum(1 for event in database.get("traceEvents", []) if event.get("name") == "pipe_kernel")
    else:
        gf, _, _, _ = viewer.get_raw_metrics(database)
        kernel_frame = gf.filter("MATCH ('*', c) WHERE c.'name' =~ '.*pipe_kernel.*' AND c IS LEAF").dataframe
        kernel_count += int(kernel_frame["count"].sum()) if len(kernel_frame) else 0
    return messages, kernel_count


def main():
    fd = int(sys.argv[1])
    output_format = sys.argv[2]
    threshold = int(sys.argv[3])
    target = int(sys.argv[4])
    messages = 0
    kernel_count = 0
    threshold_emitted = False
    os.set_blocking(fd, False)
    decoder = json.JSONDecoder()
    buffer = ""
    unpacker = None
    if output_format == "hatchet_msgpack":
        unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)

    while messages < target:
        ready, _, _ = select.select([fd], [], [], 30)
        if not ready:
            raise TimeoutError("Timed out waiting for periodic profile data on pipe")
        chunk = os.read(fd, 65536)
        if not chunk:
            raise RuntimeError("Pipe closed before all MessagePack documents were received")
        if output_format == "hatchet_msgpack":
            unpacker.feed(chunk)
            for database in unpacker:
                messages, kernel_count = handle_document(database, output_format, messages, kernel_count)
                if not threshold_emitted and messages >= threshold:
                    print(json.dumps({
                        "event": "threshold",
                        "messages": messages,
                        "kernel_count": kernel_count,
                    }), flush=True)
                    threshold_emitted = True
                if messages >= target:
                    print(json.dumps({
                        "event": "final",
                        "messages": messages,
                        "kernel_count": kernel_count,
                    }), flush=True)
                    return
        else:
            buffer += chunk.decode("utf-8")
            while buffer:
                try:
                    database, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                messages, kernel_count = handle_document(database, output_format, messages, kernel_count)
                if not threshold_emitted and messages >= threshold:
                    print(json.dumps({
                        "event": "threshold",
                        "messages": messages,
                        "kernel_count": kernel_count,
                    }), flush=True)
                    threshold_emitted = True
                if messages >= target:
                    print(json.dumps({
                        "event": "final",
                        "messages": messages,
                        "kernel_count": kernel_count,
                    }), flush=True)
                    return
                buffer = buffer[end:].lstrip()


if __name__ == "__main__":
    main()
