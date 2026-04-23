import pathlib
import struct

_PERFETTO_FLOW_ID_BASE = 1 << 32


def parse_perfetto_trace(path: pathlib.Path | str, normalize_event):

    def read_varint(data: bytes, offset: int):
        value = 0
        shift = 0
        while True:
            byte = data[offset]
            offset += 1
            value |= (byte & 0x7f) << shift
            if byte < 0x80:
                return value, offset
            shift += 7

    def fields(data: bytes):
        offset = 0
        while offset < len(data):
            tag, offset = read_varint(data, offset)
            field_id = tag >> 3
            wire_type = tag & 0x7
            if wire_type == 0:
                value, offset = read_varint(data, offset)
            elif wire_type == 1:
                value = struct.unpack_from("<d", data, offset)[0]
                offset += 8
            elif wire_type == 2:
                size, offset = read_varint(data, offset)
                value = data[offset:offset + size]
                offset += size
            else:
                raise ValueError(f"Unsupported protobuf wire type: {wire_type}")
            yield field_id, value

    def decode_string(value: bytes):
        return value.decode("utf-8")

    def interned_string(entry: bytes):
        iid = None
        name = None
        for field_id, value in fields(entry):
            if field_id == 1:
                iid = value
            elif field_id == 2:
                name = decode_string(value)
        return iid, name

    def parse_track_descriptor(track_descriptor: bytes):
        uuid = None
        name = None
        for field_id, value in fields(track_descriptor):
            if field_id == 1:
                uuid = value
            elif field_id == 2:
                name = decode_string(value)
        return uuid, name

    def parse_debug_annotation(annotation: bytes, interned):
        name = None
        value = None
        for field_id, field_value in fields(annotation):
            if field_id == 1:
                name = interned["debug_annotation_names"].get(field_value)
            elif field_id == 2:
                value = bool(field_value)
            elif field_id == 3:
                value = field_value
            elif field_id == 4:
                value = field_value
                if value >= (1 << 63):
                    value -= 1 << 64
            elif field_id == 5:
                value = field_value
            elif field_id == 6:
                value = decode_string(field_value)
            elif field_id == 10:
                name = decode_string(field_value)
            elif field_id == 17:
                value = interned["debug_annotation_string_values"].get(field_value)
        return name, value

    def parse_track_event(track_event: bytes, timestamp, interned, track_names):
        event_type = None
        track_uuid = None
        name = None
        category = None
        flow_ids = []
        terminating_flow_ids = []
        args = {}
        for field_id, value in fields(track_event):
            if field_id == 3:
                category = interned["event_categories"].get(value)
            elif field_id == 4:
                arg_name, arg_value = parse_debug_annotation(value, interned)
                if arg_name is not None:
                    args[f"debug.{arg_name}"] = arg_value
            elif field_id == 9:
                event_type = value
            elif field_id == 10:
                name = interned["event_names"].get(value)
            elif field_id == 11:
                track_uuid = value
            elif field_id == 36:
                flow_ids.append(value)
            elif field_id == 42:
                terminating_flow_ids.append(value)

        if event_type != 1:
            return []

        call_stack = []
        for arg_key, arg_value in args.items():
            key = arg_key.split(".")[-1]
            if not key.startswith("call_stack_"):
                continue
            frame_index = int(key.removeprefix("call_stack_"))
            call_stack.append((frame_index, arg_value))
        normalized_call_stack = None
        if call_stack:
            call_stack.sort(key=lambda item: item[0])
            normalized_call_stack = [frame_name for _, frame_name in call_stack]

        events = [
            normalize_event(
                name,
                category,
                track_names.get(track_uuid, track_uuid),
                normalized_call_stack,
                args,
                timestamp=timestamp,
                track_id=track_uuid,
            )
        ]
        for phase, event_flow_ids in (("s", flow_ids), ("f", terminating_flow_ids)):
            for flow_id in event_flow_ids:
                events.append(
                    normalize_event(
                        "launch->kernel",
                        "flow",
                        track_names.get(track_uuid, track_uuid),
                        None,
                        {},
                        id=flow_id - _PERFETTO_FLOW_ID_BASE,
                        phase=phase,
                        bp="e",
                        timestamp=timestamp,
                        track_id=track_uuid,
                    ))
        return events

    def apply_interned_data(interned_data: bytes, interned):
        for field_id, value in fields(interned_data):
            if field_id == 1:
                iid, name = interned_string(value)
                interned["event_categories"][iid] = name
            elif field_id == 2:
                iid, name = interned_string(value)
                interned["event_names"][iid] = name
            elif field_id == 3:
                iid, name = interned_string(value)
                interned["debug_annotation_names"][iid] = name
            elif field_id == 29:
                iid, name = interned_string(value)
                interned["debug_annotation_string_values"][iid] = name

    def parse_packet(packet: bytes, interned, track_names):
        timestamp = 0
        sequence_flags = 0
        interned_payloads = []
        track_events = []
        parsed_track_names = {}

        for field_id, value in fields(packet):
            if field_id == 8:
                timestamp = value
            elif field_id == 11:
                track_events.append(value)
            elif field_id == 12:
                interned_payloads.append(value)
            elif field_id == 13:
                sequence_flags = value
            elif field_id == 60:
                uuid, name = parse_track_descriptor(value)
                if uuid is not None and name is not None:
                    parsed_track_names[uuid] = name

        if sequence_flags & 1:
            for table in interned.values():
                table.clear()
        for payload in interned_payloads:
            apply_interned_data(payload, interned)
        track_names.update(parsed_track_names)
        parsed_events = []
        for track_event in track_events:
            parsed_events.extend(parse_track_event(track_event, timestamp, interned, track_names))
        return parsed_events

    data = pathlib.Path(path).read_bytes()
    interned = {
        "event_categories": {},
        "event_names": {},
        "debug_annotation_names": {},
        "debug_annotation_string_values": {},
    }
    track_names = {}
    track_events = []

    for field_id, value in fields(data):
        if field_id == 1:
            track_events.extend(parse_packet(value, interned, track_names))

    return {"track_events": track_events}
