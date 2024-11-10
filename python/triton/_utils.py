from typing import Tuple, List, Any

# Poor man's PyTree


def list_list_flatten(x: List[List[Any]]) -> Tuple[List[int], List[Any]]:
    spec = []
    flat = []
    for l in x:
        spec.append(len(l))
        flat.extend(l)
    return spec, flat


def list_list_unflatten(spec: List[int], flat: List[Any]) -> List[List[Any]]:
    ret = []
    idx = 0
    for size in spec:
        ret.append(flat[idx:idx + size])
        idx += size
    assert idx == len(flat)
    return ret
