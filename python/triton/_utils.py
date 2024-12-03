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

def find_paths_if(iterable, pred):
    from .language import core
    is_iterable = lambda x: isinstance(x, (list, tuple, core.tuple, core.tuple_type))
    ret = dict()
    def _impl(current, path):
        path = (path[0], ) if len(path) == 1 else tuple(path)
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl(item, path + (idx,))
        elif pred(path, current):
            if len(path) == 1:
                ret[(path[0],)] = current
            else:
                ret[tuple(path)] = current
    if is_iterable(iterable):
        _impl(iterable, [])
    elif pred(list(), iterable):
        ret = {tuple(): iterable}
    else:
        ret = dict()
    return ret

def parse_list_string(s):
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    result = []
    current = ''
    depth = 0
    for c in s:
        if c == '[':
            depth += 1
            current += c
        elif c == ']':
            depth -= 1
            current += c
        elif c == ',' and depth == 0:
            result.append(current.strip())
            current = ''
        else:
            current += c
    if current.strip():
        result.append(current.strip())
    return result
