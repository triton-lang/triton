from collections.abc import Iterable

def find_paths_if(iterable, pred):
    ret = []
    def _impl(current, path):
        if pred(current):
            if len(path) == 1:
                ret.append(path[0])
            else:
                ret.append(tuple(path))
        elif isinstance(current, Iterable):
            for idx, item in enumerate(current):
                _impl(item, path + [idx])
    _impl(iterable, [])
    return ret

val = (0, ("x", 0, (0, (0, "x")), 0), 0, 0, "x")
pred = lambda x: x == "x"
ret = find_paths_if(val, pred)
assert ret == [(1, 0), (1, 2, 1, 1), 4]