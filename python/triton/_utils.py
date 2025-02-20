from functools import reduce


def get_iterable_path(iterable, path):
    return reduce(lambda a, idx: a[idx], path, iterable)


def set_iterable_path(iterable, path, val):
    prev = iterable if len(path) == 1 else get_iterable_path(iterable, path[:-1])
    prev[path[-1]] = val


def find_paths_if(iterable, pred):
    from .language import core
    is_iterable = lambda x: isinstance(x, (list, tuple, core.tuple, core.tuple_type))
    ret = dict()

    def _impl(current, path):
        path = (path[0], ) if len(path) == 1 else tuple(path)
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl(item, path + (idx, ))
        elif pred(path, current):
            if len(path) == 1:
                ret[(path[0], )] = None
            else:
                ret[tuple(path)] = None

    if is_iterable(iterable):
        _impl(iterable, [])
    elif pred(list(), iterable):
        ret = {tuple(): None}
    else:
        ret = dict()
    return list(ret.keys())
