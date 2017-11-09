import sys
import os
import numpy as np
import itertools
from time import time

class ProgressBar:

    def __init__(self, prefix, length = 25):
        self.length = length
        self.prefix = prefix
        sys.stdout.write("{0}: [{1}] {2: >3}%".format(prefix.ljust(17), ' '*self.length, 0))
        sys.stdout.flush()

    def __del__(self):
        sys.stdout.write("\n")

    def update(self, i, total):
        percent = float(i + 1) / total
        hashes = '#' * int(round(percent * self.length))
        spaces = ' ' * (self.length - len(hashes))
        percentformat = int(round(percent * 100))
        sys.stdout.write(("\r{0}: [{1}] {2: >3}%").format(self.prefix.ljust(17), hashes + spaces, percentformat))
        sys.stdout.flush()

def load(path, Ns):
    if os.path.exists(path):
        data = np.load(path)
        return [data[x] for x, _ in Ns]
    else:
        return [np.empty((0, n)) for _, n in Ns]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def cartesian_coord(arrays):
    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points

def cartesian_iterator(arrays):
    N = len(arrays)
    split = [np.array_split(ary, min(len(ary), 2) if i < 4 else 1) for i, ary in enumerate(arrays)]
    for x in itertools.product(*split):
        yield cartesian_coord(x)

def benchmark(fn, device, nsec):
    total, hist = 0, []
    fn()
    while total < nsec:
        #norm = device.current_sm_clock/device.max_sm_clock #* device.current_mem_clock/device.max_mem_clock
        norm = 1
        start = time()
        fn()
        end = time()
        hist.append(norm*(end - start))
        total += hist[-1]
    return min(hist)
