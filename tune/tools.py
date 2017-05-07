import sys
import os
import numpy as np

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
    try:
        data = np.load(path)
        return [data[x] for x, _ in Ns]
    except OSError:
        return [np.empty((0, n)) for _, n in Ns]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
