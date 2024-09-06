# Control Flow Graph Generator from AMDGCN assembly

The script reads an assembly file and generates a Control Flow Graph (CFG) for each function in the file. The graph can be saved in `dot`, `svg` and `pdf` formats. The nodes of a graph can be represented with 1) just labels or 2) the corresponding assembly code. The edges of a graph can help to identify cycles and, thus, to provide a better navigation through the code.


### Basic usage

```
python ./amdgcn-cfg.py -i <path to assembly file>  -o <output directory>/<output prefix> -f [dot|svg|pdf]
```

`dot`-files can be visualize with [this](https://dreampuf.github.io/GraphvizOnline) online tool. You just need to copy and paste the content of a generated `dot`-file.

By default, the nodes are named with basic block labels. Use `-v` or `--verbose` option to add assembly source code to corresponding nodes.
