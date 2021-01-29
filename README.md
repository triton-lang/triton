# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

The foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please consider citing us if you use our work!


## Installation

You can install the latest release with pip as follows:
```
sudo apt-get install llvm-10-dev
pip install triton
```

or the latest development version with:
```
 pip install -e "git+https://github.com/ptillet/triton.git#egg=triton&subdirectory=python"
```

for the C++ package:
```
git clone https://github.com/ptillet/triton.git;
cd triton;
mkdir build;
cd build;
cmake ../;
make -j8;
```


## Getting Started

You can find tutorials for Triton for [Python](https://github.com/ptillet/triton/tree/master/tutorials) and [C++](https://github.com/ptillet/triton/tree/master/python/tutorials).