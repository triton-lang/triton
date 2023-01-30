TT=~/triton 
TTC=~/triton/python/triton/tools/ttc.py 

git clone https://github.com/gaxler/triton.git
cd triton/python
git checkout aot-mlir-backend 
pip install cmake
pip install -e .
