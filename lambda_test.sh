TT=~/triton
TTC=~/triton/python/triton/tools/ttc.py

git clone https://github.com/gaxler/triton.git
cd triton/python
git checkout tmp
pip install cmake
pip install -e .

mkdir /tmp/aot_test
cp examples/* /tmp/aot_test
cp triton/tools/test.yml /tmp/aot_test
cp $TT/aot_kernel_tests/* /tmp/aot_test/build/.
cp $TT/aot_kernel_tests /tmp/aot_test

cd /tmp/aot_test
# python $TTPY/triton/tools/ttc.py vector_addition.py --infer >> dummy.yml
python $TTC vector_addition.py --infer >> auto_infer_test.yml

mkdir build
python $TTC vector_addition.py --config test.yml -o /tmp/aot_test/build
$(python build.py main.c add_kernel0.c -I /tmp/aot_test/build/ -o main); ./main