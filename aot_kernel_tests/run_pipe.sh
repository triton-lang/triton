TT=~/triton 
TTC=~/triton/python/triton/aot/ttc.py 
TTL=~/triton/python/triton/aot/ttl.py 

rm -rf /tmp/aot_test
mkdir /tmp/aot_test 
mkdir /tmp/aot_test/build 
cp $TT/aot_kernel_tests/src_to_compile/* /tmp/aot_test 
cp $TT/aot_kernel_tests/build_ex/* /tmp/aot_test/build/. 
 
cd /tmp/aot_test 
python $TTC vector_addition.py -n add_kernel --signature *fp32:16 *fp32:16 *fp32:16 i32:16 --BLOCK_SIZE 64 --out-name vec_add_64 -o /tmp/aot_test/build/add_kernel64
python $TTC vector_addition.py -n add_kernel --signature *fp32:16 *fp32:16 *fp32:16 i32:16 --BLOCK_SIZE 128 --out-name vec_add_128 -o /tmp/aot_test/build/add_kernel128
python $TTL /tmp/aot_test/build/add_kernel64.h /tmp/aot_test/build/add_kernel128.h >> /tmp/aot_test/build/dispatcher.c 
 
cd build 
# Use preprocessor to dump the linked dispatcher.c into main.c (thats why no dispatcher passed to compiler)
$(python build.py main.c add_kernel64.c add_kernel128.c -I /tmp/aot_test/build/ -o main); ./main 
