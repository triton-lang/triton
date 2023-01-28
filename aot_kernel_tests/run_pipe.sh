TT=~/triton 
TTC=~/triton/python/triton/aot/tt.py 
OUT_NAME=add_kernel0


rm -rf /tmp/aot_test
mkdir /tmp/aot_test 
mkdir /tmp/aot_test/build 
cp $TT/aot_kernel_tests/src_to_compile/* /tmp/aot_test 
cp $TT/aot_kernel_tests/build_ex/* /tmp/aot_test/build/. 
 
cd /tmp/aot_test 
# python $TTC vector_addition.py --config test_config.yml -o /tmp/aot_test/build 
python $TTC vector_addition.py -n add_kernel --signature *fp32:16 *fp32:16 *fp32:16 i32:16 --BLOCK_SIZE 64 --out-name vec_add_64 -o /tmp/aot_test/build/$OUT_NAME
 
cd build 
# $(python build.py main.c add_kernel0.c -I /tmp/aot_test/build/ -o main); ./main 
$(python build.py main.c $OUT_NAME.c -I /tmp/aot_test/build/ -o main); ./main 
