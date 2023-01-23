TT=~/triton 
TTC=~/triton/python/triton/tools/ttc.py 

rm -rf /tmp/aot_test
mkdir /tmp/aot_test 
mkdir /tmp/aot_test/build 
cp $TT/python/examples/* /tmp/aot_test 
cp $TT/aot_kernel_tests/test_config.yml /tmp/aot_test 
cp $TT/aot_kernel_tests/* /tmp/aot_test/build/. 
 
cd /tmp/aot_test 
python $TTC vector_addition.py --infer >> auto_infer_test.yml 
python $TTC vector_addition.py --config test_config.yml -o /tmp/aot_test/build 
 
cd build 
$(python build.py main.c add_kernel0.c -I /tmp/aot_test/build/ -o main); ./main 
