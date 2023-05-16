# pip install transformers
# pip install --upgrade diffusers[torch]
# cd ../stuff/stable_diff
# python run.py

cd ../pytorch_rocm/
# TORCHINDUCTOR_COMPILE_THREADS=1 pytest test/inductor/test_torchinductor.py -k "test_views4_cuda"
pytest test/inductor/test_torchinductor.py -k "test_views4_cuda"
