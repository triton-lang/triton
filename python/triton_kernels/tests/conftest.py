import os


def pytest_configure(config):
    if os.environ.get("TRITON_TEST_NUM_GPUS"):
        return
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None and worker_id.startswith("gw"):
        import torch
        gpu_id = int(worker_id[2:])  # map gw0 → 0, gw1 → 1, ...
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id % torch.cuda.device_count())
