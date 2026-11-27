import os


def pytest_configure(config):
    if os.environ.get("TRITON_TEST_NUM_GPUS"):
        return
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None and worker_id.startswith("gw"):
        gpu_id = int(worker_id[2:])  # map gw0 → 0, gw1 → 1, ...
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            # Spread the workers over the devices the caller made visible, not over all of them.
            devices = visible.split(",")
        else:
            import torch
            devices = [str(i) for i in range(torch.cuda.device_count())]
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[gpu_id % len(devices)]
