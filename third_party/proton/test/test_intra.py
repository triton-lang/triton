import triton.intraprof as profiler


def test_setting_config():
    profiler.activate()

    profiler.init()
    assert profiler.intra.state.config.max_shared > 0

    config = {"max_shared": 32, "alloc_scratch": 50}
    profiler.init(config)
    assert profiler.intra.state.config.max_shared == config["max_shared"]
    assert profiler.intra.state.config.alloc_scratch == config["alloc_scratch"]
    profiler.deactivate()


def test_setting_state():
    assert not profiler.intra.activated
    profiler.activate()
    assert profiler.intra.activated
    profiler.deactivate()
    assert not profiler.intra.activated

    profiler.activate()
    profiler.init()
    state = profiler.finalize()
    assert state.profile_mem_cpu is not None
    assert profiler.intra.global_scratch_mem is None
    assert profiler.intra.state is None
    profiler.deactivate()
