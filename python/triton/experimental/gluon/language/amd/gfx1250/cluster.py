from triton.experimental.gluon.language._core import builtin

__all__ = ["signal", "wait"]


@builtin
def signal(_semantic=None):
    """
    Signal a cluster barrier used to synchronize CTAs within the same cluster.
    """
    _semantic.builder.create_cluster_barrier_signal()


@builtin
def wait(_semantic=None):
    """
    Wait on a cluster barrier to be signaled by all CTAs within the same cluster.
    Signal and wait operations must come in pairs. Waiting before signalling or signaling more than once
    without a corresponding wait will result in undefined behavior.
    """
    _semantic.builder.create_cluster_barrier_wait()
