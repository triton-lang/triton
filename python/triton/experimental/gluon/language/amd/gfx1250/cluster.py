from triton.experimental.gluon.language._core import builtin

__all__ = ["arrive", "wait"]


@builtin
def arrive(_semantic=None):
    """
    Signals that the cluster has arrived at a cluster barrier, used to synchronize execution of CTAs within the same cluster.
    """
    _semantic.builder.create_amd_cluster_arrive()


@builtin
def wait(_semantic=None):
    """
    Wait on a cluster barrier to be arrived by all CTAs within the same cluster.
    Arrive and wait operations must come in pairs. Waiting before arriving or arriving more than once
    without a corresponding wait will result in undefined behavior.
    """
    _semantic.builder.create_amd_cluster_wait()
