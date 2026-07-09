============
Installation
============

For supported platform/OS and supported hardware, review the `Compatibility <https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility>`_ section on Github.

--------------------
Binary Distributions
--------------------

You can install the latest stable release of Triton from pip:

.. code-block:: bash

      pip install triton

Binary wheels are available for CPython 3.10-3.14.

-----------
From Source
-----------

++++++++++++++
Python Package
++++++++++++++

You can install the Python package from source by running the following commands:

.. code-block:: bash

      git clone https://github.com/triton-lang/triton.git
      cd triton

      pip install -r python/requirements.txt # build-time dependencies
      pip install -e .

Note that, if llvm is not present on your system, the setup.py script will download the official LLVM static libraries and link against that.

For building with a custom LLVM, review the `Building with a custom LLVM <https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm>`_ section on Github.

You can then test your installation by running the tests:

.. code-block:: bash

      # One-time setup
      make dev-install

      # To run all tests (requires a GPU)
      make test

      # Or, to run tests without a GPU
      make test-nogpu

--------------------------
Hardware-specific notes
--------------------------

NVIDIA Blackwell sm\_121 (GB10 / DGX Spark)
++++++++++++++++++++++++++++++++++++++++++++

PyTorch 2.9 advertises maximum compute capability ``sm_120``, but Blackwell consumer parts
identify as ``sm_121``. Triton-compiled kernels fail with:

  ``RuntimeError: Triton Error [CUDA]: no kernel image is available for execution on the device``

To run Triton on ``sm_121``, point Triton at a system ``ptxas`` that understands the target
(CUDA 13.0 or later) and set the CUDA arch list to include native ``sm_121`` with a PTX
fallback:

.. code-block:: bash

      export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
      export TORCH_CUDA_ARCH_LIST="12.1+PTX"
      unset TRITON_OVERRIDE_ARCH

The PyTorch-bundled ``ptxas`` predates ``sm_121``. Switching to the system ``ptxas`` via
``TRITON_PTXAS_PATH`` is the whole fix — every other Triton internal is unchanged.

.. warning::

   Setting ``TRITON_OVERRIDE_ARCH=sm90`` (or any non-Blackwell arch) will produce kernels
   that the Blackwell driver rejects. This is the most common misconfiguration.

Verified on: NVIDIA GB10 (DGX Spark), PyTorch 2.9.0+cu130, Triton 3.5, CUDA 13.0,
Ubuntu 24.04.
