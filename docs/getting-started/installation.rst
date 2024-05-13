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

Binary wheels are available for CPython 3.8-3.12 and PyPy 3.8-3.9.

And the latest nightly release:

.. code-block:: bash

      pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly


-----------
From Source
-----------

++++++++++++++
Python Package
++++++++++++++

You can install the Python package from source by running the following commands:

.. code-block:: bash

      git clone https://github.com/triton-lang/triton.git;
      cd triton/python;
      pip install ninja cmake wheel; # build-time dependencies
      pip install -e .

Note that, if llvm is not present on your system, the setup.py script will download the official LLVM static libraries and link against that.

For building with a custom LLVM, review the `Building with a custom LLVM <https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm>`_ section on Github.

You can then test your installation by running the unit tests:

.. code-block:: bash

      pip install -e '.[tests]'
      pytest -vs test/unit/

and the benchmarks

.. code-block:: bash

      cd bench
      python -m run --with-plots --result-dir /tmp/triton-bench
