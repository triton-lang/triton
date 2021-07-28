==============
Installation
==============

---------------------
Binary Distributions
---------------------

You can install the latest stable release of Triton from pip:

      pip install triton


And the latest nightly release:

.. code-block:: bash
  
      pip install -U --pre triton


--------------
From Source
--------------

+++++++++++++++
Python Package
+++++++++++++++

You can install the Python package from source by running the following commands:

.. code-block:: bash

      git clone https://github.com/ptillet/triton.git;
      cd triton/python;
      pip install -e .

You can then test your installation by running the unit tests:

.. code-block:: bash

      pytest -vs .

and the benchmarks

.. code-block:: bash
      
      cd bench/
      python -m run --with-plots --result-dir /tmp/triton-bench