***************
From Source
***************

Triton is a fairly self-contained package and uses its own parser (forked from `wgtcc <https://github.com/wgtdkp/wgtcc>`_) and LLVM-8.0+ for code generation. 

.. code-block:: bash

    sudo apt-get install llvm-8-dev
    git clone https://github.com/ptillet/triton.git;
    cd triton/python/;
    python setup.py develop;

This should take about 15-20 seconds to compile on  a modern machine.

You can then test your installation by running the *einsum.py* example in an environment that contains pytorch:

.. code-block:: bash

    cd examples;
    python einsum.py