Tutorials
=========

Below is a gallery of tutorials for writing various basic operations with Triton. It is recommended that you read through the tutorials in order, starting with the simplest one.

To install the dependencies for the tutorials:

.. code-block:: bash

    cd triton
    pip install -e './python[tutorials]'

**Note**: These tutorials have been updated to use the stable PyTorch device API (`torch.device("cuda:0")`) instead of the Triton driver API to ensure compatibility across different Triton versions. If you encounter device-related errors, see the `troubleshooting guide <https://triton-lang.org/main/getting-started/troubleshooting.html>`_ for more details.
