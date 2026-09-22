Welcome to Triton's documentation!
==================================

Triton_ is a language and compiler for parallel programming. It aims to provide a Python-based programming environment for productively writing custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware.


Getting Started
---------------

- Follow the :doc:`installation instructions <getting-started/installation>` for your platform of choice.
- Take a look at the :doc:`tutorials <getting-started/tutorials/index>` to learn how to write your first Triton program.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting-started/installation
   getting-started/tutorials/index


Triton API
----------

- :doc:`triton <python-api/triton>`
- :doc:`triton.language <python-api/triton.language>`
- :doc:`triton.testing <python-api/triton.testing>`
- :doc:`Triton semantics <python-api/triton-semantics>`
- :doc:`triton.language.extra.cuda <python-api/triton.language.extra.cuda>`


.. toctree::
   :maxdepth: 1
   :caption: Python API
   :hidden:

   python-api/triton
   python-api/triton.language
   python-api/triton.testing
   python-api/triton-semantics


Gluon
-----

- Learn about Gluon's lower-level programming model in the :doc:`Gluon overview <gluon/index>`.
- Browse the :doc:`Gluon tutorials <getting-started/tutorials/gluon/index>`,
  :doc:`examples <getting-started/examples/gluon/index>`, and :doc:`API reference <gluon/api/index>`.

.. toctree::
   :maxdepth: 1
   :caption: Gluon
   :hidden:

   Overview <gluon/index>
   Tutorials <getting-started/tutorials/gluon/index>
   Examples <getting-started/examples/gluon/index>
   API Reference <gluon/api/index>


Triton MLIR Dialects and Ops
----------------------------

- :doc:`Triton MLIR Dialects and Ops <dialects/dialects>`

.. toctree::
   :maxdepth: 1
   :caption: Triton MLIR Dialects
   :hidden:

   dialects/dialects

Going Further
-------------

Check out the following documents to learn more about Triton and how it compares against other DSLs for DNNs:

- Chapter 1: :doc:`Introduction <programming-guide/chapter-1/introduction>`
- Chapter 2: :doc:`Related Work <programming-guide/chapter-2/related-work>`
- Chapter 3: :doc:`Debugging <programming-guide/chapter-3/debugging>` and
  :doc:`Floating-Point Sanitizer (FpSan) <programming-guide/chapter-3/fpsan>`

.. toctree::
   :maxdepth: 1
   :caption: Programming Guide
   :hidden:

   programming-guide/chapter-1/introduction
   programming-guide/chapter-2/related-work
   programming-guide/chapter-3/debugging
   programming-guide/chapter-3/fpsan

.. _Triton: https://github.com/triton-lang/triton
