=============
 CMake Build
=============

.. contents::
   :local:

Usage
=====

.. code-block:: bash
   
   $ mkdir build
   $ cd build
   $ cmake ..

On my CentOs 6.3 linux system with a custom installation of boost, I
needed to invoke cmake with a special option as shown here to get
cmake to properly use the boost installation as referenced by the
environment variable :envvar:`BOOST_ROOT` or :envvar:`BOOST_DIR`.

.. code-block:: bash

   $ cmake -D Boost_NO_BOOST_CMAKE=ON ..

On windows I invoked cmake using:

.. code-block:: bash

   > cmake -G "Visual Studio 9 2008 Win64" ^
      -D CMAKE_INSTALL_PREFIX=c:/pkg/x64-vc90 ^
      -D CMAKE_PREFIX_PATH=c:/pkg/x64-vc90 ^
      -D CMAKE_CONFIGURATION_TYPES="Debug;Release" ^
      ..

Once you have the cmake generated build files you may build
Boost.NumPy. On linux you may build it using:

.. code-block:: bash

   $ make
   $ make install

On windows you may build it using:

.. code-block:: bash

   $ cmake --build . --config release
   $ cmake --build . --config release --target install

Note: You need to make sure that the cmake generator you use is
compatible with your python installation. The cmake scripts try to be
helpful, but the verification logic is incomplete. On both Linux and
Windows, I am using the 64-bit python from Enthought. On windows it is
built using Visual Studio 2008. I have also successfully used Visual
Studio 2010 for Boost.NumPy extension modules, but the VS 2010
generated executables that embed python do not run because of an
apparent conflict with the runtimes.

The build artifacts get installed to ``${CMAKE_INSTALL_PREFIX}``
:file:`include` :file:`lib` and :file:`boost.numpy` where the first
two are the conventional locations for header files and libraries (aka
archives, shared objects, DLLs). The last one :file:`boost.numpy` is
my guess at how to install the tests and examples in a place that is
useful. But it is likely that this will need to be tweaked once other
people start using it. Here is an outline of the installed files.

::

   boost.numpy/doc/BoostNumPy.pdf
            |   |- html/index.html
            |- example/demo_gaussian.py
            |      |-  dtype.exe
            |      |-  fromdata.exe
            |      |-  gaussian.pyd
            |      |-  ndarray.exe
            |      |-  simple.exe
            |      |-  ufunc.exe
            |      |-  wrap.exe
            |- test/dtype.py
                |-  dtype_mod.pyd
                |-  indexing.py
                |-  indexing_mod.pyd
                |-  ndarray.py
                |-  ndarray_mod.pyd
                |-  shapes.py
                |-  shapes_mod.pyd
                |-  templates.py
                |-  templates_mod.pyd
                |-  ufunc.py
                |-  ufunc_mod.pyd


You may develope and test without performing an install. The build
binary directory is configured so the executables are in the
:file:`build/bin` folder and the shared objects are in the
:file:`build/lib` folder. If you want to test then you simply need to
set the :envvar:`PYTHONPATH` environment variable to the lib folder
containing the shared object files so that python can find the
imported extension modules.

Details
=======

I borrowed from the python ``numexpr`` project the two ``.cmake``
files :file:`FindNumPy.cmake` and :file:`FindPythonLibsNew.cmake` in
:file:`libs/numpy/cmake`.

I followed a conventional structuring of the cmake
:file:`CMakeLists.txt` input files where the one at the top level
contains all of the configuration logic for the submodules that are
built.

If you want to also generate this documentation, invoke cmake with the
additional argument ``-DBUILD_DOCS=ON`` and make sure that the sphinx
package is in your path. You may build the documentation using ``make
doc-html``. If pdflatex is also in your path, then there is an
additonal target ``make doc-pdf`` that will generate the pdf manual.

CMakeLists.txt Source Files
===========================

For reference the source code of each of the new
:file:`CMakeLists.txt` files are included below.

Top-Level
---------

:file:`Boost.NumPy/CMakeLists.txt` where the parent subdirectory
:file:`Boost.NumPy` is ommited in directory references in the rest of
this section.

.. literalinclude:: ../../../CMakeLists.txt
   :language: cmake
   :linenos:

Library Source
--------------

The file :file:`libs/numpy/src/CMakeLists.txt` contains the build of the :file:`boost_numpy library`.

.. literalinclude:: ../src/CMakeLists.txt
   :language: cmake
   :linenos:

Tests
-----

The file :file:`libs/numpy/test/CMakeLists.txt` contains the python tests.

.. literalinclude:: ../test/CMakeLists.txt
   :language: cmake
   :linenos:

Examples
--------

The file :file:`libs/numpy/example/CMakeLists.txt` contains simple
examples (both an extension module and executables embedding python).

.. literalinclude:: ../example/CMakeLists.txt
   :language: cmake
   :linenos:




