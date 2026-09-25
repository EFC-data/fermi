Installation
============

Requirements
------------

Fermi requires Python 3.10 or newer. Its core dependencies include NumPy,
SciPy, pandas, scikit-learn, NetworkX, Matplotlib, Bokeh, tqdm, and WBNM.
WBNM installs PyTorch because its model solvers use tensors.

Reading XLSX files through ``MatrixProcessorCA.load()`` also requires a pandas
Excel engine, normally ``openpyxl``::

   python -m pip install openpyxl

Install a released version
--------------------------

Install the distribution from PyPI with the interpreter that will run Fermi::

   python -m pip install fermi-cref

The distribution and import names differ intentionally::

   import fermi

Install from source
-------------------

For development, clone WBNM and Fermi next to one another and install both in
editable mode::

   git clone https://github.com/lbuffa/wbnm.git
   git clone https://github.com/EFC-data/fermi.git
   python -m pip install -e ./wbnm -e ./fermi

An editable installation reflects source changes immediately; reinstalling
after every edit is unnecessary.

Development and documentation dependencies
------------------------------------------

From the Fermi repository::

   python -m pip install -r requirements-dev.txt

Run the tests and build the HTML documentation::

   python -m pytest
   python -m sphinx -W --keep-going -b html docs/source docs/build/html

Upgrade
-------

Upgrade WBNM before Fermi so the null-model API is available when Fermi is
imported::

   python -m pip install --upgrade wbnm fermi-cref

For editable checkouts, update the repositories and refresh their metadata::

   git -C wbnm pull --ff-only
   git -C fermi pull --ff-only
   python -m pip install -e ./wbnm -e ./fermi

Verify the installation
-----------------------

::

   import fermi
   import wbnm

   print(fermi.__version__)
   print(wbnm.__version__)

If several Python installations exist, prefer ``python -m pip`` over a bare
``pip`` command. This ensures that installation and execution use the same
interpreter.
