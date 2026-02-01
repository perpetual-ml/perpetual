Installation
============

You can install Perpetual via pip:

.. code-block:: bash

   pip install perpetual

You can also install Perpetual via Conda:

.. code-block:: bash

   conda install -c conda-forge perpetual

Requirements
------------

* Python >= 3.10
* numpy
* typing-extensions

Optional Dependencies
---------------------

* pandas: Enables support for training directly on Pandas DataFrames.
* polars: Enables zero-copy training support for Polars DataFrames.
* scikit-learn: Provides a scikit-learn compatible wrapper interface.
* xgboost: Enables saving and loading models in XGBoost format for interoperability.
* onnxruntime: Enables exporting and loading models in ONNX standard format.
