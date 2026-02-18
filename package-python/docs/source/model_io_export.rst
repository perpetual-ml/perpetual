Model IO & Export
=================

Perpetual provides several ways to save, load, and export your models for production use or integration with other tools.

Native Serialization
--------------------

The recommended way to save and load a ``PerpetualBooster`` is using the native ``save_booster`` and ``load_booster`` methods. These methods use a JSON-based format that includes both the model structure and any associated metadata.

.. code-block:: python

    from perpetual import PerpetualBooster

    # Save a fitted model
    model.save_booster("model.json")

    # Load the model
    loaded_model = PerpetualBooster.load_booster("model.json")

Pickling Support
----------------

``PerpetualBooster`` fully supports Python's ``pickle`` module, making it easy to use with tools like ``scikit-learn``'s ``Joblib`` or within distributed computing frameworks.

.. code-block:: python

    import pickle

    # Save with pickle
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Load with pickle
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

Metadata Persistence
--------------------

You can attach custom metadata to your model, which will be persisted when saving and restored when loading.

.. code-block:: python

    model.insert_metadata("experiment_id", "42")
    model.save_booster("model.json")

    # Later...
    loaded_model = PerpetualBooster.load_booster("model.json")
    print(loaded_model.get_metadata("experiment_id"))  # "42"

Exporting to XGBoost
--------------------

For integration with systems that already support XGBoost, you can export your ``PerpetualBooster`` to the XGBoost JSON format.

.. code-block:: python

    model.save_as_xgboost("model_xgb.json")

This allows you to load the model using the XGBoost library:

.. code-block:: python

    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model("model_xgb.json")

Exporting to ONNX
-----------------

For high-performance deployment across different platforms and languages, Perpetual supports exporting models to the ONNX (Open Neural Network Exchange) format.

.. code-block:: python

    model.save_as_onnx("model.onnx")

You can then run inference using any ONNX-compatible runtime:

.. code-block:: python

    import onnxruntime as rt
    sess = rt.InferenceSession("model.onnx")
    # ... run inference ...

Example
-------

For a complete working example demonstrating all these features, see `model_io_export.py <https://github.com/perpetual-ml/perpetual/blob/main/package-python/examples/model_io_export.py>`_.
