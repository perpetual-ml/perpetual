Quickstart
==========

Here is a simple example to get you started with Perpetual.

Classification
--------------

.. code-block:: python

    from perpetual import PerpetualBooster
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = PerpetualBooster(objective="LogLoss")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

Regression
----------

.. code-block:: python

    from perpetual import PerpetualBooster
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=20)
    model = PerpetualBooster(objective="SquaredLoss")
    model.fit(X, y)

    predictions = model.predict(X)
