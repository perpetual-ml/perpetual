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

Ranking
-------

.. code-block:: python

    import numpy as np
    from perpetual import PerpetualBooster

    # Generate synthetic ranking data
    # 100 queries, each with 10 documents
    n_queries = 100
    n_docs_per_query = 10
    total_docs = n_queries * n_docs_per_query

    X = np.random.rand(total_docs, 5)  # 5 features
    y = np.random.rand(total_docs)     # Relevance scores

    # helper to create groups
    # The 'group' parameter tells the booster which rows belong to the same query
    group = np.full(n_queries, n_docs_per_query)

    model = PerpetualBooster(objective="ListNetLoss")
    model.fit(X, y, group=group)

    predictions = model.predict(X)


More Examples
-------------

You can find more examples in the `package-python/examples <https://github.com/perpetual-ml/perpetual/tree/main/package-python/examples>`_ directory on GitHub.

