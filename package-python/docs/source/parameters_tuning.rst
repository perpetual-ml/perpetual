Parameters Tuning
=================

Perpetual is designed to be "perpetual" – it doesn't require the traditional hyperparameter tuning (like learning rate, depth, etc.) that most GBDT libraries need.

The Budget Parameter
--------------------

The most important parameter in Perpetual is ``budget``.

* **Budget**: This controls the complexity and predictive power of the model. 
    * A higher budget (e.g., > 1.0) will result in more trees and potentially higher accuracy but longer training time.
    * A lower budget (default 0.5) is usually sufficient for most tasks.

How it works
------------

Perpetual automatically adjusts the learning rate (eta) and stopping criteria based on the budget. As you increase the budget, the algorithm becomes more conservative, adding more trees with smaller step sizes.
