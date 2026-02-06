import numpy as np
from perpetual.iv import BraidedBooster
from sklearn.model_selection import train_test_split


def generate_iv_data(n=10000):
    """
    Generates synthetic data with unobserved confounding.
    y = beta * w + gamma * U + eps
    w = pi * z + delta * U + nu
    """
    U = np.random.randn(n)  # Unobserved confounder
    z = np.random.randn(n)  # Instrument

    # Treatment w is affected by z and U
    w = 0.8 * z + 0.5 * U + np.random.randn(n) * 0.1

    # Outcome y is affected by w and U (confounding)
    # True causal effect of w on y is 2.0
    y = 2.0 * w + 1.2 * U + np.random.randn(n) * 0.1

    X = np.random.randn(n, 2)  # Other covariates
    Z = z.reshape(-1, 1)  # Instrumental Variables

    return X, Z, y, w


if __name__ == "__main__":
    X, Z, y, w = generate_iv_data()
    X_train, X_test, Z_train, Z_test, y_train, y_test, w_train, w_test = (
        train_test_split(X, Z, y, w, test_size=0.2)
    )

    print("Fitting BraidedBooster (BoostIV)...")
    model = BraidedBooster(stage1_budget=0.1, stage2_budget=0.1)
    model.fit(X_train, Z_train, y_train, w_train)

    print("Estimating causal effect (Counterfactual Prediction)...")
    # To find the effect, we predict at w=1 vs w=0
    w_1 = np.ones(len(X_test))
    w_0 = np.zeros(len(X_test))

    y_1 = model.predict(X_test, w_counterfactual=w_1)
    y_0 = model.predict(X_test, w_counterfactual=w_0)

    estimated_effect = np.mean(y_1 - y_0)
    print(f"Estimated Causal Effect: {estimated_effect:.4f} (True: 2.0000)")

    # Compare with a naive model (which will be biased by U)
    from perpetual import PerpetualBooster

    naive = PerpetualBooster()
    X_naive = np.column_stack([X_train, w_train])
    naive.fit(X_naive, y_train)

    X_test_1 = np.column_stack([X_test, np.ones(len(X_test))])
    X_test_0 = np.column_stack([X_test, np.zeros(len(X_test))])
    naive_effect = np.mean(naive.predict(X_test_1) - naive.predict(X_test_0))
    print(f"Naive Model Effect (Biased): {naive_effect:.4f}")
