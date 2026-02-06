import numpy as np
from perpetual.uplift import UpliftBooster
from sklearn.model_selection import train_test_split


def generate_hillstrom_synthetic(n=10000):
    """Generates synthetic data inspired by Hillstrom dataset."""
    X = np.random.randn(n, 10)
    # Treatment effect is heterogeneous
    w = np.random.randint(0, 2, n)
    # Baseline outcome
    y_base = 0.5 * X[:, 0] + 0.3 * X[:, 1]
    # Heterogeneous treatment effect
    te = 0.4 * X[:, 0] + 0.2 * np.power(X[:, 2], 2)
    y = y_base + w * te + np.random.randn(n) * 0.1
    return X, w, y


if __name__ == "__main__":
    print("Generating synthetic Hillstrom data...")
    X, w, y = generate_hillstrom_synthetic()
    X_train, X_test, w_train, w_test, y_train, y_test = train_test_split(
        X, w, y, test_size=0.2
    )

    print("Fitting UpliftBooster (R-Learner)...")
    model = UpliftBooster(outcome_budget=0.1, propensity_budget=0.01, effect_budget=0.1)
    model.fit(X_train, w_train, y_train)

    print("Predicting CATE (Conditional Average Treatment Effect)...")
    cate_pred = model.predict(X_test)

    # In reality, we don't know the true effect, but since it's synthetic we can compare
    te_true = 0.4 * X_test[:, 0] + 0.2 * np.power(X_test[:, 2], 2)
    corr = np.corrcoef(cate_pred, te_true)[0, 1]
    print(f"Correlation with true Treatment Effect: {corr:.4f}")
