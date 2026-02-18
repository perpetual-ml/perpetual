import numpy as np
import pandas as pd
from perpetual import PerpetualBooster
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


def explain_classification():
    print("\n--- Classification Explainability ---")
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    # Train model
    model = PerpetualBooster(objective="LogLoss", budget=1.0)
    model.fit(X_train, y_train)

    # 1. Feature Importance
    print("\n1. Feature Importance (Gain):")
    importance = model.calculate_feature_importance(method="Gain", normalize=True)
    # Sort by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_importance[:5]:  # Show top 5
        print(f"{feat}: {imp:.4f}")

    # 2. Partial Dependence
    print("\n2. Partial Dependence for top feature:")
    top_feature = sorted_importance[0][0]
    pd_values = model.partial_dependence(X_train, feature=top_feature, samples=10)
    print(f"Values for {top_feature}:")
    print(pd_values)

    # 3. Predict Contributions (SHAP-like)
    print("\n3. Prediction Contributions (SHAP-like) for first 2 samples:")
    # Calculate contributions
    contributions = model.predict_contributions(X_test.iloc[:2], method="Average")

    # contributions shape is (n_samples, n_features + 1), last column is bias
    bias = contributions[:, -1]
    feat_contribs = contributions[:, :-1]

    for i in range(2):
        print(f"\nSample {i}:")
        print(f"Bias: {bias[i]:.4f}")
        print("Top 3 contributing features:")
        # Get indices of top 3 absolute contributions
        top_idxs = np.argsort(np.abs(feat_contribs[i]))[-3:][::-1]
        for idx in top_idxs:
            print(f"{feature_names[idx]}: {feat_contribs[i, idx]:.4f}")

        prediction = model.predict(X_test.iloc[[i]])[0]
        # For LogLoss, contributions sum to the log-odds prediction
        sum_contribs = np.sum(feat_contribs[i]) + bias[i]
        print(f"Sum of contributions: {sum_contribs:.4f}")
        print(f"Model prediction: {prediction:.4f}")


def explain_regression():
    print("\n--- Regression Explainability ---")
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=5, random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    model = PerpetualBooster(objective="SquaredLoss", budget=1.0)
    model.fit(X_train, y_train)

    print("\n1. Feature Importance (Cover):")
    importance = model.calculate_feature_importance(method="Cover", normalize=True)
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_importance[:5]:
        print(f"{feat}: {imp:.4f}")


if __name__ == "__main__":
    explain_classification()
    explain_regression()
