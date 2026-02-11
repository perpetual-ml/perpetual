import numpy as np
import pandas as pd
from perpetual import PerpetualBooster, PerpetualRiskEngine
from sklearn.model_selection import train_test_split


def generate_loan_data(n=5000):
    """Generates synthetic loan approval data."""
    np.random.seed(42)
    # Features: Income, FICO, Age, Debt-to-Income
    income = np.random.lognormal(11, 0.5, n)
    fico = np.random.randint(500, 850, n)
    age = np.random.randint(18, 80, n)
    dti = np.random.uniform(0.1, 0.6, n)

    X = pd.DataFrame({"income": income, "fico": fico, "age": age, "dti": dti})

    # Probability of default
    logits = -0.0001 * income - 0.02 * fico + 5.0 * dti + 10.0
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > np.random.rand(n)).astype(int)

    return X, y


if __name__ == "__main__":
    print("Generating synthetic loan data...")
    X, y = generate_loan_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Fitting PerpetualBooster with Monotonicity Constraints...")
    # Income and FICO should decrease default probability, DTI should increase it
    model = PerpetualBooster(
        objective="LogLoss",
        budget=0.5,
        monotone_constraints={"income": -1, "fico": -1, "dti": 1},
    )
    model.fit(X_train, y_train)

    print("Initializing PerpetualRiskEngine...")
    engine = PerpetualRiskEngine(model)

    # Pick a sample that was rejected (y_pred high)
    probs_test = model.predict_proba(X_test)
    rejected_idx = np.where(probs_test > 0.5)[0][0]
    X_rejected = X_test.iloc[[rejected_idx]]

    print(f"\nAnalyzing reject for applicant at idx {rejected_idx}:")
    print(X_rejected)
    print(f"Probability of Default: {probs_test[rejected_idx, 1]:.4f}")

    reasons = engine.generate_reason_codes(
        X_rejected, threshold=0.3, rejection_direction="higher"
    )

    print("\nAdverse Action Codes (Top Reasons for rejection):")
    for r in reasons[0]:
        print(f"- {r}")
