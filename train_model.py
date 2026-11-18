import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "applicant_income",
    "coapplicant_income",
    "loan_amount",
    "loan_term_months",
    "credit_score",
    "debt_to_income_ratio",
    "property_area",
    "self_employed",
]
TARGET_NAMES = ["Rejected", "Approved"]


def make_loan_dataset(n_samples=3000, seed=42):
    rng = np.random.default_rng(seed)

    applicant_income = np.clip(rng.normal(72000, 22000, n_samples), 15000, 180000)
    coapplicant_income = np.clip(rng.normal(15000, 12000, n_samples), 0, 90000)
    loan_amount = np.clip(rng.normal(165000, 50000, n_samples), 10000, 450000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 72, 84], size=n_samples)
    credit_score = np.clip(rng.normal(690, 105, n_samples), 300, 850)
    debt_to_income_ratio = np.clip(rng.uniform(0.08, 0.72, n_samples), 0.05, 0.90)
    property_area = rng.integers(0, 3, size=n_samples)
    self_employed = rng.binomial(1, 0.18, size=n_samples)

    credit_component = credit_score / 850
    income_coverage = np.clip(loan_amount / (applicant_income + coapplicant_income + 1), 0.0, 1.0)
    debt_component = np.clip(debt_to_income_ratio, 0.0, 1.0)
    term_component = np.clip(loan_term_months / 84, 0.0, 1.0)
    area_component = np.clip((2 - property_area) / 2, 0.0, 1.0)

    approval_score = (
        0.42 * credit_component
        + 0.20 * (1 - income_coverage)
        + 0.20 * (1 - debt_component)
        + 0.10 * area_component
        + 0.08 * (1 - self_employed)
    )

    approval_score += rng.normal(0, 0.05, n_samples)
    y = (approval_score >= 0.56).astype(int)

    data = np.column_stack(
        [
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term_months,
            credit_score,
            debt_to_income_ratio,
            property_area,
            self_employed,
        ]
    )
    return data, y


def main():
    X, y = make_loan_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Feature set: {FEATURE_NAMES}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

    model_package = {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
    }
    joblib.dump(model_package, "model.pkl")
    print("Saved trained loan approval model to model.pkl")


if __name__ == "__main__":
    main()
