from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model_package = joblib.load("model.pkl")
model = model_package["model"]
scaler = model_package["scaler"]
feature_names = model_package["feature_names"]
target_names = model_package["target_names"]
VALID_TERM_MONTHS = {12, 24, 36, 48, 60, 72, 84}
VALID_PROPERTY_AREAS = {0, 1, 2}
VALID_SELF_EMPLOYED = {0, 1}


def validate_values(values):
    applicant_income = values[0]
    coapplicant_income = values[1]
    loan_amount = values[2]
    loan_term_months = values[3]
    credit_score = values[4]
    debt_to_income_ratio = values[5]
    property_area = values[6]
    self_employed = values[7]

    if applicant_income <= 0 or coapplicant_income < 0 or loan_amount <= 0:
        return False, "Income and loan amount must be positive values."

    if loan_term_months not in VALID_TERM_MONTHS:
        return False, "Loan term months must be one of 12, 24, 36, 48, 60, 72, or 84."

    if credit_score < 300 or credit_score > 850:
        return False, "Credit score must stay between 300 and 850."

    if debt_to_income_ratio < 0.05 or debt_to_income_ratio > 0.9:
        return False, "Debt-to-income ratio must stay between 0.05 and 0.90."

    if property_area not in VALID_PROPERTY_AREAS:
        return False, "Property area must be 0, 1, or 2."

    if self_employed not in VALID_SELF_EMPLOYED:
        return False, "Self-employed flag must be either 0 or 1."

    return True, ""


def build_summary(values, prediction_label):
    applicant_income = values[0]
    coapplicant_income = values[1]
    loan_amount = values[2]
    loan_term_months = values[3]
    credit_score = values[4]
    debt_to_income_ratio = values[5]
    property_area = values[6]
    self_employed = values[7]

    monthly_income = (applicant_income + coapplicant_income) / 12
    monthly_emi = loan_amount / loan_term_months
    emi_to_income = monthly_emi / monthly_income if monthly_income > 0 else 0
    loan_to_income = loan_amount / (applicant_income + coapplicant_income) if applicant_income + coapplicant_income > 0 else 0
    satisfaction_score = round(
        (credit_score / 850) * 40
        + max(0, (1 - debt_to_income_ratio)) * 35
        + max(0, (1 - loan_to_income)) * 15
        + (1 if property_area > 0 else 0.5) * 10,
        1,
    )

    risk_state = "Low Risk" if prediction_label == "Approved" else "Manual Review"

    return {
        "income_total": round(applicant_income + coapplicant_income, 2),
        "loan_amount": round(loan_amount, 2),
        "loan_to_income_ratio": round(loan_to_income, 3),
        "emi_to_income_ratio": round(emi_to_income, 3),
        "credit_score": round(credit_score, 1),
        "debt_to_income_ratio": round(debt_to_income_ratio, 3),
        "property_area": int(property_area),
        "self_employed": int(self_employed),
        "satisfaction_score": satisfaction_score,
        "risk_state": risk_state,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        feature_names=feature_names,
        prediction=None,
        probability=None,
        values=None,
        summary=None,
        status_class=None,
        result_message=None,
        error_message=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    values = []
    for feature in feature_names:
        raw_value = request.form.get(feature)
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            return render_template(
                "index.html",
                feature_names=feature_names,
                prediction="Please enter valid numbers for all fields.",
                probability=None,
                values=None,
                summary=None,
                status_class="failure",
                result_message="Please complete each applicant field with a valid numeric value.",
                error_message="All applicant fields must be filled using the allowed range-safe inputs.",
            )

    is_valid, validation_message = validate_values(values)
    if not is_valid:
        return render_template(
            "index.html",
            feature_names=feature_names,
            prediction=None,
            probability=None,
            values=values,
            summary=None,
            status_class="failure",
            result_message=validation_message,
            error_message=validation_message,
        )

    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction_index = int(model.predict(X_scaled)[0])
    prediction_label = target_names[prediction_index]
    probability = float(model.predict_proba(X_scaled)[0][prediction_index])

    if prediction_index == 1:
        status_class = "success"
        result_message = (
            f"Approved for financing with {probability * 100:.1f}% confidence. "
            "The applicant profile aligns well with the bank's approval thresholds."
        )
    else:
        status_class = "failure"
        result_message = (
            f"Application requires manual review with {probability * 100:.1f}% confidence. "
            "Consider improving income coverage, reducing debt burden, or boosting credit profile."
        )

    summary = build_summary(values, prediction_label)

    return render_template(
        "index.html",
        feature_names=feature_names,
        prediction=prediction_label,
        probability=probability,
        result_message=result_message,
        values=values,
        summary=summary,
        status_class=status_class,
        error_message=None,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
