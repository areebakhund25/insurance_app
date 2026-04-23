from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model (saved with joblib)
with open("insurance_linear_model.joblib", "rb") as file:
    model = joblib.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        age = int(request.form["age"])
        sex = int(request.form["sex"])          # female=1, male=0
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = int(request.form["smoker"])    # yes=1, no=0

        # Interaction features
        age_smoker = age * smoker
        bmi_smoker = bmi * smoker

        # Prepare features for prediction
        features = np.array([[age, sex, bmi, children, smoker, age_smoker, bmi_smoker]])
        prediction = model.predict(features)[0]

        # Render template with prediction
        return render_template(
            "index.html",
            prediction_text=f"Estimated Insurance Charges: ${prediction:.2f}"
        )

    except Exception as e:
        # In case of error
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's port or default 5000
    app.run(host="0.0.0.0", port=port, debug=False)
