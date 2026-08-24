from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Load trained AI model
model = joblib.load("models/predictive_maintenance_model.pkl")

# Create history folder/file
HISTORY_FILE = "prediction_history.csv"


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    health_score = None
    recommendation = None
    risk_level = None

    if request.method == "POST":

        # Get sensor values
        air_temperature = float(request.form["air_temperature"])
        process_temperature = float(request.form["process_temperature"])
        rotational_speed = float(request.form["rotational_speed"])
        torque = float(request.form["torque"])
        tool_wear = float(request.form["tool_wear"])

        # Create input data
        input_data = pd.DataFrame([{
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])

        # AI Prediction
        result = model.predict(input_data)[0]

        # Failure probability
        probability = model.predict_proba(input_data)[0][1] * 100

        # Health Score
        health_score = 100 - probability

        # Prediction + Recommendation
        if result == 1:
            prediction = "Machine Failure Risk"
            recommendation = "Immediate maintenance inspection is recommended."
        else:
            prediction = "Machine Healthy"
            recommendation = "Machine is operating normally. Continue regular monitoring."

        # Risk Level
        if probability < 30:
            risk_level = "Low Risk"
        elif probability < 70:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        # Save Prediction History
        history_data = pd.DataFrame([{
            "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Air Temperature": air_temperature,
            "Process Temperature": process_temperature,
            "Rotational Speed": rotational_speed,
            "Torque": torque,
            "Tool Wear": tool_wear,
            "Prediction": prediction,
            "Risk Level": risk_level,
            "Failure Probability (%)": round(probability, 2),
            "Health Score (%)": round(health_score, 2)
        }])

        # Create CSV or append new data
        if os.path.exists(HISTORY_FILE):
            history_data.to_csv(
                HISTORY_FILE,
                mode="a",
                header=False,
                index=False
            )
        else:
            history_data.to_csv(
                HISTORY_FILE,
                index=False
            )

    # Dashboard Statistics
    total_predictions = 0
    healthy_count = 0
    failure_count = 0
    average_probability = 0

    if os.path.exists(HISTORY_FILE):

        history = pd.read_csv(HISTORY_FILE)

        total_predictions = len(history)

        healthy_count = len(
            history[history["Prediction"] == "Machine Healthy"]
        )

        failure_count = len(
            history[
                history["Prediction"] == "Machine Failure Risk"
            ]
        )

        average_probability = round(
            history["Failure Probability (%)"].mean(),
            2
        )

    return render_template(
        "index.html",

        prediction=prediction,
        probability=probability,
        health_score=health_score,
        recommendation=recommendation,
        risk_level=risk_level,

        total_predictions=total_predictions,
        healthy_count=healthy_count,
        failure_count=failure_count,
        average_probability=average_probability
    )


if __name__ == "__main__":

    print("Starting Predictive Maintenance AI System...")

    app.run(
        debug=True
    )