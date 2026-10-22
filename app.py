import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load your model (Replace 'model.pkl' with your actual filename if different)
MODEL_FILE = "svmc.pkl"
try:
  with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
except Exception as e:
  model = None
  print(f"Error loading model: {e}")

# Expected features based on your model's training features
FEATURE_COLUMNS = [
    "age",
    "gender",
    "city",
    "bmi",
    "family_history_diabetes",
    "physical_activity_level",
    "diet_type",
    "smoking_status",
    "alcohol_consumption",
    "hours_sleep_per_night",
    "stress_level",
    "fasting_blood_sugar",
    "hba1c_level",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "waist_circumference_cm",
    "income_bracket",
]

# HTML template embedded directly as a string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVC Model Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f9; margin: 0; padding: 20px; display: flex; justify-content: center; }
        .container { width: 100%; max-width: 600px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); }
        h2 { text-align: center; color: #333; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
        button { background: #007BFF; color: white; padding: 12px; border: none; border-radius: 4px; cursor: pointer; width: 100%; font-size: 16px; font-weight: bold; margin-top: 10px; }
        button:hover { background: #0056b3; }
        .result { margin-top: 25px; padding: 15px; background: #e9ecef; border-left: 5px solid #007BFF; font-size: 18px; font-weight: bold; text-align: center; color: #333; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>SVC Model Prediction Form</h2>
        <form action="/predict" method="POST">
            {% for feature in features %}
            <div class="form-group">
                <label for="{{ feature }}">{{ feature | replace('_', ' ') | title }}:</label>
                <input type="text" id="{{ feature }}" name="{{ feature }}" required>
            </div>
            {% endfor %}
            <button type="submit">Run Prediction</button>
        </form>

        {% if prediction_text %}
        <div class="result">
            {{ prediction_text }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/")
def home():
  return render_template_string(HTML_TEMPLATE, features=FEATURE_COLUMNS)


@app.route("/predict", methods=["POST"])
def predict():
  if model is None:
    return render_template_string(
        HTML_TEMPLATE,
        features=FEATURE_COLUMNS,
        prediction_text="Error: Model failed to load on server.",
    )

  try:
    # Extract features from form submission and convert to numeric format
    features = []
    for col in FEATURE_COLUMNS:
      val = request.form.get(col)
      features.append(float(val) if val else 0.0)

    # Make prediction using the SVC model
    prediction = model.predict([features])
    result = int(prediction[0])

    return render_template_string(
        HTML_TEMPLATE,
        features=FEATURE_COLUMNS,
        prediction_text=f"Prediction Result: {result}",
    )
  except Exception as e:
    return render_template_string(
        HTML_TEMPLATE,
        features=FEATURE_COLUMNS,
        prediction_text=f"Error: {str(e)}",
    )


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
