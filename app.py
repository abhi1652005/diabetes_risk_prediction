import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load your model
MODEL_FILE = "svmc.pkl"
try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Features metadata with appropriate input types and dropdown options to prevent conversion errors
FEATURES_METADATA = [
    {"name": "age", "label": "Age", "type": "number", "step": "1", "default": "30"},
    {"name": "gender", "label": "Gender", "type": "select", "options": [("0", "Male"), ("1", "Female"), ("2", "Other")]},
    {"name": "city", "label": "City Code", "type": "number", "step": "1", "default": "0"},
    {"name": "bmi", "label": "BMI", "type": "number", "step": "0.1", "default": "24.5"},
    {"name": "family_history_diabetes", "label": "Family History of Diabetes", "type": "select", "options": [("0", "No"), ("1", "Yes")]},
    {"name": "physical_activity_level", "label": "Physical Activity Level", "type": "select", "options": [("0", "Low"), ("1", "Moderate"), ("2", "High")]},
    {"name": "diet_type", "label": "Diet Type", "type": "select", "options": [("0", "Standard"), ("1", "Vegetarian"), ("2", "Vegan"), ("3", "Keto")]},
    {"name": "smoking_status", "label": "Smoking Status", "type": "select", "options": [("0", "Non-Smoker"), ("1", "Smoker"), ("2", "Former")]},
    {"name": "alcohol_consumption", "label": "Alcohol Consumption", "type": "select", "options": [("0", "None"), ("1", "Low"), ("2", "Moderate"), ("3", "High")]},
    {"name": "hours_sleep_per_night", "label": "Hours Sleep Per Night", "type": "number", "step": "0.5", "default": "7"},
    {"name": "stress_level", "label": "Stress Level (1-10)", "type": "number", "step": "1", "default": "5"},
    {"name": "fasting_blood_sugar", "label": "Fasting Blood Sugar", "type": "number", "step": "0.1", "default": "95"},
    {"name": "hba1c_level", "label": "HbA1c Level", "type": "number", "step": "0.1", "default": "5.5"},
    {"name": "blood_pressure_systolic", "label": "BP Systolic", "type": "number", "step": "1", "default": "120"},
    {"name": "blood_pressure_diastolic", "label": "BP Diastolic", "type": "number", "step": "1", "default": "80"},
    {"name": "waist_circumference_cm", "label": "Waist Circumference (cm)", "type": "number", "step": "0.1", "default": "85"},
    {"name": "income_bracket", "label": "Income Bracket", "type": "select", "options": [("0", "Low"), ("1", "Medium"), ("2", "High")]},
]

FEATURE_COLUMNS = [f["name"] for f in FEATURES_METADATA]

# Attractive UI Template with Dark Gradient & Responsive Grid
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVC Health & Disease Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 30px;
            color: #334155;
        }
        .container {
            width: 100%;
            max-width: 800px;
            background: rgba(255, 255, 255, 0.96);
            backdrop-filter: blur(12px);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .header { text-align: center; margin-bottom: 30px; }
        .header h2 {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .header p { color: #64748b; font-size: 14px; }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
        }
        .form-group { display: flex; flex-direction: column; }
        label {
            font-size: 13px;
            font-weight: 600;
            color: #475569;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        input, select {
            width: 100%;
            padding: 12px 14px;
            background: #f8fafc;
            border: 1.5px solid #e2e8f0;
            border-radius: 10px;
            font-size: 14px;
            color: #1e293b;
            transition: all 0.2s ease;
            outline: none;
        }
        input:focus, select:focus {
            border-color: #6366f1;
            background: #ffffff;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
        }
        button {
            grid-column: span 2;
            margin-top: 15px;
            padding: 14px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: #ffffff;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
            background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
        }
        .result {
            margin-top: 25px;
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: 700;
            background: #e0e7ff;
            color: #3730a3;
            border: 1px solid #c7d2fe;
            animation: fadeIn 0.3s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 650px) {
            .form-grid { grid-template-columns: 1fr; }
            button { grid-column: span 1; }
            .container { padding: 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>SVC Model Prediction Portal</h2>
            <p>Provide the health and lifestyle parameters below</p>
        </div>
        <form action="/predict" method="POST" class="form-grid">
            {% for item in metadata %}
            <div class="form-group">
                <label for="{{ item.name }}">{{ item.label }}:</label>
                {% if item.type == 'select' %}
                <select id="{{ item.name }}" name="{{ item.name }}" required>
                    {% for val, label in item.options %}
                    <option value="{{ val }}">{{ label }}</option>
                    {% endfor %}
                </select>
                {% else %}
                <input type="number" step="{{ item.step }}" id="{{ item.name }}" name="{{ item.name }}" value="{{ item.default }}" required>
                {% endif %}
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
    return render_template_string(HTML_TEMPLATE, metadata=FEATURE_METADATA)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE,
            metadata=FEATURE_METADATA,
            prediction_text="Error: Model failed to load on server.",
        )

    try:
        # Safely extract numerical and encoded categorical values from form submission
        features = []
        for col in FEATURE_COLUMNS:
            val = request.form.get(col)
            features.append(float(val) if val else 0.0)

        # Make prediction using the SVC model
        prediction = model.predict([features])
        result = int(prediction[0])

        return render_template_string(
            HTML_TEMPLATE,
            metadata=FEATURE_METADATA,
            prediction_text=f"Prediction Result: {result}",
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            metadata=FEATURE_METADATA,
            prediction_text=f"Error: {str(e)}",
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
