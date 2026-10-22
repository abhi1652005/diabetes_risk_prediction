import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model without changing the pkl file name (update 'model.pkl' if your file has a different name)
with open('svmc.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form submission matching your model's expected inputs
        features = [
            float(request.form['age']),
            float(request.form['gender']), # Ensure your frontend/preprocessing handles categorical encoding if needed
            float(request.form['city']),
            float(request.form['bmi']),
            float(request.form['family_history_diabetes']),
            float(request.form['physical_activity_level']),
            float(request.form['diet_type']),
            float(request.form['smoking_status']),
            float(request.form['alcohol_consumption']),
            float(request.form['hours_sleep_per_night']),
            float(request.form['stress_level']),
            float(request.form['fasting_blood_sugar']),
            float(request.form['hba1c_level']),
            float(request.form['blood_pressure_systolic']),
            float(request.form['blood_pressure_diastolic']),
            float(request.form['waist_circumference_cm']),
            float(request.form['income_bracket'])
        ]
        
        # Reshape for single prediction
        final_input = np.array([features])
        prediction = model.predict(final_input)[0]
        
        return render_template('index.html', prediction_text=f'Prediction Result: {prediction}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error in processing: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
