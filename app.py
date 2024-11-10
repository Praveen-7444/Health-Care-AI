from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and binner
model_file_path = './models/best_model.pkl'
scaler_file_path = './models/scaler.pkl'
binner_file_path = './models/binner.pkl'

with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)
with open(scaler_file_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open(binner_file_path, 'rb') as binner_file:
    binner = pickle.load(binner_file)

# Home route to render input form
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        blood_pressure = float(request.form['blood_pressure'])
        
        # Feature engineering
        bmi = weight / (height / 100) ** 2
        bp_binned = binner.transform(np.array([[blood_pressure]]))

        # Combine the features
        features = np.array([[age, weight, height, bmi, bp_binned[0, 0]]])
        features_scaled = scaler.transform(features)

        # Make a prediction
        prediction = model.predict(features_scaled)[0]
        
        # Render result page
        return render_template('result.html', name=name, age=age, weight=weight,
                               height=height, blood_pressure=blood_pressure, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
