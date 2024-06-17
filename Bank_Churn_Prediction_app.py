from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('churn_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("Received data:", data)  # Debugging statement

        # Extract input features from request
        input_data = [
            int(data['CreditScore']),
            1 if data['Geography'] == 'France' else 0,
            1 if data['Geography'] == 'Germany' else 0,
            1 if data['Geography'] == 'Spain' else 0,
            1 if data['Gender'] == 'Male' else 0,
            1 if data['Gender'] == 'Female' else 0,
            int(data['Age']),
            int(data['Tenure']),
            float(data['Balance']),
            int(data['NumOfProducts']),
            int(data['HasCrCard']),
            int(data['IsActiveMember']),
            float(data['EstimatedSalary'])
        ]

        # Reshape and predict
        input_array = np.array([input_data])
        prediction = model.predict(input_array)
        binary_prediction = (prediction > 0.5).astype(int)[0][0]

        print("Prediction:", binary_prediction)  # Debugging statement

        return jsonify({'prediction': int(binary_prediction)})
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)



