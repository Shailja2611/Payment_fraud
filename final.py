from flask import Flask, request, jsonify, render_template
import pickle  # Using pickle instead of joblib
import numpy as np
import pandas as pd  # Import pandas to create DataFrame

app = Flask(__name__)

# Load the pre-trained model (assuming model.pkl is in the same directory as server.py)
with open('model/log_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the type mapping for encoding
type_mapping = {
    'PAYMENT': 0,
    'CASH_OUT': 1,
    'CASH_IN': 2,
    'TRANSFER': 3,
    'DEBIT': 4,  # 'DEBIT' comes lastj
}

@app.route('/')
def index():
    # This will render the index.html template when the user visits the root URL
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.get_json()

    # Print the input data for debugging
    print(f"Received data: {data}")

    # Extract the input fields
    try:
        transaction_type = data['type']
        amount = data['amount']
        old_balance_org = data['oldbalanceOrg']
        new_balance_orig = data['newbalanceOrig']
    except KeyError:
        return jsonify({'error': 'Missing required fields'}), 400

    # Check if the provided type is valid
    if transaction_type not in type_mapping:
        return jsonify({'error': 'Invalid transaction type'}), 400

    # Encode transaction type as per the mapping
    encoded_type = type_mapping[transaction_type]

    # Prepare the input features for prediction
    features = np.array([encoded_type, amount, old_balance_org, new_balance_orig]).reshape(1, -1)

    # Convert the features into a DataFrame with column names matching the model's expected format
    feature_columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
    features_df = pd.DataFrame(features, columns=feature_columns)

    # Use the model to make a prediction
    try:
        prediction = model.predict(features_df)
        # Convert prediction to a Python native int
        prediction_value = int(prediction[0])
        # Print the prediction result for debugging
        print(f"Prediction: {prediction_value}")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Return the prediction result as a response
    return jsonify({'prediction': prediction_value}), 200


if __name__ == '__main__':
    app.run(debug=True)
