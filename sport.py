import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset from a CSV file

import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder# Flask app
app = Flask(__name__)

# Load the trained model and label encoder
try:
    with open('fitness_modelx.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('label_encoderx.pkl', 'rb') as le_file:
        le = pickle.load(le_file)
except FileNotFoundError:
    print("Error: Model or label encoder files not found. Make sure they are in the same directory or provide the correct path.")
    exit()  # Exit if files not found

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Input validation: Check if data is provided
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        print(f"Received data: {data}")  # Debugging line to check received data
        # Convert the data into a DataFrame
        try:
            input_data = pd.DataFrame([data])  # Wrap data in a list for single prediction
        except ValueError:
            return jsonify({'error': 'Invalid input data format. Data should be a JSON object.'}), 400
        # Ensure the input data has the same columns as the training data
        required_columns = ['Age', 'Height (cm)', 'Weight (kg)', 'Resting HR', 'Workout HR']
        missing_columns = [col for col in required_columns if col not in input_data.columns]

        if missing_columns:
            return jsonify({'error': f'Missing required columns: {missing_columns}'}), 400

        input_data = input_data[required_columns]  # Order columns to match training data

        # Data type validation
        for col in required_columns:
            try:
                input_data[col] = pd.to_numeric(input_data[col])  # Attempt conversion
            except ValueError:
                return jsonify({'error': f'Invalid data type for column {col}. Must be numeric.'}), 400

        # Make predictions
        predictions = model.predict(input_data)

        # Decode the predictions
        predicted_sport = le.inverse_transform(predictions)

        return jsonify({'suggested_sport': predicted_sport[0]})

    except Exception as e:
        print(f"An unexpected error occurred: {e}")  # Log the error for debugging
        return jsonify({'error': 'An unexpected error occurred. Please check the server logs.'}), 500
# Define the API endpoint
@app.route('/api/sensor_data', methods=['POST'])
def add_sensor_data_api():
    try:
        data = request.get_json()  # Get JSON data from the request body

        if not data:
            return jsonify({"message": "No data provided"}), 400

        rfid = data.get('rfid')  # Get rfid from the JSON data
        weight = data.get('weight')
        height = data.get('height')
        ecg = data.get('ecg')

        if not rfid or not weight or not height or not ecg:
            return jsonify({"message": "Missing required fields (rfid, weight, height, ecg)"}), 400

        # Get user by RFID
        current_user = User.query.filter_by(rfid=rfid).first()
        if not current_user:
            return jsonify({"message": "User not found"}), 404

        # Process the data to compute BMI
        weight = float(weight)
        height = float(height)
        bmi = weight / (height / 100) ** 2
        date_time = datetime.now()

        # Use the model to predict the suggested sport based on the user's BMI, height, and weight
        input_data = pd.DataFrame([{
            'Age': current_user.age,  # Assuming you have this in the user model
            'Height (cm)': height,
            'Weight (kg)': weight,
            'Resting HR': ecg,  # Assuming this is also stored in the user model
            'Workout HR': ecg  # Same assumption as above
        }])

        # Ensure the input data columns match the model's training data
        required_columns = ['Age', 'Height (cm)', 'Weight (kg)', 'Resting HR', 'Workout HR']
        input_data = input_data[required_columns]  # Reorder the columns if necessary

        # Predict the suggested sport
        prediction = model.predict(input_data)
        suggested_sport = le.inverse_transform(prediction)[0]

        # Create a new sensor data entry
        new_sensor_data = SensorData(
            rfid=current_user.rfid,
            weight=weight,
            height=height,
            bmi=bmi,
            ecg=ecg,
            suggested_sport=suggested_sport,
            datetime=date_time
        )

        # Save the sensor data into the database
        db.session.add(new_sensor_data)
        db.session.commit()

        return jsonify({"message": "Sensor data added successfully", "suggested_sport": suggested_sport}), 201  # 201 Created

    except (ValueError, TypeError) as e:
        return jsonify({"message": f"Invalid input: {e}"}), 400

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"message": "An error occurred"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)  # Set debug=False for production
