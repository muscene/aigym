import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset from a CSV file
df = pd.read_csv('fitness_data.csv')  # Update the path to your actual CSV file
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Sample dataset you provided (you can load it from a CSV file instead)
# data = {
#     'Name': ['Emily Powell', 'Trevor Miller', 'Jessica Houston'],
#     'Age': [43, 34, 42],
#     'Gender': ['Female', 'Female', 'Male'],
#     'Height (cm)': [163, 186, 172],
#     'Weight (kg)': [100, 90, 54],
#     'Resting HR': [62, 76, 61],
#     'Workout HR': [139, 166, 147],
#     'Fitness Goals': ['Muscle Gain', 'Endurance Training', 'Rehabilitation'],
#     'Preferred Exercises': ['Martial Arts', 'Cardio', 'Swimming'],
#     'Suggested Sports': ['Martial Arts', 'Running', 'Swimming']
# }

# Convert the data into a pandas DataFrame
df = pd.DataFrame(df)

# Select features and target
X = df[['Age', 'Height (cm)', 'Weight (kg)', 'Resting HR', 'Workout HR']]  # Features
y = df['Suggested Sports']  # Target variable

# Encode the target variable (Suggested Sports)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model and label encoder
try:
    with open('fitness_modelx.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('label_encoderx.pkl', 'wb') as le_file:
        pickle.dump(le, le_file)

    print("Model and label encoder saved successfully.")
except Exception as e:
    print(f"Error saving model or label encoder: {e}")

