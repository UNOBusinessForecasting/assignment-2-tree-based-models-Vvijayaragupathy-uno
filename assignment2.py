import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the training data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train_data = pd.read_csv(train_url)

# Prepare the data
X = train_data.drop(['meal'], axis=1)
y = train_data['meal']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
modelFit = model.fit(X_train, y_train)

# Validate the model
val_predictions = modelFit.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {accuracy:.4f}")

# Save the model
joblib.dump(modelFit, 'random_forest_model.joblib')

# Load the test data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_data = pd.read_csv(test_url)

# Make predictions on the test data
pred = modelFit.predict(test_data)

# Convert predictions to integer (0 or 1)
pred = pred.astype(int)

# Print the first few predictions
print("First few predictions:")
print(pred[:10])

# Print the total number of predictions
print(f"Total number of predictions: {len(pred)}")

# Optional: Save predictions to a CSV file
pd.DataFrame(pred, columns=['meal_prediction']).to_csv('predictions.csv', index=False)
