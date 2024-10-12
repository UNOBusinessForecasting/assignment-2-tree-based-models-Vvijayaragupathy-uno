from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(url):
    return pd.read_csv(url)

def prepare_features(df):
    target = df["meal"]
    features = df.drop(["meal", "id", "DateTime"], axis=1)
    return features, target

def create_model():
    return DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, random_state=42)

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    return round(100 * accuracy_score(y, predictions), 2)

# Initialize global variables
model = None
modelFit = None
pred = None

def main():
    global model, modelFit, pred
    
    # Load and prepare training data
    train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
    df = load_data(train_url)
    features, target = prepare_features(df)

    # Split the data
    train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size=0.33, random_state=42)

    # Create and train the model
    model = create_model()
    modelFit = model.fit(train_X, train_Y)

    # Evaluate the model
    print(f"\n\nIn-sample accuracy: {evaluate_model(model, train_X, train_Y)}%\n\n")
    print(f"\n\nOut-of-sample accuracy: {evaluate_model(model, test_X, test_Y)}%\n\n")

    # Load and prepare test data
    test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
    test_df = load_data(test_url)
    new_test, _ = prepare_features(test_df)

    # Make predictions
    pred = model.predict(new_test)

    # Add predictions to the test dataframe
    test_df["predicted_meal"] = pred

    # Display the first few predictions
    print(test_df[["id", "predicted_meal"]].head())

    # Save predictions to a CSV file
    test_df[["id", "predicted_meal"]].to_csv("meal_predictions.csv", index=False)

if __name__ == "__main__":
    main()
else:
    # If the script is imported, run main() to initialize the global variables
    main()
