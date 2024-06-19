import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Get the working directory of the main.py file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(working_dir)

# Step 1: Read the data
def read_data(file_name):
    file_path = f"{parent_dir}/data/{file_name}"
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
        return df
    

# Step 2: Preprocess the data
def preprocess_data(df, target_column, scaler_type):
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if there are only numerical or categorical columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(numerical_cols) > 0:
        # Impute missing values for numerical columns (mean imputation)
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        # Scale the numerical features based on scaler_type
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()

        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    if len(categorical_cols) > 0:
        # Impute missing values for categorical columns (mode imputation)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        # One-hot encode categorical features
        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])
        X_train_encoded = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
        X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
        
        # Reset indices to ensure proper alignment
        X_train_encoded.reset_index(drop=True, inplace=True)
        X_test_encoded.reset_index(drop=True, inplace=True)
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)

        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

    return X_train, X_test, y_train, y_test


# Step 3: Train the model
def train_model(X_train, y_train, model, model_name):
    # training the selected model
    model.fit(X_train, y_train)
    # saving the trained model
    model_dir = f"{parent_dir}/trained_model"
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/{model_name}.pkl", 'wb') as file:
        pickle.dump(model, file)
    return model


# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 2)
    return accuracy


# Example usage
if __name__ == "__main__":
    file_name = "your_dataset.csv"  # Replace with your actual file name
    target_column = "target"  # Replace with your actual target column name
    scaler_type = "standard"  # Choose either 'standard' or 'minmax'

    df = read_data(file_name)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)

    # Logistic Regression
    lr_model = train_model(X_train, y_train, LogisticRegression(), "logistic_regression")
    print(f"Logistic Regression Accuracy: {evaluate_model(lr_model, X_test, y_test)}")

    # Support Vector Classifier
    svc_model = train_model(X_train, y_train, SVC(), "svc")
    print(f"Support Vector Classifier Accuracy: {evaluate_model(svc_model, X_test, y_test)}")

    # Random Forest Classifier
    rf_model = train_model(X_train, y_train, RandomForestClassifier(), "random_forest")
    print(f"Random Forest Classifier Accuracy: {evaluate_model(rf_model, X_test, y_test)}")

    # XGBoost Classifier
    xgb_model = train_model(X_train, y_train, XGBClassifier(), "xgb")
    print(f"XGBoost Classifier Accuracy: {evaluate_model(xgb_model, X_test, y_test)}")

    # Naive Bayes
    nb_model = train_model(X_train, y_train, GaussianNB(), "naive_bayes")
    print(f"Naive Bayes Accuracy: {evaluate_model(nb_model, X_test, y_test)}")

    # Decision Tree Classifier
    dt_model = train_model(X_train, y_train, DecisionTreeClassifier(), "decision_tree")
    print(f"Decision Tree Accuracy: {evaluate_model(dt_model, X_test, y_test)}")
