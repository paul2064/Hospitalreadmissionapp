import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import os

# Load cleaned dataset
def load_data():
    # Use the raw URL from your GitHub repository
    file_path = 'https://raw.githubusercontent.com/paul2064/Hospitalreadmissionapp/main/hospital_readmission_cleaned.csv'
    return pd.read_csv(file_path)

# Drop unnecessary columns and analyze feature importance
def preprocess_data(df):
    df = df.drop(columns=['A1test', 'change', 'A1Ctest'], errors='ignore')  # Drop specified columns if they exist
    X = df.drop(columns=['readmitted'])  # Features
    y = df['readmitted']  # Target
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    
    # Standardize numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['number']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Feature importance analysis using RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    significant_features = feature_importances[feature_importances > 0.01].index.tolist()
    X = X[significant_features]  # Keep only significant features
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoders, target_encoder, scaler, significant_features, feature_importances

# Hyperparameter tuning
def tune_random_forest(X_train, y_train):
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    }
    rf = RandomForestClassifier(random_state=42)
    rf_search = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    best_model = rf_search.best_estimator_
    joblib.dump(best_model, "random_forest_model.pkl")
    return best_model

# Load model if exists, otherwise train a new one
def load_or_train_model(X_train, y_train):
    if os.path.exists("random_forest_model.pkl"):
        try:
            return joblib.load("random_forest_model.pkl")
        except Exception:
            pass  # In case of loading error, retrain the model
    return tune_random_forest(X_train, y_train)

def main():
    st.title("Hospital Readmission Prediction")
    st.write("Predict if a patient will be readmitted based on their medical records.")
    
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, label_encoders, target_encoder, scaler, feature_names, feature_importances = preprocess_data(df)
    
    # Load or train model
    rf_model = load_or_train_model(X_train, y_train)
    
    # Model evaluation
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    
    st.write(f"Random Forest Accuracy: {rf_accuracy:.2f}")
    
    # Feature importance visualization
    st.subheader("Feature Importance")
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importances.sort_values(ascending=False).index, 
                y=feature_importances.sort_values(ascending=False).values)
    plt.xticks(rotation=90)
    plt.ylabel("Importance Score")
    plt.xlabel("Feature")
    st.pyplot(plt)
    
    # User input
    st.sidebar.subheader("Enter Patient Details")
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.sidebar.number_input(f"{feature}", value=0, step=1)
    
    input_df = pd.DataFrame([user_input])
    
    # Encode user input
    for col, le in label_encoders.items():
        if col in input_df:
            input_df[col] = input_df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    # Scale numerical features
    numerical_cols = input_df.select_dtypes(include=['number']).columns
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Ensure input has the same features as training
    missing_cols = set(feature_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Assign default value for missing features
    input_df = input_df[feature_names]  # Reorder columns to match training data
    
    # Prediction
    if st.sidebar.button("Predict Readmission"):
        rf_prediction = rf_model.predict(input_df)[0]
        rf_proba = rf_model.predict_proba(input_df)[0][rf_prediction]
        
        st.subheader("Prediction Results")
        st.write(f"Random Forest Prediction: {'Readmitted' if target_encoder.inverse_transform([rf_prediction])[0] == 1 else 'Not Readmitted'}")
        st.write(f"Confidence: {rf_proba:.2f}")
    
if __name__ == "__main__":
    main()
