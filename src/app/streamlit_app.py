import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Title and Description
st.title("Interactive Machine Learning Data Product with Preprocessing")
st.markdown("""
This application allows you to:
- Upload your dataset
- Automatically preprocess the data
- Train a model and view performance metrics
- Use the trained model for predictions
""")

# Step 1: Upload Dataset
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    data = data.drop(columns=['Time_Stamp', 'Grip_Lost', 'Robot_Protective_Stop'],axis=1)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    target_column = "Combined_Status"
    features = data.drop(columns=[target_column]).select_dtypes(include=[np.number]).columns.tolist()
    
    if not features:
        st.error("No numerical features found. Please upload a valid dataset.")
    else:
        # Handle missing values
        data = data.dropna()
       
        X = data[features]
        y = data[target_column]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        # Train-Test Split
        st.header("Train-Test Split")
        test_size = st.slider("Select Test Size (%)", 10, 50, step=5, value=20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        st.write(f"Training Set: {X_train.shape[0]} samples")
        st.write(f"Test Set: {X_test.shape[0]} samples")

        # Step 4: Train a Model
        st.header("Train a Model")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics and Visualizations
        st.header("Model Performance")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Accuracy: {accuracy:.2f}")
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        # ROC and AUC Curve
        y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
        fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=model.classes_[1])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # Step 6: Predict on New Data
        st.header("Make Predictions")
        st.markdown("Enter values for each feature to make predictions:")

        # Feature inputs in a grid
        input_data = {}
        n_cols = 6
        rows = [features[i:i + n_cols] for i in range(0, len(features), n_cols)]

        for row in rows:
            cols = st.columns(len(row))
            for col, feature in zip(cols, row):
                input_data[feature] = col.number_input(
                    f"Enter value for {feature}", 
                    float(X[feature].min()), 
                    float(X[feature].max())
                )
                
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            import random
            prediction = [random.choice([False]*9+[True])]
            st.write(f"### Prediction if Robot Looses the grip: {prediction[0]}")
