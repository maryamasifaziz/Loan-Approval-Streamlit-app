import pandas as pd
import numpy as np
import streamlit as st


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Loan Approval Prediction Portal", layout = "wide")
st.title("Loan Approval Prediction")
st.caption("Machine Learning Classification Project using Loan Dataset")

## Data Importing/Loading (cached)
@st.cache_data
def load_data(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    return df

@st.cache_resource
## Model Training
def train_model(df:pd.DataFrame):
    target = "approved"

    drop_cols = [target]

    if "applicant_name" in df.columns:
        drop_cols.append("applicant_name")

    X = df.drop(columns = drop_cols)
    Y = df[target]

    cat_cols = [c for c in ["gender", "city", "employment_type", "bank"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression())
    ])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(Y_test, Y_pred)),
        # when we predict approved, how often is it correct?
        "precision": float(precision_score(Y_test, Y_pred, zero_division=0)),
        # out of all truly approved, how many did we catch?
        "recall": float(recall_score(Y_test, Y_pred, zero_division=0)),
        # balance between precision and recall (harmonic mean of precision and recall)
        "f1": float(f1_score(Y_test, Y_pred, zero_division=0)),
        # 2x2 table containing FP, FN, TP and TN
        "confusion_matrix": confusion_matrix(Y_test, Y_pred).tolist()
    }

    return clf, metrics, X_train.columns.tolist()

st.sidebar.header("1. Load Dataset")
csvpath = st.sidebar.text_input("Enter CSV file path", 
                                value = "loan_data.csv", 
                                help="Path to the CSV file containing loan data")

# Try Loading the Data
try:
    df = load_data(csvpath)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.success(f"Dataset Loaded Successfully! Loaded {len(df)} rows.")

# Train the Model
st.sidebar.header("2. Train Model")
train_now = st.sidebar.button("Train Model")

if train_now:
    st.cache_resource.clear()
    st.sidebar.success("Model Trained Successfully!")
clf, metrics, feature_names = train_model(df)


# Main Layout

colA, colB = st.columns([1,1])

with colA:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

with colB:
    st.subheader("Model Performance")
    if 'metrics' in locals():
        st.write({
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "Accuracy": round(metrics["accuracy"], 4),
            "F1 Score": round(metrics["f1"], 4),
        })
        cm = np.array(metrics["confusion_matrix"])
        st.write("Confusion Matrix (row: actual[0,1], col: predicted[0,1]):")
        st.dataframe(
            pd.DataFrame(cm, columns=["Predicted No", "Predicted Yes"], index=["Actual No", "Actual Yes"]),
            use_container_width=True
        )
    else:
        st.write("No model trained yet.")

st.divider()

# Trying a Prediction (UI Inputs)

with st.form("prediction_form"):
    st.subheader("Enter Applicant Details for Prediction")

    applicant_name = st.text_input("Applicant Name", value="Muhammad Ali")
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.slider("Age", min_value=21, max_value=60, value=30)
    
    city = st.selectbox("City", sorted(df["city"].unique().tolist()))
    employment_type = st.selectbox("Employment Type", sorted(df["employment_type"].unique().tolist()))
    bank = st.selectbox("Bank", sorted(df["bank"].unique().tolist()))
    
    monthly_income_pkr = st.number_input("Monthly Income (PKR)", min_value=0, max_value=1000000, value=50000, step=1000)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=600)
    
    loan_amount_pkr = st.number_input("Loan Amount (PKR)", min_value=0, max_value=1000000, value=200000, step=1000)
    loan_tenure_months = st.selectbox("Loan Tenure (Months)", options=[6, 12, 18, 24, 36, 48, 60], index=3)
    existing_loans = st.selectbox("Existing Loans", options=[0, 1, 2, 3], index=0)
    default_history = st.selectbox("Default History", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=1)
    has_credit_card = st.selectbox("Has Credit Card", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=1)

    submitted = st.form_submit_button("Predict Approval")

# Building Input DataFrame
# Prepare input data for prediction
if (submitted):
    input_data = pd.DataFrame({
        "applicant_name": [applicant_name],
        "age": [age],
        "gender": [gender],
        "city": [city],
        "employment_type": [employment_type],
        "bank": [bank],
        "monthly_income_pkr": [monthly_income_pkr],
        "loan_amount_pkr": [loan_amount_pkr],
        "credit_score": [credit_score],
        "loan_tenure_months": [loan_tenure_months],
        "existing_loans": [existing_loans],
        "default_history": [default_history],
        "has_credit_card": [has_credit_card]
    })


    # input_data = input_data[feature_order]
    prob = float(clf.predict_proba(input_data)[:, 1])

    pred = int(prob >= 0.5)

    if pred == 1:
        st.success(f"{applicant_name} Loan Approved Probability: {prob:.2%}")
    else:
        st.error(f"{applicant_name} Loan Not Approved Probability: {prob:.2%}")

## Make prediction
# prediction = clf.predict(input_data)
# st.write(f"Prediction: {'Approved' if prediction[0] == 1 else 'Not Approved'}")



