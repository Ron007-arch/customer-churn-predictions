import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 Customer Churn Prediction System")
st.write(
    "This application predicts customer churn using a Machine Learning model "
    "and explains the key factors influencing churn."
)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(
    "data/Customer-Churn-Project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ==============================
# DATA PREPROCESSING
# ==============================
data = df.copy()

data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data.dropna(inplace=True)
data.drop(columns=["customerID"], inplace=True)

data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop("Churn", axis=1)
y = data_encoded["Churn"]

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MODEL TRAINING
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.subheader("📊 Model Performance")
st.metric("Accuracy", f"{accuracy:.2%}")

# ==============================
# FEATURE IMPORTANCE
# ==============================
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.coef_[0]
}).sort_values(by="Importance", ascending=False)

st.subheader("🔍 Top Factors Affecting Churn")
st.dataframe(importance.head(10))
st.bar_chart(importance.head(10).set_index("Feature"))

# ==============================
# PREDICTION SECTION (ONLY ONCE)
# ==============================
st.subheader("🔮 Predict Customer Churn")

tenure = st.slider(
    "Tenure (months)", 0, 72, 12, key="tenure_slider"
)

monthly_charges = st.slider(
    "Monthly Charges", 20.0, 120.0, 70.0, key="monthly_slider"
)

total_charges = tenure * monthly_charges

# Create input dataframe with SAME columns as training data
input_df = pd.DataFrame(0, index=[0], columns=X_train.columns)

if "tenure" in input_df.columns:
    input_df["tenure"] = tenure
if "MonthlyCharges" in input_df.columns:
    input_df["MonthlyCharges"] = monthly_charges
if "TotalCharges" in input_df.columns:
    input_df["TotalCharges"] = total_charges

# ==============================
# PREDICT BUTTON
# ==============================
if st.button("Predict", key="predict_churn"):
    churn_prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")
    st.write(f"📈 **Churn Probability:** {churn_prob:.2%}")

    if churn_prob >= 0.5:
        st.error("❌ Will Churn")
    else:
        st.success("✅ Will Not Churn")

    # ==============================
    # WHY THIS PREDICTION?
    # ==============================
    st.subheader("🔍 Why this prediction?")

    coef_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Weight": model.coef_[0]
    })

    coef_df["Impact"] = coef_df["Weight"] * input_df.iloc[0]
    top_reasons = coef_df.reindex(
        coef_df["Impact"].abs().sort_values(ascending=False).index
    )

    st.dataframe(top_reasons.head(3))

    # ==============================
    # BUSINESS INSIGHT
    # ==============================
    st.subheader("💼 Business Insight")

    if churn_prob >= 0.5:
        st.warning(
            "High churn risk detected.\n\n"
            "✅ Offer discounts\n"
            "✅ Upgrade contracts\n"
            "✅ Proactive customer support"
        )
    else:
        st.success(
            "Low churn risk.\n\n"
            "✅ Maintain engagement\n"
            "✅ Loyalty rewards\n"
            "✅ Upsell premium services"
        )
