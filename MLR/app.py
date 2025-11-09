import streamlit as st
import pickle
import numpy as np

# ==============================
# Load the saved model
# ==============================
model = pickle.load(open(r"C:\Users\mohan\Desktop\data science naresh it\class work\MLR\investment_model.pkl", 'rb'))

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(
    page_title="Investment Profit Prediction",
    page_icon="💰",
    layout="centered"
)

# ==============================
# Custom CSS
# ==============================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: #FAFAFA;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1 {
    text-align: center;
    color: #00C9A7;
}
.stButton button {
    background-color: #00C9A7;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton button:hover {
    background-color: #00B894;
    color: white;
}
.prediction-box {
    padding: 15px;
    border-radius: 12px;
    background-color: #1B4332;
    color: #D8F3DC;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Title & Description
# ==============================
st.title("💼 Investment Profit Prediction")
st.markdown("### Predict company **Profit** based on Marketing, Promotion, Research, and State.")

# ==============================
# Inputs
# ==============================
digital = st.number_input("💻 Digital Marketing Spend", min_value=0.0, value=100000.0, step=1000.0)
promotion = st.number_input("📢 Promotion Spend", min_value=0.0, value=50000.0, step=1000.0)
research = st.number_input("🔬 Research Spend", min_value=0.0, value=200000.0, step=1000.0)

state = st.selectbox("🏙️ Select State", ["Hyderabad", "Bangalore", "Chennai"])

# ==============================
# Encode State (one-hot encoding)
# ==============================
if state == "Hyderabad":
    state_encoded = [1, 0, 0]
elif state == "Bangalore":
    state_encoded = [0, 1, 0]
else:  # Chennai
    state_encoded = [0, 0, 1]

# ==============================
# Prediction Button
# ==============================
if st.button("🔮 Predict Profit"):
    input_data = np.array([[digital, promotion, research, *state_encoded]])
    prediction = model.predict(input_data)

    st.markdown(
        f"<div class='prediction-box'>"
        f"✅ Predicted Profit: <br><br> **₹{prediction[0]:,.2f}**"
        f"</div>",
        unsafe_allow_html=True
    )

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown(" *This app uses a Multiple Linear Regression model trained on investment data.*")
