import streamlit as st
import pickle
import numpy as np

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Salary Prediction App 💼",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================
# Custom CSS Styling
# ==============================
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: #FAFAFA;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .main {
            background-color: #0E1117;
        }

        h1 {
            text-align: center;
            color: #00C9A7;
            font-size: 42px;
            font-weight: bold;
        }

        .subtext {
            text-align: center;
            font-size: 18px;
            color: #D3D3D3;
        }

        .stNumberInput > div > input {
            background-color: #1E1E1E !important;
            color: white !important;
            border-radius: 8px;
            border: 1px solid #00C9A7 !important;
            font-size: 18px !important;
            text-align: center;
        }

        .stButton > button {
            background-color: #00C9A7;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.8em 0;
            font-size: 20px;
            font-weight: bold;
            width: 100%;
            transition: 0.3s;
        }

        .stButton > button:hover {
            background-color: #00B894;
            transform: scale(1.02);
        }

        .result-box {
            background-color: #1B4332;
            color: #D8F3DC;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            font-size: 22px;
            margin-top: 30px;
            box-shadow: 0px 0px 20px rgba(0, 201, 167, 0.4);
        }

        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==============================
# Load the Trained Model
# ==============================
model = pickle.load(open(r"C:\Users\mohan\Desktop\data science naresh it\class work\simpleliner_refresion\linear_regression_model.pkl", 'rb'))

# ==============================
# App Title
# ==============================
st.markdown("<h1>💼 Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Predict salary based on years of experience using a trained Linear Regression model.</p>", unsafe_allow_html=True)

st.markdown("---")

# ==============================
# Input Section
# ==============================
years_experience = st.number_input("Enter Years of Experience 👇", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# ==============================
# Prediction Button
# ==============================
if st.button("🔮 Predict Salary"):
    experience_input = np.array([[years_experience]])
    prediction = model.predict(experience_input)

    st.markdown(
        f"""
        <div class='result-box'>
            ✅ Predicted Salary for <b>{years_experience}</b> years of experience:<br><br>
            💰 <b>₹{prediction[0]:,.2f}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>🚀 Built with ❤️ using Streamlit & Machine Learning (Linear Regression)</p>",
    unsafe_allow_html=True
)
