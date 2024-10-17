import streamlit as st
import numpy as np
import pickle
import time

# Load the model
pickle_in = open("xgb_model_.pkl", "rb")
model = pickle.load(pickle_in)

# Abiodun Sidebar 
st.sidebar.title("Loan Information")
st.sidebar.image("image.jpg", caption="Understand Loan Predictions", )

# CSS 
st.markdown("""
    <style>
        body {
            font-family: 'Arial';
        }
        .main-title {
            color: #2d6a4f;
            font-size: 50px;
            font-family: 'Helvetica';
            text-align: center;
            margin-bottom: 10px;
            margin-top: 0px;  /* Remove top margin */
        }
        .subtitle {
            color: #34495e;
            font-size: 22px;
            text-align: center;
            margin-bottom: 30px;
        }
        .predict-button {
            background-color: #28a745;
            color: white;
            padding: 15px;
            font-size: 24px;
            text-align: center;
            border-radius: 5px;
        }
        .container {
            padding: 20px;
            border-radius: 10px;
        }
        .header {
            color: #16a085;
            font-size: 28px;
            margin-bottom: 10px;
        }
        .description {
            color: #555;
            font-size: 18px;
            margin-bottom: 20px;
            text-align: justify;
        }
        .image-section {
            text-align: center;
            margin-top: 20px;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)


if "show_form" not in st.session_state:
    st.session_state.show_form = False
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# Landing Page without background
st.markdown("<div>", unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>Loan Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict the risk of loan default with ease and confidence.</p>", unsafe_allow_html=True)

#  about loans
st.markdown("""
<div class='container'>
    <div class='header'>Understanding Loans</div>
    <div class='description'>
        Loans are a crucial part of financial systems, providing capital to individuals and businesses. However, loan defaults are a risk that financial institutions face. 
        With advanced machine learning algorithms, lenders can now accurately predict the risk of default, minimizing losses and optimizing lending strategies.
    </div>
</div>
""", unsafe_allow_html=True)

# Button to move to input form
if not st.session_state.show_form:
    if st.button("Predict Loan Default Risk", key="main_cta"):
        st.session_state.show_form = True

st.markdown("</div>", unsafe_allow_html=True)  # Close main div

# Prediction Input Form
if st.session_state.show_form:
    st.markdown("<h2>Enter Applicant's Information</h2>", unsafe_allow_html=True)
    
    # Input fields
    home = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    loan_intent = st.selectbox('Loan Intent', ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'OTHER'])
    Age = int(st.number_input("Customer Age", step=1))
    customer_income = st.number_input("Customer Income", step=500)
    Emp_length = st.number_input("Employment Duration", step=1)
    Amount = st.number_input("Loan Amount", step=1000)
    Rate = st.number_input("Rate")
    Cred_length = st.number_input("Credit Length", step=1)
    Status = st.selectbox("Status", [1, 0])
    Percent_income = st.number_input("Percent Income", min_value=0.0, max_value=1.0, step=0.1)

    # Prediction button
    if st.button("Predict", key="predict_button"):

        # Show progress bar during analysis
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1.5)
        my_bar.empty()

        # Convert inputs for prediction
        home_val = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}[home]
        loan_intent_val = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'OTHER': 5}[loan_intent]
        
        # passing the input by user
        query = np.array([Age, customer_income, home_val, Emp_length, loan_intent_val, Amount, Rate, Status, Percent_income, Cred_length]).reshape(1, -1)

        # Make prediction
        result = model.predict(query)[0]
        my_bar.empty()       
        

        #  session state
        st.session_state.prediction_made = True
        st.session_state.result = result

# Show prediction result if available
if st.session_state.prediction_made:
    result = model.predict_proba(query)[:, 1]  # Probability of class 1
    if result > 0.5:
        st.error("The applicant is likely to default on the loan.")
    else:
        st.success("The applicant is likely to repay the loan.")
