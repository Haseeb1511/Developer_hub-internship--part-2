import streamlit as st
import pickle
import pandas as pd

st.title("Credit Loan Approval")
with open("model\gadient_boosting.pkl","rb") as f:
    gradient_boosting = pickle.load(f)

with open("model/random_fores.pkl","rb") as f:
    random_forest = pickle.load(f)

choice = st.sidebar.selectbox("Chose Model:",["Gradient Boosting","Random Forest"])

if choice == "Gradient Boosting":
    model = gradient_boosting
else:
    model = random_forest


columns = []

person_age = st.slider("Chose age :",18,64,22)
person_income = st.number_input("Enter salory:",min_value=1000)
person_home_ownership = st.radio("select house ownership type",['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
person_emp_length = st.slider("Chose person emp length :",0.0,123.0,4.0)
loan_intent = st.radio("Select loan Intention:",['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT','DEBTCONSOLIDATION'])
loan_grade = st.radio("Select loan grade :",['D', 'B', 'C', 'A', 'E', 'F', 'G'])
loan_amnt = st.slider("chose ammout of loan(Rs):",500,35000,10000)
loan_int_rate = st.slider("chose interest rate",5.22,23.22,10.99)
loan_percent_income = st.slider("chose loan percent income",0.0,0.83,0.1)
cb_person_default_on_file = st.radio("chose person default on file :",["Y","N"])
cb_person_cred_hist_length = st.slider("chose person credit history length :",2,30,3)

user_input = {
    "person_age":person_age,
        	"person_income"	:person_income,
            "person_home_ownership":person_home_ownership,
            "person_emp_length":person_emp_length,
            "loan_intent":loan_intent,
            "loan_grade":loan_grade,
            "loan_amnt":loan_amnt,
            "loan_int_rate":loan_int_rate,
            "loan_percent_income":loan_percent_income,
            "cb_person_default_on_file":cb_person_default_on_file,
            "cb_person_cred_hist_length":cb_person_cred_hist_length
}

if st.button("Check Status"):

    input = pd.DataFrame([user_input])

    prediction = model.predict(input)[0]

    if prediction == 1:
        st.success("Loan Status : Approved")
    else:
        st.error("Loan Status : Not Approved")
