import pickle
import pandas as pd


# Load saved model

with open("credit_model.pkl", "rb") as file:
    model = pickle.load(file)


print("Model loaded successfully!")



# New customer data

customer = pd.DataFrame({

    "income": [55000],

    "debt": [10000],

    "loan_amount": [25000],

    "late_payments": [0],

    "credit_history": [1],

    "debt_income_ratio": [10000/55000]

})



# Prediction

prediction = model.predict(customer)



if prediction[0] == 1:

    print("Creditworthy Customer")

else:

    print("Not Creditworthy Customer")
