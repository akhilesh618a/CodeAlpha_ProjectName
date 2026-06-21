import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    # classification_report
    classification_report,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
# ============================
# Step 1: Load Dataset
# ============================

data = pd.read_csv("credit_data.csv")


print("First 5 rows:")
print(data.head())


# ============================
# Step 2: Data Information
# ============================

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nData Information:")
data.info()


print("\nMissing Values:")
print(data.isnull().sum())


print("\nDuplicate Rows:")
print(data.duplicated().sum())


print("\nStatistics:")
print(data.describe())


# ============================
# Step 3: Feature Engineering
# ============================

# Convert text into numbers

data["credit_history"] = data["credit_history"].map(
    {
        "good": 1,
        "bad": 0
    }
)


# Create new feature

data["debt_income_ratio"] = (
    data["debt"] / data["income"]
)


print("\nAfter Feature Engineering:")
print(data.head())


# ============================
# Step 4: Separate X and y
# ============================

X = data.drop("result", axis=1)

y = data["result"]


print("\nInput Data:")
print(X.head())


print("\nOutput Data:")
print(y.head())


# ============================
# Step 5: Train Test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Data:", X_train.shape)

print("Testing Data:", X_test.shape)



# ============================
# Step 6: Logistic Regression
# ============================

lr_model = LogisticRegression()

lr_model.fit(
    X_train,
    y_train
)


lr_prediction = lr_model.predict(X_test)



# ============================
# Step 7: Decision Tree
# ============================

dt_model = DecisionTreeClassifier(
    random_state=42
)


dt_model.fit(
    X_train,
    y_train
)


dt_prediction = dt_model.predict(X_test)



# ============================
# Step 8: Random Forest
# ============================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


rf_model.fit(
    X_train,
    y_train
)


# Save Random Forest Model

import pickle

with open("credit_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("Random Forest model saved successfully!")



rf_prediction = rf_model.predict(X_test)



# ============================
# Step 9: Evaluation Function
# ============================

def evaluate_model(name, y_test, prediction):

    print("\n====================")
    print(name)
    print("====================")


    print(
        "Accuracy:",
        accuracy_score(y_test, prediction)
    )


    print(
        "Precision:",
        precision_score(y_test, prediction)
    )


    print(
        "Recall:",
        recall_score(y_test, prediction)
    )


    print(
        "F1 Score:",
        f1_score(y_test, prediction)
    )



# ============================
# Step 10: Compare Models
# ============================


evaluate_model(
    "Logistic Regression",
    y_test,
    lr_prediction
)


evaluate_model(
    "Decision Tree",
    y_test,
    dt_prediction
)


evaluate_model(
    "Random Forest",
    y_test,
    rf_prediction
)



# Detailed Report for Best Model

print("\nRandom Forest Classification Report")

print(
    classification_report(
        y_test,
        rf_prediction
    )
)

# Probability prediction

lr_probability = lr_model.predict_proba(X_test)[:,1]

dt_probability = dt_model.predict_proba(X_test)[:,1]

rf_probability = rf_model.predict_proba(X_test)[:,1]


print("\nROC-AUC Scores")


print(
    "Logistic Regression:",
    roc_auc_score(
        y_test,
        lr_probability
    )
)


print(
    "Decision Tree:",
    roc_auc_score(
        y_test,
        dt_probability
    )
)


print(
    "Random Forest:",
    roc_auc_score(
        y_test,
        rf_probability
    )
)


# Create ROC Curve


lr_fpr, lr_tpr, _ = roc_curve(
    y_test,
    lr_probability
)


dt_fpr, dt_tpr, _ = roc_curve(
    y_test,
    dt_probability
)


rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probability
)



plt.figure(figsize=(8,6))


plt.plot(
    lr_fpr,
    lr_tpr,
    label="Logistic Regression"
)


plt.plot(
    dt_fpr,
    dt_tpr,
    label="Decision Tree"
)


plt.plot(
    rf_fpr,
    rf_tpr,
    label="Random Forest"
)


plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)


plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Credit Scoring Model")


plt.legend()

plt.show()


