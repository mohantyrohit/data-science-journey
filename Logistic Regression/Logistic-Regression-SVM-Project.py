# -*- coding: utf-8 -*-
"""
Logistic Regression & SVM Classification Project.
Author: Mohan
"""


# 1. IMPORT ALL REQUIRED LIBRARIES


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-Learn modules for machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



# 2. LOAD THE DATASET


# Read the CSV file (make sure the path is correct on your machine)
dataset = pd.read_csv(r"C:\Users\mohan\Desktop\data science naresh it\class work\data\logit classification.csv")

# Select features (X) and target (y)
# Here columns 2 and 3 are independent variables
# Last column is the output variable
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values



# 3. SPLIT DATA INTO TRAINING AND TESTING SETS


# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)



# 4. FEATURE SCALING


# Scaling helps Logistic Regression and SVM perform better.
scaler = StandardScaler()

# Fit scaler on training data and transform both train and test
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# 5. LOGISTIC REGRESSION MODEL


print("\n================ LOGISTIC REGRESSION ================\n")

# Build the model
log_model = LogisticRegression()

# Train the model
log_model.fit(X_train, y_train)

# Predict test results
y_pred = log_model.predict(X_test)

# Show Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy score
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)

# Detailed performance report
print("\nClassification Report:\n", classification_report(y_test, y_pred))



# 6. PREDICTING ON A NEW DATASET (final1.csv)


print("\n================ PREDICTING NEW DATASET ================\n")

dataset1 = pd.read_csv(r"C:\Users\mohan\Downloads\9th, 10th - logistic, pca\2.LOGISTIC REGRESSION CODE\final1.csv")

# Make a safe copy
d2 = dataset1.copy()

# Select the required columns for prediction
X_new = dataset1.iloc[:, [3, 4]].values

# Scale new data using the same scaler
X_new_scaled = scaler.transform(X_new)

# Predict using the trained logistic regression model
d2["Logistic_Prediction"] = log_model.predict(X_new_scaled)

# Save output to a new CSV
d2.to_csv("final1_output.csv", index=False)

print("New predictions saved to final1_output.csv")



# 7. SUPPORT VECTOR MACHINE (SVM) CLASSIFICATION


print("\n================ SVM CLASSIFICATION ================\n")

# Create SVM model
classifier_svm = SVC()

# Train SVM on training data
classifier_svm.fit(X_train, y_train)

# Predict using SVM
y_pred_svm = classifier_svm.predict(X_test)

# Show SVM accuracy and results
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))
