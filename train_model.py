import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# 1. Load Dataset
# ==============================

dataset = pd.read_csv("Salary_Data.csv")

X = dataset[["YearsExperience"]].values
y = dataset["Salary"].values

# ==============================
# 2. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ==============================
# 3. Train Model
# ==============================

model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# 4. Predictions
# ==============================

y_pred = model.predict(X_test)

# ==============================
# 5. Model Evaluation
# ==============================

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

print("R² Score (Train):", model.score(X_train, y_train))
print("R² Score (Test):", model.score(X_test, y_test))

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# ==============================
# 6. Manual Prediction
# ==============================

def predict_salary(years):
    return model.predict([[years]])[0]

print("Salary for 12 years:", predict_salary(12))
print("Salary for 15 years:", predict_salary(15))

# ==============================
# 7. Visualization
# ==============================

plt.scatter(X_train, y_train)
plt.plot(X_train, model.predict(X_train))
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# ==============================
# 8. Save Model
# ==============================

with open("linear_regression_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved as linear_regression_model.pkl")
