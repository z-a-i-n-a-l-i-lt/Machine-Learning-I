# Titanic Survival Prediction - Logistic Regression from Scratch

This project uses a logistic regression model implemented **from scratch using NumPy** to predict survival outcomes on the Titanic dataset. The goal is to demonstrate how a basic machine learning algorithm works without relying on high-level libraries like `scikit-learn`.

## 🚢 Dataset

The dataset used is the famous [Titanic dataset from Kaggle](https://www.kaggle.com/competitions/titanic), which includes passenger information like age, class, fare, number of siblings/spouses aboard, and survival status.

- `train.csv`: Contains labeled training data.
- `test.csv`: Contains test data without survival labels.

## 📦 Files

- `train.csv` and `test.csv` – Required data files (upload via Colab).
- `submission.csv` – Generated file with survival predictions for test passengers.
- `titanic_logistic_regression.ipynb` or `.py` – The main code file.
- `README.md` – Project documentation.

## 📊 Features Used

- `Pclass` (Passenger Class)
- `Sex` (converted to numeric)
- `Age` (filled missing values with median)
- `SibSp` (Siblings/Spouses aboard)
- `Parch` (Parents/Children aboard)
- `Fare` (filled missing values with median)
- `Embarked` (Port of Embarkation, filled with mode and mapped to numeric)

## 🧠 Model

- **Algorithm**: Logistic Regression
- **Implementation**: Manual gradient descent using NumPy
- **Normalization**: Feature scaling using z-score normalization
- **Training Iterations**: 10,000
- **Learning Rate**: 0.01

