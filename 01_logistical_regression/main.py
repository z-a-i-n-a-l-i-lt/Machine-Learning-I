#run the code at colab.research.google.com
from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

combined = pd.concat([train, test], sort=False)

combined['Age'] = combined['Age'].fillna(combined['Age'].median())
combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])
combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())

combined['Sex'] = combined['Sex'].map({'male': 0, 'female': 1})
combined['Embarked'] = combined['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X_train = combined.loc[combined['Survived'].notnull(), features].values
y_train = combined.loc[combined['Survived'].notnull(), 'Survived'].values.reshape(-1, 1)
X_test = combined.loc[combined['Survived'].isnull(), features].values
test_ids = combined.loc[combined['Survived'].isnull(), 'PassengerId']

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

def sigmoid(z):
    z = np.array(z)
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.01, iterations=10000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    for _ in range(iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        theta -= lr * gradient
    return theta

def predict(X, theta):
    return (sigmoid(np.dot(X, theta)) >= 0.5).astype(int)

theta = train_logistic_regression(X_train, y_train)

for i, weight in enumerate(theta.ravel()):
    print(f"w_{i} ({features[i]}): {weight:.4f}")

predictions = predict(X_test, theta)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Survived": predictions.ravel().astype(int)
})

submission.to_csv('submission.csv', index=False)

files.download('submission.csv')

def predict_single_person_fixed(theta, X_mean, X_std):
    pclass = 3
    sex = 0
    age = 22.0
    sibsp = 1
    parch = 0
    fare = 7.25
    embarked = 0

    x = np.array([pclass, sex, age, sibsp, parch, fare, embarked])
    x = (x - X_mean) / X_std
    pred = predict(x.reshape(1, -1), theta)

    if pred[0] == 1:
        print("Prediction: The passenger is likely to SURVIVE.")
    else:
        print("Prediction: The passenger is likely NOT to survive.")

predict_single_person_fixed(theta, X_mean, X_std)

