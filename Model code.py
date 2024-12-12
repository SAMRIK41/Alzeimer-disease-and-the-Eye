from google.colab import drive
drive.mount('/content/gdrive')


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
from sklearn.svm import SVR
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold, cross_val_score


file_path = ('/content/Protien Data main.xlsx')

# Read the Excel file
data = pd.read_excel(file_path)

# Display the first few rows of the dataframe
print(data.head())


from sklearn.preprocessing import LabelEncoder

l_Outlook=LabelEncoder()
l_Outlook=LabelEncoder()
l_Outlook=LabelEncoder()
l_Outlook=LabelEncoder()
l_Outlook=LabelEncoder()


data['Age_n']=l_Outlook.fit_transform(data['Age'])
data['Gender_n']=l_Outlook.fit_transform(data['Gender'])
data['Retinal Nerve Fiber Layer (RNFL) Thickness (Âµm)_n']=l_Outlook.fit_transform(data['Retinal Nerve Fiber Layer (RNFL) Thickness (Âµm)'])
data['Macular Thickness (Âµm)_n']=l_Outlook.fit_transform(data['Macular Thickness (Âµm)'])
data['Retinal Amyloid Presence_n']=l_Outlook.fit_transform(data['Retinal Amyloid Presence'])

print(data.head())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Variables (use your encoded data)
X = data[['Age_n', 'Gender_n',
          'Retinal Nerve Fiber Layer (RNFL) Thickness (Âµm)_n',
          'Macular Thickness (Âµm)_n']]  # Features
y = data['Retinal Amyloid Presence_n']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
