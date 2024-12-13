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



import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the Logistic Regression model has been trained and predictions made
# Predict probabilities and classify based on threshold
y_prob = model.predict_proba(X)[:, 1]  # Probability for class '1' (Present)
y_pred = model.predict(X)  # Predicted class labels

# Add predicted values and probabilities to the dataset for plotting
data['Predicted Presence'] = y_pred
data['Prediction Probability'] = y_prob

# Create scatter plot
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    x=data['Retinal Nerve Fiber Layer (RNFL) Thickness (Âµm)_n'],
    y=data['Macular Thickness (Âµm)_n'],
    hue=data['Predicted Presence'],
    size=data['Prediction Probability'],
    sizes=(20, 200),
    palette={1: 'blue', 0: 'orange'},
    style=data['Predicted Presence'],
    markers={1: 'o', 0: 'X'}
)

# Customize the plot
plt.title("Logistic Regression Predictions: RNFL vs Macular Thickness", fontsize=16)
plt.xlabel("Encoded RNFL Thickness", fontsize=12)
plt.ylabel("Encoded Macular Thickness", fontsize=12)
plt.legend(title="Predicted Amyloid Presence", labels=["Absent", "Present"], fontsize=10)
# plt.grid(True)
plt.show()
