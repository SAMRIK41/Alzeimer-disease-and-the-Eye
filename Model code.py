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
