import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#df = pd.read_csv("/kaggle/input/diabetes-prediction-dataset/diabetes_prediction_dataset.csv")
#df = pd.read_csv('diabetes_prediction_dataset.csv', encoding='latin')
df = pd.read_csv('diabetes_prediction_dataset.csv', encoding='latin')
df.head()

df.info()

# Convert gender to numerical format
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

df.info()

df["smoking_history"].unique()

# Convert smoking history to numerical format
smoking_history_mapping = {'never': 0, 'No Info': -1, 'current': 2, 'former': 1, 'ever': 2, 'not current': 0}
df['smoking_history'] = df['smoking_history'].map(smoking_history_mapping)

df.head()

# Check the shape of the data
print('Shape of the dataset: \n', df.shape)

# Check the data types
print('\nData types of each column:\n', df.dtypes)

# Check for missing values
print('\nNumber of missing values in each column:\n', df.isnull().sum())

# Check for duplicates
print('\nNumber of duplicate rows in the dataset:', df.duplicated().sum())

# Check the distribution of numerical variables
print('\nSummary statistics for numerical variables:\n', df.describe())

# Visualize the distribution of numerical variables using histograms
df.hist(bins=10, figsize=(10, 8))
plt.show()

# Check for outliers in numerical variables using box plots
sns.boxplot(data=df, orient='h', palette='Set2')
plt.show()

# Check the distribution of categorical variables using frequency tables
print('Frequency table for categorical variables:\n\n', df['gender'].value_counts())

# Visualize the distribution of categorical variables using bar plots
sns.countplot(x='gender', data=df, palette='Set2')
plt.show()

# Check for correlations between variables using a correlation matrix
corr_matrix = df.corr()
print('Correlation matrix:\n', corr_matrix)

# Visualize the correlations using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Visualize the relationships between variables using scatter plots
sns.scatterplot(x='age', y='blood_glucose_level', hue='diabetes', data=df, palette='Set2')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
#df = pd.read_csv('/kaggle/input/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
df = pd.read_csv('diabetes_prediction_dataset.csv', encoding='latin')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('diabetes', axis=1), df['diabetes'], test_size=0.2,
                                                    random_state=42)

# Select the relevant features
numerical_features = ['age', 'bmi', 'blood_glucose_level']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'HbA1c_level']

# Preprocess the data using a pipeline
numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_pipeline = Pipeline(
    [('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', numerical_pipeline, numerical_features),
                                                       ('cat', categorical_pipeline, categorical_features)])

# Train a logistic regression model
model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=42))])
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)