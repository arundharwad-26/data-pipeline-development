import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load Dataset

# use relative path so the script works regardless of parent folder name
# the CSV file is located in the same directory as this script
import os

data_path = os.path.join(os.path.dirname(__file__), "Titanic-Dataset.csv")
df = pd.read_csv(data_path)
print("Initial dataset shape:", df.shape)
print(df.head())

X=df[['Pclass','Sex','Age','Fare']]
y=df["Survived"]

# Define Column Types
numeric_features=['Age','Fare']
categorical_features=['Pclass','Sex']

# Create Preprocessing Pipelines

# Numeric Pipeline
numeric_pipline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

# Categorical Pipeline
categorical_pipeline=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

# Combine Both Pipelines
preprocessor=ColumnTransformer(transformers=[
    ('num',numeric_pipline,numeric_features),
    ('cat',categorical_pipeline,categorical_features)
])

# Apply Transformation
X_processed=preprocessor.fit_transform(X)
print("Processed Data Shape:",X_processed.shape)

# Train_Test_Split

X_train, X_test, y_train, y_test= train_test_split(
    X_processed,y,test_size=0.2, random_state=42
)
print("Train shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# Save Processed Data
# save using the same directory as the script to avoid permission issues
out_dir = os.path.dirname(__file__)
np.save(os.path.join(out_dir, "x_train.npy"), X_train)
np.save(os.path.join(out_dir, "X_test.npy"), X_test)

print("ETL Pipeline Completed Successfully")