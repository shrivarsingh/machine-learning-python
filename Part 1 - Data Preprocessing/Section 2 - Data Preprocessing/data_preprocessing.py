# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
print("\nImporting the dataset\n")
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values  # iloc = locate index
y = dataset.iloc[:, -1].values
print(f"\t---X---\n{X}\n\n\t---y---\n{y}")

# taking care of missing data (numerical values)
print("\nTaking care of missing values\n")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(f"\t---X---\n{X}\n\n\t---y---\n{y}")

# encoding catergorical data (non-numerical values)
print("\nEncoding catergorical data\n")
# independant variable
print("\nIndependant Variable\n")
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(f"\t---X---\n{X}\n\n\t---y---\n{y}")
# dependant variable
print("\nDependant Variable\n")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(f"\t---X---\n{X}\n\n\t---y---\n{y}")

# Feature scaling is to be applied after we split into train and test set

# spliting the data into the training set and test set
print("\nSpliting the data into the Training set and Test set\n")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"\t---X_train---\n{X_train}")
print(f"\t---X_test---\n{X_test}")
print(f"\t---y_train---\n{y_train}")
print(f"\t---y_test---\n{y_test}")

# Feature scaling - avoids having certain features to be dominated by other features
print("\nFeature Scaling\n")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(f"\t---X_train---\n{X_train}")
print(f"\t---X_test---\n{X_test}")