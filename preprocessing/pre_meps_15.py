import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from aif360.datasets.meps_dataset_panel19_fy2015 import MEPSDataset19
cd = MEPSDataset19()
le = LabelEncoder()
df = pd.DataFrame(cd.features)
# binning
df[0] = pd.cut(df[0],9, labels=[i for i in range(1,10)])
df[2] = pd.cut(df[2],10, labels=[i for i in range(1,11)])
df[3] = pd.cut(df[3],10, labels=[i for i in range(1,11)])
# drop another sex column
df = df.astype('int').drop(columns=[10])
# encode categorical column
df[4] = le.fit_transform(df[4])

X = np.array(df.to_numpy(), dtype=int)
y = np.array(cd.labels, dtype=int).reshape(-1)
# Y = np.eye(2)[Y.reshape(-1)]
# Y = np.array(Y, dtype=int)
# input_shape = (None, len(X[0]))
# nb_classes = 2

# y = np.array(cd.labels).reshape(-1)


# split data into training data, validation data and test data
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

# set constraints for each attribute
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T

# for meps15 data, age(0), race(1) and gender(9) are protected attributes in 137 features
# protected_attribs = [0, 1, 9]