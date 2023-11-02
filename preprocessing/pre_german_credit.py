"""
This python file preprocesses the German Credit Dataset.
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
import preprocess_utilities
"""
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Q8MAW8
"""

# make outputs stable across runs
np.random.seed(42)
tf.random.set_seed(42)


# load german credit risk dataset
absolute_dir_path = os.path.dirname(os.path.abspath(__file__))
last_dir = os.path.dirname(absolute_dir_path)
data_path = os.path.join(last_dir, 'datasets', 'proc_german_num_02 withheader-2.csv')
# data_path = ('datasets/proc_german_num_02 withheader-2.csv')
df = pd.read_csv(data_path)


# preprocess data
data = df.values.astype(np.int32)
data[:,0] = (data[:,0]==1).astype(np.int64)
bins_loan_nurnmonth = [0] + [np.percentile(data[:,2], percent, axis=0) for percent in [25, 50, 75]] + [80]
bins_creditamt = [0] + [np.percentile(data[:,4], percent, axis=0) for percent in [25, 50, 75]] + [200]
bins_age = [15, 25, 45, 65, 120]
list_index_num = [2, 4, 10]
list_bins = [bins_loan_nurnmonth, bins_creditamt, bins_age]
for index, bins in zip(list_index_num, list_bins):
    data[:, index] = np.digitize(data[:, index], bins, right=True)


# split data into training data and test data
X = data[:, 1:]
y = data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# set constraints for each attribute, 839808 data points in the input space
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T


# for german credit data, gender(6) and age(9) are protected attributes in 24 features
protected_attribs = [6, 9]

# intial_input for AEQUITAS
# ADF中默认采用的initial_input = [1, 2, 24, 1, 60, 1, 3, 2, 2, 1, 22, 3, 2, 2, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0]
initial_input = preprocess_utilities.generate_instance(constraint)
# print("Generated instance:", initial_input)

# configurations for SG
configurations = {
    'num_attributes': len(X[0]),
    'feature_name': df.columns[:-1].tolist(),
    'class_name': ['output'],
    'categorical_features': list(range(len(X[0]))),
}
# print(configurations)