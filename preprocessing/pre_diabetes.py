import numpy as np
import sys
from sklearn.model_selection import train_test_split
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
import preprocess_utilities
sys.path.append("../")

X = []
y = []
i = 0

absolute_dir_path = os.path.dirname(os.path.abspath(__file__))
last_dir = os.path.dirname(absolute_dir_path)
data_path = os.path.join(last_dir, 'datasets', 'diabetes')
feature_name = []

# with open("datasets/diabetes", "r") as ins:
with open(data_path, "r") as ins:
    for line in ins:
        line = line.strip()
        line1 = line.split(',')
        if (i == 0):
            feature_name = line1[:-1]
            i += 1
            continue
        # L = map(int, line1[:-1])
        L = [int(i) for i in line1[:-1]]
        X.append(L)
        y.append(int(line1[-1]))

X = np.array(X, dtype=float)
y = np.array(y, dtype=float)
X = X.astype(np.int32)
y = y.astype(np.int32)

# split data into training data, validation data and test data
# X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# set constraints for each attribute
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T

# for heart health data, age(0), gender(1) are protected attributes in 13 features
# protected_attribs = [0, 1]

# intial_input for AEQUITAS
initial_input = preprocess_utilities.generate_instance(constraint)
# print("Generated instance:", initial_input)

# configurations for SG
configurations = {
    'num_attributes': len(X[0]),
    'feature_name': feature_name,
    'class_name': ['output'],
    'categorical_features': list(range(len(X[0]))),
}
# print(configurations)