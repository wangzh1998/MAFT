import numpy as np
import sys
from sklearn.model_selection import train_test_split
sys.path.append("../")

X = []
y = []
i = 0

with open("datasets/diabetes", "r") as ins:
    for line in ins:
        line = line.strip()
        line1 = line.split(',')
        if (i == 0):
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