"""
This python file constructs and trains the model for MEPS15 Dataset.
"""


import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_heart_heath
from tensorflow import keras
# from keras.metrics import Precision, Recall, Accuracy

# precision = Precision(name="precision")
# recall = Recall(name="recall")
# accuracy = Accuracy(name="accuracy")

# create and train a six-layer neural network for the binary classification task
model = keras.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=pre_heart_heath.X_train.shape[1:]),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(15, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(5, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])


# uncomment for training
# history = model.fit(pre_heart_heath.X_train, pre_heart_heath.y_train, epochs=30, validation_data=(pre_heart_heath.X_val, pre_heart_heath.y_val))
# model.evaluate(pre_heart_heath.X_test, pre_heart_heath.y_test) # 70.00% accuracy
# model.save("../models/original_models/heart_model.h5")
history = model.fit(pre_heart_heath.X_train, pre_heart_heath.y_train, epochs=30)
model.evaluate(pre_heart_heath.X_test, pre_heart_heath.y_test) # 74.79% accuracy
model.save("../models/original_models/heart_model.h5")