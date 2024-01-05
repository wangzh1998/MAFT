"""
This python file implement different gradient methods: ADF, EIDIG and MAFT.
Implement this for compare of ADF(original gradient), EIDIG (real gradient) and MAFT (estimated gradient)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

def compute_grad_adf(x, model, loss_func=keras.losses.binary_crossentropy):
    # compute the gradient of loss w.r.t input attributes

    x = tf.constant([x], dtype=tf.float32)
    y_pred = tf.cast(model(x) > 0.5, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = loss_func(y_pred, model(x))
    gradient = tape.gradient(loss, x)
    return gradient[0].numpy()

def compute_grad_eidig(x, model):
    # compute the gradient of model perdictions w.r.t input attributes

    x = tf.constant([x], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        # change 1: switch gradient from gradient(loss/x) to gradient(y/x)
        y_pred = model(x)
    gradient = tape.gradient(y_pred, x)
    return gradient[0].numpy() if model(x) > 0.5 else -gradient[0].numpy()

def compute_grad_maft(x, model, perturbation_size=1e-4):
    # compute the gradient of model perdictions w.r.t input attributes
    h = perturbation_size
    n = len(x)
    e = np.empty(n)
    e.fill(h)
    E = np.diag(e)
    X = np.repeat([x], n, axis=0)
    X = X + E
    X = tf.constant(X, dtype=tf.float32)
    Y = model(X)
    x = tf.constant([x], dtype=tf.float32)
    y_pred = model(x)
    gradient = (Y - y_pred) / h
    gradient = tf.reshape(gradient, [1, -1])
    return gradient[0].numpy() if model(x) > 0.5 else -gradient[0].numpy()

def compute_grad_maft_non_vectorized(x, model, perturbation_size=1e-4):
    h = perturbation_size
    n = len(x)
    y_pred = model(tf.constant([x], dtype=tf.float32))
    gradient = np.empty(n)
    for i in range(n):
        # perturb the i_th attribute
        x_perturbed = np.copy(x)
        x_perturbed[i] += h
        # calculate the model output after perturbation
        x_perturbed = tf.constant([x_perturbed], dtype=tf.float32)
        y_perturbed = model(x_perturbed)
        # calculate the gradient on the i_th attribute
        gradient[i] = (y_perturbed - y_pred) / h

    return gradient if model(tf.constant([x])) > 0.5 else -gradient

# compare the real gradient and estimated gradient
# same seeds should be used for all methods
def adf_gradient_generation(seeds, num_attribs, model):
    g_num = len(seeds)
    adf_gradients = np.empty(shape=(0, num_attribs))
    for i in range(g_num):
        x1 = seeds[i]
        real_grad = compute_grad_adf(x1, model)
        adf_gradients = np.append(adf_gradients, [real_grad], axis=0)
    return adf_gradients

# def gradient_generation(X, seeds, num_attribs, protected_attribs, constraint, model, decay, max_iter, s_g):
def eidig_gradient_generation(seeds, num_attribs, model):

    g_num = len(seeds)
    eidig_gradients = np.empty(shape=(0, num_attribs))
    for i in range(g_num):
        x1 = seeds[i]
        real_grad = compute_grad_eidig(x1, model)
        eidig_gradients = np.append(eidig_gradients, [real_grad], axis=0)
    return eidig_gradients

def maft_gradient_generation(seeds, num_attribs, model, perturbation_size=1e-4):

    g_num = len(seeds)
    maft_gradients = np.empty(shape=(0, num_attribs))
    for i in range(g_num):
        x1 = seeds[i]
        estimated_grad = compute_grad_maft(x1, model, perturbation_size)
        maft_gradients = np.append(maft_gradients, [estimated_grad], axis=0)
    return maft_gradients

def maft_gradient_generation_non_vec(seeds, num_attribs, model, perturbation_size=1e-4):

    g_num = len(seeds)
    maft_gradients_non_vec = np.empty(shape=(0, num_attribs))
    for i in range(g_num):
        x1 = seeds[i]
        estimated_grad = compute_grad_maft_non_vectorized(x1, model, perturbation_size)
        maft_gradients_non_vec = np.append(maft_gradients_non_vec, [estimated_grad], axis=0)
    return maft_gradients_non_vec