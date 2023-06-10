"""
This python file implement our approach EIDIG
Modify this for compare of EIDIG (real gradient) and MAFT (estimated gradient)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import cluster
import itertools
import time
import generation_utilities

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
    # 将x中每个属性+h，形成n个新实例，得到X。将X投入model，得到Y。再根据Y求对X的模拟偏导数，得到n个偏导数后，合并起来构成模拟导数。
    # Y是Tensor(n,1) n行1列，每一行的y值是输出的概率值，而不是实数标签 如果用实数标签，算出来的梯度就不准
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

# 对seeds的真实梯度和模拟梯度进行对比
# 三个方法取的种子应该是一样的，这样才方便进行后续对比

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