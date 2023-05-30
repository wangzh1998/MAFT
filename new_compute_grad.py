import tensorflow as tf
import numpy as np
import time

# keep different compute_grad method

# v1 v2效果接近，v3效果不太好

# BBDIG
# 模拟y/x
# v1 h = 1e-4
def compute_grad_v1(x, model):
    # compute the gradient of model perdictions w.r.t input attributes
    # 将x中每个属性+h，形成n个新实例，得到X。将X投入model，得到Y。再根据Y求对X的模拟偏导数，得到n个偏导数后，合并起来构成模拟导数。
    # unit = 1
    h = 1e-4
    n = len(x)
    X = np.tile(x, (n, 1))
    for i in range(n):
        e = np.zeros_like(x)
        # e[i] = unit * h
        e[i] = h
        X[i] += e
    X = tf.constant(X, dtype=tf.float32) # 不做这个转换可以吗？不可以。
    Y = model(X)
    x = tf.constant([x], dtype=tf.float32)
    y_pred = model(x)
    gradient = (Y - y_pred) / h
    gradient = tf.reshape(gradient, [1, -1])
    return gradient[0].numpy() if model(x) > 0.5 else -gradient[0].numpy()

# v2 h=1e-4 修改计算X的过程
def compute_grad_v2(x, model):
    # compute the gradient of model perdictions w.r.t input attributes
    # 将x中每个属性+h，形成n个新实例，得到X。将X投入model，得到Y。再根据Y求对X的模拟偏导数，得到n个偏导数后，合并起来构成模拟导数。
    # unit = 1
    h = 1e-4
    n = len(x)
    e = np.empty(n)
    e.fill(h)
    E = np.diag(e)
    X = np.repeat(x.reshape(1, n), n, axis=0)
    X = X + E
    X = tf.constant(X, dtype=tf.float32)
    Y = model(X)
    x = tf.constant([x], dtype=tf.float32)
    y_pred = model(x)
    gradient = (Y - y_pred) / h
    gradient = tf.reshape(gradient, [1, -1])
    return gradient[0].numpy() if model(x) > 0.5 else -gradient[0].numpy()

# v3 h=1e-4 修改(Y - y_pred) / h 为 （Y1 - Y2）/ (2 * h)
def compute_grad_v3(x, model):
    # compute the gradient of model perdictions w.r.t input attributes
    # 将x中每个属性+h，形成n个新实例，得到X。将X投入model，得到Y。再根据Y求对X的模拟偏导数，得到n个偏导数后，合并起来构成模拟导数。
    # unit = 1
    h = 1e-4
    n = len(x)
    e = np.empty(n)
    e.fill(h)
    E = np.diag(e)
    X = np.repeat(x.reshape(1, n), n, axis=0)
    X1 = X + E
    X2 = X - E
    X1 = tf.constant(X1, dtype=tf.float32)
    X2 = tf.constant(X2, dtype=tf.float32)
    Y1 = model(X1)
    Y2 = model(X2)
    x = tf.constant([x], dtype=tf.float32)
    # y_pred = model(x)
    gradient = (Y1 - Y2) / (2 * h)
    gradient = tf.reshape(gradient, [1, -1])
    return gradient[0].numpy() if model(x) > 0.5 else -gradient[0].numpy()

# v2_2
def compute_grad(x, model, perturbation_size=1e-4):
    # compute the gradient of model perdictions w.r.t input attributes
    # 将x中每个属性+h，形成n个新实例，得到X。将X投入model，得到Y。再根据Y求对X的模拟偏导数，得到n个偏导数后，合并起来构成模拟导数。
    # Y是Tensor(n,1) n行1列，每一行的y值是输出的概率值，而不是实数标签 如果用实数标签，算出来的梯度就不准
    h = perturbation_size
    n = len(x)
    e = np.empty(n)
    e.fill(h)
    E = np.diag(e)
    X = np.repeat([x], n, axis=0) # v2_2 相较于V2，GPT4改了这里
    X = X + E
    X = tf.constant(X, dtype=tf.float32)
    Y = model(X)
    x = tf.constant([x], dtype=tf.float32)
    y_pred = model(x)
    gradient = (Y - y_pred) / h
    gradient = tf.reshape(gradient, [1, -1])
    return gradient[0].numpy() if model(x) > 0.5 else -gradient[0].numpy()

def test_time(x, n):
    # h = 1e-4
    # t1 = time.time()
    # for _ in range(10000):
    #     e = np.empty(n)
    #     e.fill(h)
    #     E = np.diag(e)
    # t2 = time.time()
    # print(t2 - t1)
    #
    # t1 = time.time()
    # for _ in range(10000):
    #     e = np.full(n, h)
    #     E = np.diag(e)
    # t2 = time.time()
    # print(t2 - t1)

    y = x.reshape(1, n)
    X = np.repeat(y, n, axis=0)
    print(np.repeat(x.reshape(1, n), n, axis=0))
    print(np.tile(x, (n, 1)))
    t1 = time.time()
    for _ in range(10000):
        X = np.repeat(x.reshape(1, n), n, axis=0)
    t2 = time.time()
    print(t2 - t1)

    t1 = time.time()
    for _ in range(10000):
        X = np.tile(x, (n, 1))
    t2 = time.time()
    print(t2 - t1)




x = np.array(range(5))
# compute_grad_v2(x, model='model')
test_time(x, len(x))