"""
This python file implement our approach BBDIG, and it can be simply applied to other differentiable prediction models.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import cluster
import itertools
import time
import generation_utilities


def compute_grad(x, model, perturbation_size=1e-4):
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


def global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, decay, max_iter, s_g,
                      perturbation_size):
    # global generation phase of EIDIG

    g_id = np.empty(shape=(0, num_attribs))
    all_gen_g = np.empty(shape=(0, num_attribs))
    try_times = 0
    g_num = len(seeds)
    for i in range(g_num):
        x1 = seeds[i].copy()
        grad1 = np.zeros_like(X[0]).astype(float)
        grad2 = np.zeros_like(X[0]).astype(float)
        for _ in range(max_iter):
            try_times += 1
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                g_id = np.append(g_id, [x1], axis=0)
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            # 3.2 use momentum
            grad1 = decay * grad1 + compute_grad(x1, model, perturbation_size)
            grad2 = decay * grad2 + compute_grad(x2, model, perturbation_size)
            direction = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction[attrib] = (-1) * sign_grad1[attrib]
            x1 = x1 + s_g * direction
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_g = np.append(all_gen_g, [x1], axis=0)
    g_id = np.array(list(set([tuple(id) for id in g_id])))
    return g_id, all_gen_g, try_times


def local_generation(num_attribs, l_num, g_id, protected_attribs, constraint, model, update_interval, s_l, epsilon,
                     perturbation_size):
    # local generation phase of EIDIG

    direction = [-1, 1]
    l_id = np.empty(shape=(0, num_attribs))
    all_gen_l = np.empty(shape=(0, num_attribs))
    try_times = 0
    for x1 in g_id:
        x0 = x1.copy()
        similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
        x2 = generation_utilities.max_diff(x1, similar_x1, model)
        grad1 = compute_grad(x1, model, perturbation_size)
        grad2 = compute_grad(x2, model, perturbation_size)
        p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
        p0 = p.copy()
        suc_iter = 0
        for _ in range(l_num):
            try_times += 1
            if suc_iter >= update_interval:
                similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                x2 = generation_utilities.find_pair(x1, similar_x1, model)
                grad1 = compute_grad(x1, model, perturbation_size)
                grad2 = compute_grad(x2, model, perturbation_size)
                p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
                suc_iter = 0
            suc_iter += 1
            a = generation_utilities.random_pick(p)
            s = generation_utilities.random_pick([0.5, 0.5])
            x1[a] = x1[a] + direction[s] * s_l
            x1 = generation_utilities.clip(x1, constraint)
            all_gen_l = np.append(all_gen_l, [x1], axis=0)
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                l_id = np.append(l_id, [x1], axis=0)
            else:
                x1 = x0.copy()
                p = p0.copy()
                suc_iter = 0
    l_id = np.array(list(set([tuple(id) for id in l_id])))
    return l_id, all_gen_l, try_times


def individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, decay, l_num, update_interval,
                                         max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, perturbation_size=1e-4):
    # complete implementation of EIDIG
    # return non-duplicated individual discriminatory instances generated, non-duplicate instances generated and total number of search iterations

    num_attribs = len(X[0])
    g_id, gen_g, g_gen_num = global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, decay,
                                               max_iter, s_g, perturbation_size)
    l_id, gen_l, l_gen_num = local_generation(num_attribs, l_num, g_id, protected_attribs, constraint, model,
                                              update_interval, s_l, epsilon_l, perturbation_size)
    all_id = np.append(g_id, l_id, axis=0)
    all_gen = np.append(gen_g, gen_l, axis=0)
    all_id_nondup = np.array(list(set([tuple(id) for id in all_id])))
    all_gen_nondup = np.array(list(set([tuple(gen) for gen in all_gen])))
    return all_id_nondup, all_gen_nondup, g_gen_num + l_gen_num


def seedwise_generation(X, seeds, protected_attribs, constraint, model, l_num, decay, update_interval, max_iter=10,
                        s_g=1.0, s_l=1.0, epsilon=1e-6, perturbation_size=1e-4):
    # perform global generation and local generation successively on each single seed

    num_seeds = len(seeds)
    num_gen = np.array([0] * num_seeds)
    num_ids = np.array([0] * num_seeds)
    num_attribs = len(X[0])
    ids = np.empty(shape=(0, num_attribs))
    all_gen = np.empty(shape=(0, num_attribs))
    direction = [-1, 1]
    for index, instance in enumerate(seeds):
        x1 = instance.copy()
        flag = False
        grad1 = np.zeros_like(X[0]).astype(float)
        grad2 = np.zeros_like(X[0]).astype(float)
        for j in range(max_iter):
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                ids = np.append(ids, [x1], axis=0)
                flag = True
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = decay * grad1 + compute_grad(x1, model, perturbation_size)
            grad2 = decay * grad2 + compute_grad(x2, model, perturbation_size)
            direction_g = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction_g[attrib] = (-1) * sign_grad1[attrib]
            x1 = x1 + s_g * direction_g
            x1 = generation_utilities.clip(x1, constraint)
            all_gen = np.append(all_gen, [x1], axis=0)
        if flag == True:
            x0 = x1.copy()
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = compute_grad(x1, model, perturbation_size)
            grad2 = compute_grad(x2, model, perturbation_size)
            p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
            p0 = p.copy()
            suc_iter = 0
            for _ in range(l_num):
                if suc_iter >= update_interval:
                    similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                    x2 = generation_utilities.find_pair(x1, similar_x1, model)
                    grad1 = compute_grad(x1, model, perturbation_size)
                    grad2 = compute_grad(x2, model, perturbation_size)
                    p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
                    suc_iter = 0
                suc_iter += 1
                a = generation_utilities.random_pick(p)
                s = generation_utilities.random_pick([0.5, 0.5])
                x1[a] = x1[a] + direction[s] * s_l
                x1 = generation_utilities.clip(x1, constraint)
                all_gen = np.append(all_gen, [x1], axis=0)
                similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                if generation_utilities.is_discriminatory(x1, similar_x1, model):
                    ids = np.append(ids, [x1], axis=0)
                else:
                    x1 = x0.copy()
                    p = p0.copy()
                    suc_iter = 0
        nondup_ids = np.array(list(set([tuple(id) for id in ids])))
        nondup_gen = np.array(list(set([tuple(gen) for gen in all_gen])))
        num_gen[index] = len(nondup_gen)
        num_ids[index] = len(nondup_ids)
    return num_gen, num_ids


def time_record(X, seeds, protected_attribs, constraint, model, decay, l_num, record_step, record_frequency,
                update_interval, max_iter=10, s_g=1.0, s_l=1.0, epsilon=1e-6, perturbation_size=1e-4):
    # record time consumption

    num_attribs = len(X[0])
    t = np.array([0.0] * record_frequency)
    direction_l = [-1, 1]
    threshold = record_step
    index = 0
    t1 = time.time()
    ids = np.empty(shape=(0, num_attribs))
    num_ids = num_ids_before = 0
    for instance in seeds:
        if num_ids >= record_frequency * record_step:
            break
        x1 = instance.copy()
        flag = False
        grad1 = np.zeros_like(X[0]).astype(float)
        grad2 = np.zeros_like(X[0]).astype(float)
        for i in range(max_iter + 1):
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, model):
                ids = np.append(ids, [x1], axis=0)
                flag = True
                break
            if i == max_iter:
                break
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = decay * grad1 + compute_grad(x1, model, perturbation_size)
            grad2 = decay * grad2 + compute_grad(x2, model, perturbation_size)
            direction_g = np.zeros_like(X[0])
            sign_grad1 = np.sign(grad1)
            sign_grad2 = np.sign(grad2)
            for attrib in range(num_attribs):
                if attrib not in protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                    direction_g[attrib] = (-1) * sign_grad1[attrib]
            x1 = x1 + s_g * direction_g
            x1 = generation_utilities.clip(x1, constraint)
            t2 = time.time()
        if flag == True:
            ids = np.array(list(set([tuple(id) for id in ids])))
            num_ids = len(ids)
            if num_ids > num_ids_before:
                num_ids_before = num_ids
                if num_ids == threshold:
                    t[index] = t2 - t1
                    threshold += record_step
                    index += 1
                    if num_ids >= record_frequency * record_step:
                        break
            x0 = x1.copy()
            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            x2 = generation_utilities.max_diff(x1, similar_x1, model)
            grad1 = compute_grad(x1, model, perturbation_size)
            grad2 = compute_grad(x2, model, perturbation_size)
            p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
            p0 = p.copy()
            suc_iter = 0
            for _ in range(l_num):
                if suc_iter >= update_interval:
                    similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                    x2 = generation_utilities.find_pair(x1, similar_x1, model)
                    grad1 = compute_grad(x1, model, perturbation_size)
                    grad2 = compute_grad(x2, model, perturbation_size)
                    p = generation_utilities.normalization(grad1, grad2, protected_attribs, epsilon)
                    suc_iter = 0
                suc_iter += 1
                a = generation_utilities.random_pick(p)
                s = generation_utilities.random_pick([0.5, 0.5])
                x1[a] = x1[a] + direction_l[s] * s_l
                x1 = generation_utilities.clip(x1, constraint)
                t2 = time.time()
                similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
                if generation_utilities.is_discriminatory(x1, similar_x1, model):
                    ids = np.append(ids, [x1], axis=0)
                    ids = np.array(list(set([tuple(id) for id in ids])))
                    num_ids = len(ids)
                    if num_ids > num_ids_before:
                        num_ids_before = num_ids
                        if num_ids == threshold:
                            t[index] = t2 - t1
                            threshold += record_step
                            index += 1
                            if num_ids >= record_frequency * record_step:
                                break
                else:
                    x1 = x0.copy()
                    p = p0.copy()
                    suc_iter = 0
    return t
