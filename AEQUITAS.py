"""
This python file reproduces ADF, the state-of-the-art individual discrimination generation algorithm.
The official implementation can be accessed at https://github.com/pxzhang94/ADF.
"""
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import cluster
import itertools
import time
import generation_utilities

# 添加了initial_input参数
# try_times参数等价于seeds数目
def global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g, initial_input):
    # global generation phase of AEQUITAS

    g_id = np.empty(shape=(0, num_attribs))
    all_gen_g = np.empty(shape=(0, num_attribs))
    # try_times = 0
    g_num = len(seeds)
    try_times = g_num
    x1 = initial_input.copy()
    for i in range(g_num):
        for j in range(num_attribs):
            # random select to make a new potential individual instance
            # and clip the generating instance with each feature to make sure it is valid
            x1[j] = random.randint(constraint[j][0], constraint[j][1])
        similar_x = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
        if generation_utilities.is_discriminatory(x1, similar_x, model):
            g_id = np.append(g_id, [x1], axis=0)
        all_gen_g = np.append(all_gen_g, [x1], axis=0)
    g_id = np.array(list(set([tuple(id) for id in g_id])))
    return g_id, all_gen_g, try_times

# 添加param_probability, param_probability_change_size,direction_probability, direction_probability_change_size参数
def local_generation(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l, epsilon, param_probability, param_probability_change_size,
                 direction_probability, direction_probability_change_size):
    # local generation phase of AEQUITAS

    direction = [-1, 1]
    l_id = np.empty(shape=(0, num_attribs))
    all_gen_l = np.empty(shape=(0, num_attribs))
    try_times = 0
    for x1 in g_id:
        x0 = x1.copy()
        for _ in range(l_num):
            try_times += 1
            # randomly choose the feature for perturbation
            param_choice = np.random.choice(range(num_attribs), p=param_probability)

            # randomly choose the direction for perturbation
            direction_choice = np.random.choice(direction, p=[direction_probability[param_choice], (1 - direction_probability[param_choice])])
            if (x1[param_choice] == constraint[param_choice][0]) or (
                    x1[param_choice] == constraint[param_choice][1]):
                direction_choice = np.random.choice(direction)

            # perturbation
            x1[param_choice] = x1[param_choice] + (direction_choice * s_l)

            # clip the generating instance with each feature to make sure it is valid
            x1 = generation_utilities.clip(x1, constraint)

            similar_x1 = generation_utilities.similar_set(x1, num_attribs, protected_attribs, constraint)
            is_discriminatory = generation_utilities.is_discriminatory(x1, similar_x1, model)
            if is_discriminatory:
                l_id = np.append(l_id, [x1], axis=0)
            else:
                x1 = x0.copy()
            all_gen_l = np.append(all_gen_l, [x1], axis=0)

            # update the probabilities of directions
            if (is_discriminatory and direction_choice == -1) or (not is_discriminatory and direction_choice == 1):
                direction_probability[param_choice] = min(direction_probability[param_choice] + (direction_probability_change_size * s_l), 1)
            elif (not is_discriminatory and direction_choice == -1) or (is_discriminatory and direction_choice == 1):
                direction_probability[param_choice] = max(direction_probability[param_choice] - (direction_probability_change_size * s_l), 0)

            # update the probabilities of features
            if is_discriminatory:
                param_probability[param_choice] = param_probability[param_choice] + param_probability_change_size
            else:
                param_probability[param_choice] = max(param_probability[param_choice] - param_probability_change_size, 0)
            # normalize the probabilities of features
            param_probability = param_probability / np.sum(param_probability)

    l_id = np.array(list(set([tuple(id) for id in l_id])))
    return l_id, all_gen_l, try_times


# 添加参数：不同dataset对应的initial_input
def individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, l_num, max_iter=10, s_g=1.0,
                                         s_l=1.0, epsilon=1e-6, initial_input=None):
    # complete implementation of AEQUITAS
    # return non-duplicated individual discriminatory instances generated, non-duplicate instances generated and total number of search iterations

    num_attribs = len(X[0])

    # hyper-parameters for initial probabilities of directions
    init_prob = 0.5
    direction_probability = [init_prob] * num_attribs
    direction_probability_change_size = 0.001

    # hyper-parameters for features
    param_probability = [1.0 / num_attribs] * num_attribs
    param_probability_change_size = 0.001

    if initial_input is None:
        print('Please input corresponding initial input')
        initial_input = np.zeros_like(X[0])

    g_id, gen_g, g_gen_num = global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter,
                                               s_g, initial_input)
    l_id, gen_l, l_gen_num = local_generation(num_attribs, l_num, g_id, protected_attribs, constraint, model, s_l,
                                              epsilon, param_probability, param_probability_change_size,
                 direction_probability, direction_probability_change_size)
    all_id = np.append(g_id, l_id, axis=0)
    all_gen = np.append(gen_g, gen_l, axis=0)
    all_id_nondup = np.array(list(set([tuple(id) for id in all_id])))
    all_gen_nondup = np.array(list(set([tuple(gen) for gen in all_gen])))
    return all_id_nondup, all_gen_nondup, g_gen_num + l_gen_num