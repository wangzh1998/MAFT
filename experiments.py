"""
This python file provides experimental functions backing up the claims involving efficiency and effectiveness in our paper.
"""


import os

import tensorflow as tf
import numpy as np
import generation_utilities
import time
import ADF
import EIDIG
import MAFT
import AEQUITAS
import SG
import Gradient
from experiment_config import Method, BlackboxMethod, AllMethod

# allocate GPU and set dynamic memory growth
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# make outputs stable across runs for validation
# alternatively remove them when dealing with real-world issues
np.random.seed(42)
tf.random.set_seed(42)

def gradient_comparison(benchmark, X, model, g_num=1000, perturbation_size=1e-4, l_num=1000, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    # compare different perturbation_size in terms of effectiveness and efficiency of MAFT

    print('--- START ', '---')
    if g_num >= len(X):
        seeds = X.copy()
    else:
        clustered_data = generation_utilities.clustering(X, c_num)
        seeds = np.empty(shape=(0, len(X[0])))
        for i in range(g_num):
            new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i % c_num, fashion=fashion)
            seeds = np.append(seeds, [new_seed], axis=0)

    # ADF
    t1 = time.time()
    adf_gradients = Gradient.adf_gradient_generation(seeds, len(X[0]), model)
    np.save('logging_data/gradients_comparison/' + benchmark + '_ADF_gradient' + '.npy', adf_gradients)
    t2 = time.time()
    adf_time_cost = t2 - t1
    print('ADF:', 'Generate gradients of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:', t2 - t1,
            's.')

    # EIDIG
    t1 = time.time()
    eidig_gradients = Gradient.eidig_gradient_generation(seeds, len(X[0]), model)
    np.save('logging_data/gradients_comparison/' + benchmark + '_EIDIG_gradient' + '.npy', eidig_gradients)
    t2 = time.time()
    eidig_time_cost = t2 - t1
    print('EIDIG-5:', 'Generate gradients of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:',
          t2 - t1, 's.')

    # MAFT
    t1 = time.time()
    maft_gradients = Gradient.maft_gradient_generation(seeds, len(X[0]), model, perturbation_size)
    np.save('logging_data/gradients_comparison/' + benchmark + '_MAFT_gradient' + '.npy', maft_gradients)
    t2 = time.time()
    maft_time_cost = t2 - t1
    print('MAFT-5:', 'Generate gradients of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:',
          t2 - t1, 's.')

    # MAFT non-vectorized
    t1 = time.time()
    maft_gradients_non_vec = Gradient.maft_gradient_generation_non_vec(seeds, len(X[0]), model, perturbation_size)
    np.save('logging_data/gradients_comparison/' + benchmark + '_MAFT_gradient' + '.npy', maft_gradients)
    t2 = time.time()
    maft_time_cost_non_vec = t2 - t1
    print('MAFT-5-non-vec:', 'Generate gradients of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:',
          t2 - t1, 's.')

    print('--- END ', '---')
    return adf_gradients, eidig_gradients, maft_gradients, maft_gradients_non_vec, adf_time_cost, eidig_time_cost, maft_time_cost, maft_time_cost_non_vec

def gradient_comparison_global_direction(benchmark, X, protected_attribs, constraint, model, g_num=1000, perturbation_size=1e-4, l_num=1000, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    # compare global direction direction

    print('--- START ', '---')
    if g_num >= len(X):
        seeds = X.copy()
    else:
        clustered_data = generation_utilities.clustering(X, c_num)
        seeds = np.empty(shape=(0, len(X[0])))
        for i in range(g_num):
            new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i % c_num, fashion=fashion)
            seeds = np.append(seeds, [new_seed], axis=0)

    num_attribs = len(X[0])

    # ADF
    # t1 = time.time()
    # adf_directions = Gradient.adf_gradient_generation(seeds, len(X[0]), model)
    # # np.save('logging_data/directions_comparison/' + benchmark + '_ADF_gradient' + '.npy', adf_directions)
    # t2 = time.time()
    # adf_time_cost = t2 - t1
    # print('ADF:', 'Generate', len(adf_directions), 'directions of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:', t2 - t1,
    #       's.')

    # EIDIG
    t1 = time.time()
    # eidig_directions = Gradient.eidig_gradient_generation(seeds, len(X[0]), model)
    eidig_directions = EIDIG.global_direction_comparison(X, seeds, num_attribs, protected_attribs, constraint, model, decay)
    # np.save('logging_data/directions_comparison/' + benchmark + '_EIDIG_gradient' + '.npy', eidig_directions)
    t2 = time.time()
    eidig_time_cost = t2 - t1
    print('EIDIG-5:', 'Generate', len(eidig_directions), 'directions of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:',
          t2 - t1, 's.')

    # MAFT
    t1 = time.time()
    # maft_directions = Gradient.maft_gradient_generation(seeds, len(X[0]), model, perturbation_size)
    maft_directions = MAFT.global_direction_comparison(X, seeds, num_attribs, protected_attribs, constraint, model, decay, perturbation_size)
    # np.save('logging_data/directions_comparison/' + benchmark + '_MAFT_gradient' + '.npy', maft_directions)
    t2 = time.time()
    maft_time_cost = t2 - t1
    print('MAFT-5:', 'Generate', len(maft_directions), 'directions of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:',
          t2 - t1, 's.')

    print('--- END ', '---')
    return eidig_directions, maft_directions, eidig_time_cost, maft_time_cost

def gradient_comparison_local_probability(benchmark, X, protected_attribs, constraint, model, g_num=1000, perturbation_size=1e-4, l_num=1000, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    # compare global direction direction

    print('--- START ', '---')
    if g_num >= len(X):
        seeds = X.copy()
    else:
        clustered_data = generation_utilities.clustering(X, c_num)
        seeds = np.empty(shape=(0, len(X[0])))
        for i in range(g_num):
            new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i % c_num, fashion=fashion)
            seeds = np.append(seeds, [new_seed], axis=0)

    num_attribs = len(X[0])

    # EIDIG
    t1 = time.time()
    eidig_probabilities = EIDIG.local_probability_comparision(seeds, num_attribs, protected_attribs, constraint, model, epsilon_l)
    # np.save('logging_data/directions_comparison/' + benchmark + '_EIDIG_gradient' + '.npy', eidig_directions)
    t2 = time.time()
    eidig_time_cost = t2 - t1
    print('EIDIG-5:', 'Generate', len(eidig_probabilities), 'probabilities of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:',
          t2 - t1, 's.')

    # MAFT
    t1 = time.time()
    maft_probabilities = MAFT.local_probability_comparision(seeds, num_attribs, protected_attribs, constraint, model, epsilon_l, perturbation_size)
    # np.save('logging_data/directions_comparison/' + benchmark + '_MAFT_gradient' + '.npy', maft_directions)
    t2 = time.time()
    maft_time_cost = t2 - t1
    print('MAFT-5:', 'Generate', len(maft_probabilities), 'probabilities of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:',
          t2 - t1, 's.')

    print('--- END ', '---')
    return eidig_probabilities, maft_probabilities, eidig_time_cost, maft_time_cost

def hyper_comparison(round_id, benchmark, X, protected_attribs, constraint, model, perturbation_size_list, initial_input=None, dataset_configuration = {},
                     g_num=100, l_num=100, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6,
                     fashion='RoundRobin'):
    # compare different perturbation_size in terms of effectiveness and efficiency of MAFT

    iter = '{}x{}'.format(g_num, l_num)
    dir = 'logging_data/hyper_comparison/hyper_comparison_instances/' + iter + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    base_methods_nums = len(AllMethod) - 1
    tot_methods_num = base_methods_nums + len(perturbation_size_list)
    num_ids = np.zeros(shape=(tot_methods_num))
    num_all_ids = np.zeros_like(num_ids)
    time_costs = np.zeros_like(num_ids)
    total_iters = np.zeros_like(num_ids)

    round_now = round_id
    print('--- ROUND', round_now, '---')
    if g_num >= len(X):
        seeds = X.copy()
    else:
        clustered_data = generation_utilities.clustering(X, c_num)
        seeds = np.empty(shape=(0, len(X[0])))
        for i in range(g_num):
            new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i % c_num, fashion=fashion)
            seeds = np.append(seeds, [new_seed], axis=0)

    def run_algorithm(method, perturbation_size=None):
        t1 = time.time()
        if method == AllMethod.AEQUITAS:
            ids, gen, total_iter = AEQUITAS.individual_discrimination_generation(X, seeds, protected_attribs, constraint,
                                                                            model, l_num, max_iter, s_g, s_l,
                                                                            epsilon_l, initial_input)

        elif method == AllMethod.SG:
            ids, gen, total_iter = SG.individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, dataset_configuration, l_num)
        elif method == AllMethod.ADF:
            ids, gen, total_iter = ADF.individual_discrimination_generation(X, seeds, protected_attribs, constraint,
                                                                            model, l_num, max_iter, s_g, s_l,
                                                                            epsilon_l)
        elif method == AllMethod.EIDIG:
            ids, gen, total_iter = EIDIG.individual_discrimination_generation(X, seeds, protected_attribs,
                                                                              constraint, model, decay, l_num, 5,
                                                                              max_iter, s_g, s_l, epsilon_l)
        elif method == AllMethod.MAFT:
            ids, gen, total_iter = MAFT.individual_discrimination_generation(X, seeds, protected_attribs,
                                                                             constraint, model, decay, l_num, 5,
                                                                             max_iter, s_g, s_l, epsilon_l,
                                                                             perturbation_size)
        else:
            raise ValueError("Invalid method")

        np.save(dir + benchmark + '_ids_' + method.name + '_' + str(perturbation_size) + '_' + 'round' + str(round_now) + '.npy', ids)
        t2 = time.time()
        time_cost = t2 - t1
        if method == AllMethod.MAFT:
            print('{}-{}: unique dis ins:{}, unique tot ins:{}, total iters:{}, time cost:{}, speed:{} ins/s, success rate:{}.'
              .format(method.name, perturbation_size, len(ids), len(gen), total_iter, time_cost, len(ids)/time_cost, len(ids)/total_iter))
        else:
            print('{}: unique dis ins:{}, unique tot ins:{}, total iters:{}, time cost:{}, speed:{} ins/s, success rate:{}.'
                .format(method.name, len(ids), len(gen), total_iter, time_cost, len(ids) / time_cost,
                        len(ids) / total_iter))
        return ids, gen, total_iter, time_cost

    for method in AllMethod:
        if(method == AllMethod.MAFT):
            for idx, perturbation_size_list in enumerate(perturbation_size_list):
                ids, gen, total_iter, time_cost = run_algorithm(method, perturbation_size_list)
                num_ids[method.value+idx] = len(ids)
                num_all_ids[method.value+idx] = len(gen)
                total_iters[method.value+idx] = total_iter
                time_costs[method.value+idx] = time_cost
        else:
            ids, gen, total_iter, time_cost = run_algorithm(method)
            num_ids[method.value] = len(ids)
            num_all_ids[method.value] = len(gen)
            total_iters[method.value] = total_iter
            time_costs[method.value] = time_cost

    print('\n')
    return num_ids, num_all_ids, total_iters, time_costs

# compare MAFT with white-box methods (ADF and EIDIG) in terms of effectiveness and efficiency
def comparison(round_id, benchmark, X, protected_attribs, constraint, model, g_num=1000, l_num=1000, perturbation_size=1e-4, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):

    iter = '{}x{}_H_{}'.format(g_num, l_num, perturbation_size)
    # store invividual discrimination instances
    dir = 'logging_data/complete_comparison/complete_comparison_instances/' + iter + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    method_nums = len(Method)
    num_ids = np.zeros(shape=(method_nums))
    num_all_ids = np.zeros(shape=(method_nums))
    time_costs = np.zeros(shape=(method_nums))
    total_iters = np.zeros(shape=(method_nums))

    round_now = round_id
    print('--- ROUND', round_now, '---')
    if g_num >= len(X):
        seeds = X.copy()
    else:
        clustered_data = generation_utilities.clustering(X, c_num)
        seeds = np.empty(shape=(0, len(X[0])))
        for j in range(g_num):
            new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, j%c_num, fashion=fashion)
            seeds = np.append(seeds, [new_seed], axis=0)

    def run_algorithm(method):
        t1 = time.time()

        if method == Method.ADF:
            ids, gen, total_iter = ADF.individual_discrimination_generation(X, seeds, protected_attribs, constraint,
                                                                            model, l_num, max_iter, s_g, s_l,
                                                                            epsilon_l)
        elif method == Method.EIDIG:
            ids, gen, total_iter = EIDIG.individual_discrimination_generation(X, seeds, protected_attribs,
                                                                              constraint, model, decay, l_num, 5,
                                                                              max_iter, s_g, s_l, epsilon_l)
        elif method == Method.MAFT:
            ids, gen, total_iter = MAFT.individual_discrimination_generation(X, seeds, protected_attribs,
                                                                             constraint, model, decay, l_num, 5,
                                                                             max_iter, s_g, s_l, epsilon_l,
                                                                             perturbation_size)
        else:
            raise ValueError("Invalid method")

        np.save(dir + benchmark + '_ids_' + method.name + '_' + str(round_now) + '.npy', ids)
        t2 = time.time()
        time_cost = t2 - t1
        print(
            '{}: unique dis ins:{}, unique tot ins:{}, total iters:{}, time cost:{}, speed:{} ins/s, success rate:{}.'
            .format(method.name, len(ids), len(gen), total_iter, time_cost, len(ids) / time_cost,
                    len(ids) / total_iter))
        return ids, gen, total_iter, time_cost

    for method in Method:
        ids, gen, total_iter, time_cost = run_algorithm(method)
        num_ids[method.value] = len(ids)
        num_all_ids[method.value] = len(gen)
        total_iters[method.value] = total_iter
        time_costs[method.value] = time_cost
    print('\n')
    return num_ids, num_all_ids, total_iters, time_costs

# parameter 'initial_input' for AEQUITAS and parameter 'dataset_configuration' for SG
# compare MAFT with black-box methods (AEQUITAS and SG) in terms of effectiveness and efficiency
def comparison_blackbox(round_id, benchmark, X, protected_attribs, constraint, model, g_num=1000, l_num=1000, perturbation_size=1e-4, initial_input=None, dataset_configuration = {}, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):

    iter = '{}x{}_H_{}'.format(g_num, l_num, perturbation_size)
    # store invividual discrimination instances
    dir = 'logging_data/complete_comparison/complete_comparison_instances_bb/' + iter + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    method_nums = len(BlackboxMethod)
    num_ids = np.zeros(shape=(method_nums))
    num_all_ids = np.zeros(shape=(method_nums))
    time_costs = np.zeros(shape=(method_nums))
    total_iters = np.zeros(shape=(method_nums))


    round_now = round_id
    print('--- ROUND', round_now, '---')
    if g_num >= len(X):
        seeds = X.copy()
    else:
        clustered_data = generation_utilities.clustering(X, c_num)
        seeds = np.empty(shape=(0, len(X[0])))
        for j in range(g_num):
            new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, j%c_num, fashion=fashion)
            seeds = np.append(seeds, [new_seed], axis=0)

    def run_algorithm(method):
        t1 = time.time()

        if method == BlackboxMethod.AEQUITAS:
            ids, gen, total_iter = AEQUITAS.individual_discrimination_generation(X, seeds, protected_attribs, constraint,
                                                                            model, l_num, max_iter, s_g, s_l,
                                                                            epsilon_l, initial_input)
        elif method == BlackboxMethod.MAFT:
            ids, gen, total_iter = MAFT.individual_discrimination_generation(X, seeds, protected_attribs,
                                                                             constraint, model, decay, l_num, 5,
                                                                             max_iter, s_g, s_l, epsilon_l,
                                                                             perturbation_size)
        elif method == BlackboxMethod.SG:
            ids, gen, total_iter = SG.individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, dataset_configuration, l_num)
        else:
            raise ValueError("Invalid method")

        np.save(dir + benchmark + '_ids_' + method.name + '_' + str(round_now) + '.npy', ids)
        t2 = time.time()
        time_cost = t2 - t1
        print(
            '{}: unique dis ins:{}, unique tot ins:{}, total iters:{}, time cost:{}, speed:{} ins/s, success rate:{}.'
            .format(method.name, len(ids), len(gen), total_iter, time_cost, len(ids) / time_cost,
                    len(ids) / total_iter))
        return ids, gen, total_iter, time_cost

    for method in BlackboxMethod:
        ids, gen, total_iter, time_cost = run_algorithm(method)
        num_ids[method.value] = len(ids)
        num_all_ids[method.value] = len(gen)
        total_iters[method.value] = total_iter
        time_costs[method.value] = time_cost
    print('\n')
    return num_ids, num_all_ids, total_iters, time_costs
