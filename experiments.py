"""
This python file provides experimental functions backing up the claims involving efficiency and effectiveness in our paper.
"""


import os
from enum import Enum

import tensorflow as tf
from tensorflow import keras
import numpy as np
import generation_utilities
import time
import ADF
import EIDIG
import MAFT
import AEQUITAS
import SG
import Gradient
from experiment_config import Method, BlackboxMethod

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
    # np.save('logging_data/gradients_comparison/' + benchmark + '_ADF_gradient' + '.npy', adf_gradients)
    t2 = time.time()
    adf_time_cost = t2 - t1
    print('ADF:', 'Generate gradients of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:', t2 - t1,
            's.')

    # EIDIG
    t1 = time.time()
    eidig_gradients = Gradient.eidig_gradient_generation(seeds, len(X[0]), model)
    # np.save('logging_data/gradients_comparison/' + benchmark + '_EIDIG_gradient' + '.npy', eidig_gradients)
    t2 = time.time()
    eidig_time_cost = t2 - t1
    print('EIDIG-5:', 'Generate gradients of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:',
          t2 - t1, 's.')

    # MAFT
    t1 = time.time()
    maft_gradients = Gradient.maft_gradient_generation(seeds, len(X[0]), model, perturbation_size)
    # np.save('logging_data/gradients_comparison/' + benchmark + '_MAFT_gradient' + '.npy', maft_gradients)
    t2 = time.time()
    maft_time_cost = t2 - t1
    print('MAFT-5:', 'Generate gradients of ', len(seeds), ' seeds on benchmark ', benchmark, '. Time cost:',
          t2 - t1, 's.')

    # MAFT non-vectorized
    t1 = time.time()
    maft_gradients_non_vec = Gradient.maft_gradient_generation_non_vec(seeds, len(X[0]), model, perturbation_size)
    # np.save('logging_data/gradients_comparison/' + benchmark + '_MAFT_gradient' + '.npy', maft_gradients)
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

def hyper_comparison(round_id, benchmark, X, protected_attribs, constraint, model, perturbation_size_list,
                     g_num=1000, l_num=1000, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6,
                     fashion='RoundRobin'):
    # compare different perturbation_size in terms of effectiveness and efficiency of MAFT

    iter = '{}x{}'.format(g_num, l_num)
    dir = 'logging_data/hyper_comparison/generate_instances/' + iter + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    num_ids = np.array(
        [0] * (len(perturbation_size_list) + 2))  # nums_id[i]表示第i个方法生成的的id数
    num_all_ids = np.array([0] * (len(perturbation_size_list) + 2))
    time_costs = np.array([0] * (len(perturbation_size_list) + 2))
    total_iters = np.array([0] * (len(perturbation_size_list) + 2))

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

        np.save(dir + benchmark + '_ids_' + method.name + '_' + str(perturbation_size) + '_' + 'round' + str(round_now) + '.npy', ids)
        t2 = time.time()
        time_cost = t2 - t1
        print('{}-{}: unique dis ins:{}, unique tot ins:{}, total iters:{}, time cost:{}, speed:{} ins/s, success rate:{}.'
              .format(method.name, perturbation_size, len(ids), len(gen), total_iter, time_cost, len(ids)/time_cost, len(ids)/total_iter))
        return ids, gen, total_iter, time_cost

    for method in Method:
        if(method == Method.MAFT):
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

def comparison(round_id, benchmark, X, protected_attribs, constraint, model, g_num=1000, l_num=1000, perturbation_size=1e-4, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    # compare MAFT with EIDIG and ADF in terms of effectiveness and efficiency

    iter = '{}x{}_H_{}'.format(g_num, l_num, perturbation_size)
    dir = 'logging_data/logging_data_from_tests/complete_comparison_instances/' + iter + '/'
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
        nonlocal num_ids, time_cost
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

# 添加了参数initial_input
# 为调用SG方法添加了参数dataset_configuration
def comparison_blackbox(num_experiment_round, benchmark, X, protected_attribs, constraint, model, g_num=1000, l_num=1000, perturbation_size=1e-4, initial_input=None, dataset_configuration = {}, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    # compare MAFT with AEQUITAS in terms of effectiveness and efficiency

    iter = '{}x{}_H_{}'.format(g_num, l_num, perturbation_size)
    dir = 'logging_data/logging_data_from_tests/complete_comparison_instances_bb/' + iter + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # 获取BBMethod的方法数
    num_methods = len(BlackboxMethod)
    num_ids = np.zeros(shape=(num_methods, num_experiment_round), dtype=np.float64)
    time_cost = np.zeros(shape=(num_methods, num_experiment_round), dtype=np.float64)

    for i in range(num_experiment_round):
        round_now = i + 1
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
            nonlocal num_ids, time_cost
            t1 = time.time()

            if method == BlackboxMethod.AEQUITAS:
                ids, gen, total_iter = AEQUITAS.individual_discrimination_generation(X, seeds, protected_attribs, constraint,
                                                                                model, l_num, max_iter, s_g, s_l,
                                                                                epsilon_l, initial_input)
            elif method == BlackboxMethod.SG:
                ids, gen, total_iter = SG.individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, dataset_configuration, l_num)
            elif method == BlackboxMethod.MAFT:
                ids, gen, total_iter = MAFT.individual_discrimination_generation(X, seeds, protected_attribs,
                                                                                 constraint, model, decay, l_num, 5,
                                                                                 max_iter, s_g, s_l, epsilon_l,
                                                                                 perturbation_size)
            else:
                raise ValueError("Invalid method")

            np.save(dir + benchmark + '_ids_' + method.name + '_' + str(round_now) + '.npy', ids)
            t2 = time.time()
            print(method.name, 'In', total_iter, 'search iterations', len(gen),
                  'non-duplicate instances are explored', len(ids), 'of which are discriminatory. Time cost:', t2 - t1,
                  's.')
            num_ids[method.value][i] = len(ids)
            time_cost[method.value][i] = t2 - t1

        for method in BlackboxMethod:
            run_algorithm(method)
        print('\n')

    avg_num_ids = np.mean(num_ids, axis=1)
    avg_speed = np.mean(num_ids / time_cost, axis=1) # 更新了计算平均值的方式，和后面在分析时同步
    print('Results of complete comparison on', benchmark,
          'with g_num set to {} and l_num set to {}'.format(g_num, l_num), ',averaged on', num_experiment_round,
          'rounds:')
    for method in BlackboxMethod:
        print(method.name, ':', avg_num_ids[method.value],
              'individual discriminatory instances are generated at a speed of', avg_speed[method.value],
              'per second.')
    return num_ids, time_cost

def global_comparison(num_experiment_round, benchmark, X, protected_attribs, constraint, model, decay_list, num_seeds=1000, c_num=4, max_iter=10, s_g=1.0):
    # compare the global phase given the same set of seeds

    num_ids = np.array([0] * (len(decay_list) + 1))
    num_iter = np.array([0] * (len(decay_list) + 1))
    time_cost = np.array([0] * (len(decay_list) + 1))

    for i in range(num_experiment_round):
        round_now = i + 1
        print('--- ROUND', round_now, '---')
        num_attribs = len(X[0])
        num_dis = 0
        if num_seeds >= len(X):
            seeds = X
        else:
            clustered_data = generation_utilities.clustering(X, c_num)
            seeds = np.empty(shape=(0, num_attribs))
            for i in range(num_seeds):
                x_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i%c_num, fashion='Distribution')
                seeds = np.append(seeds, [x_seed], axis=0)
        for seed in seeds:
            similar_seed = generation_utilities.similar_set(seed, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(seed, similar_seed, model):
                num_dis += 1
        print('Given', num_seeds, '(no more than 600 for german credit) seeds,', num_dis, 'of which are individual discriminatory instances.')

        t1 = time.time()
        ids_ADF, _, total_iter_ADF = ADF.global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g)
        t2 = time.time()
        num_ids_ADF = len(ids_ADF)
        print('ADF:', 'In', total_iter_ADF, 'search iterations,', num_ids_ADF, 'non-duplicate individual discriminatory instances are generated. Time cost:', t2-t1, 's.')
        num_ids[0] += num_ids_ADF
        num_iter[0] += total_iter_ADF
        time_cost[0] += t2-t1

        for index, decay in enumerate(decay_list):
            print('Decay factor set to {}:'.format(decay))
            t1 = time.time()
            ids_EIDIG, _, total_iter_EIDIG = EIDIG.global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, decay, max_iter, s_g)
            t2 = time.time()
            num_ids_EIDIG = len(ids_EIDIG)
            print('EIDIG:', 'In', total_iter_EIDIG, 'search iterations,', num_ids_EIDIG, 'non-duplicate individual discriminatory instances are generated. Time cost:', t2-t1, 's.')
            num_ids[index+1] += num_ids_EIDIG
            num_iter[index+1] += total_iter_EIDIG
            time_cost[index+1] += t2-t1
        
        print('\n')

    avg_num_ids = num_ids / num_experiment_round
    avg_speed = num_ids / time_cost
    avg_iter = num_iter / num_experiment_round / num_seeds
    print('Results of global phase comparsion on', benchmark, 'given {} seeds'.format(num_seeds), ',averaged on', num_experiment_round, 'rounds:')
    print('ADF:', avg_num_ids[0], 'individual discriminatory instances are generated at a speed of', avg_speed[0], 'per second, and the number of iterations on a singe seed is', avg_iter[0], '.')
    for index, decay in enumerate(decay_list):
        print('Decay factor set to {}:'.format(decay))
        print('EIDIG:', avg_num_ids[index+1], 'individual discriminatory instances are generated at a speed of', avg_speed[index+1], 'per second, and the number of iterations on a singe seed is', avg_iter[index+1], '.')

    return num_ids, num_iter, time_cost
    

def local_comparison(num_experiment_round, benchmark, X, protected_attribs, constraint, model, update_interval_list, num_seeds=100, l_num=1000, c_num=4, s_l=1.0, epsilon=1e-6):
    # compare the local phase given the same individual discriminatory instances set

    num_ids = np.array([0] * (len(update_interval_list) + 1))
    time_cost = np.array([0] * (len(update_interval_list) + 1))

    for i in range(num_experiment_round):
        round_now = i + 1
        print('--- ROUND', round_now, '---')
        num_attribs = len(X[0])
        clustered_data = generation_utilities.clustering(X, c_num)
        id_seeds = np.empty(shape=(0, num_attribs))
        for i in range(100000000):
            x_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i%c_num, fashion='RoundRobin')
            similar_x_seed = generation_utilities.similar_set(x_seed, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x_seed, similar_x_seed, model):
                id_seeds = np.append(id_seeds, [x_seed], axis=0)
                if len(id_seeds) >= num_seeds:
                    break
    
        t1 = time.time()
        ids_ADF, _, total_iter_ADF = ADF.local_generation(num_attribs, l_num, id_seeds.copy(), protected_attribs, constraint, model, s_l, epsilon)
        t2 = time.time()
        num_ids_ADF = len(ids_ADF)
        print('ADF:', 'In', total_iter_ADF, 'search iterations,', num_ids_ADF, 'non-duplicate individual discriminatory instances are generated. Time cost:', t2-t1, 's.')
        num_ids[0] += num_ids_ADF
        time_cost[0] += t2-t1
        
        for index, update_interval in enumerate(update_interval_list):
            print('Update interval set to {}:'.format(update_interval))
            t1 = time.time()
            ids_EIDIG, _, total_iter_EIDIG = EIDIG.local_generation(num_attribs, l_num, id_seeds.copy(), protected_attribs, constraint, model, update_interval, s_l, epsilon)
            t2 = time.time()
            num_ids_EIDIG = len(ids_EIDIG)
            print('EIDIG:', 'In', total_iter_EIDIG, 'search iterations,', num_ids_EIDIG, 'non-duplicate individual discriminatory instances are generated. Time cost:', t2-t1, 's.')
            num_ids[index+1] += num_ids_EIDIG
            time_cost[index+1] += t2-t1
        
        print('\n')

    avg_num_ids = num_ids / num_experiment_round
    avg_speed = num_ids / time_cost
    print('Results of local phase comparsion on', benchmark, 'with l_num set to {} given {} discriminatory seeds'.format(l_num, num_seeds), ',averaged on', num_experiment_round, 'rounds:')
    print('ADF:', avg_num_ids[0], 'individual discriminatory instances are generated at a speed of', avg_speed[0], 'per second.')
    for index, update_interval in enumerate(update_interval_list):
        print('Update interval set to {}:'.format(update_interval))
        print('EIDIG:', avg_num_ids[index+1], 'individual discriminatory instances are generated at a speed of', avg_speed[index+1], 'per second.')
    
    return num_ids, time_cost


def seedwise_comparison(num_experiment_round, benchmark, X, protected_attribs, constraint, model, g_num=100, l_num=100, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    # compare the number of non-duplicate individual discriminatory instances generated in a seedwise fashion

    num_gen = np.zeros([3, g_num])
    num_ids = np.zeros([3, g_num])

    for i in range(num_experiment_round):
        round_now = i + 1
        print('--- ROUND', round_now, '---')
        clustered_data = generation_utilities.clustering(X, c_num)
        seeds = np.empty(shape=(0, len(X[0])))
        for i in range(g_num):
            new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i%c_num, fashion=fashion)
            seeds = np.append(seeds, [new_seed], axis=0)

        gen_ADF, ids_ADF = ADF.seedwise_generation(X, seeds, protected_attribs, constraint, model, l_num, max_iter, s_g, s_l, epsilon_l)
        gen_EIDIG_5, ids_EIDIG_5 = EIDIG.seedwise_generation(X, seeds, protected_attribs, constraint, model, l_num, 0.5, 5, max_iter, s_g, s_l, epsilon_l)
        gen_EIDIG_INF, ids_EIDIG_INF = EIDIG.seedwise_generation(X, seeds, protected_attribs, constraint, model, l_num, 0.5, l_num+1, max_iter, s_g, s_l, epsilon_l)
        num_gen[0] += gen_ADF
        num_ids[0] += ids_ADF
        num_gen[1] += gen_EIDIG_5
        num_ids[1] += ids_EIDIG_5
        num_gen[2] += gen_EIDIG_INF
        num_ids[2] += ids_EIDIG_INF
    
    avg_num_gen = num_gen / num_experiment_round
    avg_num_ids = num_ids / num_experiment_round
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_gen_ADF.npy', num_gen[0])
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_ids_ADF.npy', num_ids[0])
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_gen_EIDIG_5.npy', num_gen[1])
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_ids_EIDIG_5.npy', num_ids[1])
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_gen_EIDIG_INF.npy', num_gen[2])
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_ids_EIDIG_INF.npy', num_ids[2])

    print('Results averaged on', num_experiment_round, 'rounds have been saved. Results on the first 10 seeds:')
    print('ADF:')
    print('Number of generated instances:', num_gen[0, :10])
    print('Number of generated discriminatory instances:', num_ids[0, :10])
    print('EIDIG-5:')
    print('Number of generated instances:', num_gen[1, :10])
    print('Number of generated discriminatory instances:', num_ids[1, :10])
    print('EIDIG-INF:')
    print('Number of generated instances:', num_gen[2, :10])
    print('Number of generated discriminatory instances:', num_ids[2, :10])


def time_cost_comparison(num_experiment_round, benchmark, X, protected_attribs, constraint, model, record_step=100, record_frequency=100, g_num=1000, l_num=1000, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    # compare the time consumption for generating a certain number of non-duplicate individual discriminatory instances

    time_cost = np.zeros([3, record_frequency])

    for i in range(num_experiment_round):
        round_now = i + 1
        print('--- ROUND', round_now, '---')
        if len(X) <= g_num:
            seeds = X.copy()
        else:
            clustered_data = generation_utilities.clustering(X, c_num)
            seeds = np.empty(shape=(0, len(X[0])))
            for i in range(g_num):
                new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i%c_num, fashion=fashion)
                seeds = np.append(seeds, [new_seed], axis=0)

        t_ADF = ADF.time_record(X, seeds, protected_attribs, constraint, model, l_num, record_step, record_frequency, max_iter, s_g, s_l, epsilon_l)
        t_EIDIG_5 = EIDIG.time_record(X, seeds, protected_attribs, constraint, model, decay, l_num, record_step, record_frequency, 5, max_iter, s_g, s_l, epsilon_l)
        t_EIDIG_INF = EIDIG.time_record(X, seeds, protected_attribs, constraint, model, decay, l_num, record_step, record_frequency, l_num+1, max_iter, s_g, s_l, epsilon_l)
        time_cost[0] += t_ADF
        time_cost[1] += t_EIDIG_5
        time_cost[2] += t_EIDIG_INF

    avg_time_cost = time_cost / num_experiment_round
    np.save('logging_data/logging_data_from_tests/time_cost_comparison/' + benchmark + '_time_every{}ids_ADF.npy'.format(record_step), avg_time_cost[0])
    np.save('logging_data/logging_data_from_tests/time_cost_comparison/' + benchmark + '_time_every{}ids_EIDIG_5.npy'.format(record_step), avg_time_cost[1])
    np.save('logging_data/logging_data_from_tests/time_cost_comparison/' + benchmark + '_time_every{}ids_EIDIG_INF.npy'.format(record_step), avg_time_cost[2])

    print('Results averaged on', num_experiment_round, 'rounds have been saved. Results on the first 10 records:')
    print('ADF:')
    print('Time cost:', avg_time_cost[0, :10])
    print('EIDIG-5:')
    print('Time cost:', avg_time_cost[1, :10])
    print('EIDIG-INF:')
    print('Time cost:', avg_time_cost[2, :10])