"""
This python file is used to run test experiments.
"""


import experiments
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing
from tensorflow import keras
import numpy as np


"""
for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
for german credit data, gender(6) and age(9) are protected attributes in 24 features
for bank marketing data, age(0) is protected attribute in 16 features
"""


# load models
adult_model = keras.models.load_model("models/original_models/adult_model.h5")
german_model = keras.models.load_model("models/original_models/german_model.h5")
bank_model = keras.models.load_model("models/original_models/bank_model.h5")


# test the implementation of ADF, EIDIG-5, EIDIG-INF
# the individual discriminatory instances generated are saved to 'logging_data/logging_data_from_tests/complete_comparison'
# todo 修改为100x100
ROUND = 3 # the number of experiment rounds
g_num = 100 # the number of seeds used in the global generation phase
l_num = 100 # the maximum search iteration in the local generation phase
perturbation_size_list = np.logspace(-10, 1, num=12, base=10.0) # 创建1e-10到10的12个数的等比数列

# save experiments info
# experiments_data = []
sum_num_ids = np.array([0] * (len(perturbation_size_list)+2))
sum_num_iter = np.array([0] * (len(perturbation_size_list)+2))
sum_time_cost = np.array([0] * (len(perturbation_size_list)+2))

# for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0,6]), ('C-a&g', [0,7]), ('C-r&g', [6,7])]:
#     print('\n', benchmark, ':\n')
#     method_num_speed_result = experiments.hyper_comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model, perturbation_size_list, g_num, l_num)
#     experiments_data.append([benchmark, method_num_speed_result])
# for benchmark, protected_attribs in [('G-g', [6]), ('G-a', [9]), ('G-g&a', [6,9])]:
#     print('\n', benchmark, ':\n')
#     method_num_speed_result = experiments.hyper_comparison(ROUND, benchmark, pre_german_credit.X_train, protected_attribs, pre_german_credit.constraint, german_model, perturbation_size_list, g_num, l_num)
#     experiments_data.append([benchmark, method_num_speed_result])
# print('\nB-a:\n')
# method_num_speed_result = experiments.hyper_comparison(ROUND, 'B-a', pre_bank_marketing.X_train, [0], pre_bank_marketing.constraint, bank_model, perturbation_size_list, g_num, l_num)
# experiments_data.append([benchmark, method_num_speed_result])

for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0,6]), ('C-a&g', [0,7]), ('C-r&g', [6,7])]:
    print('\n', benchmark, ':\n')
    num_ids, num_iter, time_cost = experiments.hyper_comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model, perturbation_size_list, g_num, l_num)
    sum_num_ids += num_ids
    sum_num_iter += num_iter
    sum_time_cost += time_cost
for benchmark, protected_attribs in [('G-g', [6]), ('G-a', [9]), ('G-g&a', [6,9])]:
    print('\n', benchmark, ':\n')
    num_ids, num_iter, time_cost = experiments.hyper_comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model, perturbation_size_list, g_num, l_num)
    sum_num_ids += num_ids
    sum_num_iter += num_iter
    sum_time_cost += time_cost
print('\nB-a:\n')
num_ids, num_iter, time_cost = experiments.hyper_comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model, perturbation_size_list, g_num, l_num)

sum_num_ids += num_ids
sum_num_iter += num_iter
sum_time_cost += time_cost

avg_num_ids = sum_num_ids / ROUND
avg_iter = sum_num_iter / ROUND / (7*(g_num * l_num)+3*(g_num * l_num))
avg_speed = sum_num_ids / sum_time_cost

print('Results of global phase comparsion, averaged on', ROUND, 'rounds:')
print('ADF:', avg_num_ids[0], 'individual discriminatory instances are generated at a speed of', avg_speed[0], 'per second.')
print('EIDIG:', avg_num_ids[1], 'individual discriminatory instances are generated at a speed of', avg_speed[1], 'per second.')
for index, perturbation_size in enumerate(perturbation_size_list):
    # print('Perturbation_size set to {}:'.format(perturbation_size))
    print('MAFT_{}'.format(perturbation_size), ':', avg_num_ids[index + 2], 'individual discriminatory instances are generated at a speed of',
          avg_speed[index], 'per second.')



import os
dir = 'logging_data/hyper_comparison/hyper_comparison_info/'
iter = '{}x{}_'.format(g_num, l_num)
if not os.path.exists(dir):
    os.makedirs(dir)

'''
保存数据
'''
np.save(dir + iter + 'hyper_comparison_avg_num_ids.npy', avg_num_ids)
np.save(dir + iter + 'hyper_comparison_avg_iter.npy', avg_iter)
np.save(dir + iter + 'hyper_comparison_avg_speed.npy', avg_speed)

'''
读取数据
'''
avg_num_ids = np.load(dir + iter + 'hyper_comparison_avg_num_ids.npy')
avg_iter = np.load(dir + iter + 'hyper_comparison_avg_iter.npy')
avg_speed = np.load(dir + iter + 'hyper_comparison_avg_speed.npy')

'''
画图
'''

import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, figsize=(20, 10))

perturbation_size_list = np.logspace(-10, 1, num=12, base=10.0) # 创建1e-10到10的12个数的等比数列
methods = ['ADF', 'EIDIG']
for perturbation_size in perturbation_size_list:
    methods.append('MAFT-{}'.format(perturbation_size))

# num图
axs[0].plot(methods, avg_num_ids, marker='o')
axs[0].set_title('Avg Num')
axs[0].legend()

# speed图
axs[1].plot(methods, avg_speed, marker='o')
axs[1].set_title('Avg Speed')
axs[1].legend()

# iter图
axs[2].plot(methods, avg_iter, marker='o')
axs[2].set_title('Avg Iter')
axs[2].legend()

plt.tight_layout()
plt.savefig(dir + iter + 'hyper_comparison_result.png')
plt.show()