"""
This python file is used to run test experiments.
"""


import experiments
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing
from tensorflow import keras


"""
for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
for german credit data, gender(6) and age(9) are protected attributes in 24 features
for bank marketing data, age(0) is protected attribute in 16 features
"""


# load models
adult_model = keras.models.load_model("models/original_models/adult_model.h5")
german_model = keras.models.load_model("models/original_models/german_model.h5")
bank_model = keras.models.load_model("models/original_models/bank_model.h5")

# save experiments info
experiments_data = []

# test the implementation of ADF, EIDIG-5, EIDIG-INF
# the individual discriminatory instances generated are saved to 'logging_data/logging_data_from_tests/complete_comparison'
ROUND = 1 # the number of experiment rounds
g_num = 1000 # the number of seeds used in the global generation phase
l_num = 1000 # the maximum search iteration in the local generation phase
perturbation_size = 1 # the perturbation size used in the compute_gradient function
for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0,6]), ('C-a&g', [0,7]), ('C-r&g', [6,7])]:
    print('\n', benchmark, ':\n')
    method_num_speed_result = experiments.comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model, g_num, l_num, perturbation_size)
    experiments_data.append([benchmark, method_num_speed_result])
for benchmark, protected_attribs in [('G-g', [6]), ('G-a', [9]), ('G-g&a', [6,9])]:
    print('\n', benchmark, ':\n')
    method_num_speed_result = experiments.comparison(ROUND, benchmark, pre_german_credit.X_train, protected_attribs, pre_german_credit.constraint, german_model, g_num, l_num, perturbation_size)
    experiments_data.append([benchmark, method_num_speed_result])
print('\nB-a:\n')
method_num_speed_result = experiments.comparison(ROUND, 'B-a', pre_bank_marketing.X_train, [0], pre_bank_marketing.constraint, bank_model, g_num, l_num, perturbation_size)
experiments_data.append([benchmark, method_num_speed_result])


'''
保存数据
'''
import pickle
import os

dir = 'logging_data/logging_data_from_tests/complete_comparison_info/'
iter = '{}x{}_H_{}'.format(g_num, l_num, perturbation_size)
if not os.path.exists(dir):
    os.makedirs(dir)
with open(dir + iter + 'complete_comparison_info.pkl', 'wb') as f:
    pickle.dump(experiments_data, f)

'''
读取数据
'''
with open(dir + iter + 'complete_comparison_info.pkl', 'rb') as f:
    loaded_experiments_data = pickle.load(f)



'''
画图
'''
import pandas as pd
import numpy as np

df = pd.DataFrame(columns=['benchmark', 'method', 'avg_num', 'avg_speed'])

for benchmark_data in loaded_experiments_data:
    benchmark = benchmark_data[0]
    for method_data in benchmark_data[1]:
        method = method_data[0]
        avg_num = method_data[1]
        avg_speed = method_data[2]
        df = df.append({'benchmark': benchmark, 'method': method, 'avg_num': avg_num, 'avg_speed': avg_speed}, ignore_index=True)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2)

# num图
for method in df['method'].unique():
    df_method = df[df['method'] == method]
    # axs[0].plot(df_method['benchmark'], df_method['avg_num'], marker='o', label=method)
    log_avg_num = np.log10(df_method['avg_num'])  # 使用numpy的log10函数计算以10为底的对数
    axs[0].plot(df_method['benchmark'], log_avg_num, marker='o', label=method)

axs[0].set_title('Log Avg Num (base 10)')
axs[0].set_xlabel('Benchmark')  # 设置x轴标签
axs[0].set_ylabel('Log Avg Num (base 10)')  # 设置y轴标签
axs[0].legend()

# speed图
for method in df['method'].unique():
    df_method = df[df['method'] == method]
    # axs[1].plot(df_method['benchmark'], df_method['avg_speed'], marker='o', label=method)
    log_avg_speed = np.log2(df_method['avg_speed'])  # 使用numpy的log2函数计算以2为底的对数
    axs[1].plot(df_method['benchmark'], log_avg_speed, marker='o', label=method)

axs[1].set_title('Log Avg Speed (base 2)')
axs[1].set_xlabel('Benchmark')  # 设置x轴标签
axs[1].set_ylabel('Log Avg Speed (base 2)')  # 设置y轴标签
axs[1].legend()

plt.tight_layout()
plt.savefig(dir + iter + 'complete_comparison_result.png')
plt.show()