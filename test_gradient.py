"""
This python file is used to run test experiments.
"""


import experiments
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing
from tensorflow import keras
import numpy as np
import os


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
g_num = 10 # the number of seeds used in the global generation phase
# todo 是否需要改成1e-4
perturbation_size = 1 # the perturbation size used in the compute_gradient function
# results of experiments to save
results = []

benchmarks_list = [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0, 6]), ('C-a&g', [0, 7]), ('C-r&g', [6, 7]),
                   ('G-g', [6]), ('G-a', [9]), ('G-g&a', [6, 9]), ('B-a', [])]

for benchmark, protected_attribs in benchmarks_list:
    print('\n', benchmark, ':\n')
    if 'G' in benchmark:
        data = pre_german_credit.X_train
        model = german_model
    elif 'B' in benchmark:
        data = pre_bank_marketing.X_train
        model = bank_model
    else:
        data = pre_census_income.X_train
        model = adult_model

    eidig_gradients, maft_gradients, eidig_time_cost, maft_time_cost = experiments.gradient_comparison(benchmark, data,
                                                                                                       model, g_num,
                                                                                                       perturbation_size)
    result = [benchmark, eidig_gradients, maft_gradients, eidig_time_cost, maft_time_cost]
    results.append(result)

# Convert list to ndarray
results = np.array(results, dtype=object)

'''
保存数据
'''

dir = 'logging_data/gradients_comparison/'
iter = 'Seeds_{}_H_{}_'.format(g_num, perturbation_size)
if not os.path.exists(dir):
    os.makedirs(dir)

'''
保存数据
'''
np.save(dir + iter + 'experiment_results.npy', results)

'''
读取数据 暂时不写
'''
# data = np.load('dir + iter + experiment_results.npy', allow_pickle=True)


'''
画图
'''
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 提取benchmark名称、EIDIG梯度、MAFT梯度
benchmark_names = results[:, 0]
eidig_grads = results[:, 1]
maft_grads = results[:, 2]

# 初始化列表来保存每个benchmark的所有实例的相似度
all_sims = []

# 初始化列表来保存每个benchmark的平均相似度
avg_sims = []

# 遍历所有的benchmarks
for i in range(len(benchmark_names)):
    # 提取出当前benchmark的EIDIG梯度和MAFT梯度
    eidig_grad = eidig_grads[i]
    maft_grad = maft_grads[i]

    # 初始化列表来保存当前benchmark的所有实例的相似度
    sims = []

    # 遍历所有的实例
    for j in range(g_num):
        # 计算当前实例的EIDIG梯度和MAFT梯度之间的cosine相似度
        sim = cosine_similarity(eidig_grad[j].reshape(1, -1), maft_grad[j].reshape(1, -1))

        # 将相似度添加到列表中
        sims.append(sim[0][0])

    # 将当前benchmark的所有实例的相似度添加到总列表中
    all_sims.append(sims)

    # 计算当前benchmark的平均相似度，并将其添加到平均相似度列表中
    avg_sims.append(np.mean(sims))

# 用条形图显示每个benchmark的平均cosine相似度
plt.figure(figsize=(10, 5))
plt.bar(benchmark_names, avg_sims)
plt.xlabel('Benchmark')
plt.ylabel('Average Cosine Similarity')
plt.title('Average Cosine Similarity for Each Benchmark')
plt.savefig('logging_data/gradients_comparison/' + iter + 'average_cosine_similarity.png')
plt.show()


# 要单独可视化每个benchmark的每个实例的相似度，可以为每个benchmark创建一个箱线图或小提琴图。这可以显示出每个benchmark中实例的相似度的分布。下面是一个使用箱线图的例子

import seaborn as sns

plt.figure(figsize=(15, 10))

# 使用seaborn的箱线图函数，将每个benchmark的所有实例的相似度进行可视化
sns.boxplot(data=all_sims)

# 设置x轴的标签为benchmark的名字
plt.xticks(range(len(benchmark_names)), benchmark_names)

plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity for Each Instance in Each Benchmark')
plt.savefig('logging_data/gradients_comparison/' + iter + 'cosine_similarity_box.png')
plt.show()


# 此图与之前的箱线图类似，但还包括了每个benchmark实例的相似度的密度分布，从而更直观地显示出数据的分布。
import seaborn as sns
import pandas as pd

# 将数据转换为DataFrame格式，以便在seaborn中使用
sim_data = pd.DataFrame(all_sims, index=benchmark_names).T

plt.figure(figsize=(15, 10))

# 使用seaborn的小提琴图函数，将每个benchmark的所有实例的相似度进行可视化
sns.violinplot(data=sim_data)

plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity for Each Instance in Each Benchmark')
plt.savefig('logging_data/gradients_comparison/' + iter + 'cosine_similarity_violin.png')
plt.show()



# 画时间对比直方图
# 提取EIDIG和MAFT的时间开销
eidig_time = results[:, 3]
maft_time = results[:, 4]

x = np.arange(len(benchmark_names))  # 设定x轴坐标

plt.figure(figsize=(15, 10))

# 使用条形图可视化EIDIG和MAFT的时间开销
plt.bar(x - 0.2, eidig_time, 0.4, label='EIDIG')
plt.bar(x + 0.2, maft_time, 0.4, label='MAFT')

# 设置x轴的标签为benchmark的名字
plt.xticks(x, benchmark_names)

plt.ylabel('Time (s)')
plt.title('Time Cost for Each Benchmark')
plt.legend()
plt.savefig('logging_data/gradients_comparison/' + iter + 'time_cost.png')
plt.show()
