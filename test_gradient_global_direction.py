"""
This python file is used to run test experiments.
"""


import experiments
import numpy as np
import os
import experiment_config

info = experiment_config.all_benchmark_info
all_benchmarks = [benchmark for benchmark in info.keys()]

g_num = 1000 # the number of seeds used in the global generation phase
perturbation_size = 1 # the perturbation size used in the compute_gradient function
# results of experiments to save
results = []

for benchmark in all_benchmarks:
    print('\n', benchmark, ':\n')
    model, dataset, protected_attribs = info[benchmark]
    data = dataset.X_train
    constraint = dataset.constraint
    eidig_directions, maft_directions, eidig_time_cost, maft_time_cost = experiments.gradient_comparison_global_direction(
        benchmark, data, protected_attribs, constraint,
        model, g_num,
        perturbation_size)
    result = [benchmark, eidig_directions, maft_directions, eidig_time_cost, maft_time_cost]
    results.append(result)

# Convert list to ndarray
results = np.array(results, dtype=object)

'''
保存数据
'''

dir = 'logging_data/directions_comparison/'
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
# data = np.load(dir + iter + 'experiment_results.npy', allow_pickle=True)
# results = data


'''
画图
'''
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 提取benchmark名称、EIDIG梯度、MAFT梯度
benchmark_names = results[:, 0]
eidig_directions = results[:, 1]
maft_directions = results[:, 2]

# 初始化列表来保存每个benchmark的所有实例的相似度
all_sims = []

# 初始化列表来保存每个benchmark的平均相似度
avg_sims = []

# 遍历所有的benchmarks
for i in range(len(benchmark_names)):
    # 提取出当前benchmark的EIDIG梯度和MAFT梯度方向
    eidig_direction = eidig_directions[i]
    maft_direction = maft_directions[i]

    # 初始化列表来保存当前benchmark的所有实例的相似度
    sims = []

    # 如果实际设置的g_num大于当前benchmark的实例数，则将g_num设置为当前benchmark的实例数（有一个数据集只有600条数据）
    if g_num >= len(eidig_direction):
        g_num = len(eidig_direction)

    # 遍历所有的实例
    for j in range(g_num):
        # 计算当前实例的EIDIG梯度和MAFT梯度之间的cosine相似度
        sim = cosine_similarity(eidig_direction[j].reshape(1, -1), maft_direction[j].reshape(1, -1))

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
plt.savefig('logging_data/directions_comparison/' + iter + 'average_cosine_similarity.png')
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
plt.savefig('logging_data/directions_comparison/' + iter + 'cosine_similarity_box.png')
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
plt.savefig('logging_data/directions_comparison/' + iter + 'cosine_similarity_violin.png')
plt.show()



# 画时间对比直方图
# 提取EIDIG和MAFT的时间开销
eidig_time = results[:, 3]
maft_time = results[:, 4]

x = np.arange(len(benchmark_names))  # 设定x轴坐标

plt.figure(figsize=(15, 10))

# 使用条形图可视化ADF、EIDIG和MAFT的时间开销
plt.bar(x - 0.15, eidig_time, 0.3, label='EIDIG')
plt.bar(x + 0.15, maft_time, 0.3, label='MAFT')

# 设置x轴的标签为benchmark的名字
plt.xticks(x, benchmark_names)

plt.ylabel('Time (s)')
plt.title('Time Cost for Each Benchmark')
plt.legend()
plt.savefig('logging_data/directions_comparison/' + iter + 'time_cost.png')
plt.show()
