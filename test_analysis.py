# todo 这个版本跑通了，可以用

import json
import numpy as np
import matplotlib.pyplot as plt

g_num = 100 # the number of seeds used in the global generation phase
l_num = 100 # the maximum search iteration in the local generation phase
perturbation_size = 1 # the perturbation size used in the compute_gradient function
iter = '{}x{}_H_{}'.format(g_num, l_num, perturbation_size)
dir = 'logging_data/logging_data_from_tests/complete_comparison_info/' + iter + '/'
filename = 'progress_data.json'

# 从JSON文件加载数据
with open(dir + filename, 'r') as file:
    data = json.load(file)

# Initialize lists to store data
methods = ['ADF', 'EIDIG', 'MAFT']  # Assuming the methods have this order in the JSON data
benchmark_data_list = data["data"]
benchmarks = [benchmark_data["benchmark"] for benchmark_data in benchmark_data_list]

avg_speeds_tot = np.zeros(shape=(len(methods), len(benchmarks)))
avg_nums_tot = np.zeros(shape=(len(methods), len(benchmarks)))

std_dev_speeds_tot = np.zeros(shape=(len(methods), len(benchmarks)))
std_dev_nums_tot = np.zeros(shape=(len(methods), len(benchmarks)))


# Process the data
for benchmark_index, benchmark_data in enumerate(benchmark_data_list):
    nums_id = np.array(benchmark_data["data"]["nums_id"]) # 3 * round_number
    time_cost = np.array(benchmark_data["data"]["time_cost"]) # 3 * round_number

    avg_nums_id = np.mean(nums_id, axis=1)
    avg_speed = np.mean(nums_id / time_cost, axis=1) # 先求每一轮的速度，再算不同轮数的速度均值 # 3 * 1 # （和之前的逻辑不一致，改成这样是为了方便计算标准差）
    # avg_speed = np.sum(nums_id, axis=1) / np.sum(time_cost, axis=1) # 用不同轮数的总实例数/总时间作为速度均值（和之前的逻辑保持一致）# 3 * 1

    std_dev_nums_id = np.std(nums_id, axis=1)
    std_dev_speed = np.std(nums_id / time_cost, axis=1)

    for method_index, method in enumerate(methods):
        avg_nums_tot[method_index][benchmark_index] = avg_nums_id[method_index]
        avg_speeds_tot[method_index][benchmark_index] = avg_speed[method_index]
        std_dev_nums_tot[method_index][benchmark_index] = std_dev_nums_id[method_index]
        std_dev_speeds_tot[method_index][benchmark_index] = std_dev_speed[method_index]

# # 对数据取对数
# avg_nums_tot = np.log2(avg_nums_tot)
# avg_speeds_tot = np.log2(avg_speeds_tot)
# # 对标准差取对数前，先加一个较小的常数，避免出现0
# # std_dev_nums_tot = np.log2(std_dev_nums_tot + 1e-10)
# # std_dev_speeds_tot = np.log2(std_dev_speeds_tot + 1e-10)
# std_dev_nums_tot = np.log2(std_dev_nums_tot)
# std_dev_speeds_tot = np.log2(std_dev_speeds_tot)

# 创建子图
fig, ax = plt.subplots(2, 1, figsize=(15, 8))

# 绘制实例数量的折线图
for i in range(len(methods)):
    ax[0].plot(benchmarks, avg_nums_tot[i], marker='o', label=methods[i])
    ax[0].fill_between(benchmarks, avg_nums_tot[i] - std_dev_nums_tot[i], avg_nums_tot[i] + std_dev_nums_tot[i], alpha=0.2)
# ax[0].set_title('Instance Numbers (log10)')
ax[0].set_title('Instance Numbers')
ax[0].legend()

# 绘制生成速度的折线图
for i in range(len(methods)):
    ax[1].plot(benchmarks, avg_speeds_tot[i], marker='o', label=methods[i])
    ax[1].fill_between(benchmarks, avg_speeds_tot[i] - std_dev_speeds_tot[i], avg_speeds_tot[i] + std_dev_speeds_tot[i], alpha=0.2)
# ax[1].set_title('Generation Speeds (log2)')
ax[1].set_title('Generation Speeds')
ax[1].legend()

# 显示图形
plt.tight_layout()
plt.show()
