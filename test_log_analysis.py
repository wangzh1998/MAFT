'''
分析重定向的log文件，画出图像
'''

import re
import matplotlib.pyplot as plt

# fname="./logging_data/log_v1/2022-11-14-21_24_21_comparison.txt"
# fname="./logging_data/log_v1/2022-11-15-20_40_01_comparison.txt"
# fname="./logging_data/log_v1/2022-11-15-21_33_32_comparison.txt"
# fname="./logging_data/log_v1/2022-11-16-11_37_22_comparison.txt"

# benchmarks = ['C-a', 'C-r', 'C-g', 'C-a&r', 'C-a&g', 'C-r&g', 'G-g', 'G-a', 'G-g&a', 'B-a']
# methods = ['ADF', 'EIDIG-5', 'EIDIG-INF', 'MAFT-5', 'MAFT-INF']
# selected_methods_idx = [0, 1, 3]
# selected_methods = ['ADF', 'EIDIG', 'MAFT']

fname="./logging_data/log_v1/2023-05-20-17_16_56_comparison.txt"
benchmarks = ['C-a', 'C-r', 'C-g', 'C-a&r', 'C-a&g', 'C-r&g', 'G-g', 'G-a', 'G-g&a', 'B-a']
methods = ['ADF', 'EIDIG-5', 'MAFT-5']
selected_methods_idx = [0, 1, 2]
selected_methods = ['ADF', 'EIDIG', 'MAFT']

with open(fname, 'r') as f:
    lines = f.readlines()

# for line in lines:
#     if line:
#         # nums = re.findall(r"\d+\.?\d*", line)
#         nums = [int(s) for s in line.split() if s.isdigit()]
#         print(nums)

benchmarks_idx = 0
methods_idx = 0
n = len(benchmarks)
m = len(methods)
avg_discriminatory_instances = [[0 for _ in range(n)] for _ in range(m)]
avg_discriminatory_instances_per_sec = [[0 for _ in range(n)] for _ in range(m)]
for line in lines:
    nums = re.findall(r"\d+\.\d*", line)  # 匹配所有浮点数
    if len(nums) == 2:  # 找到avg instances 和 avg time
        nums = [float(num) for num in nums]
        avg_discriminatory_instances[methods_idx][benchmarks_idx] = nums[0]
        avg_discriminatory_instances_per_sec[methods_idx][benchmarks_idx] = nums[1]
        methods_idx += 1
        if methods_idx == m:
            methods_idx = 0
            benchmarks_idx += 1
print(avg_discriminatory_instances)
print(avg_discriminatory_instances_per_sec)

plt.title('White-box comparison:number')
for i in range(m):
    if i in selected_methods_idx:
        plt.plot(benchmarks, avg_discriminatory_instances[i], )
plt.xlabel('benchmark')
plt.ylabel('instances')
plt.legend(selected_methods)
plt.savefig(fname[:-4] + '_number.png')
plt.show()
plt.close()

plt.title('White-box comparison:speed')
for i in range(m):
    if i in selected_methods_idx:
        plt.plot(benchmarks, avg_discriminatory_instances_per_sec[i])
plt.xlabel('benchmark')
plt.ylabel('time')
plt.legend(selected_methods)
plt.savefig(fname[:-4] + '_speed.png')
plt.show()
plt.close()
