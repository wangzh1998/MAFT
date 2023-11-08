"""
This python file is used to run test experiments.
"""

import os
from datetime import datetime
import argparse
import pandas as pd
import experiments
import experiment_config

parser = argparse.ArgumentParser(description='Experiment configuration')
# 添加参数
parser.add_argument('--round_id', type=int, default=1, help='The id of current round')
parser.add_argument('--g_num', type=int, default=10, help='The number of seeds used in the global generation phase')
parser.add_argument('--l_num', type=int, default=10, help='The maximum search iteration in the local generation phase')
parser.add_argument('--perturbation_size', type=float, default=1, help='The perturbation size used in the MAFT method')
parser.add_argument('--should_restore_progress', type=lambda x: (str(x).lower() == 'true'), default=True, help='Control whether the experiment starts from scratch')
# 解析参数
args = parser.parse_args()
# 使用参数
round_id = args.round_id
g_num = args.g_num
l_num = args.l_num
perturbation_size = args.perturbation_size
should_restore_progress = args.should_restore_progress

# the individual discriminatory instances generated are saved to 'logging_data/logging_data_from_tests/complete_comparison_instances'
iter = '{}x{}_H_{}'.format(g_num, l_num, perturbation_size)
dir = 'logging_data/logging_data_from_tests/complete_comparison_info/' + iter + '/'
if not os.path.exists(dir):
    os.makedirs(dir)
filename = dir + 'comparison_info_round_{}.csv'.format(round_id)

info = experiment_config.all_benchmark_info
all_benchmarks = [benchmark for benchmark in info.keys()]
all_methods = [method.name for method in experiment_config.Method]
all_columns = ['round_id', 'benchmark', 'method', 'num_id', 'num_all_id', 'total_iter', 'time_cost']

# 检查文件是否存在来决定是否需要跳过一些benchmarks
if should_restore_progress and os.path.exists(filename):
    # 如果文件存在，读取已经完成的benchmarks
    existing_data = pd.read_csv(filename)
    completed_benchmarks = existing_data['benchmark'].unique()
    benchmarks_to_run = [b for b in all_benchmarks if b not in completed_benchmarks]
else:
    # 如果文件不存在，我们需要运行所有的benchmarks
    completed_benchmarks = []
    benchmarks_to_run = all_benchmarks
    # 创建一个空的DataFrame，列与run_experiments函数返回的数据相匹配
    existing_data = pd.DataFrame(columns=all_columns)

# todo perturbation_size用列表，从超参实验中按照benchmark选最优值
# todo 返回T1,T1k数据
for benchmark in completed_benchmarks:
    print(datetime.now())
    print('Skipping benchmark {}'.format(benchmark))
for benchmark in benchmarks_to_run:
    print('\n', benchmark, ':\n')
    print(datetime.now())
    model, dataset, protected_attribs = info[benchmark]
    num_ids, num_all_ids, total_iter, time_cost = experiments.comparison(round_id, benchmark, dataset.X_train, protected_attribs, dataset.constraint, model, g_num, l_num, perturbation_size)
    # 为每个round/benchmark/method构建数据字典
    for method_idx, method in enumerate(all_methods):
        data_to_append = {
            all_columns[0]: round_id,
            all_columns[1]: benchmark,
            all_columns[2]: method,
            all_columns[3]: num_ids[method_idx],
            all_columns[4]: num_all_ids[method_idx],
            all_columns[5]: total_iter[method_idx],
            all_columns[6]: time_cost[method_idx]
        }
        # 追加到现有数据中
        existing_data = existing_data.append(data_to_append, ignore_index=True)
    existing_data.to_csv(filename, index=False)