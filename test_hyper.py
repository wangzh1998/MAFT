import experiments as experiments
import numpy as np
from datetime import datetime
import experiment_config
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Experiment configuration')

parser.add_argument('--round_id', type=int, default=1, help='The id of current round')
parser.add_argument('--g_num', type=int, default=10, help='The number of seeds used in the global generation phase')
parser.add_argument('--l_num', type=int, default=10, help='The maximum search iteration in the local generation phase')
parser.add_argument('--should_restore_progress', type=lambda x: (str(x).lower() == 'true'), default=True, help='Control whether the experiment starts from scratch')

args = parser.parse_args()
round_id = args.round_id
g_num = args.g_num
l_num = args.l_num
should_restore_progress = args.should_restore_progress
ps_from = -10
ps_to = 1
perturbation_size_list = np.logspace(ps_from, ps_to, num=(ps_to - ps_from)+1, base=10.0) # 创建1e-10到1e5的等比数列

# experiment results will be saved in a csv file
iter = '{}x{}'.format(g_num, l_num)
dir = 'logging_data/hyper_comparison/hyper_comparison_info/' + iter + '/'
if not os.path.exists(dir):
    os.makedirs(dir)
filename = dir + 'hyper_comparison_info_round_{}.csv'.format(round_id)

info = experiment_config.all_benchmark_info

all_benchmarks = [benchmark for benchmark in info.keys()]
all_methods = ['AEQUITAS', 'SG', 'ADF', 'EIDIG'] + ['MAFT_{}'.format(ps) for ps in perturbation_size_list] # 和experiment_config.py中的AllMethod保持一致
all_columns = ['round_id', 'benchmark', 'method', 'num_id', 'num_all_id', 'total_iter', 'time_cost']

# check if the file exists to decide whether we need to skip some benchmarks
if should_restore_progress and os.path.exists(filename):
    # if the file exists, read the completed benchmarks data
    existing_data = pd.read_csv(filename)
    completed_benchmarks = existing_data['benchmark'].unique()
    benchmarks_to_run = [b for b in all_benchmarks if b not in completed_benchmarks]
else:
    # if the file does not exist, we need to run all benchmarks
    completed_benchmarks = []
    benchmarks_to_run = all_benchmarks
    # create an empty DataFrame with columns matching the data returned by run_experiments function
    existing_data = pd.DataFrame(columns=all_columns)

for benchmark in completed_benchmarks:
    print(datetime.now())
    print('Skipping benchmark {}'.format(benchmark))
for benchmark in benchmarks_to_run:
    print('\n', benchmark, ':\n')
    print(datetime.now())
    model, dataset, protected_attribs = info[benchmark]
    num_ids, num_all_ids, total_iter, time_cost = experiments.hyper_comparison(round_id, benchmark, dataset.X_train, protected_attribs, dataset.constraint, model,
                                                                               perturbation_size_list, dataset.initial_input, dataset.configurations,
                                                                               g_num, l_num)
    # construct a dictionary for each round/benchmark/method
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
        # append to existing data
        existing_data = existing_data.append(data_to_append, ignore_index=True)
    existing_data.to_csv(filename, index=False)
print(existing_data)
