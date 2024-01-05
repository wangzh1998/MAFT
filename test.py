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

parser.add_argument('--round_id', type=int, default=1, help='The id of current round')
parser.add_argument('--g_num', type=int, default=20, help='The number of seeds used in the global generation phase')
parser.add_argument('--l_num', type=int, default=20, help='The maximum search iteration in the local generation phase')
parser.add_argument('--perturbation_size', type=float, default=1.0, help='The perturbation size used in the MAFT method')
parser.add_argument('--should_restore_progress', type=lambda x: (str(x).lower() == 'true'), default=True, help='Control whether the experiment starts from scratch')

args = parser.parse_args()
round_id = args.round_id
g_num = args.g_num
l_num = args.l_num
perturbation_size = args.perturbation_size
should_restore_progress = args.should_restore_progress

# experiment results will be saved in a csv file
iter = '{}x{}_H_{}'.format(g_num, l_num, perturbation_size)
dir = 'logging_data/complete_comparison/complete_comparison_info/' + iter + '/'
if not os.path.exists(dir):
    os.makedirs(dir)
filename = dir + 'comparison_info_round_{}.csv'.format(round_id)

info = experiment_config.all_benchmark_info
all_benchmarks = [benchmark for benchmark in info.keys()]
all_methods = [method.name for method in experiment_config.Method]
all_columns = ['round_id', 'benchmark', 'method', 'num_id', 'num_all_id', 'total_iter', 'time_cost']

# check if the file exists to decide whether to skip some benchmarks
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
    num_ids, num_all_ids, total_iter, time_cost = experiments.comparison(round_id, benchmark, dataset.X_train, protected_attribs, dataset.constraint, model, g_num, l_num, perturbation_size)
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