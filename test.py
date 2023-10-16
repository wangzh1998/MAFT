"""
This python file is used to run test experiments.
"""


import experiments
from preprocessing import pre_census_income, pre_german_credit, pre_bank_marketing
from tensorflow import keras
import json
import os
import time
from collections import OrderedDict


def save_progress_data(dir, filename, data):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_progress_data(dir, filename):
    if not os.path.exists(dir):
        return None
    try:
        with open(dir + filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


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
# experiments_data = []

# test the implementation of ADF, EIDIG-5, MAFT-5
ROUND = 3 # the number of experiment rounds
g_num = 1000 # the number of seeds used in the global generation phase
l_num = 1000 # the maximum search iteration in the local generation phase
perturbation_size = 1 # the perturbation size used in the compute_gradient function
should_restore_progress = True # control whether the experiment starts from scratch

info = OrderedDict({
    'C-a': (adult_model, pre_census_income, [0]),
    'C-r': (adult_model, pre_census_income, [6]),
    'C-g': (adult_model, pre_census_income, [7]),
    'C-a&r': (adult_model, pre_census_income, [0, 6]),
    'C-a&g': (adult_model, pre_census_income, [0, 7]),
    'C-r&g': (adult_model, pre_census_income, [6, 7]),
    'G-g': (german_model, pre_german_credit, [6]),
    'G-a': (german_model, pre_german_credit, [9]),
    'G-g&a': (german_model, pre_german_credit, [6, 9]),
    'B-a': (bank_model, pre_bank_marketing, [0]),
})

progress_data = {
    'last_completed_benchmark': None,
    'data': [{'benchmark': None, 'data': {'nums_id': [[]], 'time_cost': [[]]}}]
}
progress_data['data'] = [item for item in progress_data['data'] if item['benchmark'] is not None]

# the individual discriminatory instances generated are saved to 'logging_data/logging_data_from_tests/complete_comparison'
iter = '{}x{}_H_{}'.format(g_num, l_num, perturbation_size)
dir = 'logging_data/logging_data_from_tests/complete_comparison_info/' + iter + '/'
filename = 'progress_data.json'
loaded_progress_data = load_progress_data(dir, filename)
if should_restore_progress and loaded_progress_data is not None: # restore progress
    progress_data = loaded_progress_data
    last_completed_benchmark = loaded_progress_data['last_completed_benchmark']
    last_benchmark_index = list(info.keys()).index(last_completed_benchmark)
else:
    # progress_data = None
    last_completed_benchmark = ''
    last_benchmark_index = -1

for benchmark_index, (benchmark, (model, dataset, protected_attribs)) in enumerate(info.items()):
    print('\n', benchmark, ':\n')
    if benchmark_index <= last_benchmark_index:
        print('skip')
        continue
    num_ids, time_cost = experiments.comparison(ROUND, benchmark, dataset.X_train, protected_attribs,dataset.constraint, model, g_num, l_num, perturbation_size)
    progress_data['last_completed_benchmark'] = benchmark
    progress_data['data'].append({'benchmark': benchmark, 'data': {'nums_id': num_ids.tolist(), 'time_cost': time_cost.tolist()}})
    save_progress_data(dir, filename, progress_data)
    print('sleep ' + str(100) + 's')
    time.sleep(100)