"""
This python file is used to run test experiments.
"""


import experiments
from preprocessing import pre_census_income, pre_german_credit, pre_bank_marketing
# pre_meps_15, pre_heart_heath, pre_diabetes, pre_students
from tensorflow import keras
import json
import os
import time
from collections import OrderedDict
from datetime import datetime

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
# meps15_model = keras.models.load_model("models/original_models/meps15_model.h5")
# heart_model = keras.models.load_model("models/original_models/heart_model.h5")
# diabetes_model = keras.models.load_model("models/original_models/diabetes_model.h5")
# students_model = keras.models.load_model("models/original_models/students_model.h5")

# save experiments info
# experiments_data = []

# test the implementation of ADF, EIDIG-5, MAFT-5
ROUND = 1 # the number of experiment rounds
g_num = 1000 # the number of seeds used in the global generation phase
l_num = 100 # the maximum search iteration in the local generation phase
perturbation_size = 1 # the perturbation size used in the compute_gradient function
should_restore_progress = False # control whether the experiment starts from scratch

info = OrderedDict({
    'C-a': (adult_model, pre_census_income, [0], [7, 4, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]),
    'C-r': (adult_model, pre_census_income, [6], [7, 4, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]),
    'C-g': (adult_model, pre_census_income, [7], [7, 4, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]),
    # 'C-a&r': (adult_model, pre_census_income, [0, 6], [7, 4, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]),
    # 'C-a&g': (adult_model, pre_census_income, [0, 7], [7, 4, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]),
    # 'C-r&g': (adult_model, pre_census_income, [6, 7], [7, 4, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]),
    'G-g': (german_model, pre_german_credit, [6], [1, 2, 24, 1, 60, 1, 3, 2, 2, 1, 22, 3, 2, 2, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
    'G-a': (german_model, pre_german_credit, [9], [1, 2, 24, 1, 60, 1, 3, 2, 2, 1, 22, 3, 2, 2, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
    # 'G-g&a': (german_model, pre_german_credit, [6, 9], [1, 2, 24, 1, 60, 1, 3, 2, 2, 1, 22, 3, 2, 2, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
    'B-a': (bank_model, pre_bank_marketing, [0], [3, 11, 2, 0, 0, 5, 1, 0, 0, 5, 4, 40, 1, 1, 0, 0]),
    # 'M-a': (meps15_model, pre_meps_15, [0]),
    # 'M-r': (meps15_model, pre_meps_15, [1]),
    # 'M-g': (meps15_model, pre_meps_15, [9]),
    # 'M-a&r': (meps15_model, pre_meps_15, [0, 1]),
    # 'M-a&g': (meps15_model, pre_meps_15, [0, 9]),
    # 'M-r&g': (meps15_model, pre_meps_15, [1, 9]),
    # 'H-a': (heart_model, pre_heart_heath, [0]),
    # 'H-g': (heart_model, pre_heart_heath, [1]),
    # 'H-a&g': (heart_model, pre_heart_heath, [0, 1]),
    # 'D-a': (diabetes_model, pre_diabetes, [7]),
    # 'S-a': (students_model, pre_students, [2]),
    # 'S-g': (students_model, pre_students, [1]),
    # 'S-a&g': (students_model, pre_students, [2, 1])
})

progress_data = {
    'last_completed_benchmark': None,
    'data': [{'benchmark': None, 'data': {'nums_id': [[]], 'time_cost': [[]]}}]
}
progress_data['data'] = [item for item in progress_data['data'] if item['benchmark'] is not None]

# the individual discriminatory instances generated are saved to 'logging_data/logging_data_from_tests/complete_comparison_instances_bb/
iter = '{}x{}_H_{}'.format(g_num, l_num, perturbation_size)
dir = 'logging_data/logging_data_from_tests/complete_comparison_info_bb/' + iter + '/'
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

for benchmark_index, (benchmark, (model, dataset, protected_attribs, initial_input)) in enumerate(info.items()):
    print('\n', benchmark, ':\n')
    if benchmark_index <= last_benchmark_index:
        print('skip')
        continue
    print(datetime.now())
    num_ids, time_cost = experiments.comparison_blackbox(ROUND, benchmark, dataset.X_train, protected_attribs,dataset.constraint, model, g_num, l_num, perturbation_size, initial_input)
    progress_data['last_completed_benchmark'] = benchmark
    progress_data['data'].append({'benchmark': benchmark, 'data': {'nums_id': num_ids.tolist(), 'time_cost': time_cost.tolist()}})
    save_progress_data(dir, filename, progress_data)
    # print('sleep ' + str(100) + 's')
    # time.sleep(100)
    # print('wake up')