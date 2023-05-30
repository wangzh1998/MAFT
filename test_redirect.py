"""
This python file is used to run test experiments.
相较于test.py: 将print输出重定向到文件中
"""

import sys
import time
import experiments
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing
from tensorflow import keras

# redirect print
standard_stdout = sys.stdout
now = time.strftime("%Y-%m-%d-%H_%M_%S_", time.localtime(time.time()))
logfile_name = "./logging_data/log_v1/"+now+r"comparison.txt"
logfile = open(logfile_name, 'w+')
sys.stdout = logfile

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
ROUND = 3 # the number of experiment rounds
g_num = 20 # the number of seeds used in the global generation phase
l_num = 20 # the maximum search iteration in the local generation phase
for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0,6]), ('C-a&g', [0,7]), ('C-r&g', [6,7])]:
    print('\n', benchmark, ':\n')
    experiments.comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model, g_num, l_num)
for benchmark, protected_attribs in [('G-g', [6]), ('G-a', [9]), ('G-g&a', [6,9])]:
    print('\n', benchmark, ':\n')
    experiments.comparison(ROUND, benchmark, pre_german_credit.X_train, protected_attribs, pre_german_credit.constraint, german_model, g_num, l_num)
print('\nB-a:\n')
experiments.comparison(ROUND, 'B-a', pre_bank_marketing.X_train, [0], pre_bank_marketing.constraint, bank_model, g_num, l_num)


# resume standard output
logfile.close()
sys.stdout = standard_stdout
