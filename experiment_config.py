from tensorflow import keras
from preprocessing import pre_census_income, pre_german_credit, pre_bank_marketing, pre_meps_15, pre_heart_heath, pre_diabetes, pre_students
from collections import OrderedDict
from enum import Enum

class AllMethod(Enum):
    AEQUITAS = 0
    SG = 1
    ADF = 2
    EIDIG = 3
    MAFT = 4

class Method(Enum):
    ADF = 0
    EIDIG = 1
    # MAFT = 2

class BlackboxMethod(Enum):
    AEQUITAS = 0
    MAFT = 1
    # SG =

# load models
adult_model = keras.models.load_model("models/original_models/adult_model.h5")
german_model = keras.models.load_model("models/original_models/german_model.h5")
bank_model = keras.models.load_model("models/original_models/bank_model.h5")
meps15_model = keras.models.load_model("models/original_models/meps15_model.h5")
heart_model = keras.models.load_model("models/original_models/heart_model.h5")
diabetes_model = keras.models.load_model("models/original_models/diabetes_model.h5")
students_model = keras.models.load_model("models/original_models/students_model.h5")
all_models = [adult_model, german_model, bank_model, meps15_model, heart_model, diabetes_model, students_model]

all_benchmark_info = OrderedDict({
    'C-a': (adult_model, pre_census_income, [0]),
    'C-r': (adult_model, pre_census_income, [6]),
    'C-g': (adult_model, pre_census_income, [7]),
    'G-g': (german_model, pre_german_credit, [6]),
    'G-a': (german_model, pre_german_credit, [9]),
    'B-a': (bank_model, pre_bank_marketing, [0]),
    'M-a': (meps15_model, pre_meps_15, [0]),
    'M-r': (meps15_model, pre_meps_15, [1]),
    'M-g': (meps15_model, pre_meps_15, [9]),
    'H-a': (heart_model, pre_heart_heath, [0]),
    'H-g': (heart_model, pre_heart_heath, [1]),
    'D-a': (diabetes_model, pre_diabetes, [7]),
    'S-a': (students_model, pre_students, [2]),
    'S-g': (students_model, pre_students, [1]),
    # 'C-a&r': (adult_model, pre_census_income, [0, 6]),
    # 'C-a&g': (adult_model, pre_census_income, [0, 7]),
    # 'C-r&g': (adult_model, pre_census_income, [6, 7]),
    # 'G-g&a': (german_model, pre_german_credit, [6, 9]),
    # 'M-a&r': (meps15_model, pre_meps_15, [0, 1]),
    # 'M-a&g': (meps15_model, pre_meps_15, [0, 9]),
    # 'M-r&g': (meps15_model, pre_meps_15, [1, 9]),
    # 'H-a&g': (heart_model, pre_heart_heath, [0, 1]),
    # 'S-a&g': (students_model, pre_students, [2, 1]),
})