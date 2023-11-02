import pandas as pd


def find_feature_info(X, dataset_index, dataset_name):
    M = len(X[0])  # 特征数目
    feature_info = []

    for m in range(M):
        feature_values = [sample[m] for sample in X]
        unique_values = set(feature_values)

        info = {
            'dataset_name': dataset_name,
            'dataset_index': dataset_index,
            'feature_index': m,
            'num_unique_values': len(unique_values),
            'min_value': min(feature_values),
            'max_value': max(feature_values)
        }
        feature_info.append(info)

        print(f"Feature {m}:")
        print(f"\tNumber of unique values: {info['num_unique_values']}")
        print(f"\tMin value: {info['min_value']}")
        print(f"\tMax value: {info['max_value']}")

    return feature_info


def compute_cartesian_product_for_datasets(datasets, datasets_names, models):
    all_feature_info = []
    cartesian_products = []

    for idx, dataset in enumerate(datasets):
        feature_info = find_feature_info(dataset.X, idx, datasets_names[idx])
        all_feature_info.extend(feature_info)

        total_cartesian_product = 1
        for info in feature_info:
            total_cartesian_product *= info['num_unique_values']

        model = models[idx]
        model_accuracy = model.evaluate(dataset.X_test, dataset.y_test)
        cartesian_products.append({
            'dataset_name': datasets_names[idx],  # 'dataset_name': 'census_income
            'dataset_index': idx,
            'cartesian_product': f"{total_cartesian_product:.2e}",
            'model_name': model.name,
            'model_loss': model_accuracy[0],
            'model_accuracy': model_accuracy[1]
        })
        print(f"Total Cartesian Product of the Input Domain: {total_cartesian_product:.2e}")
        print(f"Model Loss&Accuracy: {model_accuracy}")

    return all_feature_info, cartesian_products


def save_to_csv(datasets, datasets_names, models):
    feature_info, cartesian_products = compute_cartesian_product_for_datasets(datasets, datasets_names, models)

    # 使用pandas创建DataFrame
    df_feature_info = pd.DataFrame(feature_info)
    df_cartesian_products = pd.DataFrame(cartesian_products)

    # 将DataFrame保存到CSV文件中
    df_feature_info.to_csv("feature_info.csv", index=False)
    df_cartesian_products.to_csv("cartesian_products.csv", index=False)


from preprocessing import pre_census_income, pre_german_credit, pre_bank_marketing, pre_meps_15, pre_heart_heath, pre_diabetes, pre_students
from tensorflow import keras

adult_model = keras.models.load_model("../../models/original_models/adult_model.h5")
german_model = keras.models.load_model("../../models/original_models/german_model.h5")
bank_model = keras.models.load_model("../../models/original_models/bank_model.h5")
meps15_model = keras.models.load_model("../../models/original_models/meps15_model.h5")
heart_model = keras.models.load_model("../../models/original_models/heart_model.h5")
diabetes_model = keras.models.load_model("../../models/original_models/diabetes_model.h5")
students_model = keras.models.load_model("../../models/original_models/students_model.h5")

datasets = [pre_census_income, pre_german_credit, pre_bank_marketing, pre_meps_15, pre_heart_heath, pre_diabetes, pre_students]
datasets_names = ['census', 'credit', 'bank', 'meps', 'heart', 'diabetes', 'students']
models = [adult_model, german_model, bank_model, meps15_model, heart_model, diabetes_model, students_model]

save_to_csv(datasets, datasets_names, models)
