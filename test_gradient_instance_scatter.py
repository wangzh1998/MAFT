import os
import numpy as np
import matplotlib.pyplot as plt
import experiment_config
from sklearn.decomposition import PCA

info = experiment_config.all_benchmark_info
benchmark_names = [benchmark for benchmark in info.keys()]
methods = ["ADF", "EIDIG", "MAFT"]

folder_path = 'logging_data/gradients_comparison/'

# initialize PCA
pca = PCA(n_components=2)

# store transformed data for each method
transformed_data_dict = {method: [] for method in methods}

# traverse files in the folder and perform PCA
for filename in os.listdir(folder_path):
    if filename.endswith(".npy"):
        for benchmark in benchmark_names:
            if benchmark in filename:
                for method in methods:
                    if method in filename:
                        file_path = os.path.join(folder_path, filename)
                        gradients = np.load(file_path)
                        transformed_data = pca.fit_transform(gradients)
                        transformed_data_dict[method].append(transformed_data)

# merge data for each method
for method in transformed_data_dict:
    transformed_data_dict[method] = np.vstack(transformed_data_dict[method])

# plot scatter
plt.figure(figsize=(15, 8))
colors = {'ADF': 'red', 'EIDIG': 'blue', 'MAFT': 'green'}
for method, data in transformed_data_dict.items():
    plt.scatter(data[:, 0], data[:, 1], color=colors[method], alpha=0.5, label=method)

# add legend and labels
plt.xlabel('First Principal Component (PC1)', fontsize=18)
plt.ylabel('Second Principal Component (PC2)', fontsize=18)
plt.title('PCA of Gradient Data Across Benchmarks and Methods', fontsize=20)
# plt.legend('Method', fontsize=18)
# plt.legend(title='Method', fontsize=18)
plt.legend(fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.savefig('logging_data/gradients_comparison/PCA.svg')
plt.show()
