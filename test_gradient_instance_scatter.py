import os
import numpy as np
import matplotlib.pyplot as plt
import experiment_config
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

info = experiment_config.all_benchmark_info
benchmark_names = [benchmark for benchmark in info.keys()]
methods = ["ADF", "EIDIG", "MAFT"]

# 文件夹路径
folder_path = 'logging_data/gradients_comparison/'


# 文件夹路径
folder_path = 'logging_data/gradients_comparison/'

# 初始化 PCA
pca = PCA(n_components=2)

# 存储每种方法的转换后数据
transformed_data_dict = {method: [] for method in methods}

# 遍历文件夹中的文件并进行PCA
for filename in os.listdir(folder_path):
    if filename.endswith(".npy"):
        # 解析文件名以获取benchmark和method
        for benchmark in benchmark_names:
            if benchmark in filename:
                for method in methods:
                    if method in filename:
                        # 加载数据
                        file_path = os.path.join(folder_path, filename)
                        gradients = np.load(file_path)

                        # 执行 PCA
                        transformed_data = pca.fit_transform(gradients)
                        transformed_data_dict[method].append(transformed_data)

# 合并每种方法的数据
for method in transformed_data_dict:
    transformed_data_dict[method] = np.vstack(transformed_data_dict[method])

# 绘制散点图
plt.figure(figsize=(15, 8))
colors = {'ADF': 'red', 'EIDIG': 'blue', 'MAFT': 'green'}
for method, data in transformed_data_dict.items():
    plt.scatter(data[:, 0], data[:, 1], color=colors[method], alpha=0.5, label=method)

# 添加图例和标签
plt.xlabel('First Principal Component (PC1)')
plt.ylabel('Second Principal Component (PC2)')
plt.title('PCA of Gradient Data Across Benchmarks and Methods')
plt.legend()
plt.savefig('logging_data/gradients_comparison/PCA.svg')
plt.show()

# 计算每种方法的凸包和重叠面积
convex_hulls = {method: ConvexHull(data) for method, data in transformed_data_dict.items()}




from shapely.geometry import Polygon

# 假设 convex_hulls 是之前计算的凸包字典
# 创建每个方法的凸包多边形
polygons = {method: Polygon(transformed_data_dict[method][hull.vertices]) for method, hull in convex_hulls.items()}

# 计算交集面积
# 以计算ADF和EIDIG的凸包交集面积为例
intersection_area_ae = polygons['ADF'].intersection(polygons['EIDIG']).area
intersection_area_em = polygons['EIDIG'].intersection(polygons['MAFT']).area
intersection_area_em = polygons['MAFT'].intersection(polygons['ADF']).area

print('ADF and EIDIG intersection area:', intersection_area_ae)
print('EIDIG and MAFT intersection area:', intersection_area_em)
print('MAFT and ADF intersection area:', intersection_area_em)
