import os
import random

# 数据集路径
dataset_path = '/lab/haoq_lab/12532563/xpool/data/suscape'

# 获取所有数据的列表
data_list = []
videos_path = os.path.join(dataset_path, 'third_level_labels')
for filename in os.listdir(videos_path):
    if filename.endswith('.json'):
        data_list.append(filename.removesuffix('.json'))

# 设置随机种子，保证结果可复现
random.seed(42)

# 打乱数据列表
random.shuffle(data_list)

# 计算划分点
split_point = int(len(data_list) * 0.2)

# 划分数据集
test_list = data_list[:split_point]
train_list = data_list[split_point:]

# 写入训练集
with open(dataset_path + '/train_list.txt', 'w') as f:
    for item in train_list:
        f.write(item + '\n')

# 写入测试集
with open(dataset_path + '/test_list.txt', 'w') as f:
    for item in test_list:
        f.write(item + '\n')