import os
import random

dataset_path = '/lab/haoq_lab/12532563/xpool/data/suscape'

data_list = []
videos_path = os.path.join(dataset_path, 'third_level_labels')
for filename in os.listdir(videos_path):
    if filename.endswith('.json'):
        data_list.append(filename.removesuffix('.json'))

random.seed(42)
random.shuffle(data_list)

split_point = int(len(data_list) * 0.2)

test_list = data_list[:split_point]
train_list = data_list[split_point:]

with open(dataset_path + '/train_list.txt', 'w') as f:
    for item in train_list:
        f.write(item + '\n')

with open(dataset_path + '/test_list.txt', 'w') as f:
    for item in test_list:
        f.write(item + '\n')