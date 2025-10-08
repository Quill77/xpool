import os
import random
import json

data_path = '/lab/haoq_lab/12532563/xpool/data/OpenDataLab___nuScenes-C'

        
with open(os.path.join(data_path, 'nuscenes_corruption_new.json'), 'r') as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)

split_point = int(len(data) * 0.2)

test_list = data[:split_point]
train_list = data[split_point:]

with open(data_path + '/train_list.json', 'w') as f:
    json.dump(train_list, f, indent=4)

with open(data_path + '/test_list.json', 'w') as f:
    json.dump(test_list, f, indent=4)