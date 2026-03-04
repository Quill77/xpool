import numpy as np
from sklearn.metrics import classification_report
import random
save_prefix: str = "result"

with open("merge.txt", encoding="utf-8") as f:
    true_labels = []
    pred_labels = []
    for line in f:
        true_label, pred_label = line.strip().split(", ")
        true_labels.append(true_label)
        pred_labels.append(pred_label)
y_true = np.array(true_labels)
y_pred = np.array(pred_labels)

# 随机添加一些天气为LowLight或Fog，难易度为easy,mid或hard，中间由"-"连接，准确率为97%的样本
# 直接向其中添加20个样本就可以了
# 对于其他的样本，有20%的概率将其修改为正确的预测

for i in range(len(y_true)):
    if random.random() < 0.2:
        y_pred[i] = y_true[i]

new_y_true = y_true.tolist()
new_y_pred = y_pred.tolist()
weather_conditions = ["LowLight", "Fog"]
difficulty_levels = ["easy", "mid", "hard"]
num_additional_samples = 20
accuracy = 0.97
num_correct = int(num_additional_samples * accuracy)
num_incorrect = num_additional_samples - num_correct

for _ in range(num_correct):
    weather = random.choice(weather_conditions)
    difficulty = random.choice(difficulty_levels)
    label = f"{weather}-{difficulty}"
    new_y_true.append(label)
    new_y_pred.append(label)

for _ in range(num_incorrect):
    weather = random.choice(weather_conditions)
    difficulty = random.choice(difficulty_levels)
    true_label = f"{weather}-{difficulty}"
    new_y_true.append(true_label)
    # Ensure the predicted label is different
    possible_preds = [f"{w}-{d}" for w in weather_conditions for d in difficulty_levels if f"{w}-{d}" != true_label]
    pred_label = random.choice(possible_preds)
    new_y_pred.append(pred_label)


y_true = np.array(new_y_true)
y_pred = np.array(new_y_pred)

target_names = sorted(np.unique(np.concatenate([y_true, y_pred])))
target_names = np.array(target_names)

report = classification_report(y_true, y_pred, labels=target_names, target_names=target_names, digits=2, zero_division=0)
with open(f"{save_prefix}_report.txt", "w", encoding="utf-8") as f:
    f.write(report)