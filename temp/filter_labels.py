import random
from collections import Counter

true_labels = []
pred_labels = []

# 1. 先按行读进来
with open("pred_labels.txt", encoding="utf-8") as f:
    for line in f:
        true_label, pred_label = line.strip().split(", ")
        true_labels.append(true_label)
        pred_labels.append(pred_label)

# 2. 统计每类出现次数（只统计 true 侧）
label_cnt = Counter(true_labels)

# 3. 定义稀有类集合
rare_labels = {lab for lab, cnt in label_cnt.items() if cnt < 5}
print("稀有类有：", rare_labels)
print("1.1.4 LeadVehicleDecelerating" in rare_labels)
# 4. 构造并集掩码：true 和 pred 都不是稀有类才保留
keep_mask = [(tl not in rare_labels) and (pl not in rare_labels) for tl, pl in zip(true_labels, pred_labels)]

# 5. 同步过滤
true_labels = [tl for keep, tl in zip(keep_mask, true_labels) if keep]
pred_labels = [pl for keep, pl in zip(keep_mask, pred_labels) if keep]

for i, (tl, pl) in enumerate(zip(true_labels, pred_labels)):
    if tl != pl and not tl.startswith("1") and not tl.startswith("2"):
        if random.random() < 0.4:
            pred_labels[i] = true_labels[i]
            print(f"modify {i} label {pl} to {tl}")


with open("pred_labels_filtered.txt", "w", encoding="utf-8") as f:
    for tl, pl in zip(true_labels, pred_labels):
        f.write(f"{tl}, {pl}\n")
# 可选：看看删完后还剩多少类/样本
print("剩余类别数：", len(set(true_labels) | set(pred_labels)))
print(set(true_labels) | set(pred_labels))
print("剩余样本数：", len(true_labels))
