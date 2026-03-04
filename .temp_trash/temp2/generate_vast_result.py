# 精确复现分类报告的数据生成算法 - 修复FP分配问题

# 分类报告数据
report_data = {
    '1.1 InLane': {'precision': 0.9764, 'recall': 0.9254, 'support': 134},
    '1.1.2 LeadVehicleCutOut': {'precision': 0.4444, 'recall': 0.5000, 'support': 8},
    '1.1.3 VehicleCutInAhead': {'precision': 0.9714, 'recall': 0.7727, 'support': 44},
    '1.1.4 LeadVehicleDecelerating': {'precision': 0.8750, 'recall': 0.6364, 'support': 11},
    '1.1.5 LeadVehicleStppoed': {'precision': 0.8750, 'recall': 1.0000, 'support': 14},
    '1.1.6 LeadVehicleAccelerating': {'precision': 0.7368, 'recall': 0.6667, 'support': 21},
    '1.2 ChangingLaneLeft': {'precision': 0.7500, 'recall': 0.9231, 'support': 26},
    '1.3 ChangingLaneRight': {'precision': 0.7273, 'recall': 0.5714, 'support': 14},
    '2.1.1 StopAtRedLight': {'precision': 1.0000, 'recall': 1.0000, 'support': 15},
    '2.4.1 NoVehiclesAhead': {'precision': 1.0000, 'recall': 1.0000, 'support': 30},
    '2.6.1 NoVehiclesAhead': {'precision': 1.0000, 'recall': 0.8182, 'support': 11},
    'Brightness+easy': {'precision': 1.0000, 'recall': 0.9032, 'support': 31},
    'Brightness+hard': {'precision': 1.0000, 'recall': 0.9474, 'support': 19},
    'Brightness+mid': {'precision': 0.9500, 'recall': 1.0000, 'support': 19},
    'Fog+easy': {'precision': 0.9310, 'recall': 1.0000, 'support': 27},
    'Fog+mid': {'precision': 1.0000, 'recall': 0.9286, 'support': 28},
    'LowLight+easy': {'precision': 0.6667, 'recall': 0.8800, 'support': 25},
    'LowLight+hard': {'precision': 0.9474, 'recall': 0.7500, 'support': 24},
    'LowLight+mid': {'precision': 0.7059, 'recall': 0.6667, 'support': 18},
    'Snow+easy': {'precision': 1.0000, 'recall': 0.9333, 'support': 15},
    'Snow+mid': {'precision': 0.5870, 'recall': 1.0000, 'support': 27}
}

# 计算每个类别的TP, FP, FN（使用精确的数学计算）
for category, data in report_data.items():
    support = data['support']
    precision = data['precision']
    recall = data['recall']
    
    # 计算真正例 (TP) - 使用精确的数学计算而不是round
    tp = support * recall
    
    # 计算假负例 (FN)
    fn = support - tp
    
    # 计算总预测为正类的数量 (TP + FP)
    if precision > 0:
        total_pred_positive = tp / precision
        fp = total_pred_positive - tp
    else:
        fp = 0
    
    data['tp'] = tp
    data['fp'] = fp
    data['fn'] = fn
    data['tp_int'] = int(round(tp))  # 用于计数的整数版本
    data['fp_int'] = int(round(fp))
    data['fn_int'] = int(round(fn))

# 创建结果列表
results = []

# 第一步：生成所有TP (y_true=category, y_pred=category)
for category, data in report_data.items():
    for _ in range(data['tp_int']):
        results.append(f"{category}|{category}")

# 第二步：智能分配FN和FP
# 关键洞察：我们需要一个分配算法，确保：
# 1. 每个类别的FN准确（y_true=category但y_pred≠category）
# 2. 每个类别的FP准确（y_true≠category但y_pred=category）
# 3. 不会出现类别预测到自身的情况

# 创建分配池
fn_pool = []  # 需要分配FN的实例
fp_demands = {}  # 每个类别需要的FP数量

for category, data in report_data.items():
    # 添加FN到池子
    fn_pool.extend([category] * data['fn_int'])
    
    # 记录FP需求
    if data['fp_int'] > 0:
        fp_demands[category] = data['fp_int']

# 智能分配算法
import random
random.seed(42)  # 固定随机种子以确保可重复性

# 为每个FN分配一个错误的预测类别
for true_category in fn_pool:
    # 找到所有可能的预测类别（不能预测到自身）
    possible_pred_categories = [cat for cat in fp_demands.keys() if cat != true_category]
    
    if not possible_pred_categories:
        # 如果没有其他类别需要FP，选择任意其他类别
        possible_pred_categories = [cat for cat in report_data.keys() if cat != true_category]
    
    if possible_pred_categories:
        # 优先选择还需要FP的类别
        candidates = [cat for cat in possible_pred_categories if fp_demands.get(cat, 0) > 0]
        if not candidates:
            candidates = possible_pred_categories
        
        # 随机选择一个类别
        pred_category = random.choice(candidates)
        
        # 添加结果
        results.append(f"{true_category}|{pred_category}")
        
        # 减少FP需求
        if pred_category in fp_demands:
            fp_demands[pred_category] -= 1
            if fp_demands[pred_category] <= 0:
                del fp_demands[pred_category]

# 处理剩余的FP需求
# 如果还有类别需要FP，但我们已经用完了FN，我们需要创建额外的错误预测
for pred_category, remaining_fp in fp_demands.items():
    for _ in range(remaining_fp):
        # 选择一个真实的类别（不能是pred_category自身）
        possible_true_categories = [cat for cat in report_data.keys() if cat != pred_category]
        if possible_true_categories:
            true_category = random.choice(possible_true_categories)
            results.append(f"{true_category}|{pred_category}")

# 验证总数
print(f"总样本数: {len(results)}")
print(f"期望样本数: 561")

# 验证每个类别的support
support_check = {}
for line in results:
    true_label = line.split('|')[0]
    support_check[true_label] = support_check.get(true_label, 0) + 1

print("\nSupport验证:")
for category, expected in [(cat, data['support']) for cat, data in report_data.items()]:
    actual = support_check.get(category, 0)
    print(f"{category}: 期望={expected}, 实际={actual}, 差异={actual-expected}")

# 验证precision和recall
print("\nMetrics验证:")
for category, data in report_data.items():
    actual_tp = sum(1 for line in results if line.startswith(f"{category}|{category}"))
    actual_fp = sum(1 for line in results if line.split('|')[1] == category and not line.startswith(f"{category}|"))
    actual_fn = sum(1 for line in results if line.split('|')[0] == category and not line.endswith(f"|{category}"))
    
    precision = actual_tp / (actual_tp + actual_fp) if (actual_tp + actual_fp) > 0 else 0
    recall = actual_tp / (actual_tp + actual_fn) if (actual_tp + actual_fn) > 0 else 0
    
    print(f"{category}:")
    print(f"  Precision: 期望={data['precision']:.4f}, 实际={precision:.4f}, 差异={precision-data['precision']:.4f}")
    print(f"  Recall: 期望={data['recall']:.4f}, 实际={recall:.4f}, 差异={recall-data['recall']:.4f}")

# 保存结果
with open('result.txt', 'w', encoding='utf-8') as f:
    for line in results:
        f.write(line + '\n')

print(f"\n数据已保存到 result.txt")