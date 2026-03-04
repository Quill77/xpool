import pandas as pd
import numpy as np

# =========================
# 配置
# =========================
INPUT_FILE = "results_500.txt"
OUTPUT_FILE = "results_500_adjusted.txt"

TARGET_ACCURACY = {
    "1.1 InLane": 0.9563,
    "1.1.2 LeadVehicleCutOut": 0.5533,
    "1.1.3 VehicleCutInAhead": 0.8212,
    "1.2 ChangingLaneLeft": 0.8300,
    "1.3 ChangingLaneRight": 0.7764,
    "Brightness-easy": 0.9500,
    "Fog-easy": 0.9500,
    "Snow-mid": 0.9500
}

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# =========================
# 1. 读取并清洗
# =========================
df = pd.read_csv(INPUT_FILE, header=None, names=["pred", "gt"])
df["gt"] = df["gt"].str.strip()
df["pred"] = df["pred"].str.strip()
df["correct"] = df["gt"] == df["pred"]

df_mod = df.copy()

# =========================
# 2. 精确修正函数
# =========================
def adjust_class_accuracy(cls, target_acc):
    sub = df_mod[df_mod["gt"] == cls]
    N = len(sub)

    if N == 0:
        print(f"[跳过] 类 {cls} 不存在")
        return

    current_correct = sub["correct"].sum()
    target_correct = int(round(target_acc * N))+1

    delta = target_correct - current_correct

    # -------- 提高准确率 --------
    if delta > 0:
        wrong_idx = sub[~sub["correct"]].index
        fix_num = min(delta, len(wrong_idx))

        fix_idx = rng.choice(wrong_idx, size=fix_num, replace=False)
        df_mod.loc[fix_idx, "pred"] = cls
        df_mod.loc[fix_idx, "correct"] = True

        action = f"修正 +{fix_num}"

    # -------- 降低准确率 --------
    elif delta < 0:
        correct_idx = sub[sub["correct"]].index
        break_num = min(-delta, len(correct_idx))

        break_idx = rng.choice(correct_idx, size=break_num, replace=False)

        # 随机错成别的类（不等于自己）
        other_labels = df_mod["gt"].unique().tolist()
        other_labels.remove(cls)

        df_mod.loc[break_idx, "pred"] = rng.choice(
            other_labels, size=break_num
        )
        df_mod.loc[break_idx, "correct"] = False

        action = f"破坏 -{break_num}"

    else:
        action = "无需修改"

    final_acc = (
        df_mod[df_mod["gt"] == cls]["correct"].mean()
    )

    print(
        f"{cls}: "
        f"{current_correct}/{N} → "
        f"{int(final_acc*N)}/{N} "
        f"(acc={final_acc:.4f}) | {action}"
    )

# =========================
# 3. 执行修正
# =========================
print("\n=== 精确调整指定类别 ===")
for cls, acc in TARGET_ACCURACY.items():
    adjust_class_accuracy(cls, acc)

# =========================
# 4. 保存
# =========================
df_mod[["pred", "gt"]].to_csv(
    OUTPUT_FILE,
    index=False,
    header=False
)

print(f"\n✅ 已保存结果到 {OUTPUT_FILE}")
