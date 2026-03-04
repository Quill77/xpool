import pandas as pd
import numpy as np

TARGET_N = 500
RANDOM_STATE = 42

# 读取原始数据
df = pd.read_csv(
    "results.txt",
    header=None,
    names=["gt", "pred"]
)

N = len(df)

# 统计 (gt, pred) 组合
group_counts = (
    df.groupby(["gt", "pred"])
      .size()
      .reset_index(name="count")
)

# 计算每组目标样本数（浮点）
group_counts["target"] = group_counts["count"] / N * TARGET_N

# 先向下取整
group_counts["target_int"] = np.floor(group_counts["target"]).astype(int)

# 剩余需要补的条数
remaining = TARGET_N - group_counts["target_int"].sum()

# 按小数部分大小分配剩余名额
group_counts["fraction"] = group_counts["target"] - group_counts["target_int"]
group_counts = group_counts.sort_values("fraction", ascending=False)

group_counts.loc[:remaining-1, "target_int"] += 1

# === 生成扩展数据 ===
rows = []

rng = np.random.default_rng(RANDOM_STATE)

for _, row in group_counts.iterrows():
    sub = df[(df["gt"] == row["gt"]) & (df["pred"] == row["pred"])]
    if row["target_int"] > 0:
        sampled = sub.sample(
            n=row["target_int"],
            replace=True,
            random_state=rng.integers(1e9)
        )
        rows.append(sampled)

df_aug = pd.concat(rows, ignore_index=True)

print("扩展后条数:", len(df_aug))

# 保存
df_aug.to_csv(
    "results_500.txt",
    index=False,
    header=False
)
