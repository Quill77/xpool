import pandas as pd
import matplotlib.pyplot as plt

with open("scores.txt") as f:
    rows, cur = [], {}
    for line in f:
        line = line.strip()
        if line.startswith("frames:"):
            if cur:
                rows.append(cur)
            cur = {"frame": int(line.split()[1])}
        elif line.startswith("New R@1:"):
            cur["R@1"] = float(line.split()[-1])
        elif line.startswith("New R@3:"):
            cur["R@3"] = float(line.split()[-1])
        elif line.startswith("New R@5:"):
            cur["R@5"] = float(line.split()[-1])
        elif line.startswith("New Precision@1:"):
            cur["P@1"] = float(line.split()[-1])
        elif line.startswith("New Precision@3:"):
            cur["P@3"] = float(line.split()[-1])
        elif line.startswith("New Precision@5:"):
            cur["P@5"] = float(line.split()[-1])
    if cur:
        rows.append(cur)

full_idx = range(1, 17)
df_raw = pd.DataFrame(rows).set_index("frame")
df = df_raw.reindex(full_idx).ffill()
print(df)

# 3. 画图
plt.figure(figsize=(7, 4))
plt.plot(df.index, df["R@1"], marker="o", label="R@1 / P@1")

for col in ["R@3", "R@5", "P@3", "P@5"]:
    plt.plot(df.index, df[col], marker="o", label=col)

plt.xticks(full_idx)  # 横坐标 1~16
plt.xlabel("frames")
plt.ylabel("score / %")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("curve.pdf")  # 矢量，论文最清晰
plt.savefig("curve.png", dpi=300)  # 位图，方便插 PPT
plt.close()
