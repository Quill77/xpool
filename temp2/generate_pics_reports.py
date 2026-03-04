import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def plot_right_wrong_topk(y_true, y_pred, top_k=15, figsize=(6, 5), save_path="right_wrong_topk.png"):

    # 1. 统计频次
    uniq, cnt = np.unique(y_true, return_counts=True)
    freq = pd.Series(cnt, index=uniq).sort_values(ascending=False)
    top_cls = freq.head(top_k).index.tolist()
    y_true_top = np.where(np.isin(y_true, top_cls), y_true, "Others")
    y_pred_top = np.where(np.isin(y_pred, top_cls), y_pred, "Others")
    names_top = top_cls + ["Others"]

    y_true_top = pd.Categorical(y_true_top, categories=names_top, ordered=True)
    y_pred_top = pd.Categorical(y_pred_top, categories=names_top, ordered=True)

    # 2. 计算正误
    cm = confusion_matrix(y_true_top, y_pred_top, labels=names_top)
    right = cm.diagonal()
    wrong = cm.sum(axis=1) - right
    total = right + wrong

    # 3. 排序：小数量在上，Others 强制放最顶（DataFrame 第一行）
    df = pd.DataFrame({"wrong": wrong, "right": right, "total": total}, index=names_top)
    others_row = df.loc[["Others"]] if "Others" in df.index else None
    df_main = df.drop("Others", errors="ignore").sort_values("total", ascending=True)
    if others_row is not None:
        df = pd.concat([others_row, df_main])  # Others 放最前面
    else:
        df = df_main

    # 4. 画图
    df = df[["wrong", "right"]]
    ax = df.plot.barh(stacked=True, figsize=figsize, color=[sns.color_palette("Reds")[3], sns.color_palette("GnBu")[2]])
    ax.set_xlabel("Number of samples")
    ax.set_title(f"Correct / Incorrect  (top-{top_k} classes + Others)")
    ax.legend(["Wrong", "Correct"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    return ax


def plot_report_topk(y_true, y_pred, top_k=15, metrics=["precision", "recall", "f1-score"], figsize=(10, 4), save_path="report_topk_bar.png"):
    # 同样算频次
    uniq, cnt = np.unique(y_true, return_counts=True)
    freq = pd.Series(cnt, index=uniq).sort_values(ascending=False)
    top_cls = freq.head(top_k).index.tolist()
    # 把冷门类全部去掉，只保留 top_k 里的样本
    mask = np.isin(y_true, top_cls)
    y_true_top = y_true[mask]
    y_pred_top = y_pred[mask]

    report = classification_report(y_true_top, y_pred_top, labels=top_cls, target_names=top_cls, output_dict=True, zero_division=0)
    # 构造 DataFrame
    df = pd.DataFrame({lab: [report[lab][m] for m in metrics] for lab in top_cls}, index=metrics).T
    # 画图
    ax = df.plot.bar(figsize=figsize, width=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Classes  (top-k by frequency)")
    ax.set_title(f"Classification metrics  (top-{top_k} classes)")
    ax.legend(metrics)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    return ax, df


def sims_to_report_and_cm(y_true, y_pred, target_names: list = None, save_prefix: str = "result", top_k: int = 15):

    if target_names is None:
        target_names = sorted(np.unique(np.concatenate([y_true, y_pred])))
    target_names = np.array(target_names)

    # 1. 对/错条形图（隐藏冷门）
    plot_right_wrong_topk(y_true, y_pred, top_k=top_k, save_path=f"{save_prefix}_right_wrong_topk.png")

    # 2. 指标柱状图（隐藏冷门）
    plot_report_topk(y_true, y_pred, top_k=top_k, save_path=f"{save_prefix}_report_topk_bar.png")

    # 3. 文字报告（依旧全集，想留就留）
    report = classification_report(y_true, y_pred, labels=target_names, target_names=target_names, digits=4, zero_division=0)
    with open(f"{save_prefix}_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    return report


if __name__ == "__main__":
    with open("result.txt", encoding="utf-8") as f:
        true_labels = []
        pred_labels = []
        i = 0
        for line in f:
            i+=1
            print(i)
            true_label, pred_label = line.strip().split(",")
            true_labels.append(true_label.strip())
            pred_labels.append(pred_label.strip())
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    sims_to_report_and_cm(y_true, y_pred, save_prefix="vast_results", top_k=15)
