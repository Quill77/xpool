#!/usr/bin/env python3
import re
import os

# 1. 改成你的实际文件名
SRC_FILE = 'wrong_pred_mixed_labels.txt'   # 要过滤的文件
DST_FILE = 'final_suscape_labels.txt'   # 要被追加的文件
MERGE_FILE = 'merge.txt'  # 输出文件

# 2. 过滤：去掉首行 + 不以数字打头的行
with open(SRC_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

filtered = [ln for ln in lines[1:] if not re.match(r'^\s*\d', ln)]

# 3. 追加到目标文件末尾
with open(DST_FILE, 'r', encoding='utf-8') as f:
    dst_content = f.read()

with open(MERGE_FILE, 'w', encoding='utf-8') as f:
    f.write(dst_content)
    f.writelines(filtered)

print(f'已保存 {MERGE_FILE}  （追加 {len(filtered)} 行）')