#!/usr/bin/env python3
"""
統計 results.json 中所有 traj 的行數中位數
注意：多個連續的 \n 會被合併為單一 \n
"""

import json
import re
from pathlib import Path
import statistics


def count_lines(text):
    """
    計算文字的行數，將多個連續的 \n 合併為單一 \n
    """
    # 將多個連續的 \n 替換為單一 \n
    normalized_text = re.sub(r'\n+', '\n', text)

    # 計算行數（split 後的元素數量）
    lines = normalized_text.split('\n')

    # 如果最後一行是空的，不計算
    if lines and lines[-1] == '':
        return len(lines) - 1

    return len(lines)


def main():
    # 讀取 JSON 檔案
    json_path = Path('data/AIME2025__R10/gpt-oss/p1/results.json')

    print(f"正在讀取檔案: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"總共有 {len(data)} 筆資料")

    # 統計所有 traj 的行數
    line_counts = []

    for i, item in enumerate(data):
        if 'result' in item and 'traj' in item['result']:
            traj = item['result']['traj']
            line_count = count_lines(traj)
            line_counts.append(line_count)

            # 顯示前幾筆的統計資訊
            if i < 5:
                print(f"  [{i}] unique_id: {item.get('unique_id', 'N/A')}, 行數: {line_count}")

    # 計算中位數
    if line_counts:
        median = statistics.median(line_counts)
        mean = statistics.mean(line_counts)
        min_lines = min(line_counts)
        max_lines = max(line_counts)

        print(f"\n統計結果:")
        print(f"  總共統計了 {len(line_counts)} 個 traj")
        print(f"  行數中位數: {median}")
        print(f"  行數平均值: {mean:.2f}")
        print(f"  行數最小值: {min_lines}")
        print(f"  行數最大值: {max_lines}")
    else:
        print("沒有找到任何 traj 資料")


if __name__ == '__main__':
    main()
