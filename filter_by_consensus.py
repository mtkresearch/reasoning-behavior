#!/usr/bin/env python3
"""
篩選具有特定 method_type 共識的題目
根據模型判定數量排序並輸出 unique_id
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import random


def load_json(filepath):
    """載入 JSON 檔案"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_consensus(option, model_files):
    """
    為每個 unique_id 計算判定為指定選項的模型數量

    Args:
        option: 要查詢的選項 (例如 'A', 'B', 'C' 等)
        model_files: 模型判定檔案列表

    Returns:
        dict: {unique_id: consensus_count}
    """
    consensus_counts = defaultdict(int)

    for model_file in model_files:
        data = load_json(model_file)

        for item in data:
            unique_id = item.get('unique_id')
            method_types = item.get('method_types', [])

            if unique_id and option in method_types:
                consensus_counts[unique_id] += 1

    return consensus_counts


def main():
    parser = argparse.ArgumentParser(
        description='篩選具有特定 method_type 共識的題目',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  %(prog)s A
  %(prog)s B --top 10
  %(prog)s C --model-dir ./consistency_data
        """
    )

    parser.add_argument('option', type=str,
                        help='要查詢的選項 (例如 A, B, C, D, E, F, G, H, I)')
    parser.add_argument('--model-dir', type=str,
                        default='consistency_data',
                        help='模型判定檔案所在目錄 (預設: consistency_data)')
    parser.add_argument('--top', type=int, default=None,
                        help='只顯示前 N 筆 (預設: 全部顯示)')

    args = parser.parse_args()

    # 取得所有模型檔案
    model_dir = Path(args.model_dir)
    model_files = list(model_dir.glob('consistency_*.json'))

    if not model_files:
        print(f"錯誤: 在 {model_dir} 目錄中找不到 consistency_*.json 檔案")
        return 1

    print(f"找到 {len(model_files)} 個模型檔案:")
    for f in model_files:
        print(f"  - {f.name}")
    print()

    # 計算共識數量
    consensus_counts = count_consensus(args.option, model_files)

    if not consensus_counts:
        print(f"沒有找到任何題目被判定為選項 '{args.option}'")
        return 0

    # 排序：共識數量多的靠後（從小到大排序）
    suffled_items = list(consensus_counts.items())
    random.shuffle(suffled_items)
    sorted_items = sorted(suffled_items, key=lambda x: x[1])

    # 限制輸出數量
    if args.top:
        sorted_items = sorted_items[:args.top]

    # 輸出結果
    print(f"選項 '{args.option}' 的共識結果 (共 {len(sorted_items)} 題):")
    print(f"{'='*50}")
    for unique_id, count in sorted_items:
        print(f"[{count}] {unique_id}")

    return 0


if __name__ == '__main__':
    exit(main())
