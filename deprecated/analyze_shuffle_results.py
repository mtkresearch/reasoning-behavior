#!/usr/bin/env python3
"""分析 shuffle reasoning comparison 实验结果 - 分析 truncate ratio 与正确性关系"""

import json
import glob
from collections import defaultdict, Counter
from pathlib import Path
import re

def extract_truncate_ratio(file_path):
    """从文件路径或summary.json中提取truncate ratio"""
    # 先尝试从文件路径提取
    # 路径格式: ./data/shuffle_comparison_exp/turncate_XXX/experiment_results.json

    # 浮点数格式: f05 -> 0.5, f03 -> 0.3, f01 -> 0.1, f07 -> 0.7, f09 -> 0.9
    match = re.search(r'turncate_f0?(\d+)', file_path)
    if match:
        ratio_str = match.group(1)
        return float('0.' + ratio_str)

    # 整数格式: 3 -> 3, 5 -> 5
    match = re.search(r'turncate_(\d+)(?![a-z])', file_path)
    if match:
        return float(match.group(1))

    # 如果无法从路径提取，尝试读取summary.json
    summary_path = Path(file_path).parent / 'summary.json'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            return summary.get('del_last_line', 0)

    return 0

def analyze_by_truncate_ratio(file_paths):
    """按truncate ratio分析实验结果"""

    # 按truncate ratio分组的结果
    results_by_ratio = defaultdict(list)

    # 读取所有文件并按ratio分组
    for file_path in file_paths:
        ratio = extract_truncate_ratio(file_path)
        print(f"  {Path(file_path).parent.name}: ratio = {ratio}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results_by_ratio[ratio].extend(data)

    return results_by_ratio

def main():
    # 查找所有 experiment_results.json 文件
    pattern = 'data/shuffle_comparison_exp/turncate_*/experiment_results.json'
    file_paths = sorted(glob.glob(pattern))

    if not file_paths:
        print(f"未找到匹配的文件: {pattern}")
        return

    print(f"找到 {len(file_paths)} 个文件")
    print("=" * 80)

    # 按truncate ratio分析结果
    results_by_ratio = analyze_by_truncate_ratio(file_paths)

    # 按ratio排序
    sorted_ratios = sorted(results_by_ratio.keys())

    print("\n" + "=" * 100)
    print("Truncate Ratio vs Correctness Analysis")
    print("=" * 100)

    # 表头
    print(f"\n{'Ratio':<10} {'Total':<8} {'Full%':<8} {'Normal%':<10} {'Shuffle%':<10} "
          f"{'Same%':<8} {'N&S_Both%':<12} {'Only_N%':<10} {'Only_S%':<10}")
    print("-" * 100)

    # 为每个ratio统计
    for ratio in sorted_ratios:
        results = results_by_ratio[ratio]
        total = len(results)

        # 统计各种情况
        full_correct = sum(1 for r in results if r.get('is_full_correct'))
        normal_correct = sum(1 for r in results if r.get('is_normal_correct'))
        shuffle_correct = sum(1 for r in results if r.get('is_shuffle_correct'))
        same_answer = sum(1 for r in results if r.get('is_same_answer'))

        both_correct = sum(1 for r in results
                          if r.get('is_normal_correct') and r.get('is_shuffle_correct'))
        only_normal = sum(1 for r in results
                         if r.get('is_normal_correct') and not r.get('is_shuffle_correct'))
        only_shuffle = sum(1 for r in results
                          if not r.get('is_normal_correct') and r.get('is_shuffle_correct'))

        # 打印统计
        ratio_str = f"{ratio:.1f}" if ratio < 1 else f"{int(ratio)}"
        print(f"{ratio_str:<10} {total:<8} "
              f"{full_correct/total*100:>6.1f}% "
              f"{normal_correct/total*100:>8.1f}% "
              f"{shuffle_correct/total*100:>8.1f}% "
              f"{same_answer/total*100:>6.1f}% "
              f"{both_correct/total*100:>10.1f}% "
              f"{only_normal/total*100:>8.1f}% "
              f"{only_shuffle/total*100:>8.1f}%")

    print("-" * 100)

    # 详细的组合统计
    print("\n" + "=" * 100)
    print("Detailed Combination Analysis by Ratio")
    print("=" * 100)

    for ratio in sorted_ratios:
        results = results_by_ratio[ratio]
        total = len(results)

        print(f"\n{'='*50}")
        ratio_str = f"{ratio:.1f}" if ratio < 1 else f"{int(ratio)}"
        print(f"Truncate Ratio: {ratio_str} (Total: {total} problems)")
        print(f"{'='*50}")

        # 统计所有组合
        combinations = Counter()
        for r in results:
            full = r.get('is_full_correct')
            same = r.get('is_same_answer')
            normal = r.get('is_normal_correct')
            shuffle = r.get('is_shuffle_correct')
            key = (full, same, normal, shuffle)
            combinations[key] += 1

        # 打印组合表头
        print(f"{'Full':<8} {'Same':<8} {'Normal':<8} {'Shuffle':<8} {'Count':<8} {'%':<8}")
        print("-" * 50)

        # 按数量排序打印
        for (full, same, normal, shuffle), count in sorted(combinations.items(), key=lambda x: -x[1]):
            print(f"{str(full):<8} {str(same):<8} {str(normal):<8} {str(shuffle):<8} "
                  f"{count:<8} {count/total*100:>6.1f}%")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    main()
