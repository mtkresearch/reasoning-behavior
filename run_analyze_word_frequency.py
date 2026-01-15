#!/usr/bin/env python3
"""
分析 results.json 中 result.traj 的 word 頻率

Usage:
    python run_analyze_word_frequency.py <results.json>

Example:
    python run_analyze_word_frequency.py data/AIME2025__R10/deepseek/p1/results.json
"""
import sys
import json
from collections import Counter
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_analyze_word_frequency.py <results.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = input_path.parent / "words.tsv"

    print(f"讀取檔案: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 收集所有 traj 中的 words
    all_words = []
    for item in data:
        traj = item.get("result", {}).get("traj", "")
        # 按空白間隔拆分 (包含空格、tab、換行等)
        words = traj.split()
        all_words.extend(words)

    # 統計頻率
    word_counts = Counter(all_words)

    # 按頻率由大到小排序
    sorted_words = word_counts.most_common()

    print(f"總共找到 {len(all_words)} 個 words")
    print(f"獨特的 words: {len(sorted_words)}")

    # 寫入 TSV 檔案
    print(f"寫入檔案: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        # 寫入標題行
        f.write("word\tfrequency\n")
        # 寫入資料
        for word, count in sorted_words:
            f.write(f"{word}\t{count}\n")

    print("完成!")
    print(f"前 10 個最常見的 words:")
    for word, count in sorted_words[:10]:
        print(f"  {word}: {count}")


if __name__ == "__main__":
    main()
