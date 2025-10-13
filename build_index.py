#!/usr/bin/env python3
"""
Build index for consistency data to enable on-demand loading
"""

import json
import os

def build_index():
    # Load data
    print("Loading data...")
    with open('consistency_data/consistency_data.json', 'r', encoding='utf-8') as f:
        base_data = json.load(f)

    with open('consistency_data/consistency2_gpt-oss-reasoning.json', 'r', encoding='utf-8') as f:
        strategy_full = json.load(f)
        strategy_data = {item['idx']: item for item in strategy_full['metrics']}

    # Create output directory
    os.makedirs('consistency_data/items', exist_ok=True)

    # Create index
    index = []

    print("Building index and item files...")
    for i, base_item in enumerate(base_data):
        unique_id = base_item['unique_id']

        # Add to index
        index.append({
            'index': i,
            'unique_id': unique_id
        })

        # Merge data for this item
        strategy_item = strategy_data.get(unique_id, {})
        merged = {
            **base_item,
            'strategies_found': strategy_item.get('strategies_found', []),
            'cot_collection': strategy_item.get('cot_collection', {})
        }

        # Save individual item file
        item_file = f'consistency_data/items/{unique_id}.json'
        with open(item_file, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(base_data)} items")

    # Save index
    with open('consistency_data/index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Created index with {len(index)} items")
    print(f"Index file: consistency_data/index.json")
    print(f"Item files: consistency_data/items/")

if __name__ == '__main__':
    build_index()
