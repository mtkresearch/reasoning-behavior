"""Test HTML generation"""

from attention_visual_templates import build_complete_html, get_html_header

# Test data
test_instances = [
    {
        "question": "Test question",
        "ground_truth": "42",
        "is_correct": True,
        "tokens": ["test", "token"],
        "attention_maps": [[0.5, 0.5]]
    }
]

import json
instances_json = json.dumps(test_instances, ensure_ascii=False)

instance_options = '<option value="0">Instance 0 (CORRECT)</option>'

# Generate HTML
html = build_complete_html(instances_json, instance_options)

# Save to file
with open('/tmp/test_attention.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Generated HTML saved to /tmp/test_attention.html")
print("\nFirst 500 chars:")
print(html[:500])
print("\n\nLast 500 chars:")
print(html[-500:])
