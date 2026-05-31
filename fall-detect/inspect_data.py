import json, os
base = os.path.join(os.path.dirname(__file__), '..')

files = {
    'fall_data.jsonl': os.path.join(base, 'data', 'fall_data.jsonl'),
    'lying_data.jsonl': os.path.join(base, 'data', 'lying_data.jsonl'),
    'sitting_data.jsonl': os.path.join(base, 'data', 'sitting_data.jsonl'),
    'fall-csi.jsonl': os.path.join(base, 'data', 'fall-csi.jsonl'),
    'walk-csi.jsonl': os.path.join(base, 'data', 'walk-csi.jsonl'),
}

for name, path in files.items():
    print(f"\n=== {name} ===")
    with open(path) as f:
        lines = f.readlines()
    rec = json.loads(lines[0])
    print(f"Keys: {list(rec.keys())}")
    for k, v in rec.items():
        if isinstance(v, list):
            inner = v[0] if v else None
            if isinstance(inner, list):
                print(f"  {k}: list[{len(v)}][{len(inner)}] (2D)")
            else:
                print(f"  {k}: list[{len(v)}] (flat)")
        else:
            print(f"  {k}: {repr(v)}")
