"""Dataset verification utility for AI_MICROSCOPE

Checks a local dataset directory for expected class folders (based on
`model/class_indices.json`), counts images per class, reports missing
classes and creates a CSV summary in `exports/dataset_report.csv`.

Usage:
    PYTHONPATH=. python3 scripts/dataset_verify.py --dataset data/raw

If you don't have the dataset yet, follow `scripts/download_dataset_instructions.sh`.
"""
from pathlib import Path
import argparse
import json
import csv
import sys

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def find_images(path: Path):
    for p in path.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def load_class_indices(path: Path):
    if not path.exists():
        print(f"Warning: class indices not found at {path}")
        return []
    try:
        j = json.loads(path.read_text(encoding='utf-8'))
        # values expected to be class names (strings)
        return [v for k, v in sorted(j.items(), key=lambda x: int(x[0]))]
    except Exception as e:
        print(f"Error reading class indices: {e}")
        return []


def verify(dataset_path: Path, class_indices_path: Path, out_csv: Path):
    classes = load_class_indices(class_indices_path)
    report = []
    total = 0

    # If folder-per-class layout
    for cls in classes:
        candidates = []
        # try exact name, underscore, and lower variants
        for cand in (cls, cls.replace('_', ' '), cls.replace('_', '').lower()):
            p = dataset_path / cand
            if p.exists() and p.is_dir():
                candidates.append(p)
        count = 0
        if candidates:
            for c in candidates:
                count += sum(1 for _ in find_images(c))
        else:
            # fallback: search for files whose name contains class substring
            for img in find_images(dataset_path):
                if cls.lower() in img.name.lower():
                    count += 1
        report.append({'class': cls, 'count': count})
        total += count

    # also count any images not matched to classes
    all_imgs = list(find_images(dataset_path))
    unmatched = len(all_imgs) - total

    # write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['class', 'count'])
        writer.writeheader()
        for r in report:
            writer.writerow(r)
        writer.writerow({'class': 'TOTAL', 'count': total})
        writer.writerow({'class': 'UNMATCHED', 'count': unmatched})

    # print summary
    print(f"Dataset verification report written to: {out_csv}")
    print(f"Total images matched: {total}")
    print(f"Unmatched images (present but not assigned): {unmatched}")
    missing = [r['class'] for r in report if r['count'] == 0]
    if missing:
        print("Missing classes (0 images):")
        for m in missing:
            print(" - ", m)
    else:
        print("All classes have >=1 images (per current detection heuristics).")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='data/raw', help='Path to dataset root')
    p.add_argument('--class-indices', default='model/class_indices.json', help='Path to class indices JSON')
    p.add_argument('--out', default='exports/dataset_report.csv', help='CSV output path')
    args = p.parse_args()

    ds = Path(args.dataset)
    if not ds.exists():
        print(f"Dataset path {ds} does not exist. Create it and place images (or run download instructions).", file=sys.stderr)
        sys.exit(2)

    verify(ds, Path(args.class_indices), Path(args.out))
