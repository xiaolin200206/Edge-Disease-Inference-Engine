"""
remove_class_and_reindex.py

Fully removes "Early_Blight" (old index 1) from the dataset's class
taxonomy: shrinks data.yaml (nc 8 -> 7, names list updated) and
re-indexes every label file so class ids stay contiguous.

Old index -> New index mapping:
    0 Algal_leave    -> 0 Algal_leave
    1 Early_Blight   -> (removed; should have 0 instances already)
    2 Leaf_rot       -> 1 Leaf_rot
    3 Phomopsis      -> 2 Phomopsis
    4 Pink_Disease   -> 3 Pink_Disease
    5 early_blight   -> 4 early_blight
    6 root_disease   -> 5 root_disease
    7 Anthracnose    -> 6 Anthracnose

SAFETY: Always makes a fresh backup before touching anything. Refuses
to run if the backup folder already exists, to avoid overwriting one.

Usage:
    python remove_class_and_reindex.py <dataset_root>

Example:
    python remove_class_and_reindex.py Leave_disease
"""

import os
import sys
import glob
import shutil

REMOVE_NAME = "Early_Blight"
SPLIT_DIRS = ["train", "valid", "test"]


def backup_dataset(dataset_root):
    backup_root = dataset_root.rstrip("\\/") + "_backup_before_reindex"
    if os.path.exists(backup_root):
        print(f"ERROR: backup folder already exists at:\n  {backup_root}")
        print("Refusing to proceed to avoid overwriting a previous backup.")
        sys.exit(1)
    print(f"Backing up dataset to: {backup_root}")
    shutil.copytree(dataset_root, backup_root)
    print("Backup complete.\n")


def update_data_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        content = f.read()

    # Find the names: [...] line and parse it out manually (simple, known format)
    import re
    m = re.search(r"names:\s*\[([^\]]*)\]", content)
    if not m:
        print("ERROR: could not find a 'names: [...]' line in data.yaml")
        sys.exit(1)

    raw_names = m.group(1)
    names = [n.strip().strip("'").strip('"') for n in raw_names.split(",")]

    if REMOVE_NAME not in names:
        print(f"ERROR: '{REMOVE_NAME}' not found in names list: {names}")
        sys.exit(1)

    old_index = names.index(REMOVE_NAME)
    new_names = [n for n in names if n != REMOVE_NAME]
    new_nc = len(new_names)

    new_names_str = "[" + ", ".join(f"'{n}'" for n in new_names) + "]"
    new_content = re.sub(r"names:\s*\[[^\]]*\]", f"names: {new_names_str}", content)
    new_content = re.sub(r"nc:\s*\d+", f"nc: {new_nc}", new_content)

    with open(yaml_path, "w") as f:
        f.write(new_content)

    print(f"Updated {yaml_path}")
    print(f"  Removed '{REMOVE_NAME}' (was index {old_index})")
    print(f"  nc: {len(names)} -> {new_nc}")
    print(f"  names: {new_names}\n")

    return old_index, len(names)


def remap_labels(dataset_root, old_index, old_nc):
    # Build old_id -> new_id mapping: remove old_index, shift everything after it down by 1
    id_map = {}
    for old_id in range(old_nc):
        if old_id == old_index:
            continue  # should have 0 instances; if found, we error out below
        new_id = old_id if old_id < old_index else old_id - 1
        id_map[old_id] = new_id

    total_files_changed = 0
    total_lines_changed = 0
    found_removed_class_instances = 0

    for split in SPLIT_DIRS:
        label_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.isdir(label_dir):
            print(f"[skip] no such folder: {label_dir}")
            continue

        files = glob.glob(os.path.join(label_dir, "*.txt"))
        files_changed = 0
        lines_changed = 0

        for fpath in files:
            with open(fpath, "r") as f:
                lines = f.readlines()

            new_lines = []
            file_changed = False
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    new_lines.append(line)
                    continue
                parts = stripped.split()
                old_id = int(parts[0])

                if old_id == old_index:
                    found_removed_class_instances += 1
                    print(f"  WARNING: {fpath} still has an instance of "
                          f"the removed class (id {old_index}) -- skipping this line!")
                    file_changed = True
                    continue  # drop this line entirely

                new_id = id_map[old_id]
                if new_id != old_id:
                    parts[0] = str(new_id)
                    file_changed = True
                    lines_changed += 1
                new_lines.append(" ".join(parts) + "\n")

            if file_changed:
                with open(fpath, "w") as f:
                    f.writelines(new_lines)
                files_changed += 1

        print(f"[{split}] files changed: {files_changed} | lines re-indexed: {lines_changed}")
        total_files_changed += files_changed
        total_lines_changed += lines_changed

    print(f"\nTotal files changed:   {total_files_changed}")
    print(f"Total lines re-indexed: {total_lines_changed}")
    if found_removed_class_instances:
        print(f"\n*** WARNING: found and DROPPED {found_removed_class_instances} "
              f"leftover instance(s) of the removed class. ***")
        print("This should not happen if the casing-merge step ran first --")
        print("double check your earlier merge_casing_duplicate.py run.")
    else:
        print("No leftover instances of the removed class found (as expected).")


def main():
    if len(sys.argv) != 2:
        print("Usage: python remove_class_and_reindex.py <dataset_root>")
        sys.exit(1)

    dataset_root = sys.argv[1]
    if not os.path.isdir(dataset_root):
        print(f"ERROR: folder not found: {dataset_root}")
        sys.exit(1)

    yaml_path = os.path.join(dataset_root, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"ERROR: data.yaml not found at: {yaml_path}")
        sys.exit(1)

    backup_dataset(dataset_root)
    old_index, old_nc = update_data_yaml(yaml_path)
    remap_labels(dataset_root, old_index, old_nc)

    print("\nDone. Re-run count_classes.py to sanity-check the new distribution,")
    print("then re-run training -- class indices have shifted, so this is NOT")
    print("optional, you must retrain (a stale best.pt would misinterpret ids).")


if __name__ == "__main__":
    main()
