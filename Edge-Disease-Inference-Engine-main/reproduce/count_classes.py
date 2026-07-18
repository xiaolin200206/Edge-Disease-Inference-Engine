"""
统计 Leave_disease 数据集(YOLO格式)里每个类别的instance数量。
分别统计 train 和 valid,并额外核对 Early_Blight / early_blight 这两个
疑似大小写重复的类别。

用法:
    直接双击运行,或者在命令行里:
        python count_classes.py

如果路径不对,改下面 BASE_DIR 这一行就好。
"""

import os
from collections import Counter

# ====== 按需修改这里的路径 ======
BASE_DIR = r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper\second paper\Leave_disease"
# ================================

NAMES = [
    'Algal_leave', 'Leaf_rot', 'Phomopsis',
    'Pink_Disease', 'early_blight', 'root_disease', 'Anthracnose'
]


def count_split(split_name: str):
    """统计 train 或 valid 里每个类别的 instance 数(以 bbox 行数为单位)。
    同时统计每个类别出现在多少张不同的图片里(image-level 计数)。"""
    label_dir = os.path.join(BASE_DIR, split_name, "labels")

    if not os.path.isdir(label_dir):
        print(f"  [跳过] 找不到目录: {label_dir}")
        return None

    instance_counter = Counter()   # 每个类别总共出现了几次(bbox数)
    image_counter = Counter()      # 每个类别出现在几张不同的图里
    empty_files = 0
    bad_lines = 0
    total_files = 0

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        total_files += 1
        fpath = os.path.join(label_dir, fname)

        classes_in_this_file = set()
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            parts = line.split()
            try:
                cls_id = int(parts[0])
                cls_name = NAMES[cls_id]
            except (ValueError, IndexError):
                bad_lines += 1
                continue
            instance_counter[cls_name] += 1
            classes_in_this_file.add(cls_name)

        for cls_name in classes_in_this_file:
            image_counter[cls_name] += 1

    print(f"\n  总标注文件数: {total_files}  (空标注文件: {empty_files}, 异常行: {bad_lines})")
    print(f"  {'类别':<15} {'instance数(bbox)':>15} {'出现的图片数':>12}")
    print(f"  {'-'*15} {'-'*15} {'-'*12}")
    for name in NAMES:
        inst = instance_counter.get(name, 0)
        img = image_counter.get(name, 0)
        print(f"  {name:<15} {inst:>15} {img:>12}")

    return instance_counter, image_counter


def main():
    print("=" * 60)
    print("Leave_disease 数据集类别分布统计")
    print("=" * 60)

    results = {}
    for split in ["train", "valid", "test"]:
        print(f"\n[{split}]")
        r = count_split(split)
        if r is not None:
            results[split] = r

    # ---- 核对 Early_Blight / early_blight 大小写重复问题 ----
    print("\n" + "=" * 60)
    print("Early_Blight vs early_blight 核对(疑似大小写重复类)")
    print("=" * 60)
    for split, (inst_counter, img_counter) in results.items():
        eb_upper = inst_counter.get('Early_Blight', 0)
        eb_lower = inst_counter.get('early_blight', 0)
        print(f"  [{split}] Early_Blight: {eb_upper} instances | "
              f"early_blight: {eb_lower} instances | "
              f"合计若合并为一类: {eb_upper + eb_lower}")

    # ---- 汇总:哪些类别整体样本量过少,可能是AP=0.00的元凶 ----
    print("\n" + "=" * 60)
    print("汇总:各类别在 train+valid 合计的 instance 数(从小到大排序)")
    print("=" * 60)
    combined = Counter()
    for split, (inst_counter, _) in results.items():
        if split == "test":
            continue
        combined.update(inst_counter)

    for name, count in sorted(combined.items(), key=lambda x: x[1]):
        flag = "  <-- 样本量偏少,警惕" if count < 20 else ""
        print(f"  {name:<15} {count:>6}{flag}")

    print("\n完成。可以把上面这段输出截图或复制给我,我帮你分析下一步。")


if __name__ == "__main__":
    main()
