#!/usr/bin/env python3
"""
check_leakage_leafrot.py
Paper 2 重塑 —— 数据泄露 + Leaf_rot 分布双重检查

用途：
  1. 检查 train / val 之间是否存在完全重复(MD5)或近似重复(感知哈希)的图像
     —— 特别关注 Phomopsis 那张 IMG-20251006-WA0002 是否与训练集重叠
  2. 追溯 Leaf_rot 的 26 个 instance 落在哪几张图上，判断是否连拍/同视角

依赖：
  pip install pillow imagehash

用法（在能看到数据集图片和标注的目录下运行）：
  python3 check_leakage_leafrot.py \
      --train_images path/to/train/images \
      --val_images   path/to/val/images \
      --train_labels path/to/train/labels \
      --val_labels   path/to/val/labels \
      --classes_yaml path/to/data.yaml   # 可选，用于把 class id 映射成类名

输出：
  - leakage_report.csv  : 所有跨集(train↔val)近似重复对，含感知汉明距离
  - leafrot_distribution.txt : Leaf_rot / Phomopsis 每个类别的 instance→图片分布
"""

import argparse, hashlib, os, glob, sys
from collections import defaultdict

try:
    from PIL import Image
    import imagehash
except ImportError:
    print("请先安装依赖: pip install pillow imagehash")
    sys.exit(1)

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def list_images(d):
    if not d or not os.path.isdir(d):
        return []
    out = []
    for e in IMG_EXT:
        out += glob.glob(os.path.join(d, f'*{e}'))
        out += glob.glob(os.path.join(d, f'*{e.upper()}'))
    return sorted(set(out))

def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()

def phash(path):
    try:
        return imagehash.phash(Image.open(path))
    except Exception:
        return None

def load_class_names(yaml_path):
    if not yaml_path or not os.path.exists(yaml_path):
        return {}
    names = {}
    try:
        import yaml
        y = yaml.safe_load(open(yaml_path))
        n = y.get('names', {})
        if isinstance(n, dict):
            names = {int(k): v for k, v in n.items()}
        elif isinstance(n, list):
            names = {i: v for i, v in enumerate(n)}
    except Exception:
        # 手动粗解析
        for line in open(yaml_path):
            line = line.strip()
            if line and line[0].isdigit() and ':' in line:
                k, v = line.split(':', 1)
                try: names[int(k)] = v.strip().strip("'\"")
                except: pass
    return names

def check_leakage(train_imgs, val_imgs, thresh=10):
    print(f"\n{'='*60}\n[1] 数据泄露检查: train({len(train_imgs)}) vs val({len(val_imgs)})\n{'='*60}")
    # 精确重复
    train_md5 = {md5(f): f for f in train_imgs}
    exact = []
    for vf in val_imgs:
        m = md5(vf)
        if m in train_md5:
            exact.append((vf, train_md5[m]))
    if exact:
        print(f"\n⚠️  发现 {len(exact)} 组【完全相同】(MD5一致)的 train↔val 图像 —— 这是严格的数据泄露:")
        for v, t in exact:
            print(f"    VAL {os.path.basename(v)}  ==  TRAIN {os.path.basename(t)}")
    else:
        print("\n✓ 无 MD5 完全重复。")

    # 近似重复
    train_ph = [(f, phash(f)) for f in train_imgs]
    train_ph = [(f, h) for f, h in train_ph if h is not None]
    near = []
    for vf in val_imgs:
        vh = phash(vf)
        if vh is None: continue
        for tf, th in train_ph:
            d = vh - th
            if d < thresh:
                near.append((vf, tf, d))
    near.sort(key=lambda x: x[2])
    if near:
        print(f"\n⚠️  发现 {len(near)} 组【近似重复】(感知汉明距离<{thresh})的 train↔val 图像:")
        for v, t, d in near:
            print(f"    VAL {os.path.basename(v)}  ≈  TRAIN {os.path.basename(t)}  (距离={d})")
    else:
        print(f"\n✓ 无感知近似重复(阈值<{thresh})。")

    with open('leakage_report.csv', 'w') as f:
        f.write("type,val_image,train_image,hamming_distance\n")
        for v, t in exact:
            f.write(f"exact,{os.path.basename(v)},{os.path.basename(t)},0\n")
        for v, t, d in near:
            f.write(f"near,{os.path.basename(v)},{os.path.basename(t)},{d}\n")
    print("\n→ 明细已写入 leakage_report.csv")
    return exact, near

def analyze_class_distribution(label_dir, class_names, targets):
    """统计每个目标类别的 instance 落在哪些图片上。"""
    print(f"\n{'='*60}\n[2] 类别→图片分布 (labels: {label_dir})\n{'='*60}")
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    # class_id -> {image_stem -> instance_count}
    cls_img = defaultdict(lambda: defaultdict(int))
    for lf in label_files:
        stem = os.path.splitext(os.path.basename(lf))[0]
        for line in open(lf):
            parts = line.split()
            if not parts: continue
            try: cid = int(parts[0])
            except: continue
            cls_img[cid][stem] += 1

    out_lines = []
    for cid in sorted(cls_img):
        name = class_names.get(cid, f'class_{cid}')
        total = sum(cls_img[cid].values())
        nimg = len(cls_img[cid])
        flag = ""
        if nimg == 1:
            flag = "  ⚠️ 全部来自单张图!"
        elif total / max(nimg, 1) > 8:
            flag = "  ⚠️ 每图instance密度极高,疑似连拍/同源"
        header = f"\n[{name}] (id={cid}): {total} instances 分布于 {nimg} 张图{flag}"
        out_lines.append(header)
        print(header)
        for stem, cnt in sorted(cls_img[cid].items(), key=lambda x: -x[1]):
            line = f"    {stem}: {cnt} instances"
            out_lines.append(line)
            print(line)

    with open('leafrot_distribution.txt', 'w') as f:
        f.write("\n".join(out_lines))
    print("\n→ 完整分布已写入 leafrot_distribution.txt")
    print("\n提示: 重点看 Phomopsis(应全部来自 IMG-20251006-WA0002 单图) 和 Leaf_rot(26 instances / 5 图)")
    print("      若 Leaf_rot 的 5 张图文件名是连续编号或时间相近,基本可判定为连拍。")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_images', required=True)
    ap.add_argument('--val_images', required=True)
    ap.add_argument('--train_labels', default=None)
    ap.add_argument('--val_labels', default=None)
    ap.add_argument('--classes_yaml', default=None)
    ap.add_argument('--near_thresh', type=int, default=10,
                    help='感知汉明距离阈值,<该值视为近似重复(默认10)')
    args = ap.parse_args()

    class_names = load_class_names(args.classes_yaml)

    train_imgs = list_images(args.train_images)
    val_imgs = list_images(args.val_images)
    if not train_imgs or not val_imgs:
        print("⚠️ 未找到图片,检查 --train_images / --val_images 路径")
        sys.exit(1)

    exact, near = check_leakage(train_imgs, val_imgs, args.near_thresh)

    if args.val_labels:
        analyze_class_distribution(args.val_labels, class_names,
                                   ['Phomopsis', 'Leaf_rot'])

    print(f"\n{'='*60}\n结论提示:")
    if exact:
        print("  ✗ 存在严格泄露(MD5重复) —— 论文 Section 4.2.1 与 Limitation 必须严肃承认")
    elif near:
        print("  △ 存在近似重复 —— 需在 Limitation 中说明 early-stage random split 缺陷")
    else:
        print("  ✓ 未检出 train↔val 泄露 —— 可用'per-class instance≠scene diversity'框架温和陈述")
    print(f"{'='*60}\n把这段终端输出(或两个生成文件)发我,我据此定稿措辞。")
