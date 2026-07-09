#!/usr/bin/env python3
"""
clean_val_and_revalidate.py  —  量化数据泄露对 AP 的影响

做什么（三步，全自动）:
  1. 读 leakage_report.csv（上次 check_leakage_leafrot.py 生成的）
  2. 从验证集里剔除所有泄露图（1 组 MD5 完全重复 + 117 组近重复）
  3. 用现成权重在【原始验证集】和【干净验证集】上各跑一次 validation

不训练、不重新标注、不需要原图。就是拿同一个模型考两次试：
一次用有泄露的卷子，一次用干净的卷子。

用法（在 second paper 目录下）:
    py -m pip install ultralytics pyyaml
    py clean_val_and_revalidate.py

可选:
    py clean_val_and_revalidate.py --weights yolo11s.pt --dataset Leave_disease
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='Leave_disease')
ap.add_argument('--weights', default='yolo11s.pt')
ap.add_argument('--leakage', default='leakage_report.csv')
ap.add_argument('--imgsz', type=int, default=640)
args = ap.parse_args()

ROOT      = Path(args.dataset).resolve()
VAL_IMG   = ROOT / 'valid' / 'images'
VAL_LBL   = ROOT / 'valid' / 'labels'
DATA_YAML = ROOT / 'data.yaml'
CLEAN_DIR = ROOT / 'valid_clean'
WEIGHTS   = Path(args.weights).resolve()
LEAK_CSV  = Path(args.leakage).resolve()

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def die(msg):
    print(f'\n[FAIL] {msg}')
    sys.exit(1)


print('=' * 70)
print('  Dataset Integrity Audit — 量化泄露对 AP 的影响')
print('=' * 70)
for p, name in ((ROOT, '数据集目录'), (VAL_IMG, 'valid/images'),
                (VAL_LBL, 'valid/labels'), (DATA_YAML, 'data.yaml'),
                (WEIGHTS, '权重'), (LEAK_CSV, 'leakage_report.csv')):
    if not p.exists():
        die(f'{name} 不存在: {p}')
    print(f'  [OK] {name:20} {p}')
print('-' * 70)

# ---------- 步骤 1: 读泄露清单 ----------
leaked = set()
with open(LEAK_CSV, newline='', encoding='utf-8-sig') as f:
    rdr = csv.DictReader(f)
    if 'val_image' not in (rdr.fieldnames or []):
        die(f'leakage_report.csv 没有 val_image 列。实际列: {rdr.fieldnames}')
    for row in rdr:
        v = (row.get('val_image') or '').strip()
        if v:
            leaked.add(Path(v).stem)
print(f'STEP 1  读到 {len(leaked)} 张唯一的泄露 val 图')

# ---------- 步骤 2: 构建干净验证集 ----------
if CLEAN_DIR.exists():
    shutil.rmtree(CLEAN_DIR)
(CLEAN_DIR / 'images').mkdir(parents=True)
(CLEAN_DIR / 'labels').mkdir(parents=True)

all_imgs = [p for p in VAL_IMG.iterdir() if p.suffix.lower() in IMG_EXT]
if not all_imgs:
    die(f'{VAL_IMG} 里没有图片')

kept, removed, removed_names = 0, 0, []
for img in all_imgs:
    s = img.stem
    is_leaked = any(s.startswith(L) or L.startswith(s) for L in leaked)
    if is_leaked:
        removed += 1
        removed_names.append(s[:55])
        continue
    shutil.copy2(img, CLEAN_DIR / 'images' / img.name)
    lbl = VAL_LBL / (s + '.txt')
    if lbl.exists():
        shutil.copy2(lbl, CLEAN_DIR / 'labels' / lbl.name)
    kept += 1

print(f'STEP 2  验证集 {len(all_imgs)} 张 -> 保留 {kept}, 剔除 {removed}')
if removed == 0:
    die('剔除 0 张 —— 文件名匹配失败。把 leakage_report.csv 前几行发我。')
if removed > len(all_imgs) * 0.8:
    print(f'  [WARN] 剔除了 {removed/len(all_imgs)*100:.0f}%，匹配可能过宽')
print(f'  剔除样例: {removed_names[:3]}')
print('-' * 70)

# ---------- 步骤 3: 两份绝对路径 yaml（口径一致） ----------
import yaml

with open(DATA_YAML, encoding='utf-8') as f:
    y = yaml.safe_load(f)


def build(val_dir):
    d = dict(y)
    d.pop('path', None)                       # 去掉 path 根键，避免与绝对路径冲突
    d['val'] = str(Path(val_dir).resolve()).replace('\\', '/')
    for k in ('train', 'test'):
        if k in d and not os.path.isabs(str(d[k])):
            p = (ROOT / str(d[k]).lstrip('./')).resolve()
            if p.exists():
                d[k] = str(p).replace('\\', '/')
    return d


YAML_ORIG  = Path('data_orig_abs.yaml').resolve()
YAML_CLEAN = Path('data_clean.yaml').resolve()
with open(YAML_ORIG, 'w', encoding='utf-8') as f:
    yaml.safe_dump(build(VAL_IMG), f, allow_unicode=True)
with open(YAML_CLEAN, 'w', encoding='utf-8') as f:
    yaml.safe_dump(build(CLEAN_DIR / 'images'), f, allow_unicode=True)

# ---------- 步骤 4: 两次 validation ----------
from ultralytics import YOLO


def run(data_yaml, tag):
    print(f'\n{"=" * 70}\n[{tag}]\n{"=" * 70}')
    m = YOLO(str(WEIGHTS))
    r = m.val(data=str(data_yaml), imgsz=args.imgsz, split='val',
              verbose=False, plots=False)
    names = r.names if hasattr(r, 'names') else m.names
    out = {}
    try:
        ap50 = r.box.ap50
        idx = r.box.ap_class_index
        print(f'\n[{tag}] Per-class AP@0.5:')
        for i, c in enumerate(idx):
            out[names[c]] = float(ap50[i])
            print(f'    {names[c]:<16} AP@0.5 = {ap50[i]:.3f}')
        out['__mAP50__'] = float(r.box.map50)
        out['__mAP50_95__'] = float(r.box.map)
        print(f'    {"mAP@0.5 (all)":<16} = {r.box.map50:.3f}')
        print(f'    {"mAP@0.5:0.95":<16} = {r.box.map:.3f}')
    except Exception as e:
        print(f'  读取 per-class AP 失败: {e}')
        print(f'  results_dict: {getattr(r, "results_dict", None)}')
    return out


orig  = run(YAML_ORIG,  'A — ORIGINAL (含泄露)')
clean = run(YAML_CLEAN, 'B — CLEANED (去泄露)')

# ---------- 步骤 5: 对照表 ----------
print(f'\n{"=" * 70}')
print('  Table A vs Table B — 泄露对 AP 的影响')
print('=' * 70)
print(f'  {"Class":<18}{"A 含泄露":>12}{"B 清洗后":>12}{"Δ":>10}{"虚高%":>10}')
print('  ' + '-' * 62)
for k in [k for k in orig if not k.startswith('__')]:
    a = orig.get(k, float('nan'))
    b = clean.get(k, float('nan'))
    d = a - b
    infl = (d / b * 100) if (b == b and b > 0) else float('nan')
    infl_s = f'{infl:+.1f}%' if infl == infl else '—'
    print(f'  {k:<18}{a:>12.3f}{b:>12.3f}{d:>+10.3f}{infl_s:>10}')

for lbl, key in (('mAP@0.5', '__mAP50__'), ('mAP@0.5:0.95', '__mAP50_95__')):
    a, b = orig.get(key), clean.get(key)
    if a and b:
        print(f'  {lbl:<18}{a:>12.3f}{b:>12.3f}{a-b:>+10.3f}{(a-b)/b*100:>+9.1f}%')

print(f'\n  验证集: {len(all_imgs)} 张 -> {kept} 张 (剔除 {removed} 张泄露图)')
print('\n  把以上全部输出发给 Claude，写进论文的 Dataset Integrity Audit。')
print('=' * 70)
