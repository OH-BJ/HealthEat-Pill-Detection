# src/split_dataset.py

import os
import random
import shutil
from pathlib import Path

def split_dataset(
    img_dir='data/raw/raw_train_images',
    label_dir='data/raw/seg_labels/train_labels',
    output_dir='data/yolo',
    val_ratio=0.1,
    test_ratio=0.05,
    seed=42
):
    random.seed(seed)
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    assert img_dir.exists(), f"이미지 경로가 존재하지 않음: {img_dir}"
    assert label_dir.exists(), f"라벨 경로가 존재하지 않음: {label_dir}"

    all_images = sorted([f for f in img_dir.glob('*.png')])
    all_labels = {f.stem: f for f in label_dir.glob('*.txt')}
    print(f"총 이미지: {len(all_images)}, 총 라벨: {len(all_labels)}")

    # 라벨이 있는 이미지만 사용
    valid_images = [img for img in all_images if img.stem in all_labels]
    print(f"라벨 매칭된 이미지: {len(valid_images)}")

    random.shuffle(valid_images)
    n_total = len(valid_images)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)

    splits = {
        'train': valid_images[: n_total - n_val - n_test],
        'val': valid_images[n_total - n_val - n_test: n_total - n_test],
        'test': valid_images[n_total - n_test:],
    }

    for split, imgs in splits.items():
        img_out = output_dir / 'images' / split
        lbl_out = output_dir / 'labels' / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            lbl_path = all_labels.get(img_path.stem)
            if not lbl_path:
                continue
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

        print(f"{split}: {len(imgs)}장 저장 완료 ({img_out})")

    print("✅ 데이터셋 split 완료.")
    print(f"train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")

if __name__ == "__main__":
    split_dataset()
