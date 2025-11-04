import os
import random
from PIL import Image
import numpy as np
from pathlib import Path
import shutil

# 설정
original_images_dir = "data/yolo_seg/images/train/"  # 원본 알약 이미지 (1500개)
masks_dir = "data/yolo_seg/masks/train/"  # 마스크 (img.png → img_mask.png)
labels_dir = "data/yolo_seg/labels/train/"  # 원본 라벨 (.txt)
background_base_dir = "data/yolo_seg/background/"  # 배경 폴더
train_aug_base_dir = "data/yolo_seg/train_aug/"  # 증강 저장 (6종별 폴더)
train_aug_labels_base_dir = "data/yolo_seg/labels/train_aug/"  # 증강 라벨 (6종별 폴더)

# 6종 배경 타입
background_types = ["steel", "glass", "marble_floor", "fabric", "plastic", "wooden"]

# 폴더 생성
os.makedirs(train_aug_base_dir, exist_ok=True)
os.makedirs(train_aug_labels_base_dir, exist_ok=True)

# 원본 이미지 목록 (이름순)
original_images = sorted([f for f in os.listdir(original_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

num_to_test = len(original_images)  # 테스트로 10장만 (전체로 하려면 len(original_images)로 변경)
print(f"Found {len(original_images)} original images. Testing with {num_to_test} images, generating 6x = {num_to_test * 6} augmented images sequentially...")

for bg_type in background_types:
    bg_dir = os.path.join(background_base_dir, bg_type)
    aug_type_dir = os.path.join(train_aug_base_dir, bg_type)
    aug_labels_type_dir = os.path.join(train_aug_labels_base_dir, bg_type)
    os.makedirs(aug_type_dir, exist_ok=True)
    os.makedirs(aug_labels_type_dir, exist_ok=True)
    
    bg_images = sorted([f for f in os.listdir(bg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])  # 배경 30장 목록
    if len(bg_images) < 30:
        print(f"Warning: Only {len(bg_images)} background images in {bg_type}. Need 30.")
        continue
    
    for i in range(num_to_test):
        original_name = Path(original_images[i]).stem  # e.g., K-001900-...
        img_file = original_images[i]
        original_img_path = os.path.join(original_images_dir, img_file)
        mask_file = original_name + '_mask.png'
        mask_path = os.path.join(masks_dir, mask_file)
        label_file = original_name + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(mask_path):
            print(f"No mask for {img_file}, skipping")
            continue
        
        # 원본 알약 이미지 로드
        original_img = Image.open(original_img_path).convert('RGBA')
        
        # 마스크 로드 (alpha 채널로 변환)
        mask_img = Image.open(mask_path).convert('L')
        alpha_mask = np.array(mask_img) / 255.0  # 0-1 (알약=1, 배경=0)
        
        # 랜덤 배경 선택
        random_bg_file = random.choice(bg_images)
        random_bg_path = os.path.join(bg_dir, random_bg_file)
        background_img = Image.open(random_bg_path).convert('RGB').resize(original_img.size)
        
        # numpy 배열로 합성 (PIL 대신 – 빠름)
        bg_array = np.array(background_img)
        original_array = np.array(original_img)
        
        # 배경에 알약 합성 (alpha 블렌드)
        alpha_channel = alpha_mask[:, :, np.newaxis]  # (H, W, 1)
        blended = (1 - alpha_channel) * bg_array + alpha_channel * original_array[:, :, :3]  # 배경 * (1-alpha) + 알약 * alpha (RGB만)
        blended = np.concatenate([blended, original_array[:, :, 3:4]], axis=2)  # RGBA 유지
        
        blended_img = Image.fromarray(blended.astype(np.uint8)).convert('RGB')  # RGB로 저장
        
        # 새 증강 이미지 저장
        new_aug_name = f"{original_name}_{bg_type}.png"
        new_aug_path = os.path.join(aug_type_dir, new_aug_name)
        blended_img.save(new_aug_path)
        
        # 파일 생성 확인
        if os.path.exists(new_aug_path):
            print(f"Type {bg_type}: Created {new_aug_name} (random bg: {random_bg_file})")
        else:
            print(f"Type {bg_type}: Failed to create {new_aug_name}")
            continue
        
        # 라벨 복사 (내용 동일)
        if os.path.exists(label_path):
            new_label_name = f"{original_name}_{bg_type}.txt"
            new_label_path = os.path.join(aug_labels_type_dir, new_label_name)
            shutil.copy2(label_path, new_label_path)
            print(f"Type {bg_type}: Copied label {new_label_name}")
    
    print(f"Type {bg_type}: {num_to_test} files augmented. Check train_aug/{bg_type}/ and labels/train_aug/{bg_type}/.")

print(f"All 6 types augmented for {num_to_test} images. Total augmented: {num_to_test * 6} files.")
print("Test complete! For full (1500 images), change num_to_test = len(original_images) and rerun.")
print("Next: Update data.yaml and train!")