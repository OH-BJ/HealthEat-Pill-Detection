from ultralytics import YOLO, SAM
from PIL import Image
import numpy as np
import os

# 설정
dataset_dir = "data/yolo_seg/images/train/"
mask_dir = "data/yolo_seg/masks/train/"
os.makedirs(mask_dir, exist_ok=True)

yolo_model_path = "best.pt"  # 실제 YOLO seg 모델 경로
sam_model_path = "sam2.1_b.pt"  # SAM 모델 경로

yolo_model = YOLO(yolo_model_path)
sam_model = SAM(sam_model_path)

for img_file in os.listdir(dataset_dir):
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    img_path = os.path.join(dataset_dir, img_file)
    yolo_results = yolo_model(img_path)
    if len(yolo_results[0].boxes) == 0:
        print(f"No detections in {img_file}, skipping")
        continue
    
    bboxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    print(f"Detected {len(bboxes)} pills in {img_file}")
    
    orig_h, orig_w = yolo_results[0].orig_shape
    combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    for i, bbox in enumerate(bboxes):
        print(f"Processing pill {i+1}: bbox {bbox}")
        
        sam_results = sam_model(img_path, bboxes=[bbox])
        if len(sam_results[0].masks) > 0:
            mask = sam_results[0].masks.data[0].cpu().numpy()
            mask = (mask > 0.3).astype(np.uint8) * 255
        else:
            mask_idx = i % len(yolo_results[0].masks.data) if len(yolo_results[0].masks.data) > 0 else 0
            mask = yolo_results[0].masks.data[mask_idx].cpu().numpy()
            mask = (mask > 0.3).astype(np.uint8) * 255
        
        area = np.sum(mask == 255)
        if area < 100:
            print(f"Pill {i+1} mask area too small ({area}), skipping")
            continue
        
        combined_mask = np.maximum(combined_mask, mask)
    
    # 마스크를 RGBA로 변환 (알파 채널: 알약=0 고정, 배경=255 변형)
    alpha_mask = 255 - combined_mask  # 배경 = 255 (투명 변형), 알약 = 0 (불투명 고정)
    alpha_mask = np.stack([combined_mask, combined_mask, combined_mask, alpha_mask], axis=-1)  # RGBA (RGB=흑백, A=알파)
    alpha_mask = Image.fromarray(alpha_mask, mode='RGBA')
    
    mask_path = os.path.join(mask_dir, img_file.replace('.png', '_alpha_mask.png'))
    alpha_mask.save(mask_path)
    print(f"Alpha mask saved to {mask_path} (area: {np.sum(combined_mask == 255)} pixels)")

print("All alpha masks generated! Check data/yolo_seg/masks/train/ folder.")