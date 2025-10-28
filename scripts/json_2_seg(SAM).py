import json
import os
import cv2
import numpy as np
from ultralytics import SAM
from pathlib import Path

# 동적 경로 설정 (스크립트 위치 기준)
script_dir = Path(__file__).resolve().parent  # data/processed/
base_dir = script_dir.parent  # data/
JSON_FILE = base_dir / "processed" / "train_annotations_integrated.json"
TRAIN_IMAGES_DIR = base_dir / "raw" / "train_images"
SEG_PROC_DIR = base_dir / "raw" / "seg_proc"

# 출력 폴더 생성
os.makedirs(SEG_PROC_DIR, exist_ok=True)

# 경로 확인 & 로그
print(f"스크립트 위치: {script_dir}")
print(f"JSON 파일: {JSON_FILE} (존재: {JSON_FILE.exists()})")
print(f"이미지 폴더: {TRAIN_IMAGES_DIR} (존재: {TRAIN_IMAGES_DIR.exists()})")
print(f"Seg 출력 폴더: {SEG_PROC_DIR} (존재: {SEG_PROC_DIR.exists()})")

if not JSON_FILE.exists():
    raise FileNotFoundError(f"JSON 파일을 찾을 수 없어요: {JSON_FILE}. data/processed/에 확인하세요.")

# 1. JSON 로드 & 데이터 그룹화
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 카테고리 매핑 (e.g., 1899 → 'K-001900')
categories = {cat['id']: cat['name'].replace(' ', '_') for cat in data['categories']}

# 이미지 & 어노테이션 딕셔너리
images = {img['id']: img for img in data['images']}
anns_by_image = {}  # image_id → list of anns (멀티 알약 지원)
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in anns_by_image:
        anns_by_image[img_id] = []
    anns_by_image[img_id].append(ann)

# 클래스 ID 목록 (전체 클래스 처리)
all_class_ids = list(set(ann['category_id'] for ann in data['annotations']))
print(f"처리할 클래스 수: {len(all_class_ids)} (e.g., {all_class_ids[:3]})")

# SAM 모델 로드 (sam_l 복귀: 정확도 ↑)
try:
    sam = SAM('sam_l.pt')  # 원본 크기 최적화
except Exception as e:
    print(f"SAM 로드 오류: {e}. 'sam_b.pt' 시도하세요.")
    exit(1)

# 2. 이미지별 처리 (멀티 알약 지원)
processed_count = 0
skipped_count = 0
PAD_RATIO = 0.05  # 패딩 줄임 (5% – bbox 과도 적용 방지)
AREA_THRESHOLD = 30  # 작은 pill 허용

for img_id, img_info in images.items():
    if img_id not in anns_by_image: 
        skipped_count += 1
        continue
    
    file_name = img_info['file_name']
    img_path = TRAIN_IMAGES_DIR / file_name
    if not img_path.exists():
        print(f"이미지 없음 (스킵): {img_path}")
        skipped_count += 1
        continue
    
    # 이미지 로드 (원본 크기 유지: 976x1280)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"이미지 로드 실패 (스킵): {img_path}")
        skipped_count += 1
        continue
    
    h, w = img.shape[:2]  # 원본 976x1280 (h x w)
    txt_lines = []  # 멀티 객체용 .txt 라인들
    
    # 이미지 내 어노테이션(알약)별 SAM 처리
    for ann in anns_by_image[img_id]:
        class_id = ann['category_id']
        class_name = categories.get(class_id, f"Class_{class_id}")
        print(f"처리 중: {file_name} - {class_name} (ID: {class_id})")
        
        bbox = ann.get('bbox', [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            print(f"  → Skipped invalid bbox for ann {ann.get('id', 'unknown')}: {bbox}")
            continue
        
        # BBox 패딩: 최소 확장으로 prompt 강화
        pad_w, pad_h = bbox[2] * PAD_RATIO, bbox[3] * PAD_RATIO
        x1 = max(0, bbox[0] - pad_w / 2)
        y1 = max(0, bbox[1] - pad_h / 2)
        x2 = min(w, bbox[0] + bbox[2] + pad_w / 2)
        y2 = min(h, bbox[1] + bbox[3] + pad_h / 2)
        box_prompt = np.array([x1, y1, x2, y2])
        
        # Point Prompt: bbox 중심 포인트 (원형 알약 경계 강화)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        point_prompt = np.array([[center_x, center_y]])
        
        # SAM 예측 (box + point prompt)
        results = sam(img, bboxes=box_prompt.reshape(1, -1), points=point_prompt, save=False)
        if len(results[0].masks.data) == 0:
            print(f"  → 마스크 생성 실패 (스킵): {class_name}")
            continue
        
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
        print(f"  → 마스크 생성 성공: {class_name} (area: {np.sum(mask > 0)} 픽셀)")
        
        # 마스크 저장: 클래스별 별도 PNG (오버레이: 원본 + 녹색 마스크)
        overlay = img.copy()
        overlay[mask > 0] = [0, 255, 0]  # 녹색 표시
        mask_path = SEG_PROC_DIR / f"{file_name.replace('.png', f'_{class_id}_mask.png')}"
        cv2.imwrite(str(mask_path), overlay)
        print(f"  → Mask PNG 저장: {mask_path}")
        
        # 마스크 → Polygon (YOLO 형식) – 원본 정규화
        orig_h, orig_w = img_info['height'], img_info['width']
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < AREA_THRESHOLD:
                print(f"  → 작은 마스크 스킵: {class_name}")
                continue
            polygon = largest_contour.flatten().tolist()
            
            # 정규화 (원본 w/h)
            norm_polygon = [coord / orig_w if i % 2 == 0 else coord / orig_h for i, coord in enumerate(polygon)]
            txt_line = f"{class_id} {' '.join(map(str, norm_polygon))}\n"
            txt_lines.append(txt_line)
            print(f"  → Polygon 추가: {len(norm_polygon)//2} 포인트")
        
        processed_count += 1
        if processed_count % 5 == 0:
            print(f"진행: {processed_count}개 어노테이션 처리됨")
    
    # .txt 파일 저장 (이미지당 하나, 멀티 라인)
    if txt_lines:
        txt_path = SEG_PROC_DIR / f"{file_name.replace('.png', '.txt')}"
        with open(txt_path, 'w') as f:
            f.writelines(txt_lines)
        print(f"저장됨: {txt_path} ({len(txt_lines)} 객체)")

print(f"완료! 총 처리: {processed_count}개 어노테이션. 스킵: {skipped_count}개. data/raw/seg_proc 폴더 확인하세요.")
print("팁: PAD_RATIO=0.05로 bbox 과도 적용 ↓. area 로그로 마스크 품질 확인 – 4개 알약 PNG 재테스트!")