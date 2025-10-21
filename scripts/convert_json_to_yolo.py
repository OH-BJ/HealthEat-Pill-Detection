import os
import json
import glob
import shutil

# 원본 Category ID 리스트
ORIGINAL_CATEGORY_IDS = [
    1899, 2482, 3350, 3482, 3543, 3742, 3831, 4377, 4542, 5093, 5885, 6191, 6562, 10220, 
    12080, 12246, 12419, 12777, 13394, 13899, 16231, 16261, 16547, 16550, 16687, 18109, 
    18146, 18356, 19231, 19551, 19606, 19860, 20013, 20237, 20876, 21025, 21324, 21770, 
    22073, 22346, 22361, 22626, 23202, 23222, 24849, 25366, 25437, 25468, 27652, 27732, 
    27776, 27925, 27992, 28762, 29344, 29450, 29666, 29870, 30307, 31704, 31862, 31884, 
    32309, 33008, 33207, 33877, 33879, 34596, 35205, 36636, 38161, 41767, 44198
]


ID_TO_YOLO_ID = {
    original_id: new_yolo_id 
    for new_yolo_id, original_id in enumerate(ORIGINAL_CATEGORY_IDS)
}


BASE_DIR = 'data/ai05-level1-project'
INPUT_JSON_DIR = os.path.join(BASE_DIR, 'train_annotations') # 입력: JSON 파일 위치
OUTPUT_LABEL_DIR = os.path.join(BASE_DIR, 'train_labels') # 출력: YOLO TXT 저장 위치

def convert_coco_to_yolo():
    """COCO 스타일 JSON 파일을 이미지 파일 이름별로 통합하여 YOLO TXT 포맷으로 변환합니다."""
    
    # 1. 출력 디렉토리 초기화
    if os.path.exists(OUTPUT_LABEL_DIR):
        shutil.rmtree(OUTPUT_LABEL_DIR)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    
    print(f"출력 디렉토리 생성: {OUTPUT_LABEL_DIR}")
    
    json_files = glob.glob(os.path.join(INPUT_JSON_DIR, '**', '*.json'), recursive=True)
    
    if not json_files:
        print(f"오류: 입력 JSON 파일을 {INPUT_JSON_DIR}에서 찾을 수 없습니다.")
        return

    print(f"총 {len(json_files)}개의 JSON 파일을 스캔하여 주석을 통합합니다...")
    
    # { 이미지_파일명: { 'width': w, 'height': h, 'annotations': [anno1, anno2, ...] } }
    image_data_map = {}


    # 모든 JSON을 스캔하여 주석을 이미지 파일별로 통합(Aggregation)
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            # print(f"파일 로드 실패: {json_path}")
            continue

        if not data.get('images') or not data.get('annotations'):
            continue
            
        # JSON 구조상 images 배열에 1개의 요소만 있고, 
        # annotations 배열에도 1개의 객체 주석만 있다고 가정합니다.
        img_info = data['images'][0]
        img_filename = img_info['file_name']
        
        if img_filename not in image_data_map:
            # 첫 주석인 경우, 이미지 메타데이터를 저장
            image_data_map[img_filename] = {
                'width': img_info['width'],
                'height': img_info['height'],
                'annotations': []
            }
        
        # 현재 JSON의 모든 주석을 통합된 리스트에 추가합니다.
        # (하나의 JSON에 여러 주석이 있다면 모두 추가, 하나만 있다면 그것만 추가)
        image_data_map[img_filename]['annotations'].extend(data['annotations'])

    print(f"✅ 스캔 및 통합 완료. 총 {len(image_data_map)}개의 고유 이미지에 대한 주석이 준비되었습니다.")

    # =================================================
    # 2단계: 통합된 주석을 YOLO TXT 포맷으로 변환 및 저장
    # =================================================
    
    for img_filename, img_data in image_data_map.items():
        img_w = img_data['width']
        img_h = img_data['height']
        
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(OUTPUT_LABEL_DIR, label_filename)
        
        yolo_lines = []

        for annot in img_data['annotations']:
            # COCO BBOX: [x_min, y_min, w, h] (픽셀)
            x_min, y_min, w, h = annot['bbox']
            original_cat_id = annot['category_id']
            
            # 💡 재매핑: 원본 ID -> YOLO ID (0~72)
            yolo_cat_id = ID_TO_YOLO_ID.get(original_cat_id)
            
            if yolo_cat_id is None:
                continue 

            # COCO to YOLO (정규화된 중앙 좌표 및 크기)
            x_center_norm = (x_min + w / 2) / img_w
            y_center_norm = (y_min + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            # YOLO TXT 포맷
            yolo_line = f"{yolo_cat_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(yolo_line)

        # TXT 파일 저장 (주석이 하나라도 있는 경우)
        if yolo_lines:
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
        # 주석이 하나도 없는 경우 (필요 시 빈 파일을 만들 수도 있지만, YOLO는 보통 라벨이 없는 이미지를 무시합니다.)

    print("모든 주석이 이미지 파일별로 통합되어 YOLO TXT 포맷으로 변환 완료.")


if __name__ == '__main__':
    convert_coco_to_yolo()