import os
import pandas as pd
from ultralytics import YOLO
import glob
from tqdm import tqdm
import re

# =================================================================
# 1. 설정 변수
# =================================================================

# 학습된 YOLO 모델 가중치 파일 경로
MODEL_WEIGHTS_PATH = 'runs/yolov8_exp_251021_1426/weights/best.pt'

# 테스트 이미지가 있는 폴더 경로
TEST_IMAGES_DIR = 'data/ai05-level1-project/test_images' 

# 제출 파일 저장 경로
OUTPUT_CSV_PATH = 'submission.csv'

# =================================================================
# 2. 클래스 ID 매핑 (YOLO ID -> 원본 ID)
#    : convert_json_to_yolo.py 스크립트의 ORIGINAL_CATEGORY_IDS를 사용
# =================================================================

# YOLO TXT 변환 시 사용했던 73개의 원본 Category ID 리스트
# YOLO ID (0, 1, 2, ...)가 이 리스트의 인덱스를 가리킵니다.
ORIGINAL_CATEGORY_IDS = [
    1899, 2482, 3350, 3482, 3543, 3742, 3831, 4377, 4542, 5093, 5885, 6191, 6562, 10220, 
    12080, 12246, 12419, 12777, 13394, 13899, 16231, 16261, 16547, 16550, 16687, 18109, 
    18146, 18356, 19231, 19551, 19606, 19860, 20013, 20237, 20876, 21025, 21324, 21770, 
    22073, 22346, 22361, 22626, 23202, 23222, 24849, 25366, 25437, 25468, 27652, 27732, 
    27776, 27925, 27992, 28762, 29344, 29450, 29666, 29870, 30307, 31704, 31862, 31884, 
    32309, 33008, 33207, 33877, 33879, 34596, 35205, 36636, 38161, 41767, 44198
]

def get_image_id(filename):
    """
    이미지 파일 이름에서 숫자 부분(image_id)을 추출합니다.
    (수정됨) 파일명이 '1.png', '2.png'와 같이 숫자 파일명일 경우, 
    확장자를 제외한 순수 파일명(숫자)을 image_id로 반환합니다.
    """
    # 파일명에서 확장자 제거
    name_without_ext = os.path.splitext(filename)[0]
    
    # 파일명 전체가 image_id (예: '1.png' -> 1)
    try:
        # 정수형으로 변환하여 반환
        return int(name_without_ext)
    except ValueError:
        # 파일명이 숫자가 아닌 경우를 대비한 예외 처리 (필요시)
        print(f"경고: 파일명 '{filename}'이 순수 숫자가 아닙니다. 첫 번째 숫자 부분을 사용합니다.")
        numbers = re.findall(r'^\d+', name_without_ext)
        if numbers:
            return int(numbers[0])
        else:
            return -1 # 유효하지 않은 ID 반환

def generate_submission_csv():
    """학습된 YOLO 모델을 사용하여 테스트 이미지에 추론을 수행하고 제출 파일을 생성합니다."""
    
    # 1. 모델 로드
    try:
        model = YOLO(MODEL_WEIGHTS_PATH)
        print(f"모델 로드 성공: {MODEL_WEIGHTS_PATH}")
    except Exception as e:
        print(f"모델 로드 실패. 경로를 확인하십시오: {MODEL_WEIGHTS_PATH}")
        print(f"오류 내용: {e}")
        return

    # 2. 테스트 이미지 목록 가져오기
    # .png, .jpg 등 모든 이미지 확장자를 포함하도록 glob 사용
    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.png')) + \
                  glob.glob(os.path.join(TEST_IMAGES_DIR, '*.jpg'))
    
    if not image_paths:
        print(f"🚨 오류: 테스트 이미지 폴더({TEST_IMAGES_DIR})에서 이미지를 찾을 수 없습니다.")
        return

    # 파일명 기준 오름차순 정렬 (1.png, 2.png, ... 순서 보장)
    image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) if os.path.splitext(os.path.basename(x))[0].isdigit() else os.path.basename(x))

    print(f"총 {len(image_paths)}개의 테스트 이미지에 대해 추론을 시작합니다.")

    submission_data = []
    annotation_id_counter = 1 # 제출 파일의 고유한 annotation_id

    # 3. 추론 및 결과 변환
    for image_path in tqdm(image_paths, desc="추론 및 CSV 변환 중"):
        file_name = os.path.basename(image_path)
        
        # Image ID 추출 (파일명의 숫자를 그대로 사용)
        image_id = get_image_id(file_name)
        
        if image_id == -1:
             print(f"⚠️ 파일 '{file_name}'의 image_id를 추출할 수 없어 건너뜁니다.")
             continue
        
        # YOLO 추론 실행
        # imgsz는 학습 시 사용한 크기(예: 640)와 동일하게 지정하는 것이 좋습니다.
        # conf(confidence threshold)는 필요에 따라 조정 가능합니다.
        results = model(image_path, imgsz=640, conf=0.001, augment=True) 
        
        # 결과를 순회하며 submission_data에 추가
        for result in results:
            if result.boxes is None:
                continue

            # YOLOv8의 boxes 객체에서 바운딩 박스, 신뢰도, 클래스 ID 추출
            # xyxy: [x_min, y_min, x_max, y_max] (픽셀 좌표)
            boxes = result.boxes.xyxy.cpu().numpy() 
            scores = result.boxes.conf.cpu().numpy()
            yolo_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, yolo_id in zip(boxes, scores, yolo_ids):
                # 💡 YOLO ID (0~72)를 원본 Category ID로 변환
                if yolo_id >= len(ORIGINAL_CATEGORY_IDS):
                    print(f"경고: 알 수 없는 YOLO ID {yolo_id}가 감지되었습니다. 건너뜁니다.")
                    continue
                
                original_cat_id = ORIGINAL_CATEGORY_IDS[yolo_id]
                
                # BBox 좌표 변환 (xyxy -> COCO [x, y, w, h])
                x_min, y_min, x_max, y_max = box
                
                # 대회 요구 포맷: bbox_x, bbox_y, bbox_w, bbox_h (픽셀)
                bbox_x = round(x_min)
                bbox_y = round(y_min)
                bbox_w = round(x_max - x_min)
                bbox_h = round(y_max - y_min)
                
                # 스코어는 소수점 4자리까지 표시
                score_formatted = f"{score:.4f}"
                
                submission_data.append([
                    annotation_id_counter,
                    image_id,
                    original_cat_id,
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h,
                    score_formatted
                ])
                
                annotation_id_counter += 1

    # 4. CSV 파일로 저장
    if submission_data:
        df = pd.DataFrame(submission_data, columns=[
            'annotation_id', 'image_id', 'category_id', 
            'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score'
        ])
        
        # image_id 기준으로 정렬 (제출 편의를 위해)
        df = df.sort_values(by=['image_id', 'score'], ascending=[True, False])

        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\n✅ 제출 파일 생성 완료: {OUTPUT_CSV_PATH}에 총 {len(submission_data)}개의 객체 저장.")
    else:
        print("\n⚠️ 경고: 감지된 객체가 없어 제출 파일이 생성되지 않았습니다.")


if __name__ == '__main__':
    generate_submission_csv()
