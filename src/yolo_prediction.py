import os
from ultralytics import YOLO
import glob
from tqdm import tqdm
import shutil
import os.path

# =================================================================
# 1. 설정 변수
# =================================================================

RUN_DIR = 'runs/yolov8_exp_251023_0925'
# 학습된 YOLO 모델 가중치 파일 경로 (NOTE: 실제 best.pt 경로로 수정하세요.)
MODEL_WEIGHTS_PATH = os.path.join(RUN_DIR, 'weights/best.pt')

# 테스트 이미지가 있는 폴더 경로
TEST_IMAGES_DIR = 'data/ai05-level1-project/test_images' 

# 시각화 결과 이미지를 저장할 폴더 경로
OUTPUT_VISUALS_DIR = os.path.join(RUN_DIR, 'test_visualizations')

# =================================================================
# 2. 추론 하이퍼파라미터 (🌟 시각적 검증 파라미터)
# =================================================================

# 훈련 시 사용한 해상도 (1280으로 통일)
IMG_SIZE = 1280 

# 신뢰도 임계값: 시각적 명확성을 위해 적당한 값 (0.25) 사용 권장. 
# 제출 파일의 모든 박스를 보려면 conf=0.01로 설정하세요.
CONF_THRESHOLD = 0.25

# NMS IoU 임계값
IOU_THRESHOLD = 0.7 

# TTA 적용 여부: 시각화의 명확성을 위해 TTA는 끄는 것이 좋습니다.
AUGMENT = False

# =================================================================
# 3. 시각화 실행 함수
# =================================================================

def visualize_predictions():
    """
    학습된 YOLO 모델을 사용하여 테스트 이미지 디렉토리에 추론을 수행하고
    결과 이미지를 하나의 폴더에 저장합니다.
    """
    
    # 1. 모델 로드
    try:
        model = YOLO(MODEL_WEIGHTS_PATH)
        print(f"✅ 모델 로드 성공: {MODEL_WEIGHTS_PATH}")
    except Exception as e:
        print(f"🚨 모델 로드 실패. 경로를 확인하십시오: {MODEL_WEIGHTS_PATH}")
        print(f"오류 내용: {e}")
        return
    
    # 3. 테스트 이미지 목록 확인
    image_paths = glob.glob(os.path.join(TEST_IMAGES_DIR, '*.png'))
    
    if not image_paths:
        print(f"🚨 오류: 테스트 이미지 폴더({TEST_IMAGES_DIR})에서 이미지를 찾을 수 없습니다.")
        return

    print(f"총 {len(image_paths)}개의 테스트 이미지에 대해 시각화를 시작합니다.")
    print(f"시각화 설정: imgsz={IMG_SIZE}, conf={CONF_THRESHOLD}, iou={IOU_THRESHOLD}, augment={AUGMENT}")

    # 4. 추론 및 시각화 저장 (디렉토리 전체를 source로 전달)
    # project와 name 설정을 사용하여 최종 저장 경로를 runs/yolov8_exp_251023_0925/test_visualizations/predict로 고정합니다.
    model.predict(
        source=TEST_IMAGES_DIR, # 디렉토리 전체를 소스로 지정
        imgsz=IMG_SIZE, 
        conf=CONF_THRESHOLD,      
        iou=IOU_THRESHOLD,        
        augment=AUGMENT,  
        save=True,      # 시각화된 이미지 저장 활성화
        project=os.path.dirname(OUTPUT_VISUALS_DIR), # 'runs/yolov8_exp_251023_0925'
        name=os.path.basename(OUTPUT_VISUALS_DIR),   # 'test_visualizations'
        verbose=True # 추론 진행 상황을 자세히 표시
    )

    print(f"\n시각화 완료")


if __name__ == '__main__':
    visualize_predictions()
