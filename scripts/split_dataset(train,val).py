import os
import random
import shutil
import glob

BASE_PATH = 'data/yolo'
TRAIN_IMG_DIR = os.path.join(BASE_PATH, 'images/train_images')
TRAIN_LABEL_DIR = os.path.join(BASE_PATH, 'labels_curated/train_images') # YOLO TXT 파일 위치

VAL_IMG_DIR = os.path.join(BASE_PATH, 'images/val_images')
VAL_LABEL_DIR = os.path.join(BASE_PATH, 'labels_curated/val_images')

VAL_RATIO = 0.20 # 20%를 검증 데이터로 사용

def split_data():
    """train_images 폴더의 파일을 무작위로 train/val로 분할합니다."""
    
    # 1. 분할 전 폴더 상태 확인 및 Val 폴더 생성
    if not os.path.isdir(TRAIN_IMG_DIR) or not os.path.isdir(TRAIN_LABEL_DIR):
        print("오류: train/images 또는 train/labels 폴더가 존재하지 않습니다. 이전 단계를 확인하십시오.")
        return

    # Val 폴더 생성 (이미 존재하면 건너뜁니다)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)
    
    # 2. 모든 이미지 파일 목록 가져오기
    all_images = glob.glob(os.path.join(TRAIN_IMG_DIR, '*.png'))
    
    if not all_images:
        print("🚨 오류: train_images 폴더에 이미지 파일이 없습니다.")
        return
        
    total_count = len(all_images)
    val_count = int(total_count * VAL_RATIO)
    train_count = total_count - val_count
    
    print(f"총 이미지 수: {total_count}개")
    print(f"▶ Train 세트: {train_count}개 ({100-VAL_RATIO*100:.0f}%)")
    print(f"▶ Val 세트:   {val_count}개 ({VAL_RATIO*100:.0f}%)")

    # 3. Validation 세트에 사용할 파일 무작위 선택
    val_files = random.sample(all_images, val_count)

    # 4. 파일 이동 실행 (이미지 및 라벨 동시 이동)
    print("\n Validation 데이터셋을 이동 중...")
    
    for img_path in val_files:
        # 파일 이름 (확장자 포함): 'image_001.png'
        file_name = os.path.basename(img_path)
        
        # 파일 이름 (확장자 제외): 'image_001'
        base_name = os.path.splitext(file_name)[0]
        
        # 4-1. 라벨 TXT 파일 경로 설정
        label_file_name = base_name + '.txt'
        src_label_path = os.path.join(TRAIN_LABEL_DIR, label_file_name)
        
        # 4-2. 이미지 이동
        shutil.move(img_path, os.path.join(VAL_IMG_DIR, file_name))
        
        # 4-3. 라벨 이동 (라벨 파일이 실제로 존재하는지 확인해야 안전합니다)
        if os.path.exists(src_label_path):
             shutil.move(src_label_path, os.path.join(VAL_LABEL_DIR, label_file_name))
        else:
             print(f"경고: 라벨 파일 {label_file_name}을 찾을 수 없습니다. 이미지만 이동되었습니다.")

    print("✅ 데이터 분할 및 파일 이동 완료!")


if __name__ == '__main__':
    split_data()