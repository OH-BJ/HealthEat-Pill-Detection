import os
import glob
import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# =================================================================
# 1. 설정 변수
# =================================================================
# 원본 이미지와 라벨(.txt) 파일이 있는 루트 디렉토리. 해당 폴더에 분할 목록 파일을 저장하여 YOLO data.yaml에서 참조함.
DATA_DIR = 'data/yolo' 

# 분할 비율
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15 # 합계는 1.0 이어야 합니다.

# 시드 설정 (재현성 확보)
RANDOM_SEED = 42

# =================================================================
# 2. 층화 분할 로직
# =================================================================

def load_and_group_data():
    """
    모든 이미지 경로를 수집하고, 라벨 파일을 읽어 클래스 ID를 추출하여
    이미지 경로들을 클래스별로 그룹화합니다 (층화 표집을 위함).
    """
    label_files = glob.glob(os.path.join(DATA_DIR, 'labels', '*.txt'))
    
    # 딕셔너리: {클래스_ID: [해당 클래스를 포함하는 이미지 경로 목록]}
    data_by_class = defaultdict(list)
    
    all_image_paths = []

    print(f"총 {len(label_files)}개의 라벨 파일을 로드합니다.")

    for label_path in label_files:
        # 이미지 경로를 구성합니다. (예: .../images/0001.png)
        img_filename = os.path.basename(label_path).replace('.txt', '.png')
        img_path = os.path.join(DATA_DIR, 'images', img_filename)
                
        all_image_paths.append(img_path)
        
        # 라벨 파일에서 포함된 클래스 ID를 추출합니다.
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        # YOLO 라벨 형식: class_id x_c y_c w h (class_id는 0부터 시작)
                        class_id = int(line.split()[0])
                        # 각 클래스를 포함하는 이미지 경로를 추가
                        data_by_class[class_id].append(img_path)
        except Exception as e:
            print(f"라벨 파일 로드 오류 ({label_path}): {e}")
            continue

    # 중복된 이미지 경로 제거 (하나의 이미지가 여러 클래스에 포함될 수 있음)
    unique_image_paths = sorted(list(set(all_image_paths)))
    
    # 각 이미지에 포함된 모든 클래스 ID 리스트를 생성합니다. (Stratify 기준)
    image_class_labels = []
    for img_path in unique_image_paths:
        label_path = img_path.replace('images', 'labels').replace('.png', '.txt')
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        classes.append(line.split()[0])
        # 이미지를 대표하는 클래스 라벨(들)을 문자열로 결합 (stratify 인자용)
        image_class_labels.append(','.join(sorted(classes))) 
        
    return unique_image_paths, image_class_labels

def save_list_files(paths, filename):
    """경로 리스트를 텍스트 파일로 저장합니다."""
    file_path = os.path.join(DATA_DIR, filename)
    
    # YOLO 형식에 맞게 상대 경로로 변환 (ROOT_DATA_DIR 하위)
    relative_paths = [os.path.relpath(p, DATA_DIR) for p in paths]
    
    with open(file_path, 'w') as f:
        f.write('\n'.join(relative_paths))
    print(f"✅ {filename} 파일 저장 완료: 총 {len(paths)}개 경로")
    return file_path


def split_data_and_save():
    """메인 분할 실행 함수"""
    
    all_paths, class_labels = load_and_group_data()
    
    if not all_paths:
        print("🚨 오류: 데이터셋 폴더에서 처리할 이미지를 찾을 수 없습니다.")
        return

    # 1. Train과 임시 (Val + Test) 분할
    # Stratify 인자로 클래스 라벨을 사용하여 층화 분할
    X_temp, X_train, y_temp, y_train = train_test_split(
        all_paths, class_labels, 
        test_size=TRAIN_RATIO, 
        random_state=RANDOM_SEED, 
        stratify=class_labels
    )

    # 2. 임시 세트를 Val과 Test로 분할
    # Test 세트 비율 계산: TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    test_size_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=test_size_ratio, 
        random_state=RANDOM_SEED, 
        stratify=y_temp
    )
    
    # 3. 결과 저장
    save_list_files(X_train, 'train.txt')
    save_list_files(X_val, 'val.txt')
    save_list_files(X_test, 'test.txt') # Test 세트 경로 저장
    
    print("\n--- 데이터 분할 결과 ---")
    print(f"총 데이터: {len(all_paths)}개")
    print(f"Train 세트: {len(X_train)}개 ({len(X_train)/len(all_paths):.1%})")
    print(f"Validation 세트: {len(X_val)}개 ({len(X_val)/len(all_paths):.1%})")
    print(f"Test 세트: {len(X_test)}개 ({len(X_test)/len(all_paths):.1%})")
    print("-------------------------")


if __name__ == '__main__':
    # scikit-learn이 설치되어 있지 않다면 설치 메시지 출력
    try:
        import sklearn
    except ImportError:
        print("🚨 필수 라이브러리: `scikit-learn`이 설치되어 있지 않습니다. 설치하십시오: pip install scikit-learn")
        exit()
        
    split_data_and_save()
