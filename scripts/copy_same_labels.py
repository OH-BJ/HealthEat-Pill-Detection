import os
import shutil
import glob
from tqdm import tqdm

# =================================================================
# 1. 설정 변수 (사용 환경에 맞게 경로를 수정하십시오)
# =================================================================

# 1-1. 분할 기준이 되는 라벨 목록 (큐레이션된 라벨 파일(.txt)이 있는 위치)
REFERENCE_DATA_ROOT = 'data/yolo'
REFERENCE_CURATED_LABELS_TRAIN = os.path.join(REFERENCE_DATA_ROOT, 'labels_curated/train_images')
REFERENCE_CURATED_LABELS_VAL = os.path.join(REFERENCE_DATA_ROOT, 'labels_curated/val_images')

# 1-2. 원본(Raw) 라벨이 저장된 위치 (복사할 원본 .txt 파일의 소스 폴더)
RAW_LABEL_SOURCE = 'data\yolo\labels_raw\images' # 예: 오토 라벨링으로 생성된 수정 전의 모든 .txt 파일

# 1-3. 분할된 원본 라벨이 저장될 최종 출력 위치 (스크립트가 생성할 폴더)
OUTPUT_ROOT = 'data\yolo\labels_raw_split\images' 


# 파일 확장자 설정
LABEL_EXT = '.txt'

# =================================================================
# 2. 파일 복사 및 분할 함수
# =================================================================

def copy_raw_labels_by_list(ref_label_folder, raw_label_src, split_name):
    """
    큐레이션된 라벨 폴더의 파일 목록을 기반으로 원본 라벨 파일을 대상 폴더로 복사합니다.
    """
    # 2-1. 대상 폴더 설정 및 생성 (출력 폴더명에 '_labels' 명시)
    output_label_dir = os.path.join(OUTPUT_ROOT, f'{split_name}_labels')
    
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 2-2. 참조 라벨 파일명 목록 가져오기 (.txt 파일만 참조)
    ref_label_paths = glob.glob(os.path.join(ref_label_folder, f'*{LABEL_EXT}'))
    
    if not ref_label_paths:
        print(f"⚠️ 경고: 참조 라벨 폴더 '{ref_label_folder}'에서 라벨 파일을 찾을 수 없습니다. 건너뜁니다.")
        return 0

    copied_count = 0
    
    # 2-3. 파일 복사 실행
    print(f"\n[{split_name.upper()}]: 총 {len(ref_label_paths)}개 원본 라벨 파일 복사 시작...")
    
    for ref_path in tqdm(ref_label_paths, desc=f"복사 중 ({split_name})"):
        ref_filename_with_ext = os.path.basename(ref_path)
        ref_filename_base, _ = os.path.splitext(ref_filename_with_ext)
        
        # 원본 라벨 파일 찾기 (원본 폴더에서 라벨 파일 검색)
        raw_label_path = os.path.join(raw_label_src, ref_filename_base + LABEL_EXT)
        
        if os.path.exists(raw_label_path):
            # 라벨 복사
            shutil.copy2(raw_label_path, os.path.join(output_label_dir, os.path.basename(raw_label_path)))
            copied_count += 1
        else:
            print(f"🚨 오류: 원본 라벨 파일 '{ref_filename_base}{LABEL_EXT}'을(를) 찾을 수 없습니다. (경로: {raw_label_src})")


    return copied_count

# =================================================================
# 3. 메인 실행 블록
# =================================================================

if __name__ == '__main__':
    total_copied = 0
    
    print(f"--- 원본 라벨셋 재분할 시작 ---")
    print(f"라벨 분할 기준 폴더: {REFERENCE_DATA_ROOT}")
    print(f"원본 라벨 소스: {RAW_LABEL_SOURCE}")
    print(f"출력 폴더: {OUTPUT_ROOT}")
    
    # 1. Train 라벨 복사
    train_count = copy_raw_labels_by_list(
        ref_label_folder=REFERENCE_CURATED_LABELS_TRAIN,
        raw_label_src=RAW_LABEL_SOURCE,
        split_name='train'
    )
    total_copied += train_count
    
    # 2. Val 라벨 복사
    val_count = copy_raw_labels_by_list(
        ref_label_folder=REFERENCE_CURATED_LABELS_VAL,
        raw_label_src=RAW_LABEL_SOURCE,
        split_name='val'
    )
    total_copied += val_count

    print(f"\n--- 재분할 완료 ---")
    print(f"총 {total_copied}개의 원본 라벨 파일(.txt)을 성공적으로 복사했습니다.")
    print(f"결과 위치: {os.path.abspath(OUTPUT_ROOT)}")
