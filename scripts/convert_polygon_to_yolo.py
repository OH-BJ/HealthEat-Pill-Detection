# convert_polygon_to_yolo.py
import os

# YOUR CLASS CODES (정렬된 리스트)
ORIGINAL_CATEGORY_IDS = [
    1899, 2482, 3350, 3482, 3543, 3742, 3831, 4377, 4542, 5093, 5885, 6191, 6562, 10220, 12080, 12246, 12419, 12777,
    13394, 13899, 16231, 16261, 16547, 16550, 16687, 18109, 18146, 18356, 19231, 19551, 19606, 19860, 20013, 20237,
    20876, 21025, 21324, 21770, 22073, 22346, 22361, 22626, 23202, 23222, 24849, 25366, 25437, 25468, 27652, 27732,
    27776, 27925, 27992, 28762, 29344, 29450, 29666, 29870, 30307, 31704, 31862, 31884, 32309, 33008, 33207, 33877, 33879, 34596, 35205, 36636, 38161, 41767, 44198
]

def convert_label(txt_path, out_path):
    with open(txt_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            parts = line.strip().split()
            class_id = int(parts[0])
            try:
                class_idx = ORIGINAL_CATEGORY_IDS.index(class_id)
            except ValueError:
                print(f"[변환실패] {os.path.basename(txt_path)} {i+1}번째 줄 - 미등록 class_id: {class_id}")
                continue
            new_parts = [str(class_idx)] + parts[1:]

            # 디버깅 로그: 원본→맵핑 결과(5개만 샘플)
            if i < 5:
                print(f"[확인] {os.path.basename(txt_path)} line {i+1} 변환: {class_id} → {class_idx}")

            fout.write(' '.join(new_parts) + '\n')

def check_converted_labels(label_dir):
    for fname in os.listdir(label_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(label_dir, fname), 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    label_idx = int(parts[0])
                    if not (0 <= label_idx < 73):
                        print(f"[오류] {fname} {idx+1}번째 라벨 인덱스: {label_idx}")

# 변환 후 label 디렉토리 검증
check_converted_labels('data/yolo/labels/train')

def convert_dir(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(in_dir):
        if fname.endswith('.txt'):
            convert_label(os.path.join(in_dir, fname), os.path.join(out_dir, fname))

# 예시 실행 (경로 수정!!)
if __name__ == '__main__':
    convert_dir(
        'data/raw/seg_labels/train_labels',   # 원본 polygon txt 위치
        'data/yolo/labels/train'              # YOLO용 폴리곤 라벨로 변환 결과 경로
    )
