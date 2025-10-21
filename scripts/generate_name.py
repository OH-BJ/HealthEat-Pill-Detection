import os
import json
import glob

# 73개의 원본 Category ID
ORIGINAL_CATEGORY_IDS = [
    1899, 2482, 3350, 3482, 3543, 3742, 3831, 4377, 4542, 5093, 5885, 6191, 6562, 10220, 
    12080, 12246, 12419, 12777, 13394, 13899, 16231, 16261, 16547, 16550, 16687, 18109, 
    18146, 18356, 19231, 19551, 19606, 19860, 20013, 20237, 20876, 21025, 21324, 21770, 
    22073, 22346, 22361, 22626, 23202, 23222, 24849, 25366, 25437, 25468, 27652, 27732, 
    27776, 27925, 27992, 28762, 29344, 29450, 29666, 29870, 30307, 31704, 31862, 31884, 
    32309, 33008, 33207, 33877, 33879, 34596, 35205, 36636, 38161, 41767, 44198
]

def generate_yolo_names_list(annotation_root_dir, original_ids):
    """
    모든 JSON 파일을 스캔하여 원본 ID와 알약 이름(name)을 매핑하고,
    이를 0부터 시작하는 YOLO ID 순서에 맞게 리스트로 반환합니다.
    """
    # 원본 ID와 이름 쌍을 저장할 딕셔너리
    # {원본_ID: 알약_이름}
    id_to_name_map = {}
    
    # 주석 파일 검색 (하위 폴더 포함)
    json_files = glob.glob(os.path.join(annotation_root_dir, '**', '*.json'), recursive=True)
    
    if not json_files:
        print("경고: 주석 JSON 파일을 찾을 수 없습니다. 경로를 확인하십시오.")
        return []

    print(f"총 {len(json_files)}개의 JSON 파일을 스캔합니다...")

    # 모든 JSON 파일을 순회하며 'categories' 정보를 추출
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # COCO 포맷은 'categories' 섹션에 ID와 Name 매핑 정보가 있습니다.
            for category in data.get('categories', []):
                cat_id = category.get('id')
                cat_name = category.get('name')
                
                # ID가 이전에 확인된 73개 리스트에 포함되어 있는지 확인하고 저장
                if cat_id in original_ids and cat_name:
                    id_to_name_map[cat_id] = cat_name
                    
        except Exception as e:
            # 파일 읽기 오류는 건너뜁니다.
            print(f"파일 처리 오류 ({file_path}): {e}")
            continue

    # 3. 최종 YOLO names 리스트 생성 (0부터 72 순서)
    # 이전에 정렬된 원본 ID 순서에 따라 이름을 추출합니다.
    yolo_names_list = []
    
    for original_id in original_ids:
        # 맵에서 이름을 찾습니다. (이름이 없는 경우를 대비해 예외 처리)
        name = id_to_name_map.get(original_id)
        if name:
            yolo_names_list.append(name)
        else:
            # 이름 매핑이 누락된 경우, 임시 이름으로 대체하고 경고를 출력합니다.
            yolo_names_list.append(f"CLASS_{original_id}_NAME_MISSING")
            print(f"경고: 원본 ID {original_id}에 대한 알약 이름이 누락되었습니다.")

    return yolo_names_list


if __name__ == '__main__':
    # 프로젝트 루트(project#1)를 기준으로 상대 경로 설정
    ANNOTATION_DIR = 'data/ai05-level1-project/train_annotations'
    
    names_list = generate_yolo_names_list(ANNOTATION_DIR, ORIGINAL_CATEGORY_IDS)
    
    if names_list:
        print("\n========================================================")
        print(f"✅ 최종 YOLO names 목록 (총 {len(names_list)}개):")
        print("========================================================")
        
        # data.yaml 포맷에 맞춰 출력
        for i, name in enumerate(names_list):
            print(f"  {i}: \"{name}\"")
        
        print("\n이 목록을 'pill_data.yaml' 파일의 'names' 섹션에 복사하여 사용")
    else:
        print("\nnames 목록 생성에 실패했습니다. 경로와 JSON 파일 구조를 확인하십시오.")