import json
import os
import hashlib
from collections import defaultdict

def get_image_id(file_name):
    return int(hashlib.md5(file_name.encode()).hexdigest(), 16) % (2**31)

def get_base_file_name(file_name):
    """file_name에서 subfolder 영향 제거 (공통 이미지 이름)"""
    return file_name.split('_json')[0] if '_json' in file_name else file_name

def merge_annotations(annotations_dir, output_json_path):
    all_images = []
    all_annotations = []
    all_categories = set()  # (id, name) 튜플
    image_to_ann = defaultdict(list)  # base_file_name → list of anns (멀티 클래스 합침)
    image_to_info = {}  # base_file_name → img_info (중복 피함)
    ann_id_counter = 1
    skipped_files = []
    loaded_files = 0

    for root, dirs, files in os.walk(annotations_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                print(f"Loading: {json_path}")
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # bbox 누락 검사
                    if not data.get('annotations') or not any('bbox' in ann and ann['bbox'] for ann in data['annotations']):
                        print(f"  - Skipping: No valid bbox in {file}")
                        skipped_files.append(file)
                        continue
                    
                    loaded_files += 1
                    img_info = data['images'][0]
                    base_file_name = get_base_file_name(img_info['file_name'])
                    
                    # img_info 저장 (중복 피함, 첫 번째 사용)
                    if base_file_name not in image_to_info:
                        img_info['id'] = get_image_id(img_info['file_name'])
                        image_to_info[base_file_name] = img_info
                        all_images.append(img_info)
                    
                    # Categories 쌓기: dl_idx/dl_name unique
                    dl_idx = img_info.get('dl_idx', None)
                    dl_name = img_info.get('dl_name', 'Unknown_Drug')
                    print(f"  - dl_idx: '{dl_idx}', dl_name: '{dl_name}'")
                    if dl_idx and dl_idx.isdigit():
                        class_id = int(dl_idx)
                        class_name = dl_name.replace(' ', '_')
                        tuple_key = (class_id, class_name)
                        if tuple_key not in all_categories:
                            all_categories.add(tuple_key)
                            print(f"  - Added unique category: ID={class_id}, name={class_name}")
                    
                    # Annotations 추가 (멀티 클래스 합침)
                    for ann in data['annotations']:
                        ann['id'] = ann_id_counter
                        ann['image_id'] = image_to_info[base_file_name]['id']
                        # category_id = dl_idx (원본 ID)
                        dl_idx = img_info.get('dl_idx', 1)
                        ann['category_id'] = int(dl_idx) if dl_idx.isdigit() else 1
                        ann_id_counter += 1
                        image_to_ann[base_file_name].append(ann)
                    
                    print(f"  - Loaded: {len(data['annotations'])} anns for base {base_file_name}, category_id={ann['category_id']}")
                
                except json.JSONDecodeError as e:
                    print(f"  - JSON Error in {file}: {e}")
                    skipped_files.append(file)

    # Categories: ID 오름차순 정렬
    categories = [{'supercategory': 'pill', 'id': cid, 'name': name} for cid, name in sorted(all_categories, key=lambda x: x[0])]
    
    # Annotations: 멀티 ann 합침
    merged_annotations = []
    for base_file_name, anns in image_to_ann.items():
        merged_annotations.extend(anns)
        print(f"  - Merged for {base_file_name}: {len(anns)} anns from multi classes")  # 멀티 bbox 확인
    
    integrated_data = {
        'images': all_images,
        'type': 'instances',
        'annotations': merged_annotations,
        'categories': categories
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(integrated_data, f, indent=2, ensure_ascii=False)
    
    print(f"통합 완료: {len(all_images)} 이미지, {len(merged_annotations)} 어노테이션, {len(categories)} 클래스")
    print(f"Loaded files: {loaded_files}, Skipped: {len(skipped_files)} ({skipped_files[:5] if skipped_files else []})")
    print(f"Categories sample: {categories[:10]}...")
    return integrated_data

# 실행
annotations_dir = "data/raw/train_labels/"
output_path = "data/processed/train_annotations_integrated.json"
data = merge_annotations(annotations_dir, output_path)