1. convert_json_to_yolo.py
COCO 스타일의 주석 파일을 YOLOv8 학습을 위한 YOLO TXT 포맷으로 변환하고 통합함.

2. split_dataset.py
변환된 이미지와 라벨을 훈련(Train) 세트와 검증(Validation) 세트로 무작위 분할함.

3. generate_name.py
목적
YOLOv8 설정 파일(data.yaml)에 필요한 73개 알약의 이름 목록(names list)을 생성.