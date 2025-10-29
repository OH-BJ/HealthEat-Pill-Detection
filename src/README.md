1. convert_json_to_yolo.py
COCO 스타일의 주석 파일을 YOLOv8 학습을 위한 YOLO TXT 포맷으로 변환하고 통합함.

2. split_dataset.py
변환된 이미지와 라벨을 훈련(Train) 세트와 검증(Validation) 세트로 무작위 분할함.

3. generate_name.py
목적
YOLOv8 설정 파일(data.yaml)에 필요한 73개 알약의 이름 목록(names list)을 생성.

4. split_dataset_seg.py
seg 전용 이미지와 라벨을 훈련(Train) 세트와 검증(Validataion), 테스트(Test) 세트로 무작위 분할합니다.

5. convert_polygon_to_yolo.py
폴리콘 어노테이션들을 yolo-seg 환경에서 구동되게끔 전환해줍니다. 실행시 최하단에서 경로 설정 해주셔야 합니다.

--- 하단은 이제 쓰이지 않는 스크립트지만 추가로 더 해보고 싶은 사람들을 위해 남겼습니다

- merge_anns_for_seg.py
어노테이션 파일을 모아주는 역할. 첫 실행시 json_2_seg(SAM).py 이전에 실행하길 권장하며 테스트시 코드 끝 부분에서 경로 설정을 다시 해주셔야 합니다.

- json_2_seg(SAM).py
merge_anns_for_seg.py를 통해 생성된 json 파일의 bbox를 읽고 자동으로 segmentation 처리를 해줍니다.
