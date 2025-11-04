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

6. generate_mask.py
알파채널 마스크 생성 스크립트

7. composite_aug.py
백그라운드와 알약 이미지 합성 스크립트

--- 하단은 이제 쓰이지 않는 스크립트지만 추가로 더 해보고 싶은 사람들을 위해 남겼습니다

- merge_anns_for_seg.py
어노테이션 파일을 모아주는 역할. 첫 실행시 json_2_seg(SAM).py 이전에 실행하길 권장하며 테스트시 코드 끝 부분에서 경로 설정을 다시 해주셔야 합니다.

- json_2_seg(SAM).py
merge_anns_for_seg.py를 통해 생성된 json 파일의 bbox를 읽고 자동으로 segmentation 처리를 해줍니다.

- diffusion_aug.py
알약 마스크 생성 및 디퓨전-inpainting 모델로 하여금 배경 변형 스크립트


---

이 밑으로는 이미지 생성 예시입니다.

---

### 🪡 Fabric 예시
<figure>
  <img width="1848" height="1185" alt="fabric 예시" src="https://github.com/user-attachments/assets/aa7fa554-accc-4471-b83e-21f4473276ac" />
  <figcaption align="center">– 섬유 재질의 합성 예시</figcaption>
</figure>

---

### 🧊 Glass 예시
<figure>
  <img width="1789" height="1183" alt="glass 예시" src="https://github.com/user-attachments/assets/0492cef2-6e02-4b54-bdb1-c01570ceae1d" />
  <figcaption align="center">– 유리 텍스쳐 합성 예시</figcaption>
</figure>

---

### 🪞 Marble Floor 예시
<figure>
  <img width="1794" height="1183" alt="marble_floor 예시" src="https://github.com/user-attachments/assets/c25746ff-6070-4f16-b7ff-cd7136bad655" />
  <figcaption align="center">– 대리석 반사 합성 예시</figcaption>
</figure>

---

### 🧴 Plastic 예시
<figure>
  <img width="1784" height="1184" alt="plastic 예시" src="https://github.com/user-attachments/assets/993d18e2-9ba3-4d4c-8844-ecc4243a38d8" />
  <figcaption align="center">– 플라스틱 표면 합성 예시</figcaption>
</figure>

---

### ⚙️ Steel 예시
<figure>
  <img width="1780" height="1186" alt="steel 예시" src="https://github.com/user-attachments/assets/264c9eae-e29f-444a-bc47-438d020632c7" />
  <figcaption align="center">– 금속성 합성 예시</figcaption>
</figure>

---

### 🪵 Wooden 예시
<figure>
  <img width="1799" height="1171" alt="wooden 예시" src="https://github.com/user-attachments/assets/ab0953d2-646b-43e6-b464-84a799e1abc8" />
  <figcaption align="center">– 목재 질감 합성 예시</figcaption>
</figure>
