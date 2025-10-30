* /images: 모든 알약 png images
* /labels: 아래 labels 폴더 중, train 시 사용할 폴더를 labels로 이름을 변경해주어야함.
* /labels_raw: 통합 후 YOLO 형식으로 변환한 labels
* /labels_curated: cvat를 통해 auto labeling을 거친 labels

train.txt, val.txt, test.txt 파일에 경로를 담아 split된 상태.
data.yaml에 해당 txt 파일의 경로를 지정해 사용함.