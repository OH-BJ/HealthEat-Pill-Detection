# 헬스잇(Health Eat) 경구약제 이미지 인식 프로젝트

> 사용자가 촬영한 약 사진으로부터 알약의 종류와 위치를 탐지하는 객체 탐지(Object Detection) 모델 개발 프로젝트입니다.

---

## 주요 기능
- 이미지 속 최대 4개 알약의 클래스(73종) 탐지
- 이미지 속 알약의 위치(Bounding Box) 탐지

---

## 실행 방법 (How-to-Run)

### A. 환경 설정 (최초 1회)
1.  Python 3.11.9 버전을 권장합니다.
2.  새 가상환경 (`venv-gpu`)을 생성하고 활성화합니다.
    ```bash
    # 3.11 버전으로 가상환경 생성
    py -3.11 -m venv venv-gpu
    
    # 가상환경 활성화
    .\venv-gpu\Scripts\activate
    ```
3.  필수 라이브러리를 설치합니다. (GPU용 PyTorch를 먼저 설치해야 합니다.)
    ```bash
    # 1. GPU용 PyTorch (CUDA 12.1) 설치
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    
    # 2. 나머지 필수 라이브러리 설치
    pip install -r requirements.txt
    ```

### B. 데이터 준비 (최초 1회)
1.  원본 데이터(`data/raw/train_images` 등)가 프로젝트 폴더 내에 준비되었는지 확인합니다.
2.  YOLO 훈련용 데이터(`data/yolo/images/train_images`, `data/yolo/labels/train_images` 등)가 준비되었는지 확인합니다. (YOLO는 labels 폴더 하위 경로명이 images 폴더 하위 경로명과 같아야 합니다.)

### C. 모델 훈련 (공식 스크립트)
1.  **`yolo_config.yaml`** 파일을 열어 훈련 설정을 수정합니다.
    *(예: `architecture`, `input_size`, `epochs`, `batch_size` 등)*
2.  "공식 훈련 스크립트" (`src/yolo_train.py`)를 실행합니다.
    ```bash
    # data.yaml 사용 (또는 yolo_config.yaml에 지정된 경로)
    python src/yolo_train.py 
    ```
3.  훈련 결과는 `runs/detect/` 폴더 내에 저장됩니다.

### D. 캐글 제출 (공식 스크립트)
1.  **`src/yolo_submission.py`** 파일을 엽니다.
2.  **`RUN_DIR`**에 훈련이 완료된 폴더 경로(예: `runs/detect/train_result`)를 정확히 입력합니다.
3.  **`TEST_IMAGES_DIR`**에 테스트 이미지 폴더 경로(`data/raw/test_images`)가 맞는지 확인합니다.
4.  "공식 제출 스크립트"를 실행합니다.
    ```bash
    python src/yolo_submission.py
    ```
5.  `RUN_DIR` 폴더 안에 생성된 `submission.csv` 파일을 캐글에 제출합니다.

---

## 팀 멤버 및 역할
| 역할 | 이름 | GitHub |
| :--- | :--- | :--- |
| **Project Manager** | [오병주] | [@OH-BJ](https://github.com/OH-BJ) |
| **Data Engineer** | [이상윤] | [@SYLforge](https://github.com/SYLforge) |
| **Model Architect** | [서준범] | [@Seo-Junbeom](https://github.com/Seo-Junbeom) |
| **Experimentation Lead** | [김승우] | [@carsy078-maker](https://github.com/carsy078-maker) |

---

## 최종 결과물 링크
| 구분 | 링크 |
| :--- | :--- |
| **최종 보고서** | [보고서 링크 삽입 예정] |
| **오병주 협업 일지** | [개인별 링크 삽입 예정] |
| **이상윤 협업 일지** | [개인별 링크 삽입 예정] |
| **서준범 협업 일지** | [개인별 링크 삽입 예정] |
| **김승우 협업 일지** | [개인별 링크 삽입 예정] |

---

## 개발 환경
- **언어**: `Python 3.11.9` (권장)
- **프레임워크**: `PyTorch` (`torch==2.9.0`)
- **핵심 모델**: `YOLOv8` (`ultralytics==8.3.218`)
- **주요 라이브러리**: `opencv-python`, `pandas`, `numpy`, `PyYAML`, `tqdm` (자세한 목록은 `requirements.txt` 참고)
- **라벨링 도구**: `CVAT` (누락 라벨 보완용)
- **개발 도구**: `Visual Studio Code` (`ipykernel` 포함), `Git`

---

## 프로젝트 폴더 구조 (주요 폴더)

```
├── data/
│   ├── raw/         # (1. 캐글 원본 데이터: train/test 이미지)
│   └── yolo/        # (2. YOLO 훈련/스크립트용 데이터)
│       ├── images/  (train/val - B에서 생성)
│       ├── labels/  (train/val - B에서 생성)
│       └── labeled_data/ # (B의 split 스크립트가 읽는 소스 폴더)
│           └── images/ # (.txt 라벨 원본) 
├── runs/
│   └── detect/      # (4. 훈련/제출 결과 저장)
│       ├── train_result/ (훈련 결과 폴더 예시)
│       │   ├── weights/best.pt
│       │   └── submission.csv
│
├── src/               # (A. 공식 메인 스크립트)
│   ├── yolo_train.py
│   └── yolo_submission.py
├── scripts/           # (B. 유틸리티 스크립트)
│   ├── split_dataset.py # (데이터 분할 스크립트)
│   └── ... (기타 스크립트)
│
├── venv-gpu/          # (GPU용 가상환경)
├── .gitignore
├── data.yaml          # (C. YOLO 데이터 경로 설정)
├── yolo_config.yaml   # (D. 공식 훈련 설정 파일)
└── requirements.txt   # (필수 라이브러리 목록)
```