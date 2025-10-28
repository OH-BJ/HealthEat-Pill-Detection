# 헬스잇(Health Eat) 경구약제 이미지 인식 프로젝트

> 사용자가 촬영한 약 사진으로부터 알약의 종류와 위치를 탐지하는 객체 탐지(Object Detection) 모델 개발 프로젝트입니다.

---

## 주요 기능
- 이미지 속 최대 4개 알약의 클래스(73종) 탐지
- 이미지 속 알약의 위치(Bounding Box) 탐지

---

## 실행 방법 (How-to-Run)

### 환경 설정 (최초 1회)
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

### 데이터 준비 (현재 데이터 이슈 확인)
1.  **주의:** 현재 `data/ai05-level1-project/train_images` 폴더에 테스트 이미지가 포함된 문제가 확인되었습니다.
2.  원본 데이터(`data/ai05-level1-project/train_images` 등) 정제 작업을 진행합니다. (테스트 이미지 제거 등)
3.  정제된 데이터를 사용하여 `scripts/split_dataset.py` 스크립트로 YOLO 훈련용 데이터(`data/yolo/images/train_images`, `data/yolo/labels/train_images` 등)를 생성합니다. (YOLO는 labels 폴더 하위 경로명이 images 폴더 하위 경로명과 같아야 합니다.)

### 모델 훈련 (공식 스크립트)
1.  **`yolo_config.yaml`** 파일을 열어 훈련 설정을 수정합니다.
    *(예: `architecture`, `input_size`, `epochs`, `batch_size` 등)*
2.  "공식 훈련 스크립트" (`src/yolo_train.py`)를 실행합니다.
    ```bash
    # data.yaml 사용 (또는 yolo_config.yaml에 지정된 경로)
    python src/yolo_train.py
    ```
3.  훈련 결과는 `runs/` 폴더 내에 저장됩니다.

### 캐글 제출 (공식 스크립트)
1.  **`src/yolo_submission.py`** 파일을 엽니다.
2.  **`RUN_DIR`**에 훈련이 완료된 폴더 경로(예: `runs/ex)훈련결과`)를 정확히 입력합니다.
3.  **`TEST_IMAGES_DIR`**에 테스트 이미지 폴더 경로(`data/ai05-level1-project/test_images`)가 맞는지 확인합니다.
4.  (선택) 스크립트 상단의 **`CONF_THRESHOLD`**(예: 0.005), **`IOU_THRESHOLD`**(예: 0.7) 값을 조정합니다.
5.  "공식 제출 스크립트"를 실행합니다.
    ```bash
    python src/yolo_submission.py
    ```
6.  `RUN_DIR` 폴더 안에 생성된 `submission.csv` 파일을 캐글에 제출합니다.

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
.
├── data/
│   ├── ai05-level1-project/ # (1. 캐글 원본 데이터)
│   │   ├── test_images/
│   │   └── train_images/  # (주의: 현재 test 이미지가 포함된 상태)
│   └── yolo/               # (2. YOLO 훈련용 데이터)
│       ├── images/  (train/val)
│       ├── labels/  (train/val)
│       └── labeled_data/ # (split 스크립트용 라벨 소스 - 확인 필요)
│           └── images/ # (.txt 라벨)
├── runs/                # (4. 훈련/제출 결과 저장 - 훈련 시 자동 생성)
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