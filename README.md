# [AI05] 헬스잇(Health Eat) 경구약제 이미지 인식 프로젝트

**[3팀] 최종 결과 보고서**

---

## 1. 프로젝트 개요

본 프로젝트는 BBox(Object Detection)로 시작했으나, 데이터셋의 근본적인 한계(GT 비일관성, 데이터 누수)를 발견하고 **Segmentation(픽셀 단위 분할)**으로 전략을 전면 수정한 도전적인 과정을 담고 있습니다.

* **1단계 (BBox 시도):** `data/yolo` 데이터셋과 `src/yolo_train.py`를 기반으로 Baseline 모델(`runs/yolov8_exp...`)을 구축하고 최적화를 시도했습니다.
* **2단계 (한계 발견):** `train/test` 데이터 누수 및 BBox 라벨의 근본적 한계를 발견하여 BBox 전략을 폐기했습니다.
* **3단계 (전략 수정):** `scripts/json_2_seg(SAM).py` 등으로 `data/yolo_seg`라는 고품질 Segmentation 데이터셋을 원점에서부터 신규 구축했습니다.
* **4단계 (고도화):** `src/cbam.py` 모듈을 커스텀 적용한 `src/yolo_train_seg.py`로 Segmentation 모델(`runs/yolo11n...seg`)을 훈련시켰습니다.
* **5단계 (시스템화):** 완성된 모델을 **FastAPI(백엔드)** 및 **Android(프론트엔드)**와 연동하여 실시간 서빙 시스템을 구축했습니다.

---

## 2. 프로젝트 최종 보고서

프로젝트의 전체 과정, 기술적 접근, 핵심 성과 및 교훈을 담은 최종 보고서(발표 자료) PDF입니다.

* **[최종 발표자료 PDF 링크 (여기를 수정하세요)](./[AI05]경구약제_최종보고서_3팀.pdf)**

---

## 3. 팀원 및 역할

| 이름 | 역할 (담당) | GitHub (선택) |
| :--- | :--- | :--- |
| **오병주 (PM)** | 프로젝트 총괄, 방향 제시, 발표/문서 | [@OH-BJ] |
| **이상윤 (DE)** | Segmentation 데이터셋(SAM) 구축, CBAM 커스텀 아키텍처, API 서버(FastAPI) 구축, 데이터 누수 발견 | [@SYLforge] |
| **서준범 (MA)** | BBox 파이프라인(`src/`) 및 Baseline 모델 구축, BBox 아키텍처 테스트 android-poc 구축 | [@Seo-Junbeom] |
| **김승우 (EL)** | BBox/Segmentation 모델 튜닝 및 데이터 증강 실험 (HSV), SAM활용 train데이터 추가 | [@carsy078-maker] |

---

## 4. 코드 실행 방법

**설치 (Installation)**
```bash
# 1. 저장소 복제
git clone [https://github.com/OH-BJ/HealthEat-Pill-Detection.git](https://github.com/OH-BJ/HealthEat-Pill-Detection.git)
cd HealthEat-Pill-Detection

# 2. 패키지 설치
pip install -r requirements.txt

**BBox 모델 훈련 (Baseline)**
* `src/yolo_train.py`: BBox 훈련 스크립트
* `yolo_train_config.yaml`: BBox 훈련용 설정 파일 (모델, 하이퍼파라미터)
* `data.yaml`: BBox 데이터셋 경로 설정 파일

```bash
# BBox 훈련 실행
python src/yolo_train.py --config yolo_train_config.yaml

**Segmentation 모델 훈련 (최종)**
* `src/yolo_train_seg.py`: Segmentation 훈련 스크립트
* `data_seg.yaml`: Segmentation 데이터셋 경로 및 설정 파일

```bash
# Segmentation (CBAM) 훈련 실행
python src/yolo_train_seg.py --config data_seg.yaml

**Kaggle 제출**
* `src/yolo_prediction.py` / `src/yolo_submission.py`: Kaggle 제출 파일(`submission.csv`) 생성 스크립트

---

## 5. 백엔드(FastAPI) & 프론트엔드(Android)

AI 모델을 서빙하는 API 서버 및 모바일 앱 코드입니다. (모델 가중치 등 용량이 큰 파일 포함)

* **[FastAPI(백엔드) 및 Android(프론트엔드) 코드 Google Drive 링크 (백엔드: https://drive.google.com/file/d/1b6CUoio83wnIOBJbvn7J6RtKiy3RAre1/view?usp=sharing
프론트엔드: https://drive.google.com/file/d/10S_qu4dR2eQZ-d1eRflEOLTdBLZ3xVht/view?usp=sharing)]**

---

## 6. 팀원별 협업 일지 (필수 제출)

* **오병주:** [여기에 오병주 님의 협업 일지 링크를 넣어주세요]
* **이상윤:** [여기에 이상윤 님의 협업 일지 링크를 넣어주세요]
* **서준범:** [여기에 서준범 님의 협업 일지 링크를 넣어주세요]
* **김승우:** [여기에 김승우 님의 협업 일지 링크를 넣어주세요]

---

## 7. 최종 성과 (BBox vs Segmentation)

| 성능 지표 | BBox Baseline 모델 (실패) | Segmentation (CBAM) 모델 (최종) |
| :--- | :---: | :---: |
| **`mAP`** (Mask/Box) | 0.** | **0.** |
| **`mIoU`** (픽셀 정밀도) | - (측정 불가) | **0.** |
| **`Latency`** (추론 속도) | 0.** ms | 0.** ms |