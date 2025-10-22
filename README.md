# 헬스잇(Health Eat) 경구약제 이미지 인식 프로젝트

> 사용자가 촬영한 약 사진으로부터 알약의 종류와 위치를 탐지하는 객체 탐지(Object Detection) 모델 개발 프로젝트입니다.

---

## ⚠️ 주요 데이터 이슈 및 해결 전략 (2025-10-22 기준)
- **발견된 문제**: EDA 결과, 라벨 누락(추정 1/3), 희귀 클래스 이미지 링크 깨짐(404), 블러 이미지(약 7.8%) 등 심각한 데이터 품질 문제 식별됨.
- **최종 해결 방안**: **CVAT + 베이스라인 YOLO 모델**을 활용한 **'모델 기반 자동 라벨링'**으로 누락/오류 라벨을 보완하기로 결정함.

---

##  주요 기능
- 이미지 속 최대 4개 알약의 클래스(이름) 탐지
- 이미지 속 알약의 위치(Bounding Box) 탐지
- [추후 추가될 기능]
- [추후 추가될 기능]

---

##  팀 멤버 및 역할
| 역할 | 이름 | GitHub |
| :--- | :--- | :--- |
|  **Project Manager** | [오병주] | [@OH-BJ](https://github.com/OH-BJ) |
|  **Data Engineer** | [이상윤] | [@SYLforge](https://github.com/SYLforge) |
|  **Model Architect** | [서준범] | [@Seo-Junbeom](https://github.com/Seo-Junbeom) |
|  **Experimentation Lead** | [김승우] | [@carsy078-maker](https://github.com/carsy078-maker) |

---

##  최종 결과물 링크
| 구분 | 링크 |
| :--- | :--- |
|  **최종 보고서** | [보고서 링크 삽입 예정] |
|  **오병주 협업 일지** | [개인별 링크 삽입 예정] |
|  **이상윤 협업 일지** | [개인별 링크 삽입 예정] |
|  **서준범 협업 일지** | [개인별 링크 삽입 예정] |
|  **김승우 협업 일지** | [개인별 링크 삽입 예정] |

---

##  개발 환경
- **언어**: `Python 3.x`
- **프레임워크**: `PyTorch`
- **주요 라이브러리**: `OpenCV`, `Pandas`, `Numpy`
- **개발 도구**: `Visual Studio Code`

---

##  폴더 구조
├── data/ # 데이터셋 (raw, processed) 
├── notebooks/ # EDA 및 실험용 노트북 
├── output/ # EDA 결과물 (그래프 등) 
├── runs/ # YOLOv8 학습 결과 (모델 가중치, 로그) 
├── scripts/ # 데이터 변환 등 유틸리티 스크립트 
├── src/ # YOLO 관련 핵심 소스 코드 
├── fonts/ # 한글 폰트 파일 
├── .gitignore # Git 무시 목록 
├── README.md # 프로젝트 설명서 
├── data.yaml # YOLO 데이터셋 설정 파일 
└── yolo_config.yaml # YOLO 모델 학습 설정 파일