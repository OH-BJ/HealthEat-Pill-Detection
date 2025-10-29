# src/yolo_train.py

import torch
from ultralytics import YOLO
import yaml
import os
from datetime import datetime
from cbam import CBAM
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -----------------------------
# 한글 폰트 설정
# -----------------------------
font_path = "fonts/NanumBarunGothic.ttf"
if Path(font_path).exists():
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumBarunGothic')
else:
    print(f"⚠️ 경고: {font_path} 없음. 기본 폰트로 진행.")
    plt.rc('font', family='DejaVu Sans')

# -----------------------------
# 설정 로드
# -----------------------------
def load_config(config_path='yolo_config.yaml'):
    config_path = Path(__file__).parent / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"config 파일 없음: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# -----------------------------
# 모델 훈련 함수
# -----------------------------
def train_model(config):
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"✅ Using device: {device}")

    # 1. YOLOv11-seg 모델 로드
    arch = config.get('architecture', 'yolo11n')
    model = YOLO(f'{arch}-seg.pt')
    print(f"✅ 모델 로드 완료: {arch}-seg.pt")

    # 2. CBAM 추가 (YOLOv11n backbone 확인 후 삽입)
    try:
        layer_index = config.get('cbam_layer', 6)
        target_layer = model.model.model[layer_index]
        if hasattr(target_layer, 'add_module'):
            cbam_layer = CBAM(planes=128, ratio=16)
            target_layer.add_module('cbam', cbam_layer)
            print(f"✅ CBAM 모듈 layer {layer_index}에 추가 완료.")
        else:
            print(f"⚠️ layer {layer_index}에 CBAM 추가 불가 (구조 확인 필요).")
    except Exception as e:
        print(f"⚠️ CBAM 추가 중 오류: {e}")

    # 3. 실험 이름 및 로그 디렉토리
    exp_name = f"{arch}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    print(f"📁 Experiment name: {exp_name}")

    # 4. 훈련 실행
    print("🚀 YOLOv11 segmentation 훈련 시작...")
    results = model.train(
        data=config['data_yaml_path'],
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['input_size'],
        lr0=config['learning_rate'],
        device=device,
        project='runs',
        name=exp_name,
        pretrained=True,
        patience=config.get('patience', 50),
        save_period=10,  # 10 epoch마다 가중치 저장
        verbose=True
    )

    # 5. mAP 플롯 저장
    if hasattr(results, 'metrics'):
        metrics = results.metrics
        if 'mAP50' in metrics and 'mAP50-95' in metrics:
            plt.figure(figsize=(8, 5))
            plt.plot(metrics['mAP50'], label='mAP@0.5')
            plt.plot(metrics['mAP50-95'], label='mAP@0.5:0.95')
            plt.xlabel('Epoch')
            plt.ylabel('mAP')
            plt.title('YOLOv11 Pill Segmentation Training Performance')
            plt.legend()
            out_path = f"runs/{exp_name}/mAP_plot.png"
            plt.savefig(out_path)
            print(f"📊 mAP 플롯 저장 완료: {out_path}")
        else:
            print("⚠️ metrics에서 mAP 키를 찾을 수 없습니다.")
    else:
        print("⚠️ results.metrics 속성이 없습니다. (ultralytics 버전 확인 필요)")

    print("✅ 훈련 완료. 결과는 runs/ 디렉토리에 저장됨.")


# -----------------------------
# 실행
# -----------------------------
if __name__ == '__main__':
    try:
        cfg = load_config()
        train_model(cfg)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
