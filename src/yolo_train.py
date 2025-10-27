# src/yolo_train.py

import torch
from ultralytics import YOLO
import yaml
import os
from datetime import datetime

# YAML 설정 파일을 읽어 딕셔너리로 반환하는 함수
def load_config(config_path='yolo_config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train_model(config):
    # 1. 환경 설정
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # 2. 모델 로드
    model = YOLO(f'{config["architecture"]}.pt') 
    print(f"YOLO Model '{config['architecture']}' loaded.")


    # 3. 학습 실행
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    experiment_name = f"yolov8_exp_{timestamp}"

    results = model.train(
        data=config['data_yaml_path'], 
        epochs=config['epochs'],
        patience=config['patience'], 
        batch=config['batch_size'],
        imgsz=config['input_size'],
        device=device.type, 
        lr0=config['learning_rate'], 
        
        # YAML 파일에서 읽어온 증강 하이퍼파라미터를 직접 전달
        mosaic=config.get('mosaic', 1.0),
        hsv_h=config.get('hsv_h', 0.015),
        hsv_s=config.get('hsv_s', 0.7),
        hsv_v=config.get('hsv_v', 0.4),
        degrees=config.get('degrees', 0),
        
        project='runs',
        name=experiment_name,
    )
    
    print("Training finished. Results saved.")
    
if __name__ == '__main__':
    try:
        config = load_config()
        train_model(config)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 파일을 확인하십시오.")