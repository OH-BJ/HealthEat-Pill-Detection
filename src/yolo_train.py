# src/yolo_train.py

import torch
from ultralytics import YOLO
import yaml
import os
from datetime import datetime

MODEL_NAME = 'yolo11n'


def train_model():
   
    # 2. 모델 로드
    model = YOLO(f'{MODEL_NAME}.pt') 


    # 3. 학습 실행
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    experiment_name = f"{MODEL_NAME}_exp_{timestamp}"

    results = model.train(cfg = 'yolo_train_config.yaml',
        name= experiment_name
    )
    
    print("Training finished. Results saved.")
    
if __name__ == '__main__':
    train_model()
    