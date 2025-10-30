# src/yolo_train.py

from ultralytics import YOLO
from datetime import datetime

MODEL_NAME = 'yolo11n'


def train_model():
    model = YOLO(f'{MODEL_NAME}.pt') 

    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    experiment_name = f"{MODEL_NAME}_exp_{timestamp}"

    results = model.train(cfg = 'yolo_train_config.yaml',
        name= experiment_name
    )
    
    print("Training finished. Results saved.")
    
if __name__ == '__main__':
    train_model()
    