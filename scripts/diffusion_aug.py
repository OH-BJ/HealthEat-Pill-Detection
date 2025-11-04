import torch
from diffusers import StableDiffusionXLInpaintPipeline
from ultralytics import YOLO, SAM
from PIL import Image
import numpy as np
import os

# 1. 모든 알약 마스크 생성 (다중 객체 합치기)
def get_pill_masks(img_path, yolo_model_path="best.pt", sam_model_path="sam2.1_b.pt"):
    yolo_model = YOLO(yolo_model_path)
    sam_model = SAM(sam_model_path)
    
    yolo_results = yolo_model(img_path)
    if len(yolo_results[0].boxes) == 0:
        raise ValueError("No pill detections")
    
    # 모든 알약 bbox 추출
    bboxes = yolo_results[0].boxes.xyxy.cpu().numpy()  # [N, 4] 배열 (N=3~4)
    
    combined_mask = np.zeros((yolo_results[0].orig_shape[0], yolo_results[0].orig_shape[1]), dtype=np.uint8)  # 원본 크기 마스크 초기화
    
    for bbox in bboxes:
        # SAM으로 각 알약 마스크 생성
        sam_results = sam_model(img_path, bboxes=[bbox])
        if len(sam_results[0].masks) > 0:
            mask = sam_results[0].masks.data[0].cpu().numpy()
        else:
            # Fallback: YOLO seg masks (인덱스 맞춤)
            mask_idx = np.where((yolo_results[0].boxes.xyxy.cpu().numpy() == bbox).all(axis=1))[0][0]
            mask = yolo_results[0].masks.data[mask_idx].cpu().numpy()
        
        mask = (mask > 0.5).astype(np.uint8) * 255
        combined_mask = np.maximum(combined_mask, mask)  # 모든 마스크 합치기 (오버랩 OK)
    
    # Inverse: 배경 마스크 (알약=0 고정, 배경=255 변형)
    background_mask = Image.fromarray(255 - combined_mask).convert("L")
    return background_mask

# 2. Inpainting으로 배경 변형 (원본 크기 유지)
def generate_background_aug(original_image, background_mask, prompt="realistic kitchen table with soft shadows, no pills", strength=0.75, num_inference_steps=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Pipe 로드: 기본 CLIP text_encoder_2 자동 (custom T5 제거)
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,  # SDXL repo safetensors 풀셋
        variant=None,
        safety_checker=None  # 경고 무시
    ).to(device)
    
    pipe.enable_model_cpu_offload()
    
    print(f"Pipeline loaded on {device} (cu126, 4090) – inpainting size {original_image.size} with CLIP encoders...")
    
    augmented = pipe(
        prompt=prompt,
        image=original_image,
        mask_image=background_mask,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.5,
        height=original_image.height,
        width=original_image.width
    ).images[0]
    return augmented
# 3. 메인 루프 (원본 크기 유지)
dataset_dir = "data/yolo_seg/images/train/"
output_dir = "data/yolo_seg/images/train_aug/"
os.makedirs(output_dir, exist_ok=True)

yolo_model_path = "best.pt"  # 실제 경로

for img_file in os.listdir(dataset_dir)[:5]:
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img_path = os.path.join(dataset_dir, img_file)
    original_img = Image.open(img_path)  # 리사이즈 제거 – 원본 976x1280 유지
    
    try:
        print(f"Processing {img_file} (size: {original_img.size})...")
        bg_mask = get_pill_masks(img_path, yolo_model_path)  # 모든 알약 마스크 합침
        
        prompts = [
            "realistic wooden table with warm sunset light, clean background",
            "dark kitchen counter with overhead lamp, realistic shadows",
            "white marble surface with bright daylight, minimalistic"
        ]
        for i, prompt in enumerate(prompts):
            print(f"Generating {i+1}/3 for {img_file}...")
            aug_img = generate_background_aug(original_img, bg_mask, prompt)
            aug_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_bg_aug_{i}.png")
            aug_img.save(aug_path)
            print(f"Generated: {aug_path} (size: {aug_img.size})")
    except Exception as e:
        print(f"Error on {img_file}: {e}")

print("배경 증강 완료! output_dir PNG 크기/마스크 확인하세요.")