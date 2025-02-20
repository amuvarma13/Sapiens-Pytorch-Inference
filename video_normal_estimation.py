from datetime import timedelta
import torch
import cv2
import os
from cap_from_youtube import cap_from_youtube
from sapiens_inference.normal import SapiensNormal, SapiensNormalType, draw_normal_map

# Ensure output directory exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

videoUrl = 'https://youtu.be/comTX7mxSzU?si=LL2ilfJ6tDXeFTkQ'
cap = cap_from_youtube(videoUrl, start=timedelta(minutes=2, seconds=54))

dtype = torch.float32  # Ensure dtype is defined
estimator = SapiensNormal(SapiensNormalType.NORMAL_03B, dtype=dtype)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    normal_map = estimator(frame)
    normal_image = draw_normal_map(normal_map)
    combined = cv2.addWeighted(frame, 0.3, normal_image, 0.8, 0)

    output_path = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
    cv2.imwrite(output_path, combined)
    
    frame_count += 1
