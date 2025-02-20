import os
from datetime import timedelta
import torch
import cv2
from cap_from_youtube import cap_from_youtube
from sapiens_inference.normal import SapiensNormal, SapiensNormalType, draw_normal_map

videoUrl = 'https://youtu.be/comTX7mxSzU?si=LL2ilfJ6tDXeFTkQ'
cap = cap_from_youtube(videoUrl, start=timedelta(minutes=2, seconds=54))

# Define your desired dtype if needed (e.g., torch.float16)
dtype = torch.float16
estimator = SapiensNormal(SapiensNormalType.NORMAL_03B, dtype=dtype)

# Create an output directory for the saved images
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame to obtain the normal map
    normal_map = estimator(frame)
    normal_image = draw_normal_map(normal_map)
    combined = cv2.addWeighted(frame, 0.3, normal_image, 0.8, 0)

    # Construct a filename and save the image
    filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    cv2.imwrite(filename, combined)
    frame_count += 1

cap.release()
