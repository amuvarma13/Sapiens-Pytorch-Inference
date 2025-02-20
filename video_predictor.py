from datetime import timedelta
import torch
import cv2
from cap_from_youtube import cap_from_youtube

from sapiens_inference import SapiensPredictor, SapiensConfig, SapiensNormalType, SapiensDepthType

config = SapiensConfig()
# config.dtype = torch.float16
config.normal_type = SapiensNormalType.NORMAL_1B  # Disabled by default
# config.depth_type = SapiensDepthType.DEPTH_03B  # Disabled by default
predictor = SapiensPredictor(config)

videoUrl = 'https://youtube.com/shorts/lXfX9qw0yAo?si=SrMq4-PGhBEau91l'
cap = cap_from_youtube(videoUrl, start=timedelta(minutes=0, seconds=0))

# Ensure the capture is open
if not cap.isOpened():
    print("Error: Cannot open video stream")
    exit()

# Retrieve frame dimensions and FPS from the capture
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback if FPS is not available

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame using the predictor
    results = predictor(frame)

    # Write the processed frame to the output file
    out.write(results)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
