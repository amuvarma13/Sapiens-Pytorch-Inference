from datetime import timedelta
import torch
import cv2
import numpy as np
from cap_from_youtube import cap_from_youtube
from sapiens_inference import SapiensPredictor, SapiensConfig, SapiensNormalType, SapiensDepthType

# Initialize the configuration and predictor
config = SapiensConfig()
# config.dtype = torch.float16
config.normal_type = SapiensNormalType.NORMAL_1B  # Disabled by default
# config.depth_type = SapiensDepthType.DEPTH_03B  # Disabled by default
predictor = SapiensPredictor(config)

# Define the YouTube video URL and start time
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
    fps = 30  # Fallback if FPS is not available

# Attempt to get total frame count (may not be available for some streams)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames > 0:
    print(f"Total frames: {total_frames}")
else:
    print("Total frame count not available.")

# Define the codec and create VideoWriter object to save as MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    print("Error: VideoWriter did not open properly")
    exit()

# Limit processing to 10 frames (or 100 frames as desired)
limit_frames = 10
current_frame = 0

while True:
    ret, frame = cap.read()
    if not ret or current_frame >= limit_frames:
        break

    current_frame += 1
    print(f"Processing frame {current_frame}/{limit_frames}", end='\r')

    # Process the frame using the predictor
    results = predictor(frame)

    # Debug: print shape and dtype of results
    # Uncomment the next line to see the output frame properties
    # print(results.shape, results.dtype)

    # Ensure the results are in the proper format (uint8, 3 channels, correct size)
    # Convert results to uint8 if they are not already (adjust conversion if needed)
    if results.dtype != 'uint8':
        # Assuming results are normalized in [0, 1] or are float, scale to 255
        results = (results * 255).clip(0, 255).astype('uint8')

    # If the predictor returns an image in RGB, convert it to BGR
    if results.shape[2] == 3:
        results = cv2.cvtColor(results, cv2.COLOR_RGB2BGR)

    # Resize the processed frame if its dimensions don't match the expected size
    if (results.shape[1], results.shape[0]) != (frame_width, frame_height):
        results = cv2.resize(results, (frame_width, frame_height))

    # Write the processed frame to the output video file
    out.write(results)

print("\nProcessing complete.")

# Release all resources
cap.release()
out.release()
cv2.destroyAllWindows()
