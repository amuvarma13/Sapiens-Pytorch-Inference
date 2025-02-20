import os
from datetime import timedelta
import torch
import cv2
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

# Create output directories for video and images
if not os.path.exists("output_images"):
    os.makedirs("output_images")

# Define the codec and create VideoWriter object to save as MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# Limit processing to a certain number of frames (e.g., 10)
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
    print(results.shape)

    # Write the processed frame to the output video file
    out.write(results)

    # Save each processed frame as an image file in 'output_images/' folder
    image_filename = f"output_images/frame_{current_frame:03d}.png"
    cv2.imwrite(image_filename, results)

print("\nProcessing complete.")

# Release all resources
cap.release()
out.release()
cv2.destroyAllWindows()
