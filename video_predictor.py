import os
from datetime import timedelta
import torch
import cv2
from cap_from_youtube import cap_from_youtube
from sapiens_inference import SapiensPredictor, SapiensConfig, SapiensNormalType, SapiensDepthType

# Initialize the configuration and predictor
config = SapiensConfig()
config.normal_type = SapiensNormalType.NORMAL_1B  # Normal configuration
predictor = SapiensPredictor(config)

# Define the YouTube video URL and start time
videoUrl = 'https://youtube.com/shorts/lXfX9qw0yAo?si=SrMq4-PGhBEau91l'
cap = cap_from_youtube(videoUrl, start=timedelta(minutes=0, seconds=0))

# Ensure the capture is open
if not cap.isOpened():
    print("Error: Cannot open video stream")
    exit()

# Get FPS (fallback to 30 if not available)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

# Create output directories for images and video if they don't exist
if not os.path.exists("output_images"):
    os.makedirs("output_images")
if not os.path.exists("output_images/video"):
    os.makedirs("output_images/video")

# Process a single frame to determine processed frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read frame.")
    exit()

results = predictor(frame)
result_height, result_width = results.shape[:2]
print(f"Processed frame dimensions: {result_width}x{result_height}")

# Initialize VideoWriter using the processed frame dimensions.
# Save the video inside 'output_images/video/'
video_path = os.path.join("output_images", "video", "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, (result_width, result_height))

# Save the first processed frame as an image as well
cv2.imwrite("output_images/frame_001.png", results)
out.write(results)

# Set frame processing limit (including the first processed frame)
limit_frames = 10
current_frame = 1

all_results = []

while current_frame < limit_frames:
    ret, frame = cap.read()
    if not ret:
        break

    results = predictor(frame)
    all_results.append(results)
    current_frame += 1
    print(f"Processing frame {current_frame}/{limit_frames}", end='\r')

    # Write processed frame to video and save as image
    out.write(results)
    image_filename = f"output_images/frame_{current_frame:03d}.png"
    cv2.imwrite(image_filename, results)

print("\nProcessing complete.")

# Release all resources
cap.release()
out.release()
cv2.destroyAllWindows()


all_results_tensor = torch.stack(all_results)
print(all_results_tensor.shape)