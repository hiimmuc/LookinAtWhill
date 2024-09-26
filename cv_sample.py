import cv2
from ultralytics import YOLO

from utils.predictor import Predictor

# Load the YOLOv8 model
# model = YOLO()
predictor = Predictor("./checkpoints/YOLO/yolov8n-pose.pt", device="cuda:0")
# Open the video file
source = 0
cap = cv2.VideoCapture(source)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    width, height = frame.shape[:2]

    if success:
        # Visualize the results on the frame
        annotated_frame = predictor.predict(frame)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
