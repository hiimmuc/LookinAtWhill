import math
import os
import time
from decimal import *
from decimal import Decimal

import cv2
import numpy as np

# import pycuda.autoinit
import pyrealsense2 as rs
from PIL import Image
from ultralytics import YOLO

from utils.predictor import Predictor

THETA = 0.0  # angle of the camera with respect to the x-axis

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# get camera intrinsics
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# intr = pipeline.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)


# Start streaming
pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
# Load the YOLOv8 model

predictor = Predictor("./checkpoints/YOLO/yolov8n-pose.pt", device="cuda:0")


try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Visualize the results on the frame
        annotated_frame, boxes, keypoints, pred_labels = predictor.predict(color_image)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # get distance from the camera to center of the bounding box
        try:
            if len(boxes) > 0:

                box = boxes[0][0]
                x, y, w, h = box
                center_x = x + w / 2
                center_y = y + h / 2
                dist = depth_frame.get_distance(int(center_x), int(center_y))
                print(f"Distance to center of the bounding box: {dist} meters")
                #

                # Xtemp = dist * (x - intr.ppx) / intr.fx
                # Ytemp = dist * (y - intr.ppy) / intr.fy
                # Ztemp = dist

                # Xtarget = (
                #     Xtemp - 35
                # )  # 35 is RGB camera module offset from the center of the realsense
                # Ytarget = -(Ztemp * math.sin(THETA) + Ytemp * math.cos(THETA))
                # Ztarget = Ztemp * math.cos(THETA) + Ytemp * math.sin(THETA)

                # coordinates_text = (
                #     "("
                #     + str(
                #         Decimal(str(Xtarget)).quantize(Decimal("0"), rounding=ROUND_HALF_UP)
                #     )
                #     + ", "
                #     + str(
                #         Decimal(str(Ytarget)).quantize(Decimal("0"), rounding=ROUND_HALF_UP)
                #     )
                #     + ", "
                #     + str(
                #         Decimal(str(Ztarget)).quantize(Decimal("0"), rounding=ROUND_HALF_UP)
                #     )
                #     + ")"
                # )

                # coordinat = Decimal(str(Ztarget)).quantize(
                #     Decimal("0"), rounding=ROUND_HALF_UP
                # )
                # print(
                #     "Distance to Camera at distance : {2:0.2f} mm".format(coordinat),
                #     end="\r",
                # )

                cv2.putText(
                    annotated_frame,
                    "Distance: " + str(round(dist, 2)) + "m",
                    (int(x - 180), int(y + 30)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    1,
                )
        except Exception as e:
            pass

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(
                annotated_frame,
                dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                interpolation=cv2.INTER_AREA,
            )
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((annotated_frame, depth_colormap))

        # Show images
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", images)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:

    # Stop streaming
    pipeline.stop()
