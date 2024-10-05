import os
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

COCO_PERSON_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
]
COCO_HEAD = [[1, 2]]


def convert(data):
    """X = []
    Y = []
    C = []
    A = []
    i = 0
    print(data)
    while i < 51:
        X.append(data[i])
        Y.append(data[i+1])
        C.append(data[i+2])
        i += 3
    """
    data = data.cpu().numpy()
    X, Y, C = data[0], data[1], data[2]
    A = np.array([X, Y, C]).flatten().tolist()
    return X, Y, C, A


def normalize_by_image_(X, Y, image_size):
    """
    Normalize the image according to the paper.
    Args:
        - X: array of X positions of the keypoints
        - Y: array of Y positions of the keypoints
        - Image: Image array
    Returns:
        returns the normalized arrays
    """

    image_width, image_height = image_size
    try:
        center_p = (int((X[11] + X[12]) / 2), int((Y[11] + Y[12]) / 2))
        X_new = np.array(X) / image_width
        Y_new = np.array(Y) - center_p[1]

        width = abs(np.max(X) - np.min(X))
        height = abs(np.max(Y) - np.min(Y))

        X_new = X_new + ((np.array(X) - center_p[0]) / width)
        Y_new /= height
    except Exception as e:
        print(e)
        X_new, Y_new = X, Y

    return X_new, Y_new


def prepare_coco_kps(kps_in):
    """Convert from a list of 51 to a list of 3, 17"""

    assert len(kps_in) % 3 == 0, "keypoints expected as a multiple of 3"
    xxs = kps_in[0:][::3]
    yys = kps_in[1:][::3]  # from offset 1 every 3
    ccs = kps_in[2:][::3]

    return [xxs, yys, ccs]


def preprocess_coco(annotations, im_size=None, enlarge_boxes=True, min_conf=0.0):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []
    enlarge = 1 if enlarge_boxes else 2  # Avoid enlarge boxes for social distancing

    for result in annotations:
        kps = prepare_coco_kps(
            result.keypoints.data[0].T.cpu().numpy().flatten().tolist()
        )
        box = result.boxes.xywh[0].cpu().numpy().flatten().tolist()

        try:
            # print(result)
            conf = result.boxes.conf
            # Enlarge boxes
            delta_h = (box[3]) / (10 * enlarge)
            delta_w = (box[2]) / (5 * enlarge)
            # from width height to corners
            x, y, w, h = box
            box = [x - w / 2, y - h / 2, w, h]
            box[2] += box[0]
            box[3] += box[1]

        except KeyError:
            all_confs = np.array(kps[2])
            score_weights = np.ones(17)
            score_weights[:3] = 3.0
            score_weights[5:] = 0.1
            # conf = np.sum(score_weights * np.sort(all_confs)[::-1])
            conf = float(np.mean(all_confs))
            # Add 15% for y and 20% for x
            delta_h = (box[3] - box[1]) / (7 * enlarge)
            delta_w = (box[2] - box[0]) / (3.5 * enlarge)
            assert delta_h > -5 and delta_w > -5, "Bounding box <=0"

        # box[0] -= delta_w
        # box[1] -= delta_h
        # box[2] += delta_w
        # box[3] += delta_h
        print(result.boxes.xyxy[0], box)

        # Put the box inside the image
        if im_size is not None:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(box[2], im_size[0])
            box[3] = min(box[3], im_size[1])

        if conf >= min_conf:
            box.append(conf)
            boxes.append(box)
            keypoints.append(kps)

    return boxes, keypoints


def draw_face(img, kps, color):
    X, Y, C, _ = convert(kps)

    height = abs(Y[0] - Y[-1])

    head = COCO_HEAD[0]
    c1, c2 = head

    # radius = abs(int(X[c1])- int(X[c2]))
    radius = int(0.4 * height)
    # print(X, Y)
    try:
        center = (
            (int((X[c1] + X[c2]) / 2), int((Y[c1] + Y[c2]) / 2))
            if any(e > 0 for e in [X[c1], X[c2], Y[c1], Y[c2]])
            else (X[0], Y[0])
        )
        img = cv2.circle(img, center, radius, color, -1)
        img = cv2.circle(img, center, radius, (255, 255, 255), 2)
    except Exception as e:
        pass
    return img


def run_and_kps(img, kps, label):
    blk = np.zeros(img.shape, np.uint8)
    X, Y, C, _ = convert(kps)
    if label > 0.5:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    for i in range(len(Y)):
        blk = cv2.circle(blk, (int(X[i]), int(Y[i])), 1, color, 2)
    # img = cv2.addWeighted(img, 1.0, blk, 0.55, 1)
    return blk
