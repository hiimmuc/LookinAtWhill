import time

import PIL
from PIL import Image, ImageFile
from tqdm import tqdm
from ultralytics import YOLO

from utils.looking_dataset import *
from utils.looking_network import *
from utils.looking_predict_utils import *

INPUT_SIZE = 51
FONT = cv2.FONT_HERSHEY_SIMPLEX
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Predictor:
    def __init__(self, model_path, device="cpu"):
        self.predictor = YOLO(model_path)
        self.device = device
        self.mode = "joints"
        self.looking_model = self.get_model()

    def get_model(self):
        if self.mode == "joints":
            model = LookingModel(INPUT_SIZE)
            print(self.device)
            if not os.path.isfile(
                f"{parent_path}/checkpoints/LOOKING/LookingModel_LOOK+PIE.p"
            ):
                """
                DOWNLOAD(LOOKING_MODEL, os.path.join(self.path_model, 'Looking_Model.zip'), quiet=False)
                with ZipFile(os.path.join(self.path_model, 'Looking_Model.zip'), 'r') as zipObj:
                    # Extract all the contents of zip file in current directory
                    zipObj.extractall()
                exit(0)"""
                raise NotImplementedError
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        f"{parent_path}/checkpoints/LOOKING/LookingModel_LOOK+PIE.p"
                    ),
                    map_location=self.device,
                )
            )
            model.eval()
        else:
            pass
        return model.to(self.device)

    def predict_look(self, boxes, keypoints, im_size, batch_wise=True):
        try:
            label_look = []
            final_keypoints = []
            if batch_wise:
                if len(boxes) != 0:
                    for i in range(len(boxes)):
                        kps = keypoints[i].cpu().numpy()

                        kps_final = (
                            np.array([kps[0], kps[1], kps[2]]).flatten().tolist()
                        )

                        X, Y = kps_final[:17], kps_final[17:34]

                        X, Y = normalize_by_image_(X, Y, im_size)

                        # X, Y = normalize(X, Y, divide=True, height_=False)
                        kps_final_normalized = (
                            np.array([X, Y, kps_final[34:]]).flatten().tolist()
                        )
                        final_keypoints.append(kps_final_normalized)
                    tensor_kps = torch.Tensor([final_keypoints]).to(self.device)

                    out_labels = (
                        self.looking_model(tensor_kps.squeeze(0))
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                else:
                    out_labels = []
            else:
                if len(boxes) != 0:
                    for i in range(len(boxes)):
                        kps = keypoints[i]
                        # move to cpu
                        kps_final = (
                            np.array([kps[0], kps[1], kps[2]]).flatten().tolist()
                        )
                        X, Y = kps_final[:17], kps_final[17:34]
                        X, Y = normalize_by_image_(X, Y, im_size)
                        # X, Y = normalize(X, Y, divide=True, height_=False)
                        kps_final_normalized = (
                            np.array([X, Y, kps_final[34:]]).flatten().tolist()
                        )
                        # final_keypoints.append(kps_final_normalized)
                        tensor_kps = torch.Tensor(kps_final_normalized).to(self.device)

                        out_labels = (
                            self.looking_model(tensor_kps.unsqueeze(0))
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )
                else:
                    out_labels = []
            return out_labels
        except Exception as e:
            print(e)
            return []

    def post_process(
        self,
        image,
        boxes,
        keypoints,
        pred_labels,
        transparency,
        eyecontact_thresh,
    ):
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        scale = 0.007
        imageWidth, imageHeight, _ = open_cv_image.shape
        font_scale = min(imageWidth, imageHeight) / (10 / scale)

        mask = np.zeros(open_cv_image.shape, dtype=np.uint8)
        for i, label in enumerate(pred_labels):

            if label > eyecontact_thresh:
                color = (0, 255, 0)
                mask = draw_face(mask, keypoints[i], color)
            else:
                color = (255, 0, 0)
                mask = draw_face(mask, keypoints[i], color)

        mask = cv2.erode(mask, (7, 7), iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        # open_cv_image = cv2.addWeighted(open_cv_image, 0.5, np.ones(open_cv_image.shape, dtype=np.uint8)*255, 0.5, 1.0)
        # open_cv_image = cv2.addWeighted(open_cv_image, 0.5, np.zeros(open_cv_image.shape, dtype=np.uint8), 0.5, 1.0)
        open_cv_image = cv2.addWeighted(open_cv_image, 1, mask, transparency, 1.0)
        return open_cv_image

    def predict(
        self,
        image,
        min_conf=0.3,
        iou=0.7,
        imgsz=320,
        transparency=0.5,
        eyecontact_thresh=0.5,
    ):
        results = self.predictor.predict(
            source=image, imgsz=imgsz, conf=min_conf, iou=iou, device=self.device
        )
        result_img = results[0].plot()
        boxes = []
        keypoints = []

        for result in results:
            boxes.append(result.boxes.xywh)
            keypoints.append(result.keypoints.data[0].T)
            # print(result.boxes)
        im_size = result_img.shape[:2]

        pred_labels = self.predict_look(boxes, keypoints, im_size)

        annotated_frame = self.post_process(
            result_img,
            boxes,
            keypoints,
            pred_labels,
            transparency=transparency,
            eyecontact_thresh=eyecontact_thresh,
        )

        return annotated_frame, boxes, keypoints, pred_labels
