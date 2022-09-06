import time

import cv2
import torch
import numpy as np
from dotenv import load_dotenv
from imutils.video import VideoStream

from utils.download_model import safe_download
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.logger import logger


load_dotenv()


@torch.no_grad()
def predict_object(frame):
    # Padded resize
    img = letterbox(frame, imgsz, stride=stride)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, visualize=False)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
    predicted_classes = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame[i].shape).round()
            # Collect results
            for c in det[:, -1].unique():
                ct = (det[:, -1] == c).sum()
                predicted_classes.append({"name": names[int(c)], "qty": int(ct)})
    return predicted_classes


if __name__ == '__main__':
    # Object detection model file path
    weights = "weights/best.pt"
    imgsz = 640
    conf_thres = 0.8
    iou_thres = 0.45
    max_det = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Download and load object detection model
    safe_download("https://github.com/meontechno/frictionless_weights/raw/main/v10.0/best.pt")
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names

    vid_stream = VideoStream(src="/dev/video0").start()

    logger.info(f"Device selected: {device}")
    c = 0
    fl = 10
    fc = np.zeros(fl)
    li = 0

    logger.info(f"Predictions:")
    while True:
        # Object detection
        t1 = time.perf_counter()
        frame = vid_stream.read()

        if frame is None:
            logger.error("Empty camera frame")
            time.sleep(0.1)

        detected_objects = predict_object(frame)
        logger.info(f"{detected_objects}")

        cv2.imshow("Original Frame", frame)
        if li > fl-1:
            li = 0
        fc[li] = 1/(time.perf_counter()-t1)

        if cv2.waitKey(1) == ord('q'):
            logger.info(f"FPS: {np.average(fc):.2f}")
            break
        li += 1
