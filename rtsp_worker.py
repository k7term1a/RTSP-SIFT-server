import cv2
import time
import numpy as np
from sift_matcher import match_sift_with_boxes

def rtsp_process(queue, shared_images, pattern_path):
    rtsp_url = "rtsp://your_stream_url"
    cap = cv2.VideoCapture(rtsp_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        shared_images['original'] = frame.copy()

        try:
            pattern = cv2.imread(pattern_path)
            if pattern is None:
                continue

            result_img, matched, num_matches = match_sift_with_boxes(pattern, frame)
            shared_images['processed'] = result_img

        except Exception as e:
            print(f"SIFT 比對錯誤：{e}")
            continue

        time.sleep(0.1)
