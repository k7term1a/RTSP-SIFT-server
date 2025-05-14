import cv2
import time
from sift_matcher import match_sift_with_boxes

def rtsp_process(queue, shared_images, pattern_path):
    # rtsp_url = "rtsp://YOUR_CAMERA_ADDRESS"
    # cap = cv2.VideoCapture(rtsp_url)
    camera_device = "/dev/video0"
    cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("❌ 無法開啟 RTSP 串流")
        return

    pattern_img = cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 影像抓取失敗，重試中...")
            time.sleep(1)
            cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
            continue

        shared_images["original"] = frame.copy()

        if pattern_img is not None:
            processed = match_sift_with_boxes(pattern_img, frame)
            shared_images["processed"] = processed

        time.sleep(0.05)  # 每 50ms 處理一次
