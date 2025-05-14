import cv2
import time

def rtsp_reader_process(shared_images):
    # rtsp_url = "rtsp://YOUR_CAMERA_ADDRESS"  # ← 替換成你的攝影機串流網址
    # cap = cv2.VideoCapture(rtsp_url)
    camera_device = "/dev/video0"
    cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("❌ 無法開啟 RTSP 串流")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 影像抓取失敗，重試中...")
            time.sleep(1)
            cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
            continue

        shared_images['original'] = frame
        time.sleep(0.01)
