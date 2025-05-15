import cv2
import time

def rtsp_reader_process(shared_images):
    cap = cv2.VideoCapture(10)

    if not cap.isOpened():
        print("❌ 無法開啟 RTSP 串流")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 影像抓取失敗，重試中...")
            time.sleep(1)
            cap = cv2.VideoCapture(10)
            continue

        shared_images['original'] = frame
        time.sleep(0.01)
