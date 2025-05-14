import cv2
import time

def rtsp_reader_process(shared_images):
    rtsp_url = "rtsp://YOUR_CAMERA_ADDRESS"  # ← 替換成你的攝影機串流網址
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("❌ 無法開啟 RTSP 串流")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 影像抓取失敗，重試中...")
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url)
            continue

        shared_images['original'] = frame
        time.sleep(0.01)
