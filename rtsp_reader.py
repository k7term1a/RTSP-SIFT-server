import cv2
import time

def rtsp_reader_process(shared_images, image_queue):
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

        # 推入處理佇列（不阻塞）
        try:
            image_queue.put_nowait(frame)
        except:
            pass  # queue 滿了就丟掉

        time.sleep(0.01)  # 控制 RTSP 取樣頻率
