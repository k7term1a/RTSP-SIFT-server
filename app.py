from flask import Flask, render_template, request, redirect, url_for, send_file, Response
import os
import cv2
import numpy as np
from multiprocessing import Process, Queue, Value, Manager
from rtsp_worker import rtsp_process

app = Flask(__name__)
UPLOAD_FOLDER = "static"
PATTERN_PATH = os.path.join(UPLOAD_FOLDER, "pattern.png")

# 多處共享影像緩衝區（共享圖片）
manager = Manager()
shared_images = manager.dict()  # {'original': ..., 'processed': ...}

# 啟動背景處理程序
frame_queue = Queue(maxsize=5)
p = Process(target=rtsp_process, args=(frame_queue, shared_images, PATTERN_PATH))
p.daemon = True
p.start()

def generate_stream(img_type):
    while True:
        frame = shared_images.get(img_type)
        if frame is None:
            continue

        # 將 frame 編碼成 JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # 將 JPEG 回傳給瀏覽器
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

RTSP_URL = "rtsp://<username>:<password>@<ip>:<port>/<path>"  # 改成你的攝影機串流網址

def generate_rtsp_stream():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("❌ 無法連線到 RTSP 串流")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ RTSP 串流中斷，重連中...")
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/stream_rtsp')
def stream_rtsp():
    return Response(generate_rtsp_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename != "":
        file.save(PATTERN_PATH)
    return redirect(url_for('index'))

@app.route('/stream/<img_type>')
def stream(img_type):
    return Response(generate_stream(img_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/pattern")
def pattern():
    return send_file(PATTERN_PATH, mimetype='image/png')

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if not os.path.exists(PATTERN_PATH):
        # 預設 pattern
        cv2.imwrite(PATTERN_PATH, 255 * np.ones((100, 100, 3), dtype=np.uint8))
    app.run(host="0.0.0.0", port=8080)
