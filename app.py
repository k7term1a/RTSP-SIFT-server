from flask import Flask, render_template, request, redirect, url_for, send_file, Response
import os
import cv2
import numpy as np
from multiprocessing import Process, Manager, Queue
from rtsp_reader import rtsp_reader_process
from sift_processor import sift_process_worker

app = Flask(__name__)
UPLOAD_FOLDER = "static"
PATTERN_PATH = os.path.join(UPLOAD_FOLDER, "pattern.png")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 如果沒有 pattern.png，就產生一張白圖
if not os.path.exists(PATTERN_PATH):
    blank = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(PATTERN_PATH, blank)

# 建立共享記憶體與佇列
manager = Manager()
shared_images = manager.dict()
image_queue = Queue(maxsize=10)

# 啟動 RTSP 擷取程序
p_rtsp = Process(target=rtsp_reader_process, args=(shared_images, image_queue))
p_rtsp.daemon = True
p_rtsp.start()

# 啟動多個 SIFT Worker（可調整數量）
NUM_WORKERS = 4
for _ in range(NUM_WORKERS):
    p = Process(target=sift_process_worker, args=(image_queue, shared_images, PATTERN_PATH))
    p.daemon = True
    p.start()

# 串流影像產生器
def generate_stream(img_type):
    while True:
        frame = shared_images.get(img_type)
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# 路由設定
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and file.filename != "":
        file.save(PATTERN_PATH)
    return redirect(url_for('index'))

@app.route("/stream/<img_type>")
def stream(img_type):
    return Response(generate_stream(img_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/pattern")
def pattern():
    return send_file(PATTERN_PATH, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
