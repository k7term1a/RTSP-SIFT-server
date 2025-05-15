import time
import cv2
from sift_matcher import match_sift_with_boxes

def sift_process_worker(image_queue, shared_images, pattern_path, draw_keypoints_flag):
    sift = cv2.SIFT_create()

    while True:
        # 每次都重新載入 pattern，未來可優化
        pattern = cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)
        if pattern is None:
            time.sleep(0.5)
            continue

        try:
            frame = image_queue.get(timeout=1)
        except:
            continue

        processed = match_sift_with_boxes(pattern, frame, draw_keypoints_flag)
        shared_images['processed'] = processed
