import time
import cv2
from sift_matcher import match_sift_with_boxes

def sift_process(shared_images, pattern_path):
    while True:
        pattern_img = cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)
        if pattern_img is None:
            time.sleep(0.5)
            continue

        if 'original' in shared_images:
            frame = shared_images['original']
            processed = match_sift_with_boxes(pattern_img, frame)
            shared_images['processed'] = processed

        time.sleep(0.05)
